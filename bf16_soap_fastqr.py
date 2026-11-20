"""Refresh variants for the BF16 compiled SOAP (fast-QR study, 2026-11-20).

``bf16_soap.py`` is imported, never edited. ``FastRefreshSOAP`` subclasses ``CompiledSOAP`` and only
changes which function runs at a refresh event:

refresh kinds at QR events (``qr_kind``)
  'householder'  production ``qr_basis`` (FP32 Householder island). Default.
  'cholqr'       C1 of research/reviews/panel_20261120/CANDIDATES.md: Gram-shifted CholeskyQR3 with no
                 operator ridge. Same Z = C Q, same descending Rayleigh-quotient sort and the same V
                 permutation as ``qr_basis``; then one Cholesky pass on Z^T Z + s I with
                 s = shift_rel * lambda_max(Z^T Z) (power iteration) and ``unshifted_passes`` plain
                 CholeskyQR passes. Every FP32 product in the event region runs as IEEE FP32
                 (``allow_tf32`` is switched off around the compiled call only). A factor whose Cholesky
                 reports ``info != 0`` in any pass, or whose output is non-finite, falls back to Householder
                 QR of the same sorted Z, and its moment is re-transported. Fallbacks are counted.
warm step (``warm_ns_steps``)
  2   production ``warm_basis`` (two Newton-Schulz corrections). Default.
  1   C2: one Newton-Schulz correction.
schedules
  ``precondition_frequency`` K and ``hard_reset_interval`` k as in production: a refresh every K ages, QR
  (of ``qr_kind``) when the age is a multiple of k, otherwise warm. k == K gives QR-K with no warm step.

With the defaults every event dispatches to the production compiled functions themselves
(``bf16_soap.ordinary_group`` / ``warm_group`` / ``qr_group``), so the default optimizer is production by
construction; ``benchmarks/fastqr_checks.py`` also checks it bit for bit.
"""
from functools import partial
import math

import torch

from bf16_soap import CompiledSOAP, covariance_ema, mm32, transport_moment


QR_KINDS = ('householder', 'cholqr')
CHOLQR_DEFAULTS = dict(shift_rel=1e-6, unshifted_passes=2, power_iters=8)


# ----------------------------------------------------------------------------- basis functions
def warm_basis_ns(cov, q, compute_dtype, basis_dtype, ns_steps=2):
    """``bf16_soap.warm_basis`` verbatim, with the Newton-Schulz count as a parameter (2 = production)."""
    qb, cb = q.to(compute_dtype), cov.to(compute_dtype)
    b = mm32(torch.bmm(qb.transpose(-1,-2), cb), qb)
    b = .5*(b+b.transpose(-1,-2))
    diagonal = b.diagonal(dim1=-2,dim2=-1)
    scale = diagonal.abs().amax(-1,keepdim=True).clamp_min(1e-30)
    gap = (diagonal.unsqueeze(-1)-diagonal.unsqueeze(-2))/scale.unsqueeze(-1)
    a = -.5*(b/scale.unsqueeze(-1))*gap/(gap.square()+.01**2)
    bound = torch.linalg.vector_norm(a,dim=(-2,-1),keepdim=True)/math.sqrt(2)
    a = a*(.5/bound.clamp_min(1e-12)).clamp(max=1)
    y = (q.float()+mm32(qb,a.to(compute_dtype))).to(basis_dtype)
    for _ in range(ns_steps):
        if basis_dtype == torch.bfloat16:
            gram = mm32(y.transpose(-1,-2), y)
        else:
            gram = torch.bmm(y.transpose(-1,-2), y)
        eye = torch.eye(y.shape[-1],device=y.device,dtype=gram.dtype)
        residual = gram-eye
        y = (y.float()-.5*mm32(y.to(compute_dtype),residual.to(compute_dtype))).to(basis_dtype)
    return y.contiguous()


def sorted_projection(cov, q):
    """The first lines of ``bf16_soap.qr_basis``: FP32 Z = C Q, columns sorted by descending q_j^T C q_j."""
    c, old = cov.float(), q.float()
    z = torch.bmm(c, old)
    order = (old*z).sum(-2).argsort(dim=-1, descending=True)
    ordered = z.gather(-1, order.unsqueeze(-2).expand_as(z))
    return ordered, order


def cholqr3(x, shift_rel=1e-6, unshifted_passes=2, power_iters=8):
    """Gram-shifted CholeskyQR3 of FP32 x (b, m, n) -> (FP32 Q, ok[b]).

    Run with IEEE FP32 matmuls (TF32 Grams caused the earlier failures). Pass 1 factors
    X^T X + s I with s = shift_rel * lambda_max(X^T X); the remaining passes are unshifted. A batch element
    whose Cholesky fails keeps an identity factor from then on, so the graph stays finite; ``ok`` marks it."""
    b, n = x.shape[0], x.shape[-1]
    eye = torch.eye(n, device=x.device, dtype=torch.float32)
    ok = torch.ones(b, dtype=torch.bool, device=x.device)
    g = torch.bmm(x.transpose(-1, -2), x)
    v = torch.full((b, n, 1), 1/math.sqrt(n), device=x.device, dtype=torch.float32)
    for _ in range(power_iters):
        v = torch.bmm(g, v)
        v = v/v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
    lam = torch.bmm(v.transpose(-1, -2), torch.bmm(g, v))
    for k in range(1+unshifted_passes):
        if k:
            g = torch.bmm(x.transpose(-1, -2), x)
            a = g
        else:
            a = g+(shift_rel*lam)*eye
        u, info = torch.linalg.cholesky_ex(a, upper=True)
        ok = ok & (info == 0)
        u = torch.where(ok.view(-1, 1, 1), u, eye)
        x = torch.linalg.solve_triangular(u, x, upper=True, left=False)
    ok = ok & torch.isfinite(x).flatten(1).all(1)
    return x, ok


def cholqr_basis(cov, q, basis_dtype, shift_rel=1e-6, unshifted_passes=2, power_iters=8):
    """C1 refresh: ``qr_basis`` with Householder replaced by CholQR3. Returns (Q, order, ok)."""
    ordered, order = sorted_projection(cov, q)
    x, ok = cholqr3(ordered, shift_rel, unshifted_passes, power_iters)
    return x.to(basis_dtype).contiguous(), order, ok


def householder_from_order(cov, q, order, basis_dtype):
    """Fallback: Householder QR of the sorted Z for a given column order (the one V was permuted by)."""
    z = torch.bmm(cov.float(), q.float())
    ordered = z.gather(-1, order.unsqueeze(-2).expand_as(z))
    return torch.linalg.qr(ordered).Q.to(basis_dtype).contiguous()


# ----------------------------------------------------------------------------- event functions
def fast_matrix_step(p, g, state, age, lr, *, beta1, beta2, cov_beta, eps, weight_decay, event,
                     compute_dtype, basis_dtype, warm_ns_steps, shift_rel, unshifted_passes, power_iters):
    """``bf16_soap.matrix_step`` for the two non-production events, 'warm' (any NS count) and 'cholqr'.

    The ordinary part is copied verbatim. 'cholqr' also returns the pre-transport moment, the column
    orders and the ok flags, which the host-side fallback needs."""
    m,v,cl,cr,ql,qr = state
    gb = g.to(compute_dtype)
    projected = torch.bmm(ql.to(compute_dtype).transpose(-1,-2),gb)
    projected = torch.bmm(projected,qr.to(compute_dtype))
    mf = beta1*m.float()+(1-beta1)*projected.float()
    vf = beta2*v.float()+(1-beta2)*projected.float().square()
    u = (mf/(vf.sqrt()+eps)).to(compute_dtype)
    u = torch.bmm(ql.to(compute_dtype),u)
    u = mm32(u,qr.to(compute_dtype).transpose(-1,-2))
    correction = torch.sqrt(1-beta2**age)/(1-beta1**age)
    new_p = (p-lr*correction*u.float())*(1-lr*weight_decay)
    new_m,new_v = mf.to(m.dtype),vf.to(v.dtype)
    new_cl = covariance_ema(cl,gb,cov_beta)
    new_cr = covariance_ema(cr,gb.transpose(-1,-2),cov_beta)
    extras = ()
    if event == 'warm':
        new_ql = warm_basis_ns(new_cl,ql,compute_dtype,basis_dtype,warm_ns_steps)
        new_qr = warm_basis_ns(new_cr,qr,compute_dtype,basis_dtype,warm_ns_steps)
    elif event == 'cholqr':
        new_ql,left_order,left_ok = cholqr_basis(new_cl,ql,basis_dtype,shift_rel,unshifted_passes,power_iters)
        new_qr,right_order,right_ok = cholqr_basis(new_cr,qr,basis_dtype,shift_rel,unshifted_passes,power_iters)
        new_v = new_v.gather(-2,left_order.unsqueeze(-1).expand_as(new_v))
        new_v = new_v.gather(-1,right_order.unsqueeze(-2).expand_as(new_v))
        extras = (new_m,left_order,right_order,left_ok,right_ok)
    else:
        raise ValueError(event)
    transported = transport_moment(new_m,ql,qr,new_ql,new_qr,compute_dtype)
    return new_p,(transported,new_v,new_cl,new_cr,new_ql,new_qr),extras


def _fast_group(params, grads, state, age, lr, *, event, **config):
    p,g = torch.stack(params),torch.stack(grads)
    new_p,new_state,extras = fast_matrix_step(p,g,state,age,lr,event=event,**config)
    for parameter,value in zip(params,new_p.unbind(0)):
        parameter.copy_(value)
    return new_state,extras


def warm_ns_group(params, grads, state, age, lr, **config):
    return _fast_group(params,grads,state,age,lr,event='warm',**config)[0]


def cholqr_group(params, grads, state, age, lr, **config):
    return _fast_group(params,grads,state,age,lr,event='cholqr',**config)


# ----------------------------------------------------------------------------- optimizer
class FastRefreshSOAP(CompiledSOAP):
    """``CompiledSOAP`` with a selectable QR-event kind and warm-step Newton-Schulz count.

    Bookkeeping (CholQR only): ``refresh_log`` has one row per QR event (call, factor refreshes, fallbacks);
    ``fallback_log`` one row per fallen-back factor (call, matrix shape, side, batch index)."""
    def __init__(self, params, *, qr_kind='householder', warm_ns_steps=2, cholqr_shift_rel=1e-6,
                 cholqr_unshifted_passes=2, cholqr_power_iters=8, **kwargs):
        if qr_kind not in QR_KINDS:raise ValueError(f'qr_kind must be one of {QR_KINDS}')
        if type(warm_ns_steps) is not int or warm_ns_steps<0:raise ValueError('Invalid Newton-Schulz count')
        if not math.isfinite(cholqr_shift_rel) or cholqr_shift_rel<0:raise ValueError('Invalid shift')
        if type(cholqr_unshifted_passes) is not int or cholqr_unshifted_passes<0:raise ValueError('Invalid passes')
        if type(cholqr_power_iters) is not int or cholqr_power_iters<1:raise ValueError('Invalid power iterations')
        super().__init__(params,**kwargs)
        self.qr_kind=qr_kind;self.warm_ns_steps=warm_ns_steps
        self.cholqr_config=dict(shift_rel=cholqr_shift_rel,unshifted_passes=cholqr_unshifted_passes,
                                power_iters=cholqr_power_iters)
        self.refresh_log=[];self.fallback_log=[];self._pending=[]

    @property
    def is_production(self):
        return self.qr_kind=='householder' and self.warm_ns_steps==2

    def refresh_counts(self):
        return dict(cholqr_events=len(self.refresh_log),
                    cholqr_factor_refreshes=sum(r['factors'] for r in self.refresh_log),
                    fallback_factors=sum(r['fallbacks'] for r in self.refresh_log))

    def _function(self,group,event):
        if event=='qr' and self.qr_kind=='cholqr':
            return self._fast_function(group,'cholqr')
        if event=='warm' and self.warm_ns_steps!=2:
            return self._fast_function(group,'warm')
        return super()._function(group,event)

    def _fast_function(self,group,event):
        key=('fast',event,group['betas'],group['shampoo_beta'],group['eps'],group['weight_decay'],
             group['state_dtype'],group['basis_dtype'],group['compute_dtype'],self.warm_ns_steps,
             tuple(sorted(self.cholqr_config.items())))
        if key not in self._functions:
            fn={'warm':warm_ns_group,'cholqr':cholqr_group}[event]
            config=dict(beta1=group['betas'][0],beta2=group['betas'][1],
                cov_beta=group['shampoo_beta'],eps=group['eps'],weight_decay=group['weight_decay'],
                compute_dtype=group['compute_dtype'],basis_dtype=group['basis_dtype'],
                warm_ns_steps=self.warm_ns_steps,**self.cholqr_config)
            if self.compile_kernels:
                options={'emulate_precision_casts':True}
                if group['params'][0].device.type=='cpu':options['pattern_matcher']=False
                fn=torch.compile(fn,fullgraph=True,dynamic=False,options=options)
            fn=partial(fn,**config)
            if event=='cholqr':fn=partial(self._cholqr_call,fn,group['compute_dtype'])
            self._functions[key]=fn
        return self._functions[key]

    def _cholqr_call(self,fn,compute_dtype,params,grads,state,age,lr):
        # IEEE FP32 for every FP32 product of this event region (Z, Grams, power iterations, solves);
        # cuBLAS reads the flag at call time, so it cannot be switched inside the compiled graph.
        previous=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False
        try:
            new_state,extras=fn(params,grads,state,age,lr)
        finally:
            torch.backends.cuda.matmul.allow_tf32=previous
        self._pending.append(dict(params=params,old_ql=state[4],old_qr=state[5],extras=extras,
                                  compute_dtype=compute_dtype))
        return new_state

    @torch.no_grad()
    def step(self,closure=None):
        self._pending=[]
        try:
            super().step(closure)
            if self._pending:self._resolve_cholqr()
        finally:
            self._pending=[]

    @torch.no_grad()
    def _resolve_cholqr(self):
        flags=torch.cat([torch.cat([e['extras'][3],e['extras'][4]]) for e in self._pending])
        failures=int((~flags).sum())  # the one host synchronization per QR event
        self.refresh_log.append(dict(call=self.call,factors=flags.numel(),fallbacks=failures))
        if not failures:return
        previous=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False
        try:
            for entry in self._pending:
                pre_m,left_order,right_order,left_ok,right_ok=entry['extras']
                if bool(left_ok.all()) and bool(right_ok.all()):continue
                params=entry['params'];s=self.state[params[0]]
                touched=torch.zeros_like(left_ok)
                for side,ok,ckey,qkey,old,order in ((0,left_ok,'cl','ql',entry['old_ql'],left_order),
                                                    (1,right_ok,'cr','qr',entry['old_qr'],right_order)):
                    bad=(~ok).nonzero().flatten()
                    if not bad.numel():continue
                    s[qkey][bad]=householder_from_order(s[ckey][bad],old[bad],order[bad],s[qkey].dtype)
                    touched[bad]=True
                    for i in bad.tolist():
                        self.fallback_log.append(dict(call=self.call,shape=list(params[0].shape),side='LR'[side],index=i))
                idx=touched.nonzero().flatten()
                s['m'][idx]=transport_moment(pre_m[idx],entry['old_ql'][idx],entry['old_qr'][idx],
                                             s['ql'][idx],s['qr'][idx],entry['compute_dtype'])
        finally:
            torch.backends.cuda.matmul.allow_tf32=previous

    def state_dict(self):
        result=super().state_dict()
        result['fast_refresh_meta']=dict(qr_kind=self.qr_kind,warm_ns_steps=self.warm_ns_steps,**self.cholqr_config)
        return result

    def load_state_dict(self,saved):
        meta=saved.get('fast_refresh_meta')
        if meta!=dict(qr_kind=self.qr_kind,warm_ns_steps=self.warm_ns_steps,**self.cholqr_config):
            raise ValueError('Refresh-variant mismatch')
        super().load_state_dict(saved)


class FrontLoadedSOAP(CompiledSOAP):
    """C6 front-loaded schedule (panel PB7): a Householder QR refresh at every ``precondition_frequency``-th age up
    to and including ``qr_until``, then the base schedule on the global age grid (warm every frequency, QR every
    ``hard_reset_interval``), so the late QR repairs still land on multiples of the repair interval.

    Only the event choice changes. QR-phase events run production's own ``qr`` event function, so a run reproduces
    ``CompiledSOAP(precondition_frequency=f, hard_reset_interval=f)`` bit for bit until the first refresh after
    ``qr_until``.
    """

    def __init__(self, params, *, qr_until, **kwargs):
        super().__init__(params, **kwargs)
        if type(qr_until) is not int or qr_until < 0 or qr_until % self.frequency:
            raise ValueError('qr_until must be a non-negative multiple of precondition_frequency')
        self.qr_until = qr_until
        self.late_repair = self.repair

    @torch.no_grad()
    def step(self, closure=None):
        next_age = self.call  # the base step increments call, then uses age = call - 1
        self.repair = self.frequency if next_age <= self.qr_until else self.late_repair
        try:
            return super().step(closure)
        finally:
            self.repair = self.late_repair  # state_dict metadata always reports the late schedule

    def state_dict(self):
        result = super().state_dict()
        result['front_loaded_meta'] = dict(qr_until=self.qr_until)
        return result

    def load_state_dict(self, saved):
        if saved.get('front_loaded_meta') != dict(qr_until=self.qr_until):
            raise ValueError('Front-loaded schedule mismatch')
        super().load_state_dict(saved)
