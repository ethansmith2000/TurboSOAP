"""Experimental batched SOAP with explicit storage/compute precision.

The tested FP32 ResearchSOAP/FastSOAP implementations are untouched. Each
compiled region covers packing, ordinary SOAP, optional basis refresh, moment
transport and FP32 master-parameter writes for one repeated parameter shape.
BF16 storage changes rounding/history and requires separate learning validation.
"""
from collections import OrderedDict
from functools import partial
import math

import torch


STATE_KEYS = ('m', 'v', 'cl', 'cr', 'ql', 'qr')


def mm32(a, b):
    """BF16 tensor-core operands with FP32 accumulation/output on CUDA."""
    if a.dtype == torch.bfloat16:
        if a.is_cuda:
            return torch.bmm(a, b, out_dtype=torch.float32)
        return torch.bmm(a.double(), b.double()).float()
    return torch.bmm(a, b)


def covariance_ema(old, g, beta):
    if old.dtype == g.dtype:
        # BF16 persistent input/output; cuBLAS accumulates the product in FP32.
        return torch.baddbmm(old, g, g.transpose(-1, -2), beta=beta, alpha=1-beta)
    product = mm32(g, g.transpose(-1, -2))
    return (beta*old.float()+(1-beta)*product).to(old.dtype)


def warm_basis(cov, q, compute_dtype, basis_dtype):
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
    for _ in range(2):
        if basis_dtype == torch.bfloat16:
            gram = mm32(y.transpose(-1,-2), y)
        else:
            # Optional precision island: measure the actual FP32 basis Gram.
            gram = torch.bmm(y.transpose(-1,-2), y)
        eye = torch.eye(y.shape[-1],device=y.device,dtype=gram.dtype)
        residual = gram-eye
        y = (y.float()-.5*mm32(y.to(compute_dtype),residual.to(compute_dtype))).to(basis_dtype)
    return y.contiguous()


def qr_basis(cov, q, basis_dtype):
    # FP32 decomposition island, with the nonredundant CQ/Rayleigh contraction.
    c, old = cov.float(), q.float()
    z = torch.bmm(c,old)
    order = (old*z).sum(-2).argsort(dim=-1,descending=True)
    ordered = z.gather(-1,order.unsqueeze(-2).expand_as(z))
    new = torch.linalg.qr(ordered).Q.to(basis_dtype).contiguous()
    return new,order


def transport_moment(m, old_l, old_r, new_l, new_r, compute_dtype):
    world = torch.bmm(old_l.to(compute_dtype),m.to(compute_dtype))
    world = torch.bmm(world,old_r.to(compute_dtype).transpose(-1,-2))
    projected = torch.bmm(new_l.to(compute_dtype).transpose(-1,-2),world)
    if m.dtype == torch.float32:
        return mm32(projected,new_r.to(compute_dtype)).float()
    return torch.bmm(projected,new_r.to(compute_dtype)).to(m.dtype)


def matrix_step(p, g, state, age, lr, *, beta1, beta2, cov_beta, eps,
                weight_decay, event, compute_dtype, basis_dtype):
    """Pure tensor transition; FP32 pointwise temporaries can fuse into registers.

    Updated unrounded moment temporaries compute this call's direction, while
    BF16 quantized moments persist into the next call. This is an explicit
    low-precision-state convention, not bitwise upstream SOAP equivalence.
    """
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
    if event == 'warm':
        new_ql = warm_basis(new_cl,ql,compute_dtype,basis_dtype)
        new_qr = warm_basis(new_cr,qr,compute_dtype,basis_dtype)
    elif event == 'qr':
        new_ql,left_order = qr_basis(new_cl,ql,basis_dtype)
        new_qr,right_order = qr_basis(new_cr,qr,basis_dtype)
        new_v = new_v.gather(-2,left_order.unsqueeze(-1).expand_as(new_v))
        new_v = new_v.gather(-1,right_order.unsqueeze(-2).expand_as(new_v))
    else:
        new_ql,new_qr = ql,qr
    if event != 'ordinary':
        new_m = transport_moment(new_m,ql,qr,new_ql,new_qr,compute_dtype)
    return new_p,(new_m,new_v,new_cl,new_cr,new_ql,new_qr)


def group_step(params, grads, state, age, lr, *, event, **config):
    p,g = torch.stack(params),torch.stack(grads)
    new_p,new_state = matrix_step(p,g,state,age,lr,event=event,**config)
    for parameter,value in zip(params,new_p.unbind(0)):
        parameter.copy_(value)
    return new_state


def ordinary_group(params,grads,state,age,lr,**config):
    return group_step(params,grads,state,age,lr,event='ordinary',**config)


def warm_group(params,grads,state,age,lr,**config):
    return group_step(params,grads,state,age,lr,event='warm',**config)


def qr_group(params,grads,state,age,lr,**config):
    return group_step(params,grads,state,age,lr,event='qr',**config)


class CompiledSOAP(torch.optim.Optimizer):
    """Dense two-sided, same-shape batched candidate; FP32 master weights.

    Default: M/V/C/Q stored in BF16, BF16 matrix products, FP32 accumulation,
    FP32 pointwise math and decomposition islands. No padding/blocking or
    one-sided coverage. Missing gradients and per-parameter asynchronous ages
    are deliberately unsupported. Auxiliary AdamW remains separate.
    """
    def __init__(self,params,*,lr=.0015,betas=(.95,.95),shampoo_beta=.95,
                 eps=1e-8,weight_decay=.05,precondition_frequency=10,
                 hard_reset_interval=50,state_dtype=torch.bfloat16,
                 basis_dtype=torch.bfloat16,compute_dtype=torch.bfloat16,
                 compile_kernels=True):
        if state_dtype not in (torch.float32,torch.bfloat16) or basis_dtype not in (torch.float32,torch.bfloat16):
            raise ValueError('Storage dtype must be FP32 or BF16')
        if compute_dtype not in (torch.float32,torch.bfloat16):raise ValueError('Compute dtype must be FP32 or BF16')
        if len(betas)!=2 or any(not math.isfinite(b) or not 0<=b<1 for b in betas):raise ValueError('Invalid betas')
        if not math.isfinite(shampoo_beta) or not 0<=shampoo_beta<1:raise ValueError('Invalid covariance beta')
        if any(not math.isfinite(x) for x in (lr,eps,weight_decay)) or lr<0 or eps<=0 or weight_decay<0:
            raise ValueError('Invalid scalar hyperparameters')
        if type(precondition_frequency) is not int or precondition_frequency<1:raise ValueError('Invalid cadence')
        if type(hard_reset_interval) is not int or hard_reset_interval<0 or hard_reset_interval%precondition_frequency:raise ValueError('Invalid repair cadence')
        original=list(params)
        if original and not isinstance(original[0],dict):original=[dict(params=original)]
        grouped=OrderedDict();seen=set()
        for group in original:
            for key,expected in dict(betas=tuple(betas),shampoo_beta=shampoo_beta,eps=eps,
                    state_dtype=state_dtype,basis_dtype=basis_dtype,compute_dtype=compute_dtype).items():
                if key in group and group[key]!=expected:
                    raise ValueError('Per-group precision/memory overrides are unsupported')
            values=list(group['params']);names=group.get('param_names',[str(i) for i in range(len(values))])
            if len(values)!=len(names):raise ValueError('Parameter names do not match')
            for name,p in zip(names,values):
                if p.ndim!=2 or p.dtype!=torch.float32 or not p.is_contiguous():raise ValueError('Require contiguous 2D FP32 master parameters')
                if id(p) in seen:raise ValueError('Duplicate parameter')
                seen.add(id(p))
                rate=group.get('lr',lr);decay=group.get('weight_decay',weight_decay)
                peak=group.get('peak_lr',rate)
                if any(not math.isfinite(x) or x<0 for x in (rate,decay,peak)):
                    raise ValueError('Invalid group hyperparameters')
                key=(p.shape,p.device,rate,decay,peak)
                target=grouped.setdefault(key,dict(params=[],param_names=[],lr=rate,
                         peak_lr=peak,weight_decay=decay))
                target['params'].append(p);target['param_names'].append(name)
        defaults=dict(lr=lr,betas=tuple(betas),shampoo_beta=shampoo_beta,eps=eps,
                      weight_decay=weight_decay,state_dtype=state_dtype,basis_dtype=basis_dtype,
                      compute_dtype=compute_dtype)
        super().__init__(list(grouped.values()),defaults)
        self.frequency=precondition_frequency;self.repair=hard_reset_interval;self.call=0
        self.compile_kernels=compile_kernels
        self._functions={}
        self.last_event=None

    def _function(self,group,event):
        key=(event,group['betas'],group['shampoo_beta'],group['eps'],group['weight_decay'],
             group['state_dtype'],group['basis_dtype'],group['compute_dtype'])
        if key not in self._functions:
            fn={'ordinary':ordinary_group,'warm':warm_group,'qr':qr_group}[event]
            config=dict(beta1=group['betas'][0],beta2=group['betas'][1],
                cov_beta=group['shampoo_beta'],eps=group['eps'],weight_decay=group['weight_decay'],
                compute_dtype=group['compute_dtype'],basis_dtype=group['basis_dtype'])
            if self.compile_kernels:
                options={'emulate_precision_casts':True}
                if group['params'][0].device.type=='cpu':options['pattern_matcher']=False
                fn=torch.compile(fn,fullgraph=True,dynamic=False,options=options)
            # Compile the distinct event functions themselves. Compiling a
            # partial first funnels all 12 event/shape specializations through
            # Dynamo's shared external_utils.inner (default cache limit 8).
            self._functions[key]=partial(fn,**config)
        return self._functions[key]

    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:raise ValueError('Closures unsupported')
        if torch.is_autocast_enabled():raise ValueError('Run optimizer outside autocast')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.grad.is_sparse:raise ValueError('Every matrix needs a dense gradient')
        self.call+=1
        age=self.call-1
        event='cold' if not age else 'ordinary' if age%self.frequency else 'qr' if self.repair and age%self.repair==0 else 'warm'
        self.last_event=event
        for group in self.param_groups:
            params=tuple(group['params']);grads=tuple(p.grad for p in params)
            s=self.state[params[0]]
            if not s:
                batch=len(params);m,n=params[0].shape;device=params[0].device
                dtype=group['state_dtype'];compute=group['compute_dtype'];basis=group['basis_dtype']
                s['m']=torch.zeros(batch,m,n,device=device,dtype=dtype);s['v']=torch.zeros_like(s['m'])
                g=torch.stack(grads).to(compute)
                s['cl']=covariance_ema(torch.zeros(batch,m,m,device=device,dtype=dtype),g,group['shampoo_beta'])
                s['cr']=covariance_ema(torch.zeros(batch,n,n,device=device,dtype=dtype),g.transpose(-1,-2),group['shampoo_beta'])
                for c,q in [('cl','ql'),('cr','qr')]:
                    matrix=s[c].float();matrix=.5*(matrix+matrix.transpose(-1,-2))
                    s[q]=torch.linalg.eigh(matrix).eigenvectors.flip(-1).to(basis).contiguous()
                s['age']=torch.zeros((),device=device)
                s['lr']=torch.tensor(group['lr'],device=device)
                continue
            s['age'].fill_(age);s['lr'].fill_(group['lr'])
            output=self._function(group,event)(params,grads,tuple(s[k] for k in STATE_KEYS),s['age'],s['lr'])
            for name,value in zip(STATE_KEYS,output):s[name]=value

    def state_bytes(self):
        return sum(t.numel()*t.element_size() for s in self.state.values() for t in s.values() if isinstance(t,torch.Tensor))

    def state_dict(self):
        result=super().state_dict()
        result['compiled_soap_meta']=dict(call=self.call,frequency=self.frequency,repair=self.repair,version=1)
        return result

    def load_state_dict(self,saved):
        meta=saved.get('compiled_soap_meta')
        if not meta or meta['version']!=1 or meta['frequency']!=self.frequency or meta['repair']!=self.repair:
            raise ValueError('State policy mismatch')
        if len(saved['param_groups'])!=len(self.param_groups):raise ValueError('State group mismatch')
        for previous,current in zip(saved['param_groups'],self.param_groups):
            for name in ('state_dtype','basis_dtype','compute_dtype','betas','shampoo_beta'):
                if previous[name]!=current[name]:raise ValueError('State precision/configuration mismatch')
        super().load_state_dict(saved)
        self.call=meta['call'];self._functions.clear()
        for group in self.param_groups:
            s=self.state[group['params'][0]]
            for name in STATE_KEYS:
                if name in s:s[name]=s[name].to(group['basis_dtype'] if name in ('ql','qr') else group['state_dtype'])

