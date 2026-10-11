"""Read-only large-factor diagnostics, without a second full QR refresh."""
import math
import torch
from soap import SOAP, _damped_jacobi_generator
try:
    from .cifar_tracking_probe import TrackingProbe, project, world, relative, factor_score
except ImportError:
    from cifar_tracking_probe import TrackingProbe, project, world, relative, factor_score


@torch.no_grad()
def norm_proxy(a, iterations=12):
    """Deterministic lower proxy for ||A||2; does not consume global RNG."""
    g = torch.Generator(device=a.device).manual_seed(9100+a.shape[0])
    x = torch.randn(a.shape[1], device=a.device, dtype=torch.float64, generator=g)
    x /= x.norm().clamp_min(1e-30)
    b = a.double()
    for _ in range(iterations):
        x = b.T@(b@x)
        x /= x.norm().clamp_min(1e-30)
    return float((b@x).norm())


@torch.no_grad()
def selected_checks(old, actual, eps):
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        old_world = world(old['m'], old['q'])
        expected = project(old_world, actual['Q'])
        denominator = actual['exp_avg_sq'].sqrt()+eps
        scores = [factor_score(c,q) for c,q in zip(old['cov'],actual['Q'])]
        d = dict(selected_basis_update_error=relative(world(actual['exp_avg']/denominator,actual['Q']),
                     world(expected/denominator,actual['Q'])),
                 selected_basis_m_error=relative(actual['exp_avg'],expected),
                 world_m_preservation_error=relative(world(actual['exp_avg'],actual['Q']),old_world),
                 maximum_gram=max(s['gram'] for s in scores))
        if not all(math.isfinite(x) for x in d.values()):
            raise FloatingPointError('Nonfinite selected-basis diagnostic')
        d['targeted_pass'] = (d['selected_basis_update_error']<1e-6 and
            d['world_m_preservation_error']<.01 and d['maximum_gram']<.05)
        d['full_reference_collected'] = False
        return d, scores
    finally:
        torch.set_float32_matmul_precision(previous)


@torch.no_grad()
def factor_diagnostic(cov, old_q, new_q, group, after_score, axis):
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        before = factor_score(cov,old_q)
        torch.set_float32_matmul_precision('high')  # Match the executed generator math.
        b = SOAP._basis_covariance(cov,old_q)
        a = group['basis_lr']*_damped_jacobi_generator(b,group['basis_jacobi_damping'])
        bound = float(a.norm()/2**.5)
        proxy = norm_proxy(a)
        diag = b.diagonal().sort().values
        gaps = (diag[1:]-diag[:-1]).abs()
        row = dict(axis=axis, dimension=old_q.shape[0], generator_precision='high',
            raw_frobenius_bound=bound, raw_rms=float(a.norm()/old_q.shape[0]**.5),
            hypothetical_fro_cap=.5, clipping_scale=min(1.,.5/max(bound,1e-30)),
            cap_binding=bound>.5, operator_norm_lower_proxy_12_iterations=proxy,
            bound_over_lower_proxy=None if proxy==0 else bound/proxy,
            clipped_operator_norm_lower_proxy=proxy*min(1.,.5/max(bound,1e-30)),
            covariance_trace_per_dimension=float(cov.diagonal().mean()),
            covariance_frobenius=float(cov.norm()),
            rayleigh_diagonal_abs_max=float(diag.abs().max()),
            rayleigh_adjacent_gap_median=float(gaps.median()),
            rayleigh_adjacent_gap_min=float(gaps.min()),
            scores=dict(before=before,executed=after_score),
            unaligned_basis_movement_rms=float((new_q-old_q).norm()/old_q.shape[0]**.5))
        for key,value in row.items():
            if isinstance(value,float) and not math.isfinite(value):
                raise FloatingPointError('Nonfinite generator diagnostic: '+key)
        return row
    finally:
        torch.set_float32_matmul_precision(previous)


class WideTransportProbe(TrackingProbe):
    @torch.no_grad()
    def flush(self):
        rows=[]
        try:
            names={id(self.opt.state[p]):(name,p) for g in self.opt.param_groups
                   for p,name in zip(g['params'],g['param_names'])}
            for state,group,old in self.pending:
                if any(q is None for q in old['q']):
                    raise ValueError('Wide pilot requires both factors')
                d,scores=selected_checks(old,state,group['eps'])
                if not d['targeted_pass']:
                    raise FloatingPointError('Wide selected-basis/physical-M/Gram gate failed')
                name,p=names[id(state)]
                reset=group['basis_method']=='qr' or (group['hard_reset_interval'] and
                    state['step']%group['hard_reset_interval']==0)
                factors=[factor_diagnostic(cov,q,state['Q'][axis],group,scores[axis],axis)
                         for axis,(cov,q) in enumerate(zip(old['cov'],old['q']))]
                rows.append(dict(matrix=name,age=state['step'],shape=list(old['m'].shape),
                    executed_method='qr' if reset else 'warm',gradient_rms=float(p.grad.float().square().mean().sqrt()),
                    factors=factors,**d))
        finally:
            self.pending.clear()
        return rows
