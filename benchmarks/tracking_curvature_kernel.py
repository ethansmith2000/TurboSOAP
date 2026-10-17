"""Pure fixed-cap anchors and diagonal-curvature damped energy update."""
import math
import torch
from residual_retraction import residual_retract_two
from tracking_direction_kernel import raw_direction,prefix_direction,directional_metrics
from tracking_caps_kernel import power_lower_estimate,geometry

def curvature_terms(b,a):
    d=b.diagonal();gap=d[:,None]-d[None,:]
    slope=(2*b*gap*a).sum()
    curvature=2*(gap*a).square().sum()
    return slope,curvature

def generator(b,name):
    raw=raw_direction(b,name!='jacobi');bound=raw.norm()/math.sqrt(2)
    cap_scale=(.5/bound.clamp_min(1e-12)).clamp(max=1)
    scale=cap_scale
    if name=='damped':
        slope,curvature=curvature_terms(b,raw)
        model_scale=(-slope/curvature.clamp_min(1e-24)).clamp(min=0,max=1)
        scale=torch.minimum(cap_scale,model_scale)
    return raw,bound,raw*scale,cap_scale,scale

def prefix_damped(cov,q):
    scale=(cov.trace().abs()/len(q)).clamp_min(1e-12)
    c=cov.float()/scale
    b=q.float().T@c@q.float();b=.5*(b+b.T)
    raw,bound,a,cap_scale,chosen_scale=generator(b,'damped')
    return q.float()+(q.bfloat16()@a.bfloat16()).float()

def jacobi(cov,q):return residual_retract_two(prefix_direction(cov,q,False),True)
def energy(cov,q):return residual_retract_two(prefix_direction(cov,q,True),True)
def damped(cov,q):return residual_retract_two(prefix_damped(cov,q),True)
ENTRIES={'jacobi':jacobi,'energy':energy,'damped':damped}

def reference_from_generator(q,a):
    y=q.float()+(q.bfloat16()@a.bfloat16()).float()
    return residual_retract_two(y,True)

def curvature_metrics(b,raw,a,cap_scale,chosen_scale):
    """Eager-only diagnostics; a diagonal model, not full Hessian or guarantee."""
    raw_slope,raw_h=curvature_terms(b,raw)
    capped=raw*cap_scale;capped_slope,capped_h=curvature_terms(b,capped)
    slope,h=curvature_terms(b,a)
    off=b.clone();off.diagonal().zero_();f=off.square().sum()
    return dict(raw_scale=float(chosen_scale),cap_scale=float(cap_scale),
        damping_alpha=float(chosen_scale/cap_scale.clamp_min(1e-24)),
        raw_directional_slope=float(raw_slope),raw_diagonal_curvature=float(raw_h),
        capped_directional_slope=float(capped_slope),capped_diagonal_curvature=float(capped_h),
        capped_model_optimal_alpha=float((-capped_slope/capped_h.clamp_min(1e-24)).clamp(min=0,max=1)),
        applied_diagonal_curvature=float(h),
        predicted_quadratic_fractional_energy_decrease=float((-slope-.5*h)/f.clamp_min(1e-24)),
        curvature_scope='Diagonal-only quadratic surrogate; omitted offdiag commutator terms, never a descent certificate')
