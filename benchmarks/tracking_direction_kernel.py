"""Pure complete direction entries; diagnostic stages are eager-only."""
import math
import torch
from residual_retraction import residual_retract_two
from tracking_caps_kernel import power_lower_estimate,geometry,reference_from_raw

def raw_direction(b,energy):
    diagonal=b.diagonal();gap=diagonal[:,None]-diagonal[None,:]
    if energy:
        omega=-b*gap
    else:
        tau=diagonal.abs().amax().clamp_min(1e-12)*.01
        omega=-b*gap/(gap.square()+tau.square()).clamp_min(1e-24)
    return .5*(.5*(omega-omega.T))

def prefix_direction(cov,q,energy):
    scale=(cov.trace().abs()/len(q)).clamp_min(1e-12)
    c=cov.float()/scale
    b=q.float().T@c@q.float();b=.5*(b+b.T)
    raw=raw_direction(b,energy);bound=raw.norm()/math.sqrt(2)
    a=raw*(.5/bound.clamp_min(1e-12)).clamp(max=1)
    return q.float()+(q.bfloat16()@a.bfloat16()).float()

def jacobi(cov,q):return residual_retract_two(prefix_direction(cov,q,False),True)
def energy(cov,q):return residual_retract_two(prefix_direction(cov,q,True),True)
ENTRIES={'jacobi':jacobi,'energy':energy}

def directional_metrics(b,a):
    """Local derivative for Q exp(tA), not a guarantee for a finite retraction."""
    d=b.diagonal();gradient=2*b*(d[:,None]-d[None,:])
    slope=(gradient*a).sum()
    off=b.clone();off.diagonal().zero_();f=off.square().sum()
    alignment=-slope/(gradient.norm()*a.norm()).clamp_min(1e-24)
    return dict(directional_slope=float(slope),energy_before=float(f),
        predicted_fractional_energy_decrease=float(-slope/f.clamp_min(1e-24)),
        cosine_to_negative_energy_gradient=float(alignment),
        derivative_scope='First-order derivative on exponential curve; finite mixed retraction measured separately')
