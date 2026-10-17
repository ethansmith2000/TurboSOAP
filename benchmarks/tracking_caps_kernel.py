"""Pure cap-only complete counterfactuals and a diagnostic-only norm estimator."""
import math
import torch
from residual_retraction import residual_retract_two

def prefix_cap(cov,q,cap):
    scale=(cov.trace().abs()/len(q)).clamp_min(1e-12)
    c=cov.float()/scale
    b=q.float().T@c@q.float();b=.5*(b+b.T)
    diagonal=b.diagonal();gap=diagonal[:,None]-diagonal[None,:]
    tau=diagonal.abs().amax().clamp_min(1e-12)*.01
    omega=-b*gap/(gap.square()+tau.square()).clamp_min(1e-24)
    raw=.5*(.5*(omega-omega.T));bound=raw.norm()/math.sqrt(2)
    a=raw*(cap/bound.clamp_min(1e-12)).clamp(max=1)
    return q.float()+(q.bfloat16()@a.bfloat16()).float()

def cap05(cov,q):return residual_retract_two(prefix_cap(cov,q,.5),True)
def cap2(cov,q):return residual_retract_two(prefix_cap(cov,q,2.),True)
def cap8(cov,q):return residual_retract_two(prefix_cap(cov,q,8.),True)
ENTRIES={'cap05':cap05,'cap2':cap2,'cap8':cap8}

def reference_from_raw(q,raw,bound,cap):
    a=raw*(cap/bound.clamp_min(1e-12)).clamp(max=1)
    y=q.float()+(q.bfloat16()@a.bfloat16()).float()
    return residual_retract_two(y,True)

def power_lower_estimate(a):
    n=len(a);x=torch.linspace(-1.,1.,n,device=a.device)
    vectors=torch.stack((x,torch.cos(torch.arange(n,device=a.device,dtype=a.dtype)*.731)),dim=1)
    vectors=vectors/vectors.norm(dim=0).clamp_min(1e-24)
    for _ in range(8):
        vectors=a.T@(a@vectors)
        vectors=vectors/vectors.norm(dim=0).clamp_min(1e-24)
    return (a@vectors).norm(dim=0).amax()

def geometry(q):
    if not bool(torch.isfinite(q).all()):return dict(finite=False,normalized_gram=None,absolute_gram_fro=None,admitted=False)
    gram=q.T@q;gram.diagonal().sub_(1.)
    absolute=float(gram.norm());normalized=absolute/len(q)**.5
    finite=math.isfinite(absolute)
    return dict(finite=finite,normalized_gram=normalized if finite else None,absolute_gram_fro=absolute if finite else None,
                admitted=finite and normalized<=.005 and absolute<=.05)
