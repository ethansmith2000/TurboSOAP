"""Read-only tangent energy derivatives for symmetric B and skew A.

The exact curvature describes exp(tA), not a finite mixed-precision update.
No live optimizer imports this diagnostic module.
"""
import torch


def offdiag(x):
    return x-torch.diag(x.diagonal())


def explicit_commutator_terms(b,a):
    x=b@a-a@b
    y=x@a-a@x
    e=offdiag(b)
    return 2*(e*x).sum(),2*(offdiag(x).square().sum()+(e*y).sum())


def single_product_terms(b,a):
    """One extra dense product; valid when B symmetric and A skew.

    X=[B,A]=BA+(BA).T. Trace(B(t)^2) is constant. Diagonal
    drift contributes -8 sum_i(sum_j Bij Aij)^2, which the diagonal
    model omits along with the off-diagonal commutator contribution.
    """
    product=b@a
    x=product+product.T
    d=b.diagonal();gap=d[:,None]-d[None,:]
    slope=2*(b*gap*a).sum()
    diagonal_model=2*(gap*a).square().sum()
    diagonal_drift=-8*(b*a).sum(dim=1).square().sum()
    offdiagonal_commutator=2*(gap*(x-gap*a)*a).sum()
    exact=2*(gap*x*a).sum()+diagonal_drift
    return dict(slope=slope,curvature=exact,diagonal_model=diagonal_model,
                offdiagonal_commutator=offdiagonal_commutator,
                diagonal_drift=diagonal_drift)


def quadratic_alpha(slope,curvature):
    """Minimize s*alpha+.5*h*alpha^2 on [0,1], including h<=0.

    This is only a local quadratic minimizer, never a finite-descent gate.
    """
    s=float(slope);h=float(curvature)
    if h>0:
        return max(0.,min(1.,-s/h))
    return 1. if s+.5*h<0 else 0.


def ideal_cubic_two(a):
    """CPU mathematical two-correction polar curve from I+A, all same dtype."""
    identity=torch.eye(len(a),dtype=a.dtype,device=a.device)
    z=identity+a
    for _ in range(2):
        z=z-.5*z@(z.T@z-identity)
    return z
