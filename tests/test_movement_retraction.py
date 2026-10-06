import math
import pytest
import torch
from benchmarks.movement_retraction import clip_skew_frobenius, POLICIES, warm, configure_matrix
from benchmarks.warm_retraction import retract
from soap_reference import ResearchSOAP


def test_bound_is_tight_for_rank_two_skew_and_preserves_small_steps():
    a = torch.zeros(12,12); a[0,1] = 3; a[1,0] = -3
    b = clip_skew_frobenius(a,.5)
    assert torch.linalg.matrix_norm(b,ord=2) == .5
    small = a*.01
    assert torch.equal(clip_skew_frobenius(small,.5),small)
    assert torch.equal(clip_skew_frobenius(torch.zeros_like(a),.5),torch.zeros_like(a))
    with pytest.raises(ValueError): clip_skew_frobenius(a,0)


def test_skew_bound_and_polynomial_geometry_over_random_rank_profiles():
    gen = torch.Generator().manual_seed(951)
    for d in (2,16,64):
        for rank in (1,d):
            x = torch.randn(d,rank,generator=gen);y = torch.randn(d,rank,generator=gen)
            a = x@y.T-y@x.T
            for policy in POLICIES.values():
                clipped = clip_skew_frobenius(a,policy.cap)
                assert torch.linalg.matrix_norm(clipped,ord=2) <= policy.cap+1e-6
                z = retract(torch.eye(d)+clipped,policy)
                assert torch.linalg.matrix_norm(z.T@z-torch.eye(d),ord=2)<.003


def test_adapter_matches_pure_kernel_and_keeps_input_state():
    gen=torch.Generator().manual_seed(8)
    q=torch.linalg.qr(torch.randn(16,16,generator=gen)).Q
    g=torch.randn(16,24,generator=gen);cov=g@g.T
    original=(q.clone(),cov.clone())
    opt=ResearchSOAP([torch.nn.Parameter(torch.zeros_like(q))],basis_method='warm')
    configure_matrix(opt,'fro_cubic2_cap05');group=opt.param_groups[0]
    assert torch.equal(opt._gauge_step_one(cov,q,group),warm(cov,q,group,POLICIES['fro_cubic2_cap05']))
    assert torch.equal(q,original[0]) and torch.equal(cov,original[1])
    assert group['basis_rotation_cap_mode']=='skew_frobenius'
