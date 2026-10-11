import torch
from benchmarks.wide_probe import norm_proxy,selected_checks,factor_diagnostic
from benchmarks.cifar_tracking_probe import project,world


def test_proxy_is_bounded_deterministic_and_preserves_rng():
    torch.manual_seed(19)
    a=torch.randn(8,8);a=a-a.T
    rng=torch.get_rng_state().clone()
    proxy=norm_proxy(a)
    assert torch.equal(rng,torch.get_rng_state())
    assert proxy==norm_proxy(a)
    assert 0<proxy<=float(torch.linalg.matrix_norm(a.double(),ord=2))*(1+1e-12)
    assert norm_proxy(torch.zeros(8,8))==0


def fixture():
    torch.manual_seed(7)
    old_q=[torch.linalg.qr(torch.randn(n,n)).Q for n in [4,3]]
    new_q=[torch.linalg.qr(torch.randn(n,n)).Q for n in [4,3]]
    old=dict(q=old_q,m=torch.randn(4,3),cov=[torch.eye(n) for n in [4,3]])
    actual=dict(Q=new_q,exp_avg=project(world(old['m'],old_q),new_q),exp_avg_sq=torch.ones(4,3))
    return old,actual


def test_selected_transport_accepts_formula_and_rejects_corrupted_m():
    old,actual=fixture()
    d,_=selected_checks(old,actual,1e-8)
    assert d['targeted_pass'] and not d['full_reference_collected']
    actual['exp_avg']=actual['exp_avg']+.1
    assert not selected_checks(old,actual,1e-8)[0]['targeted_pass']


def test_selected_transport_rejects_bad_geometry():
    old,actual=fixture()
    actual['Q'][0]=actual['Q'][0]*1.1
    actual['exp_avg']=project(world(old['m'],old['q']),actual['Q'])
    assert not selected_checks(old,actual,1e-8)[0]['targeted_pass']


def test_zero_generator_records_null_slack_and_restores_precision():
    torch.set_float32_matmul_precision('highest')
    q=torch.eye(4)
    row=factor_diagnostic(q,q,q,dict(basis_lr=.5,basis_jacobi_damping=.01),dict(gram=0.,residual=0.),0)
    assert row['raw_frobenius_bound']==0 and row['clipping_scale']==1
    assert row['bound_over_lower_proxy'] is None
    assert torch.get_float32_matmul_precision()=='highest'
