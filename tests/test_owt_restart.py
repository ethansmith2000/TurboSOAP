import pytest
import torch

from owt_restart import PROFILES, build_optimizer, geometry, multiplier
from transformer import Transformer


def model():
    return Transformer(dim=8,depth=1,heads=2,ff_mult=2.,ff_hidden_dim=16,
                       vocab_size=32,max_seq_len=16,dropout=0.)


@pytest.fixture(autouse=True)
def precision():
    old=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(old)


@pytest.mark.parametrize('profile',PROFILES)
def test_full_two_sided_coverage_and_profile_reaches_state(profile):
    m=model()
    opt=build_optimizer(m,profile,'qr',.001,compile_warm=False)
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(m.parameters()))
    beta1,beta2,cov=PROFILES[profile]
    for group in opt.matrix.param_groups:
        assert group['betas']==(beta1,beta2) and group['shampoo_beta']==cov
        assert group['precondition_mode']=='all' and not group['normalize_grads']
        assert group['covariance_compute_dtype']=='float32'
    for group in opt.auxiliary.param_groups:assert group['betas']==(.9,.999)
    for p in m.parameters():p.grad=torch.randn_like(p)
    opt.step()
    assert len(geometry(opt,0))==8


@pytest.mark.parametrize('profile',PROFILES)
def test_qr_and_warm_match_before_first_warm_then_have_declared_cadences(profile):
    torch.manual_seed(44)
    a=model()
    b=model()
    b.load_state_dict(a.state_dict())
    qr=build_optimizer(a,profile,'qr',.001,compile_warm=False)
    warm=build_optimizer(b,profile,'warm',.001,compile_warm=False)
    for call in range(1,62):
        for x,y in zip(a.parameters(),b.parameters()):
            x.grad=torch.randn_like(x)
            y.grad=x.grad.clone()
        qr.step();warm.step()
        if call<=10:
            for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
    for s in qr.matrix.state.values():
        assert s['research_qr_refreshes']==6 and s.get('research_warm_refreshes',0)==0
    for s in warm.matrix.state.values():
        assert s['research_qr_refreshes']==1 and s['research_warm_refreshes']==5
    geometry(qr,60);geometry(warm,60)


def test_schedule_keeps_full_horizon_for_short_calibration():
    assert multiplier(1)==1/200
    assert multiplier(200)==1
    assert .99<multiplier(500)<1
    assert multiplier(10000)==.1
