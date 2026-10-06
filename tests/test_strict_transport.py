import copy
import pytest
import torch
from soap_reference import ResearchSOAP
from benchmarks.cifar_optimizers import build_optimizer,screen_arm_config,counters
from benchmarks.cifar_strict_transport import shadow_refresh,diagnose


def optimizer(precision='inherit',method='warm'):
    p=torch.nn.Parameter(torch.randn(4,7))
    opt=ResearchSOAP([p],basis_method=method,precondition_frequency=2,hard_reset_interval=4,
                     transport_precision=precision)
    opt.param_groups[0]['basis_ns_iterations']=6
    return p,opt


def test_strict_transport_scope_and_actual_counts(monkeypatch):
    p,opt=optimizer('highest');events=[];prior=torch.get_float32_matmul_precision()
    transport=opt._transport_first_moment;gauge=opt._gauge_step_one
    def observed_m(*args):events.append(('m',torch.get_float32_matmul_precision()));return transport(*args)
    def observed_q(*args):events.append(('q',torch.get_float32_matmul_precision()));return gauge(*args)
    monkeypatch.setattr(opt,'_transport_first_moment',observed_m)
    monkeypatch.setattr(opt,'_gauge_step_one',observed_q)
    try:
        torch.set_float32_matmul_precision('high')
        for _ in range(5):
            p.grad=torch.randn_like(p);opt.step()
            assert torch.get_float32_matmul_precision()=='high'
        assert [v for k,v in events if k=='m']==['highest','highest']
        assert {v for k,v in events if k=='q'}=={'high'}
        assert opt.state[p]['research_strict_transports']==2
        assert opt.state[p]['research_qr_refreshes']==1 and opt.state[p]['research_warm_refreshes']==1
    finally:torch.set_float32_matmul_precision(prior)


def test_strict_transport_restores_precision_after_failure(monkeypatch):
    p,opt=optimizer('highest');p.grad=torch.randn_like(p);opt.step()
    prior=torch.get_float32_matmul_precision()
    def fail(*args):
        assert torch.get_float32_matmul_precision()=='highest'
        raise RuntimeError('injected transport failure')
    monkeypatch.setattr(opt,'_transport_first_moment',fail)
    try:
        torch.set_float32_matmul_precision('high')
        with pytest.raises(RuntimeError,match='injected'):opt._research_refresh(opt.state[p],opt.param_groups[0])
        assert torch.get_float32_matmul_precision()=='high'
    finally:torch.set_float32_matmul_precision(prior)


@pytest.mark.parametrize('method',['qr','warm'])
def test_shadow_uses_real_refresh_and_preserves_inputs(method):
    p,opt=optimizer('highest',method)
    for _ in range(2):p.grad=torch.randn_like(p);opt.step()
    state=opt.state[p];g=opt.param_groups[0]
    old=dict(q=state['Q'],cov=state['GG'],m=state['exp_avg'],v=state['exp_avg_sq']);saved=copy.deepcopy(old)
    actual=shadow_refresh(opt,old,g,2);d=diagnose(opt,old,g,2,actual)
    assert d['targeted_pass'] and d['selected_basis_update_error']==0
    for key in ['m','v']:assert torch.equal(saved[key],old[key])
    for key in ['q','cov']:assert all(torch.equal(a,b) for a,b in zip(saved[key],old[key]))


def test_old_checkpoint_without_precision_key_keeps_inherited_path():
    p,opt=optimizer();g=opt.param_groups[0];g.pop('transport_precision')
    for _ in range(5):p.grad=torch.randn_like(p);opt.step()
    assert 'research_strict_transports' not in opt.state[p]


def test_unknown_precision_rejected():
    with pytest.raises(ValueError,match='transport_precision'):optimizer('medium')
