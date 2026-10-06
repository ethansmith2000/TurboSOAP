import copy
import pytest
import torch
from benchmarks.cifar_refresh_efficiency import refresh, diagnose, VARIANTS, SOAP
from soap_reference import ResearchSOAP


def sample():
    torch.manual_seed(231)
    p=torch.nn.Parameter(torch.randn(4,7))
    opt=ResearchSOAP([p],precondition_frequency=20)
    p.grad=torch.randn_like(p);opt.step()
    for _ in range(3):
        p.grad=torch.randn_like(p);opt.step()
    state=opt.state[p];group={**opt.param_groups[0],'basis_ns_iterations':6}
    old=dict(q=state['Q'],cov=state['GG'],m=state['exp_avg'],v=state['exp_avg_sq'])
    return opt,state,group,old


@pytest.mark.parametrize('method,variant',[('qr','qr'),('warm','warm1')])
def test_local_control_matches_research_refresh(method,variant):
    opt,state,group,old=sample();group.update(basis_method=method,hard_reset_interval=0)
    expected=refresh(old,group,variant);opt._research_refresh(state,group)
    for a,b in zip(expected['q'],state['Q']):assert torch.equal(a,b)
    assert torch.equal(expected['m'],state['exp_avg'])
    assert torch.equal(expected['v'],state['exp_avg_sq'])


@pytest.mark.parametrize('variant',VARIANTS)
def test_candidate_does_not_mutate_inputs_and_restores_precision(variant):
    _,_,group,old=sample();saved=copy.deepcopy(old)
    precision=torch.get_float32_matmul_precision()
    result=diagnose(old,group,variant)
    assert result['passes']
    assert torch.get_float32_matmul_precision()==precision
    for key in ['m','v']:assert torch.equal(old[key],saved[key])
    for key in ['q','cov']:
        assert all(torch.equal(a,b) for a,b in zip(old[key],saved[key]))


def test_two_steps_share_basis_but_sequential_transport_applies_intermediate_gram():
    _,_,group,old=sample()
    final=refresh(old,group,'warm2_final');sequential=refresh(old,group,'warm2_sequential')
    assert all(torch.equal(a,b) for a,b in zip(final['q'],sequential['q']))
    torch.testing.assert_close(final['m'],sequential['m'],rtol=2e-5,atol=2e-7)
    assert final['v'] is old['v'] and sequential['v'] is old['v']


def test_precision_guard_rejects_a_broken_retraction(monkeypatch):
    _,_,group,old=sample()
    monkeypatch.setattr(SOAP,'_gauge_step_one',staticmethod(lambda covariance,q,group:q*.3))
    result=diagnose(old,group,'warm2_final')
    assert not result['passes']
    assert result['maximum_gram']>.05 and result['world_m_preservation_error']>.01
