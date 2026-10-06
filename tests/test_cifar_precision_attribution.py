import copy
import pytest
import torch
from soap_reference import ResearchSOAP
from benchmarks import cifar_precision_attribution as probe
from benchmarks.cifar_refresh_efficiency import diagnose


def sample():
    torch.manual_seed(392)
    p=torch.nn.Parameter(torch.randn(4,7));opt=ResearchSOAP([p],precondition_frequency=20)
    for _ in range(4):p.grad=torch.randn_like(p);opt.step()
    state=opt.state[p];group={**opt.param_groups[0],'basis_ns_iterations':6}
    return dict(q=state['Q'],cov=state['GG'],m=state['exp_avg'],v=state['exp_avg_sq']),group


@pytest.mark.parametrize('variant',probe.VARIANTS)
def test_attribution_preserves_inputs_and_reproduces_full_comparison(variant):
    old,group=sample();saved=copy.deepcopy(old)
    previous=torch.get_float32_matmul_precision()
    result=probe.attribute(old,group,variant)
    assert result['full_update_difference']==diagnose(old,group,variant)['update_shadow_error']
    assert result['chain_closure_relative']<1e-6
    assert torch.get_float32_matmul_precision()==previous
    for key in ['m','v']:assert torch.equal(old[key],saved[key])
    for key in ['q','cov']:assert all(torch.equal(a,b) for a,b in zip(old[key],saved[key]))


def test_attribution_isolates_deliberate_transport_corruption(monkeypatch):
    old,group=sample();original=probe.refresh
    def corrupted(old,group,variant):
        result=original(old,group,variant)
        if torch.get_float32_matmul_precision()=='high':result['m']=result['m']*1.02
        return result
    monkeypatch.setattr(probe,'refresh',corrupted)
    result=probe.attribute(old,group,'warm1')
    assert result['fixed_basis_transport_difference']==pytest.approx(.02,abs=2e-6)
    assert result['stages']['basis_path']<1e-6
    assert not result['fixed_basis_transport_pass']


def test_fixed_transport_keeps_sequential_intermediate_basis():
    old,group=sample()
    q=old['q'];middle=[basis*.9 for basis in q]
    final=probe.transport_at_fixed_bases(old,q)
    sequential=probe.transport_at_fixed_bases(old,q,middle)
    # Two active axes each contribute scale^2 in the intermediate round trip.
    torch.testing.assert_close(sequential,final*.9**4,rtol=2e-5,atol=2e-7)


def test_qr_fixed_order_does_not_resort_columns():
    old=dict(q=[torch.eye(3),torch.eye(2)],cov=[torch.diag(torch.tensor([1.,3.,2.])),torch.diag(torch.tensor([2.,1.]))])
    orders=probe.qr_orders(old)
    assert orders[0].tolist()==[1,2,0]
    identity_orders=[torch.arange(3),torch.arange(2)]
    fixed=probe.qr_at_fixed_order(old,identity_orders)
    assert torch.equal(fixed[0],torch.eye(3))
    reordered=probe.qr_at_fixed_order(old,orders)
    assert not torch.equal(reordered[0].abs(),fixed[0])
