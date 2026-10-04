import importlib.util
from pathlib import Path

import pytest
import torch

from soap_reference import ResearchSOAP


def upstream_class():
    path=Path(__file__).parent/'reference/soap_upstream.py'
    spec=importlib.util.spec_from_file_location('pinned_soap',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SOAP


@pytest.mark.parametrize('shape',[(4,4),(7,3),(3,7)])
@pytest.mark.parametrize('beta2,shampoo_beta,eps',[(.99,.999,1e-8),(.999,.99,.03)])
def test_qr_matches_pinned_upstream_across_refreshes(shape,beta2,shampoo_beta,eps):
    torch.manual_seed(12)
    a=torch.nn.Parameter(torch.randn(shape))
    b=torch.nn.Parameter(a.detach().clone())
    common=dict(lr=.003,betas=(.9,beta2),shampoo_beta=shampoo_beta,
                eps=eps,weight_decay=.07,precondition_frequency=3)
    original=upstream_class()([a],**common)
    reference=ResearchSOAP([b],hard_reset_interval=0,**common)
    for _ in range(37):
        grad=torch.randn(shape)
        a.grad=grad.clone();b.grad=grad.clone()
        original.step();reference.step()
        torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(original.state[a]['exp_avg_sq'],reference.state[b]['exp_avg_sq'],atol=2e-6,rtol=2e-5)
    assert reference.state[b]['research_qr_refreshes']==12


@pytest.mark.parametrize('mode',['all','smaller_side'])
@pytest.mark.parametrize('policy',['permutation','overlap'])
def test_warm_qr_safeguard_and_checkpoint_replay(mode,policy):
    import copy
    torch.manual_seed(14)
    p=torch.nn.Parameter(torch.randn(6,3))
    kwargs=dict(basis_method='warm',variance_policy=policy,precondition_mode=mode,
                precondition_frequency=2,hard_reset_interval=6)
    optimizer=ResearchSOAP([p],**kwargs)
    for _ in range(5):
        p.grad=torch.randn_like(p);optimizer.step()
    q=torch.nn.Parameter(p.detach().clone())
    restored=ResearchSOAP([q],**kwargs)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for _ in range(8):
        grad=torch.randn_like(p);p.grad=grad.clone();q.grad=grad.clone()
        optimizer.step();restored.step()
    torch.testing.assert_close(p,q,atol=0,rtol=0)
    assert optimizer.state[p]['research_qr_refreshes']==2
    assert optimizer.state[p]['research_warm_refreshes']==4
    assert all(torch.isfinite(value).all() for value in [p,optimizer.state[p]['exp_avg'],optimizer.state[p]['exp_avg_sq']])


def test_zero_gradient_bootstrap_preserves_zero_head_backprop_case():
    p=torch.nn.Parameter(torch.randn(5,3));before=p.detach().clone()
    optimizer=ResearchSOAP([p])
    p.grad=torch.zeros_like(p);optimizer.step()
    torch.testing.assert_close(p,before,atol=0,rtol=0)
    p.grad=torch.randn_like(p);optimizer.step()
    assert torch.isfinite(p).all()


@pytest.mark.parametrize('kwargs',[
    {'precondition_frequency':0}, {'precondition_frequency':2.5},
    {'hard_reset_interval':-1}, {'hard_reset_interval':11},
    {'precondition_mode':'aspect_ratio'},
])
def test_research_protocol_rejects_invalid_schedule_or_mode(kwargs):
    with pytest.raises(ValueError):ResearchSOAP([torch.nn.Parameter(torch.zeros(2,2))],**kwargs)


def test_research_reference_rejects_silently_ignored_group_controller():
    with pytest.raises(ValueError,match='Unsupported'):
        ResearchSOAP([dict(params=[torch.nn.Parameter(torch.zeros(2,2))],basis_lr_age_compensation=True)])
