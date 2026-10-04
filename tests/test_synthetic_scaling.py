from collections import Counter

import pytest
import torch

from benchmarks.benchmark_synthetic_scaling import MockBlock,event_fractions,force_event,inspect_state,summarize
from soap_reference import ResearchSOAP


@pytest.mark.parametrize('mode',['all','smaller_side'])
@pytest.mark.parametrize('method,frequency',[('qr',10),('qr',20),('warm',10)])
def test_estimated_event_weights_match_real_complete_reset_cycle(mode,method,frequency):
    torch.manual_seed(17)
    model=MockBlock(3)
    opt=ResearchSOAP(model.parameters(),basis_method=method,precondition_mode=mode,
                     precondition_frequency=frequency,hard_reset_interval=200)
    for p in model.parameters():p.grad=torch.randn_like(p)
    opt.step()  # Cold start skips the parameter update; exclude from the cycle.
    counts=Counter()
    for _ in range(200):
        for p in model.parameters():p.grad=torch.randn_like(p)
        opt.step()
        if not opt.last_basis_refreshes:event='ordinary'
        elif method=='qr':event='qr_refresh'
        elif opt.last_hard_reset_events:event='qr_reset'
        else:event='warm_refresh'
        counts[event]+=1
    assert {k:v/200 for k,v in counts.items()}==event_fractions(method,frequency)


@pytest.mark.parametrize('method',['qr','warm'])
def test_forced_events_dispatch_actual_matrix_refreshes_and_resets(method):
    torch.manual_seed(23)
    model=MockBlock(3)
    opt=ResearchSOAP(model.parameters(),basis_method=method,precondition_frequency=10,
                     hard_reset_interval=200)
    for p in model.parameters():p.grad=torch.randn_like(p)
    opt.step()
    for event in event_fractions(method,10):
        force_event(opt,event);opt.step()
        assert opt.last_basis_refreshes==(0 if event=='ordinary' else 3)
        assert opt.last_hard_reset_events==(3 if event in ['qr_refresh','qr_reset'] else 0)


def test_invalid_cycle_cannot_be_amortized():
    with pytest.raises(ValueError):event_fractions('warm',30,200)


def test_geometry_guard_checks_state_before_a_reset_can_repair_it():
    torch.manual_seed(29);model=MockBlock(3)
    opt=ResearchSOAP(model.parameters(),basis_method='warm')
    for p in model.parameters():p.grad=torch.randn_like(p)
    opt.step();assert inspect_state(opt,'all')['finite']
    prior=torch.get_float32_matmul_precision()
    next(iter(opt.state.values()))['Q'][0].mul_(2)
    with pytest.raises(FloatingPointError):inspect_state(opt,'all')
    assert torch.get_float32_matmul_precision()==prior


def test_failed_geometry_rounds_never_receive_cadence_estimates():
    report={'config':{'widths':[3],'tokens':[2],'modes':['all'],'rounds':1},
            'results':[dict(width=3,tokens=2,mode='all',method=m,status='guard_rejected') for m in ['qr','warm']]}
    assert summarize(report)==[]
