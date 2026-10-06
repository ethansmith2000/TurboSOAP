from collections import Counter

import pytest
import torch

from benchmarks.prepare_cadence_beta import arms
from benchmarks.rectangular_precision_gate import make_optimizer
from benchmarks.summarize_rectangular_precision import expected_events


@pytest.mark.parametrize('arm',arms(),ids=lambda a:a['name'])
def test_declared_beta_and_cadence_reach_actual_optimizer(arm):
    torch.manual_seed(91);p=torch.nn.Parameter(torch.randn(3,5))
    opt=make_optimizer([p],arm,'all');p.grad=torch.randn_like(p)
    first_gradient=p.grad.clone();opt.step()
    beta=arm['covariance_beta']
    torch.testing.assert_close(opt.state[p]['GG'][0],(1-beta)*(first_gradient@first_gradient.T))
    assert opt.param_groups[0]['betas']==(.9,.999)
    assert opt.param_groups[0]['basis_ns_iterations']==6
    counts=Counter()
    for _ in range(200):
        p.grad=torch.randn_like(p);opt.step()
        event=('ordinary' if not opt.last_basis_refreshes else 'qr_refresh' if arm['method']=='qr'
               else 'qr_reset' if opt.last_hard_reset_events else 'warm_refresh')
        counts[event]+=1;opt.pending.clear()
    assert dict(counts)==expected_events(arm,200)
    assert counts['ordinary']==200-200//arm['frequency']
    if arm['method']=='warm':assert counts['qr_reset']==1
