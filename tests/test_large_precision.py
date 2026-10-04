from collections import Counter

import pytest
import torch

from benchmarks.benchmark_large_precision import arms, event_name, state_tensors, summarize_case, warm_six
from soap import SOAP
from soap_reference import ResearchSOAP


@pytest.mark.parametrize('arm', arms(), ids=lambda a: a['name'])
def test_natural_event_accounting(arm):
    counts = Counter(event_name(i, arm) for i in range(1, 201))
    assert counts['ordinary'] == 200 - 200 // arm['frequency']
    if arm['method'] == 'warm':
        assert counts['warm_refresh'] == 19 and counts['qr_reset'] == 1
    else:
        assert counts['qr_refresh'] == 200 // arm['frequency']


def test_compilation_target_matches_existing_six_iteration_math():
    torch.manual_seed(51)
    x = torch.randn(16, 16); cov = x @ x.T
    q = torch.linalg.qr(torch.randn(16, 16)).Q
    parameter = torch.nn.Parameter(torch.randn(16, 16))
    opt = ResearchSOAP([parameter], basis_method='warm')
    group = dict(opt.param_groups[0], basis_ns_iterations=6)
    assert torch.equal(warm_six(cov, q), SOAP._gauge_step_one(cov, q, group))


def test_inference_optimizer_allows_backward_but_normal_step_rejects_state():
    torch.manual_seed(52)
    layer = torch.nn.Linear(4, 4, bias=False)
    opt = ResearchSOAP(layer.parameters(), basis_method='warm')
    x = torch.randn(3, 4)
    for _ in range(3):
        opt.zero_grad(); layer(x).square().mean().backward()
        with torch.inference_mode(): opt.step()
    assert any(torch.is_inference(t) for t in state_tensors(opt.state))
    opt.zero_grad(); layer(x).square().mean().backward()
    assert torch.isfinite(layer.weight.grad).all()
    with pytest.raises(RuntimeError, match='Inference'):
        opt.step()


def test_rejected_precision_arm_has_no_cost_summary():
    assert summarize_case({'status': 'guard_rejected'}) is None
