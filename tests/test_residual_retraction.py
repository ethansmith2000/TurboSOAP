"""Independent common-candidate formula and CPU perturbation checks."""
import pytest
import torch
from residual_retraction import residual_retract_two

@pytest.mark.parametrize('mixed',[False,True])
def test_formula_does_not_mutate_input_and_repairs_common_candidate(mixed):
    torch.manual_seed(146)
    q=torch.linalg.qr(torch.randn(32,32)).Q
    a=torch.randn(32,32);a=a-a.T;a*=.5/a.norm()
    y=q@(torch.eye(32)+a);before=y.clone();reference=y.clone()
    for _ in range(2):
        e=reference.T@reference-torch.eye(32)
        correction=(reference.bfloat16()@e.bfloat16()).float() if mixed else reference@e
        reference=reference-.5*correction
    got=residual_retract_two(y,mixed)
    assert torch.equal(got,reference) and torch.equal(y,before)
    initial=(y.T@y-torch.eye(32)).norm()
    final=(got.T@got-torch.eye(32)).norm()
    assert final<initial/100
