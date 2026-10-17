from pathlib import Path
import json
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'benchmarks')]
import pytest
import torch
from residual_gpu_fixture import PROTOCOL, verify, sha, column_layout, observer_prefix
from common_y_kernel import warm_prefix, retract_two
from residual_retraction import residual_retract_two


def test_layout_preserves_values_and_observer_preserves_candidate():
    torch.manual_seed(147)
    q = torch.linalg.qr(torch.randn(32, 32)).Q.contiguous()
    c = torch.randn(32, 32); c = c@c.T
    column = column_layout(q)
    assert torch.equal(q, column) and column.stride() == (1, 32)
    for value in (q, column):
        before = value.clone()
        y = warm_prefix(c, value, False, True)
        observed, norm = observer_prefix(c, value)
        assert torch.equal(observed, y) and torch.equal(norm, y.norm())
        assert torch.equal(value, before)


@pytest.mark.parametrize('mixed', [False, True])
def test_defected_basis_repair_and_independent_skinny_momentum(mixed):
    torch.manual_seed(148)
    q = torch.linalg.qr(torch.randn(32, 32)).Q
    a = torch.randn(32, 32); a = a-a.T; a *= .5/a.norm()
    y = column_layout((q@(torch.eye(32)+a))*(1+.0007*torch.linspace(-1, 1, 32)))
    original = y.clone(); reference = retract_two(y)
    actual = residual_retract_two(y, mixed)
    world = torch.randn(32, 4)
    projected = actual.T@world
    assert torch.equal(y, original)
    assert float((actual-reference).norm()/reference.norm()) < 1e-4
    assert float((actual@projected-world).norm()/world.norm()) < .001


def test_hash_and_resealed_contract_tamper_rejected(tmp_path):
    (tmp_path/'source').mkdir(); f = tmp_path/'source/kernel.py'; f.write_text('frozen\n')
    config = tmp_path/'protocol.json'; config.write_text(json.dumps(PROTOCOL))
    manifest = dict(protocol_sha256=sha(config), source_sha256={'kernel.py': sha(f)}, prepared_sha256={})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest)); verify(tmp_path)
    f.write_text('changed\n')
    with pytest.raises(AssertionError): verify(tmp_path)
    f.write_text('frozen\n')
    config.write_text(json.dumps(dict(PROTOCOL, gram_precision='high')))
    manifest['protocol_sha256'] = sha(config)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(AssertionError): verify(tmp_path)


def test_only_small_correction_products_use_bf16():
    from torch.utils._python_dispatch import TorchDispatchMode
    class Trace(TorchDispatchMode):
        def __init__(self): self.dtypes = []
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func) == 'aten.mm.default': self.dtypes.append([x.dtype for x in args])
            return func(*args, **(kwargs or {}))
    y = torch.eye(16)+.001*torch.ones(16, 16)
    for mixed in (False, True):
        trace = Trace()
        with trace: residual_retract_two(y, mixed)
        correction = torch.bfloat16 if mixed else torch.float32
        assert trace.dtypes == [[torch.float32]*2, [correction]*2]*2
