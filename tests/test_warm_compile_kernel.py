import math
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'benchmarks')]
import torch
import json
import shutil
import pytest
from movement_retraction import warm, POLICIES
from warm_compile_kernel import warm_kernel, classify, LIMITS
from warm_compile_fixture import PROTOCOL, verify, sha


def test_fp32_matches_current_fixed_policy_without_mutation():
    torch.manual_seed(987)
    for n in (8, 32):
        q = torch.linalg.qr(torch.randn(n, n)).Q
        h = torch.randn(n, n)
        c = h @ h.T
        before = (q.clone(), c.clone())
        expected = warm(c, q, dict(basis_lr=.5, basis_jacobi_damping=.01), POLICIES['fro_cubic2_cap05'])
        assert torch.equal(expected, warm_kernel(c, q))
        assert torch.equal(q, before[0]) and torch.equal(c, before[1])


def test_mixed_keeps_fp32_storage_and_addition():
    torch.manual_seed(998)
    n = 16
    q = torch.linalg.qr(torch.randn(n, n)).Q
    c = torch.randn(n, n); c = c @ c.T
    before = (q.clone(), c.clone())
    result = warm_kernel(c, q, True)
    assert result.dtype == q.dtype == c.dtype == torch.float32
    assert torch.isfinite(result).all()
    assert torch.equal(q, before[0]) and torch.equal(c, before[1])
    # Same products and explicit FP32 delta addition, independent reconstruction.
    qb = q.bfloat16(); cb = (c / (c.trace().abs() / n).clamp_min(1e-12)).bfloat16()
    b = (qb.T @ cb @ qb).float(); b = (b + b.T) * .5
    from soap import _damped_jacobi_generator
    raw = .5 * _damped_jacobi_generator(b, .01)
    a = raw * (.5 / (raw.norm() / math.sqrt(2)).clamp_min(1e-12)).clamp(max=1)
    z = q + (qb @ a.bfloat16()).float()
    for _ in range(2): z = 1.5 * z - .5 * (z @ (z.T @ z))
    assert torch.equal(result, z)


def test_each_fidelity_failure_is_retained():
    metrics = dict(basis_relative=0., movement_relative=0., movement_cosine=1., gram=0.,
                   signed_diagonal_mean=0., world_m_relative=0., projected_m_relative=0., tracking_ratio=1.)
    assert classify(metrics, False)['passed']
    for key in metrics:
        failed = dict(metrics)
        failed[key] = 0. if key == 'movement_cosine' else 100.
        assert not classify(failed, True)['passed']
    failed = dict(metrics, basis_relative=5e-4)
    assert not classify(failed, False)['passed'] and classify(failed, True)['passed']


def test_source_and_resealed_protocol_negative_controls(tmp_path):
    (tmp_path / 'source').mkdir()
    source = tmp_path / 'source/kernel.py'; source.write_text('original\n')
    protocol = tmp_path / 'protocol.json'; protocol.write_text(json.dumps(PROTOCOL))
    manifest = dict(protocol_sha256=sha(protocol), source_sha256={'kernel.py': sha(source)})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    verify(tmp_path)
    source.write_text('changed\n')
    with pytest.raises(AssertionError): verify(tmp_path)
    source.write_text('original\n')
    protocol.write_text(json.dumps(dict(PROTOCOL, cap=.6)))
    manifest['protocol_sha256'] = sha(protocol)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(AssertionError): verify(tmp_path)
