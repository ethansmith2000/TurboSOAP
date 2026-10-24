"""Independent CPU arithmetic/coordinate checks plus fake CUDA tracing."""
import pytest
import torch
from batched_soap_refresh import batched_qr_refresh, batched_warm_refresh
from fast_soap import fast_warm_refresh


@pytest.mark.parametrize('batch,dimension', [(1, 7), (3, 17)])
def test_batch_independence_and_scalar_policy(batch, dimension):
    g = torch.Generator().manual_seed(202)
    x = torch.randn(batch, dimension, dimension, generator=g)
    c = x @ x.transpose(-1, -2)
    q = torch.linalg.qr(torch.randn(batch, dimension, dimension, generator=g)).Q
    v = torch.randn(batch, dimension, 5, generator=g)
    eps = torch.arange(1, batch+1) * 1e-5
    copies = [z.clone() for z in (c, q, v, eps)]
    versions = [z._version for z in (c, q, v, eps)]
    rng = torch.get_rng_state().clone()
    out, nu = batched_warm_refresh(c, q, v, eps)
    expected = torch.stack([fast_warm_refresh(c[i]+eps[i]*torch.eye(dimension), q[i]) for i in range(batch)])
    torch.testing.assert_close(out, expected, atol=5e-5, rtol=0)
    assert nu is v
    for actual, before in zip((c, q, v, eps), copies):
        torch.testing.assert_close(actual, before, atol=0, rtol=0)
    assert versions == [z._version for z in (c, q, v, eps)]
    assert torch.equal(rng, torch.get_rng_state())
    # Changing another factor must not change the first factor's cap/scale.
    if batch > 1:
        changed = c.clone(); changed[1:] *= 1e10
        torch.testing.assert_close(batched_warm_refresh(changed, q, v, eps)[0][0], out[0], atol=0, rtol=0)


def test_qr_matches_reference_column_sort_and_variance_row_permutation():
    c = torch.diag_embed(torch.tensor([[2., 5., 1., 3.], [8., 1., 4., 2.]]))
    q = torch.eye(4).expand(2, -1, -1).clone()
    v = torch.arange(24.).reshape(2, 4, 3)
    eps = torch.tensor([1e-5, 2e-5])
    out, nu = batched_qr_refresh(c, q, v, eps)
    for i in range(2):
        pi = c[i] + eps[i] * torch.eye(4)
        est = torch.diag(q[i].T @ pi @ q[i])
        order = torch.argsort(est, descending=True)
        expected = torch.linalg.qr(pi @ q[i][:, order]).Q
        torch.testing.assert_close(out[i], expected, atol=0, rtol=0)
        torch.testing.assert_close(nu[i], v[i][order], atol=0, rtol=0)
    # A right factor permutes columns by operating on transposed variance.
    rectangular = v.transpose(-1, -2)
    _, right = batched_qr_refresh(c, q, rectangular.transpose(-1, -2), eps)
    for i in range(2):
        order = c[i].diagonal().argsort(descending=True)
        torch.testing.assert_close(right[i].T, rectangular[i][:, order], atol=0, rtol=0)


def test_padding_stays_decoupled_and_orthogonal():
    c = torch.zeros(2, 17, 17)
    c[:, :9, :9] = torch.diag(torch.linspace(.5, 1.5, 9)) + .01
    q = torch.eye(17).expand(2, -1, -1).clone()
    v = torch.ones(2, 17, 17)
    for _ in range(10):
        q, nu = batched_warm_refresh(c, q, v, torch.full((2,), 1e-10))
        assert nu is v and torch.isfinite(q).all()
        assert (q.transpose(-1, -2) @ q - torch.eye(17)).norm() < .05
        torch.testing.assert_close(q[:, 9:, :9], torch.zeros(2, 8, 9), atol=0, rtol=0)


def test_fake_cuda_has_three_direct_fp32_output_batched_products():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    initialized = torch.cuda.is_initialized()
    with FakeTensorMode():
        c = torch.empty(8, 512, 512, device='cuda')
        graph = make_fx(batched_warm_refresh)(c, c, c, torch.empty(8, device='cuda'))
        products = [n for n in graph.graph.nodes if str(n.target) in ('aten.bmm.default', 'aten.bmm.dtype')]
        direct = [n for n in products if str(n.target) == 'aten.bmm.dtype']
        assert len(products) == 7 and len(direct) == 3
        assert all(n.args[2] == torch.float32 for n in direct)
    assert torch.cuda.is_initialized() == initialized
