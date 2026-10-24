"""CPU audits only: these do not establish CUDA speed or learning parity."""
import copy
import types

import pytest
import torch

from fast_soap import FastSOAP, fast_warm_refresh
from prefix_sensitivity_kernel import prefix_stages
from soap_reference import ResearchSOAP


@pytest.fixture(autouse=True)
def high_precision():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(previous)


def fp32_output_reference(covariance, q):
    # Use the existing trace-normalized generator, then independently evaluate
    # BF16 operand products with FP64 accumulation and FP32 output rounding.
    def product(a, b):
        return (a.bfloat16().double() @ b.bfloat16().double()).float()

    a = prefix_stages(covariance, q, False, True)['a']
    candidate = q + product(q, a)
    for _ in range(2):
        gram_error = candidate.T @ candidate - torch.eye(len(q))
        candidate = candidate - .5 * product(candidate, gram_error)
    return candidate


@pytest.mark.parametrize('dimension', [1, 8, 17])
@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
def test_refresh_matches_intended_policy_and_preserves_inputs(dimension, scale):
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(dimension, dimension, generator=generator)
    covariance = (x @ x.T + .1 * torch.eye(dimension)) * scale
    q = torch.linalg.qr(torch.randn(dimension, dimension, generator=generator)).Q
    before_c, before_q = covariance.clone(), q.clone()
    versions = (covariance._version, q._version)
    rng = torch.random.get_rng_state().clone()
    result = fast_warm_refresh(covariance, q)
    reference = fp32_output_reference(covariance, q)
    # BF16 rounding can change with algebraic regrouping. This is an audit
    # tolerance on basis entries, not a training acceptance margin.
    torch.testing.assert_close(result, reference, rtol=0, atol=5e-5)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    assert (result.T @ result - torch.eye(dimension)).norm() / dimension**.5 < .005
    torch.testing.assert_close(covariance, before_c, rtol=0, atol=0)
    torch.testing.assert_close(q, before_q, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert versions == (covariance._version, q._version)
    assert torch.get_float32_matmul_precision() == 'high'


def test_zero_covariance_and_clustered_diagonal_are_finite():
    q = torch.eye(8)
    torch.testing.assert_close(fast_warm_refresh(torch.zeros(8, 8), q), q,
                               rtol=0, atol=0)
    # Equal diagonal entries imply zero Jacobi gap, even with off-diagonals.
    clustered = torch.eye(8) + .02 * torch.ones(8, 8)
    torch.testing.assert_close(fast_warm_refresh(clustered, q), q, rtol=0, atol=0)


def test_fullgraph_has_seven_products_and_no_trace_or_scalar_extraction():
    graphs = []

    def capture(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    kernel = torch.compile(fast_warm_refresh, backend=capture, fullgraph=True,
                           dynamic=False)
    cov = torch.diag(torch.arange(1., 9.)) + .01
    q = torch.eye(8)
    kernel(cov, q)
    kernel(cov * 2, q)
    assert len(graphs) == 1
    nodes = list(graphs[0].graph.nodes)
    assert sum(node.op == 'call_function' and node.target.__name__ == 'matmul'
               for node in nodes) == 7
    assert not any(str(node.target) in ('trace', 'item', '_local_scalar_dense')
                   for node in nodes)


def test_inductor_cpu_matches_own_eager_for_repeated_refreshes():
    optimizer = FastSOAP([torch.nn.Parameter(torch.zeros(8, 8))])
    kernel = optimizer._warm_kernel
    generator = torch.Generator().manual_seed(32)
    x = torch.randn(8, 8, generator=generator)
    covariance = x @ x.T
    compiled_q = eager_q = torch.eye(8)
    for _ in range(4):
        compiled_q = kernel(covariance, compiled_q)
        eager_q = fast_warm_refresh(covariance, eager_q)
        torch.testing.assert_close(compiled_q, eager_q, rtol=0, atol=1e-4)


def test_fake_cuda_trace_has_three_direct_fp32_output_products():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    initialized = torch.cuda.is_initialized()
    with FakeTensorMode():
        covariance = torch.empty(8, 8, device='cuda', dtype=torch.float32)
        q = torch.empty_like(covariance)
        graph = make_fx(fast_warm_refresh)(covariance, q)
        result = graph(covariance, q)
        products = [node for node in graph.graph.nodes
                    if str(node.target) in ('aten.mm.default', 'aten.mm.dtype')]
        direct = [node for node in products if str(node.target) == 'aten.mm.dtype']
        assert len(products) == 7 and len(direct) == 3
        assert all(node.args[2] == torch.float32 for node in direct)
        assert all(node.args[0].meta['val'].dtype == torch.bfloat16 and
                   node.args[1].meta['val'].dtype == torch.bfloat16 for node in direct)
        assert result.device.type == 'cuda' and result.dtype == torch.float32
    assert torch.cuda.is_initialized() == initialized


def test_complete_optimizer_trajectory_matches_policy_reference_without_precision_switch(monkeypatch):
    generator = torch.Generator().manual_seed(33)
    p = torch.nn.Parameter(torch.randn(7, 3, generator=generator))
    control_p = torch.nn.Parameter(p.detach().clone())
    options = dict(lr=.003, shampoo_beta=.9, precondition_frequency=2,
                   hard_reset_interval=6)
    optimizer = FastSOAP([p], compile_warm=False, **options)
    control = ResearchSOAP([control_p], basis_method='warm', **options)
    control._gauge_step_one = types.MethodType(
        lambda self, c, q, group: fp32_output_reference(c, q), control)

    def forbidden(*args, **kwargs):
        raise AssertionError('Optimizer changed global matmul precision')

    with monkeypatch.context() as patch:
        patch.setattr(torch, 'set_float32_matmul_precision', forbidden)
        for call in range(25):
            gradient = torch.randn(p.shape, generator=generator)
            p.grad = gradient.clone()
            control_p.grad = gradient.clone()
            before = p.detach().clone()
            optimizer.step()
            control.step()
            if call == 0:
                torch.testing.assert_close(p, before, rtol=0, atol=0)
            torch.testing.assert_close(p, control_p, rtol=0, atol=2e-5)
    state = optimizer.state[p]
    assert state['research_qr_refreshes'] == 4
    assert state['research_warm_refreshes'] == 8
    assert state.get('research_strict_transports', 0) == 0
    for tensor in [state['exp_avg'], state['exp_avg_sq'], *state['GG'], *state['Q']]:
        assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()


def test_warm_carries_variance_and_transport_preserves_physical_momentum():
    p = torch.nn.Parameter(torch.zeros(8, 5))
    optimizer = FastSOAP([p], compile_warm=False, precondition_frequency=2,
                         hard_reset_interval=6)
    generator = torch.Generator().manual_seed(34)
    for _ in range(2):
        p.grad = torch.randn(p.shape, generator=generator)
        optimizer.step()
    state, group = optimizer.state[p], optimizer.param_groups[0]
    state['step'] = 2
    variance = state['exp_avg_sq']
    before_v = variance.clone()
    world = optimizer._project_back_with_basis(state['exp_avg'], state['Q'], group)
    optimizer._research_refresh(state, group)
    after_world = optimizer._project_back_with_basis(state['exp_avg'], state['Q'], group)
    assert state['exp_avg_sq'] is variance
    torch.testing.assert_close(variance, before_v, rtol=0, atol=0)
    assert (after_world - world).norm() / world.norm() < .01


def test_state_replay_and_reject_control_policy_before_mutation():
    p = torch.nn.Parameter(torch.zeros(5, 3))
    options = dict(compile_warm=False, precondition_frequency=2, hard_reset_interval=6)
    optimizer = FastSOAP([p], **options)
    generator = torch.Generator().manual_seed(35)
    for _ in range(5):
        p.grad = torch.randn(p.shape, generator=generator)
        optimizer.step()
    q = torch.nn.Parameter(p.detach().clone())
    replay = FastSOAP([q], **options)
    replay.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for _ in range(8):
        gradient = torch.randn(p.shape, generator=generator)
        p.grad, q.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        replay.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    invalid = copy.deepcopy(optimizer.state_dict())
    invalid['param_groups'][0]['transport_precision'] = 'highest'
    with pytest.raises(ValueError, match='policy'):
        replay.load_state_dict(invalid)
    assert replay.param_groups[0]['transport_precision'] == 'inherit'


def test_reject_unsupported_precision_dtype_and_group_covariance():
    p = torch.nn.Parameter(torch.zeros(5, 3))
    with pytest.raises(ValueError, match='policy'):
        FastSOAP([dict(params=[p], covariance_compute_dtype='bfloat16')])
    with pytest.raises(ValueError, match='FP32'):
        FastSOAP([torch.nn.Parameter(torch.zeros(5, 3, dtype=torch.bfloat16))])
    torch.set_float32_matmul_precision('highest')
    with pytest.raises(ValueError, match='caller-owned'):
        FastSOAP([p])
