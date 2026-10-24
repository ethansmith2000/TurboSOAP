import copy
import gc
import math
import random

import numpy as np
import pytest
import torch

from reference_basis_diagnostics import BasisDiagnostics, factor_metrics, model_for_seed, resolve_factors
from reference_covariance_beta_soap import CovarianceBetaSOAP
from tensor_ownership import TensorOwnership


def test_metrics_match_closed_form_and_preserve_inputs_and_rng():
    c0 = torch.diag(torch.tensor([1., 2.]))
    c = torch.diag(torch.tensor([1., 3.]))
    q0 = torch.eye(2)
    q = torch.tensor([[1., -1.], [1., 1.]], dtype=torch.float64)/math.sqrt(2)
    saved = [x.clone() for x in (c, q, c0, q0)]
    rng = torch.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), copy.deepcopy(np.random.get_state())
    mode = torch.get_float32_matmul_precision()
    x = factor_metrics(c, q, c0, q0)
    assert x['relative_covariance_change'] == pytest.approx(1/math.sqrt(5))
    assert x['current_relative_offdiagonal'] == pytest.approx(math.sqrt(.2))
    assert x['held_relative_offdiagonal'] == 0
    assert x['participation_rank'] == pytest.approx(1.6)
    for actual, expected in zip((c, q, c0, q0), saved):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()
    assert all(np.array_equal(a, b) for a, b in zip(numpy_rng, np.random.get_state()))
    assert mode == torch.get_float32_matmul_precision()


def test_zero_reference_is_explicit_and_zero_covariance_has_zero_defect():
    z, q = torch.zeros(2, 2), torch.eye(2)
    zero = factor_metrics(z, q, z, q)
    assert zero['relative_covariance_change'] == zero['current_relative_offdiagonal'] == zero['participation_rank'] == 0
    grown = factor_metrics(q, q, z, q)
    assert grown['relative_covariance_change'] is None and grown['zero_reference_covariance']


def test_offdiagonal_metric_is_invariant_to_basis_signs_and_permutations():
    c = torch.tensor([[3., .2], [.2, 1.]])
    q = torch.tensor([[.8, -.6], [.6, .8]])
    a = factor_metrics(c, q, c, q)
    b = factor_metrics(5*c, q[:, [1, 0]]*torch.tensor([-1., 1.]), 5*c, q)
    assert a['current_relative_offdiagonal'] == pytest.approx(b['current_relative_offdiagonal'], rel=1e-6)
    assert a['participation_rank'] == pytest.approx(b['participation_rank'], rel=1e-6)


@pytest.mark.parametrize('arm', ['qr', 'warm'])
def test_observation_preserves_entire_transition_and_releases_reference_storage(arm):
    generator = torch.Generator().manual_seed(213)
    p = torch.nn.Parameter(torch.randn(5, 7, generator=generator)*.01)
    q = torch.nn.Parameter(p.detach().clone())
    args = dict(arm=arm, shampoo_beta=.99, block=4, maximum=8, compile_kernels=False)
    control = CovarianceBetaSOAP([('weight', p)], **args)
    observed = CovarianceBetaSOAP([('weight', q)], **args)
    owner = TensorOwnership()
    observer = BasisDiagnostics(observed, [dict(parameter='weight',side='l',tile=0),dict(parameter='weight',side='r',tile=1)], [[50,100]], owner)
    assert observer.calls == {50,99,100}
    measurements = []
    for call in range(1,112):
        p.grad = torch.randn(p.shape, generator=generator)*.03
        q.grad = p.grad.clone()
        assert control.step() == observed.step()
        if call in observer.calls:
            grad_id = id(q.grad)
            measurements.append(observer.measure(observed, call))
            assert observed.call == call and id(q.grad) == grad_id
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        torch.testing.assert_close(control.step_tensor, observed.step_tensor, rtol=0, atol=0)
        for key, value in control.groups[0]['state'].items():
            torch.testing.assert_close(value, observed.groups[0]['state'][key], rtol=0, atol=0)
    assert [x['elapsed_updates'] for x in measurements] == [0,49,50]
    assert [x['references_released'] for x in measurements] == [False,False,True]
    observer.close()
    gc.collect()
    release = owner.report()
    assert release['tracked_tensors'] == 4 and release['all_registered_released']
    owner.close()


def test_factor_resolution_handles_bucket_members_tiles_and_one_sided_axes():
    params = [('a',torch.nn.Parameter(torch.zeros(5,7))),('b',torch.nn.Parameter(torch.zeros(5,7))),
              ('embed',torch.nn.Parameter(torch.zeros(11,4))),('readout',torch.nn.Parameter(torch.zeros(4,11)))]
    opt = CovarianceBetaSOAP(params, block=4, maximum=8, compile_kernels=False)
    factors = resolve_factors(opt,[dict(parameter='b',side='r',tile=3),dict(parameter='embed',side='r'),dict(parameter='readout',side='l')])
    assert factors[0]['factor_index'] == 7
    with pytest.raises(AssertionError):
        resolve_factors(opt,[dict(parameter='embed',side='l')])
    with pytest.raises(AssertionError):
        resolve_factors(opt,[dict(parameter='a',side='l',tile=4)])


def test_fresh_model_seed_changes_initialization_and_repeats_exactly():
    kwargs = dict(device='cpu',width=8,depth=2,expansion=4,head_dim=4,vocab=19,sequence=7)
    a, b, old = model_for_seed(204,**kwargs), model_for_seed(204,**kwargs), model_for_seed(203,**kwargs)
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert not torch.equal(a.embed,old.embed)
