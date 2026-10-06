import pytest
import torch

from soap import SOAP, _orthogonalize_ns
from benchmarks.warm_retraction import VARIANTS, Retraction, candidate, retract, warm


def fixture():
    gen = torch.Generator().manual_seed(981)
    q = torch.linalg.qr(torch.randn(16, 16, generator=gen)).Q
    g = torch.randn(16, 24, generator=gen)
    cov = g @ g.T
    group = dict(basis_lr=.5, basis_jacobi_damping=.01, basis_rotation_cap=.1,
                 basis_rotation_cap_mode='average', basis_ns_iterations=2)
    return cov, q, group


@pytest.mark.parametrize('iterations', [2, 6])
def test_scaled_control_matches_actual_kernel(iterations):
    cov, q, group = fixture()
    group['basis_ns_iterations'] = iterations
    variant = Retraction('control', iterations=iterations, scaled=True)
    assert torch.equal(warm(cov, q, group, variant), SOAP._gauge_step_one(cov, q, group))
    x, _, _ = candidate(cov, q, group)
    assert torch.equal(retract(x, variant), _orthogonalize_ns(x, iterations))


def test_gram_identity_with_inherited_drift():
    cov, q, group = fixture()
    q = q.double() * 1.001
    a = torch.randn(16, 16, dtype=torch.float64) * .01
    a = a-a.T
    eye = torch.eye(16, dtype=torch.float64)
    x = q @ (eye+a)
    e = q.T@q-eye
    assert torch.allclose(x.T@x-eye, -a@a+(eye-a)@e@(eye+a), atol=1e-14)


@pytest.mark.parametrize('order', [3, 5])
def test_one_correction_on_known_small_rotation(order):
    x = torch.tensor([[1., .1], [-.1, 1.]])
    y = retract(x, Retraction('local', order=order))
    error = torch.linalg.matrix_norm(y.T@y-torch.eye(2), ord=2)
    assert error < (8e-5 if order == 3 else 1e-6)


def test_covariance_scaling_and_input_immutability():
    cov, q, group = fixture()
    original = (cov.clone(), q.clone())
    reference = warm(cov, q, group, VARIANTS[4])
    for scale in [.001, 1., 1000.]:
        actual = warm(cov*scale, q, group, next(v for v in VARIANTS if v.name=='rawcov_quintic1'))
        assert torch.allclose(actual, reference, atol=2e-6, rtol=2e-5)
    assert torch.equal(cov, original[0]) and torch.equal(q, original[1])


@pytest.mark.parametrize('variant', [Retraction('bad', order=7), Retraction('bad', iterations=0)])
def test_invalid_polynomial(variant):
    with pytest.raises(ValueError):
        retract(torch.eye(2), variant)


def test_spectral_cap_bounds_movement_and_improves_large_step():
    cov,q,group=fixture()
    group.update(basis_rotation_cap_mode='spectral_bound',basis_rotation_cap=.5)
    x,a,_=candidate(cov,q,group)
    assert torch.linalg.matrix_norm(a,ord=2)<=.50001
    variant=next(v for v in VARIANTS if v.name=='spectral_quintic1_cap05')
    y=warm(cov,q,group,variant)
    assert torch.linalg.matrix_norm(y.T@y-torch.eye(16),ord=2)<.01


def test_cifar_experimental_hook_uses_declared_configuration():
    from soap_reference import ResearchSOAP
    from benchmarks.cifar_retraction_screen import configure_matrix
    cov,q,group=fixture()
    opt=ResearchSOAP([torch.nn.Parameter(torch.zeros_like(q))],basis_method='warm')
    configure_matrix(opt,'spectral_quintic1_cap05')
    actual=opt._gauge_step_one(cov,q,opt.param_groups[0])
    variant=next(v for v in VARIANTS if v.name=='spectral_quintic1_cap05')
    assert torch.equal(actual,warm(cov,q,group,variant))
    assert opt.param_groups[0]['basis_rotation_cap_mode']=='spectral_bound'
    with pytest.raises(ValueError):configure_matrix(opt,'not-a-variant')
