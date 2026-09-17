import math

import torch

from soap import (
    SOAP,
    TurboSlimSOAP,
    _basis_metrics,
    _clip_rotation,
    _damped_jacobi_generator,
    _largest_pair_jacobi_step,
    _offdiag,
    _orthogonalize_ns,
    _sym,
)


def _offdiag_energy(covariance: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    b = _sym(q.T @ covariance @ q)
    return 0.5 * _offdiag(b).square().sum()


def test_alias_is_preserved():
    assert TurboSlimSOAP is SOAP


def test_damped_jacobi_generator_is_skew_and_descends():
    torch.manual_seed(7)
    n = 16
    a = torch.randn(n, n)
    covariance = a @ a.T
    covariance /= covariance.trace() / n
    q = torch.linalg.qr(torch.randn(n, n)).Q
    b = _sym(q.T @ covariance @ q)

    omega = _damped_jacobi_generator(b, damping=1e-2)
    assert torch.allclose(omega + omega.T, torch.zeros_like(omega), atol=1e-6)

    before = _offdiag_energy(covariance, q)
    rotation = 1e-3 * omega
    q_new = _orthogonalize_ns(q + q @ rotation, iterations=3)
    after = _offdiag_energy(covariance, q_new)
    assert after < before


def test_clip_only_tracker_does_not_amplify_zero_signal():
    diagonal = torch.tensor([4.0, 2.0, 1.0, 0.5])
    b = torch.diag(diagonal)
    omega = _damped_jacobi_generator(b, damping=1e-2)
    assert torch.count_nonzero(omega) == 0


def test_spectral_bound_cap_controls_concentrated_rotation():
    rotation = torch.zeros(128, 128)
    rotation[0, 1] = 100.0
    rotation[1, 0] = -100.0

    clipped = _clip_rotation(rotation, 0.1, mode="spectral_bound")

    assert clipped.abs().sum(dim=-1).amax() <= 0.100001


def test_pair_step_escapes_equal_diagonal_stationary_point():
    covariance_in_basis = torch.tensor([[1.0, 0.9], [0.9, 1.0]])
    q = torch.eye(2)

    corrected = _largest_pair_jacobi_step(q, covariance_in_basis, math.pi / 4)
    diagonalized = corrected.T @ covariance_in_basis @ corrected

    assert diagonalized[0, 1].abs() < 1e-6
    assert torch.allclose(corrected.T @ corrected, torch.eye(2), atol=1e-6)


def test_capped_pair_fallback_chooses_a_descent_angle():
    torch.manual_seed(8)
    for _ in range(50):
        value = torch.randn(2, 2)
        covariance = value @ value.T
        before = _offdiag(covariance).norm()

        corrected = _largest_pair_jacobi_step(
            torch.eye(2), covariance, max_angle=0.1
        )
        after = _offdiag(corrected.T @ covariance @ corrected).norm()

        assert after <= before + 1e-6


def test_tracker_uses_pair_fallback_when_dense_flow_stalls():
    covariance = torch.tensor([[1.0, 0.9], [0.9, 1.0]])
    q = torch.eye(2)
    group = {
        "basis_jacobi_damping": 1e-2,
        "basis_lr": 0.5,
        "basis_rotation_cap": math.pi / 4,
        "basis_rotation_cap_mode": "average",
        "basis_stall_pair_threshold": 1e-3,
        "basis_ns_iterations": 2,
    }

    corrected = SOAP._gauge_step_one(covariance, q, group)

    assert _offdiag(corrected.T @ covariance @ corrected).norm() < 1e-6


def test_damped_tracker_handles_condition_number_1e5():
    torch.manual_seed(29)
    n = 12
    eigenvalues = torch.logspace(0, -5, n)
    eigenvectors = torch.linalg.qr(torch.randn(n, n)).Q
    covariance = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    q = torch.linalg.qr(torch.randn(n, n)).Q
    group = {
        "basis_jacobi_damping": 1e-2,
        "basis_lr": 0.5,
        "basis_rotation_cap": 0.1,
        "basis_ns_iterations": 2,
    }

    before = _basis_metrics(covariance, q)["diag_err"]
    for _ in range(50):
        q = SOAP._gauge_step_one(covariance, q, group)
    after = _basis_metrics(covariance, q)

    assert after["diag_err"] < 0.01 * before
    assert after["orth_err"] < 1e-5


def test_moments_and_master_basis_remain_fp32_between_refreshes():
    torch.manual_seed(11)
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = SOAP(
        [parameter],
        lr=1e-2,
        weight_decay=0.0,
        precondition_frequency=3,
        covariance_compute_dtype="float32",
        basis_reset_frequency=0,
        basis_track_stats=False,
    )

    parameter.grad = torch.randn_like(parameter)
    optimizer.step()  # state/basis initialization
    state = optimizer.state[parameter]
    old_basis = [None if q is None else q.clone() for q in state["Q"]]

    parameter.grad = torch.randn_like(parameter)
    projected = optimizer._project_with_basis(
        parameter.grad, state["Q"], optimizer.param_groups[0]
    )
    optimizer.step()

    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    assert torch.allclose(state["exp_avg"], (1.0 - 0.95) * projected, atol=1e-6)
    for old, new in zip(old_basis, state["Q"]):
        if old is not None:
            assert new.dtype == torch.float32
            assert torch.equal(old, new)


def test_hard_reset_second_moment_transport_follows_squared_overlap():
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = SOAP(
        [parameter],
        covariance_compute_dtype="float32",
        basis_track_stats=False,
    )
    group = optimizer.param_groups[0]
    identity = torch.eye(2)
    swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    state = {"exp_avg_sq": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

    optimizer._transport_second_moment_for_reset(
        state,
        [identity, identity],
        [swap, swap],
        group,
    )
    assert torch.equal(state["exp_avg_sq"], torch.tensor([[4.0, 3.0], [2.0, 1.0]]))


def test_general_tensor_projection_round_trip():
    torch.manual_seed(17)
    shape = (2, 3, 4)
    parameter = torch.nn.Parameter(torch.zeros(shape))
    optimizer = SOAP(
        [parameter],
        covariance_compute_dtype="float32",
        basis_track_stats=False,
    )
    basis = [torch.linalg.qr(torch.randn(size, size)).Q for size in shape]
    value = torch.randn(shape)

    projected = optimizer._project_with_basis(value, basis, optimizer.param_groups[0])
    restored = optimizer._project_back_with_basis(
        projected, basis, optimizer.param_groups[0]
    )
    assert torch.allclose(restored, value, atol=2e-6, rtol=2e-6)


def test_smaller_side_mode_keeps_only_small_matrix_factor():
    parameter = torch.nn.Parameter(torch.zeros(16, 4))
    optimizer = SOAP(
        [parameter],
        precondition_mode="smaller_side",
        covariance_compute_dtype="float32",
        basis_track_stats=False,
    )

    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]

    assert state["GG"][0] is None
    assert state["Q"][0] is None
    assert state["GG"][1].shape == (4, 4)
    assert state["Q"][1].shape == (4, 4)


def test_refresh_reports_orthogonal_basis_metrics():
    torch.manual_seed(19)
    parameter = torch.nn.Parameter(torch.randn(8, 6))
    optimizer = SOAP(
        [parameter],
        lr=1e-2,
        weight_decay=0.0,
        precondition_frequency=1,
        basis_reset_frequency=0,
        covariance_compute_dtype="float32",
    )

    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["basis_refreshes"] == 1
    assert optimizer.just_gathered_basis_stats
    assert math.isfinite(optimizer.latest_basis_stats["diag_err"])
    assert optimizer.latest_basis_stats["orth_err"] < 1e-4
