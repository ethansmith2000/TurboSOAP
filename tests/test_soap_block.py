import math

import torch

from soap import _offdiag
from soap_block import (
    BlockSOAP,
    TurboBlockSOAP,
    _apply_block_axis,
    _rotate_axis,
    _rotate_basis_covariance,
    _select_cayley_blocks,
    _select_block_rotations,
    _transform_basis_covariance_blocks,
)


def test_alias_is_preserved():
    assert TurboBlockSOAP is BlockSOAP


def test_block_probability_compensation_is_opt_in():
    parameter = torch.nn.Parameter(torch.zeros(8, 8))
    optimizer = BlockSOAP([parameter], basis_block_size=4)

    assert optimizer.param_groups[0]["basis_block_unbiased_scale"] is False


def test_block_rotation_preserves_world_covariance_and_orthogonality():
    torch.manual_seed(41)
    size = 12
    q = torch.linalg.qr(torch.randn(size, size)).Q
    value = torch.randn(size, size)
    b = value @ value.T
    rotations = _select_block_rotations(
        b,
        pairs_per_refresh=4,
        angle_cap=math.pi / 4,
        min_correlation=0.0,
        candidate_multiplier=8,
    )

    q_new = _rotate_axis(q, 1, rotations)
    b_new = _rotate_basis_covariance(b, rotations)

    world_before = q @ b @ q.T
    world_after = q_new @ b_new @ q_new.T
    assert torch.allclose(world_after, world_before, atol=2e-5, rtol=2e-5)
    assert torch.allclose(q_new.T @ q_new, torch.eye(size), atol=2e-5)


def test_capped_parallel_matching_does_not_increase_offdiagonal_energy():
    torch.manual_seed(42)
    size = 32
    value = torch.randn(size, size)
    covariance = value @ value.T
    rotations = _select_block_rotations(
        covariance,
        pairs_per_refresh=0,
        angle_cap=math.pi / 8,
        min_correlation=0.0,
        candidate_multiplier=1,
        selection="round_robin",
        round_index=0,
    )

    updated = _rotate_basis_covariance(covariance, rotations)

    assert _offdiag(updated).norm() <= _offdiag(covariance).norm()


def test_cayley_blocks_preserve_world_covariance_and_reduce_error():
    torch.manual_seed(44)
    size = 32
    value = torch.randn(size, size)
    covariance = value @ value.T
    q = torch.linalg.qr(torch.randn(size, size)).Q
    blocks = _select_cayley_blocks(
        covariance,
        block_size=8,
        step_size=0.5,
        damping=1e-2,
        rotation_cap=0.1,
        min_correlation=0.0,
    )

    q_new = _apply_block_axis(q, 1, blocks)
    covariance_new = _transform_basis_covariance_blocks(covariance, blocks)

    assert _offdiag(covariance_new).norm() <= _offdiag(covariance).norm()
    assert torch.allclose(
        q_new @ covariance_new @ q_new.T,
        q @ covariance @ q.T,
        atol=3e-5,
        rtol=3e-5,
    )
    assert torch.allclose(q_new.T @ q_new, torch.eye(size), atol=2e-5)


def test_cayley_block_escapes_equal_diagonal_stall():
    covariance = torch.eye(8)
    covariance[0, 1] = covariance[1, 0] = 0.8
    offdiagonal_before = _offdiag(covariance).norm()

    blocks = _select_cayley_blocks(
        covariance,
        block_size=8,
        step_size=0.5,
        damping=1e-2,
        rotation_cap=0.25,
        min_correlation=0.0,
    )
    covariance_new = _transform_basis_covariance_blocks(covariance, blocks)

    assert _offdiag(covariance_new).norm() < offdiagonal_before


def test_cayley_block_schedule_is_reproducible_from_refresh_index():
    torch.manual_seed(51)
    value = torch.randn(24, 24)
    covariance = value @ value.T

    first = _select_cayley_blocks(
        covariance, 8, 0.5, 1e-2, 0.1, 0.0, round_index=7, seed=123
    )
    torch.rand(100)
    repeated = _select_cayley_blocks(
        covariance, 8, 0.5, 1e-2, 0.1, 0.0, round_index=7, seed=123
    )
    next_round = _select_cayley_blocks(
        covariance, 8, 0.5, 1e-2, 0.1, 0.0, round_index=8, seed=123
    )

    assert torch.equal(first[0], repeated[0])
    assert torch.allclose(first[1], repeated[1])
    assert not torch.equal(first[0], next_round[0])


def test_block_rotation_reduces_offdiagonal_covariance():
    covariance = torch.tensor(
        [
            [1.0, 0.9, 0.0, 0.0],
            [0.9, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.7],
            [0.0, 0.0, 0.7, 2.0],
        ]
    )
    rotations = _select_block_rotations(
        covariance,
        pairs_per_refresh=2,
        angle_cap=math.pi / 4,
        min_correlation=0.0,
        candidate_multiplier=4,
    )

    updated = _rotate_basis_covariance(covariance, rotations)

    assert _offdiag(updated).norm() < 1e-6


def test_round_robin_schedule_visits_every_pair():
    size = 6
    covariance = torch.ones(size, size) + torch.eye(size)
    visited = set()
    for round_index in range(size - 1):
        rotations = _select_block_rotations(
            covariance,
            pairs_per_refresh=size // 2,
            angle_cap=math.pi / 4,
            min_correlation=0.0,
            candidate_multiplier=1,
            selection="round_robin",
            round_index=round_index,
        )
        first, second, _, _ = rotations
        visited.update(
            tuple(sorted(pair))
            for pair in zip(first.tolist(), second.tolist())
        )

    assert len(visited) == size * (size - 1) // 2


def test_partial_round_robin_schedule_eventually_visits_every_pair():
    size = 6
    pairs_per_refresh = 1
    covariance = torch.ones(size, size) + torch.eye(size)
    visited = set()
    refreshes = (size - 1) * math.ceil((size // 2) / pairs_per_refresh)
    for round_index in range(refreshes):
        rotations = _select_block_rotations(
            covariance,
            pairs_per_refresh=pairs_per_refresh,
            angle_cap=math.pi / 4,
            min_correlation=0.0,
            candidate_multiplier=1,
            selection="round_robin",
            round_index=round_index,
        )
        first, second, _, _ = rotations
        visited.add(tuple(sorted((first.item(), second.item()))))

    assert len(visited) == size * (size - 1) // 2


def test_squared_rotation_transports_diagonal_variance():
    angle = torch.tensor(math.pi / 4)
    rotations = (
        torch.tensor([0]),
        torch.tensor([1]),
        angle.cos().unsqueeze(0),
        angle.sin().unsqueeze(0),
    )
    variance = torch.tensor([100.0, 1.0])

    transported = _rotate_axis(variance, 0, rotations, squared=True)

    assert torch.allclose(transported, torch.tensor([50.5, 50.5]), atol=1e-5)


def test_optimizer_tracks_covariance_in_current_basis():
    torch.manual_seed(43)
    parameter = torch.nn.Parameter(torch.randn(8, 6))
    optimizer = BlockSOAP(
        [parameter],
        lr=1e-2,
        weight_decay=0.0,
        precondition_frequency=1,
        basis_pairs_per_refresh=2,
        basis_pair_angle_cap=math.pi / 4,
        basis_pair_min_correlation=0.0,
        basis_reset_frequency=0,
        covariance_compute_dtype="float32",
        basis_track_stats=True,
    )

    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    for covariance in state["GG"]:
        assert _offdiag(covariance).norm() < 1e-4

    parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    assert state["basis_refreshes"] == 1
    assert parameter.isfinite().all()
    for q in state["Q"]:
        assert torch.allclose(q.T @ q, torch.eye(q.shape[1]), atol=2e-5)


def test_projected_covariance_ema_matches_world_coordinates():
    torch.manual_seed(45)
    parameter = torch.nn.Parameter(torch.randn(7, 5))
    optimizer = BlockSOAP(
        [parameter],
        shampoo_beta=0.9,
        precondition_frequency=100,
        covariance_compute_dtype="float32",
        basis_track_stats=False,
    )
    first_gradient = torch.randn_like(parameter)
    parameter.grad = first_gradient
    optimizer.step()
    state = optimizer.state[parameter]
    bases = [q.clone() for q in state["Q"]]

    second_gradient = torch.randn_like(parameter)
    parameter.grad = second_gradient
    optimizer.step()

    expected_left = 0.9 * (0.1 * first_gradient @ first_gradient.T)
    expected_left += 0.1 * second_gradient @ second_gradient.T
    expected_right = 0.9 * (0.1 * first_gradient.T @ first_gradient)
    expected_right += 0.1 * second_gradient.T @ second_gradient
    reconstructed_left = bases[0] @ state["GG"][0] @ bases[0].T
    reconstructed_right = bases[1] @ state["GG"][1] @ bases[1].T

    assert torch.allclose(reconstructed_left, expected_left, atol=2e-5, rtol=2e-5)
    assert torch.allclose(reconstructed_right, expected_right, atol=2e-5, rtol=2e-5)


def test_dense_reset_path_keeps_optimizer_finite():
    torch.manual_seed(47)
    parameter = torch.nn.Parameter(torch.randn(5, 4))
    optimizer = BlockSOAP(
        [parameter],
        precondition_frequency=1,
        basis_reset_frequency=1,
        basis_reset_method="eigh",
        covariance_compute_dtype="float32",
        basis_track_stats=False,
    )

    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    assert parameter.isfinite().all()
    assert optimizer.state[parameter]["basis_refreshes"] == 2


def test_dense_reset_repairs_basis_singular_value_drift():
    torch.manual_seed(48)
    size = 6
    parameter = torch.nn.Parameter(torch.zeros(size, size))
    optimizer = BlockSOAP([parameter], covariance_compute_dtype="float32")
    orthogonal = torch.linalg.qr(torch.randn(size, size)).Q
    q_old = orthogonal @ torch.diag(torch.linspace(0.8, 1.2, size))
    value = torch.randn(size, size)
    moving_covariance = value @ value.T
    world_covariance = q_old @ moving_covariance @ q_old.T

    q_new, b_new, transition = optimizer._dense_reset_in_moving_basis(
        moving_covariance, q_old, "eigh"
    )

    assert torch.allclose(q_new.T @ q_new, torch.eye(size), atol=2e-5)
    assert torch.allclose(
        q_new @ b_new @ q_new.T,
        world_covariance,
        atol=5e-5,
        rtol=5e-5,
    )
    assert torch.allclose(transition, q_old.T @ q_new, atol=1e-6)


def test_cayley_block_optimizer_path_keeps_optimizer_finite():
    torch.manual_seed(49)
    parameter = torch.nn.Parameter(torch.randn(16, 12))
    optimizer = BlockSOAP(
        [parameter],
        precondition_frequency=1,
        basis_block_size=8,
        basis_reset_frequency=0,
        transport_second_moment_on_warm=True,
        covariance_compute_dtype="float32",
        basis_track_stats=False,
    )

    for _ in range(4):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    assert parameter.isfinite().all()
    for q in optimizer.state[parameter]["Q"]:
        assert torch.allclose(q.T @ q, torch.eye(q.shape[1]), atol=2e-5)
