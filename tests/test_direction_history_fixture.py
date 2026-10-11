"""Independent oracle/convention checks, not full fixture or historical reruns."""
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import direction_history_fixture as fixture


def test_two_sided_row_major_history_oracle():
    fixture.cpu_guard()
    history = [torch.tensor([[1., 2., -3.], [.2, -.5, 4.]], dtype=torch.float64),
               torch.tensor([[-2., 1., .5], [3., .1, -1.]], dtype=torch.float64)]
    q = [fixture.rotation(2, .31), fixture.rotation(3, -.24)]
    _, direct = fixture.history_moments(history, q)
    full = fixture.full_s_diagonal(fixture.full_second_moment(history), q, (2, 3))
    assert fixture.relative(full, direct) < 1e-12
    # An independent single row/column weighted scalar check fixes axis convention.
    expected = sum((1 - fixture.BETA2) * fixture.BETA2 ** (len(history) - 1 - i)
                   * float(q[0][:, 1] @ g @ q[1][:, 2]) ** 2 for i, g in enumerate(history))
    assert abs(float(direct[1, 2]) - expected) < 1e-14


def test_update_bias_and_uncorrected_epsilon_convention():
    fixture.cpu_guard()
    g = torch.tensor([[2.]], dtype=torch.float64)
    q = [torch.eye(1, dtype=torch.float64)] * 2
    zero = torch.zeros_like(g)
    u = fixture.adaptive_update(zero, zero, g, q, 0)
    expected = 2 / (2 + fixture.EPS / math.sqrt(1 - fixture.BETA2))
    assert abs(float(u) - expected) < 1e-13


def test_known_relative_m_transition_preserves_two_sided_world():
    fixture.cpu_guard()
    old = [fixture.rotation(2, -.17), fixture.rotation(3, .23)]
    r = [fixture.rotation(2, .42), fixture.rotation(3, -.11)]
    new = [a @ b for a, b in zip(old, r)]
    m = torch.tensor([[1., -2., .3], [.7, .2, -4.]], dtype=torch.float64)
    transported = r[0].T @ m @ r[1]
    assert fixture.relative(fixture.world(transported, new), fixture.world(m, old)) < 1e-12


def test_diagonal_v_loop_has_independent_scalar_prediction():
    fixture.cpu_guard()
    theta = .35
    r = fixture.rotation(2, theta)
    original = torch.tensor([.01, 2.], dtype=torch.float64)
    forward = r.square().T @ original
    back = r.square() @ forward
    mixing = 2 * math.sin(theta) ** 2 * math.cos(theta) ** 2
    expected_first = original[0] * (1 - mixing) + original[1] * mixing
    assert abs(float(back[0] - expected_first)) < 1e-14
    assert float((back - original).norm()) > .1
