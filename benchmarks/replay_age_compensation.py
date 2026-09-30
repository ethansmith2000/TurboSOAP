"""Replay fixed and sparse dense SOAP trackers on covariance trajectories."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.replay_basis_tracking import (  # noqa: E402
    _basis_errors,
    build_covariance_trajectory,
)
from soap import SOAP  # noqa: E402


def _effective_lr(basis_lr: float, age: int, reference_age: int) -> float:
    if basis_lr == 0.0 or age == 0:
        return 0.0
    return -math.expm1((age / reference_age) * math.log1p(-basis_lr))


def run_tracker(
    covariances: list[torch.Tensor],
    *,
    base_frequency: int,
    later_frequency: int,
    warmup_steps: int,
    basis_lr: float,
    damping: float,
    rotation_cap: float,
    ns_iterations: int,
    age_compensation: bool,
) -> dict:
    group = {
        "basis_lr": basis_lr,
        "basis_jacobi_damping": damping,
        "basis_rotation_cap": rotation_cap,
        "basis_rotation_cap_mode": "average",
        "basis_stall_pair_threshold": 0.0,
        "basis_ns_iterations": ns_iterations,
    }
    q = SOAP._eigh_basis(covariances[0])
    last_refresh = 0
    refreshes = 0
    effective_lrs: list[float] = []
    errors = [_basis_errors(covariances[0], q)]
    for step, covariance in enumerate(covariances[1:], start=1):
        frequency = base_frequency if step < warmup_steps else later_frequency
        if step % frequency == 0:
            age = step - last_refresh
            effective_lr = (
                _effective_lr(basis_lr, age, base_frequency)
                if age_compensation
                else basis_lr
            )
            q = SOAP._gauge_step_one(
                covariance,
                q,
                group,
                effective_basis_lr=effective_lr,
            )
            last_refresh = step
            refreshes += 1
            effective_lrs.append(effective_lr)
        errors.append(_basis_errors(covariance, q))

    values = torch.stack(errors)
    post_warmup = values[warmup_steps:]
    labels = ("offdiagonal_error", "eigen_residual", "orthogonality_error")
    result = {
        "refreshes": refreshes,
        "mean_effective_lr": sum(effective_lrs) / max(1, len(effective_lrs)),
        "maximum_effective_lr": max(effective_lrs, default=0.0),
    }
    for index, label in enumerate(labels):
        result[f"mean_post_warmup_{label}"] = float(post_warmup[:, index].mean())
        result[f"p95_post_warmup_{label}"] = float(
            torch.quantile(post_warmup[:, index], 0.95)
        )
        result[f"final_{label}"] = float(values[-1, index])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--base-frequency", type=int, default=10)
    parser.add_argument("--later-frequency", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--basis-lr", type=float, default=0.5)
    parser.add_argument("--damping", type=float, default=1e-2)
    parser.add_argument("--rotation-cap", type=float, default=0.1)
    parser.add_argument("--ns-iterations", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    profiles = {
        "fixed_10": (args.base_frequency, False),
        "10_to_20": (args.later_frequency, False),
        "10_to_20_age_compensated": (args.later_frequency, True),
    }
    result = {"config": vars(args).copy(), "trajectories": {}}
    result["config"]["output"] = str(args.output) if args.output else None
    for kind in ("rotating", "shock", "equal_diagonal"):
        covariances = build_covariance_trajectory(
            kind, args.size, args.steps, args.seed
        )
        result["trajectories"][kind] = {
            name: run_tracker(
                covariances,
                base_frequency=args.base_frequency,
                later_frequency=later_frequency,
                warmup_steps=args.warmup_steps,
                basis_lr=args.basis_lr,
                damping=args.damping,
                rotation_cap=args.rotation_cap,
                ns_iterations=args.ns_iterations,
                age_compensation=age_compensation,
            )
            for name, (later_frequency, age_compensation) in profiles.items()
        }

    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
