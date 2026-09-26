"""Deterministic covariance replay for dense, held, pair, and block SOAP bases."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soap import SOAP, _offdiag, _sym  # noqa: E402
from soap_block import (  # noqa: E402
    _apply_block_axis,
    _rotate_axis,
    _select_block_rotations,
    _select_cayley_blocks,
)


def build_covariance_trajectory(
    kind: str,
    size: int,
    steps: int,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    base = torch.linalg.qr(torch.randn(size, size, generator=generator)).Q
    skew_seed = torch.randn(size, size, generator=generator)
    skew = skew_seed - skew_seed.T
    skew /= skew.norm().clamp_min(1e-12)
    eigenvalues = torch.linspace(0.2, 2.0, size)
    trajectory: list[torch.Tensor] = []

    for step in range(steps):
        progress = step / max(1, steps - 1)
        if kind == "rotating":
            angle = 1.2 * progress
            rotation = torch.matrix_exp(angle * skew)
            covariance = rotation @ base @ torch.diag(eigenvalues) @ base.T @ rotation.T
        elif kind == "shock":
            angle = 0.2 * progress + (0.8 if progress >= 0.5 else 0.0)
            rotation = torch.matrix_exp(angle * skew)
            covariance = rotation @ base @ torch.diag(eigenvalues) @ base.T @ rotation.T
        elif kind == "equal_diagonal":
            local = torch.diag(eigenvalues)
            if progress >= 0.375:
                shared = 1.0 + 0.2 * progress
                local[0, 0] = shared
                local[1, 1] = shared
                local[0, 1] = local[1, 0] = 0.75
            covariance = base @ local @ base.T
        else:
            raise ValueError(f"unknown trajectory: {kind!r}")
        trajectory.append(_sym(covariance))
    return trajectory


def _basis_errors(covariance: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    qf = q.float()
    b = _sym(qf.T @ covariance.float() @ qf)
    identity = torch.eye(q.shape[1], device=q.device)
    residual = covariance.float() @ qf - qf * b.diagonal().unsqueeze(0)
    return torch.stack(
        (
            _offdiag(b).norm() / b.norm().clamp_min(1e-12),
            residual.norm() / covariance.float().norm().clamp_min(1e-12),
            (qf.T @ qf - identity).norm() / math.sqrt(max(1, q.shape[1])),
        )
    )


def run_method(
    covariances: list[torch.Tensor],
    method: str,
    device: torch.device,
    reset_every: int,
    pairs_per_refresh: int,
    block_size: int,
    basis_lr: float,
    damping: float,
    rotation_cap: float,
    stall_threshold: float,
    unbiased_scale: bool,
    seed: int,
) -> dict:
    q = None
    resets = 0
    refreshes = 0
    errors: list[torch.Tensor] = []
    event_pairs = []
    cpu_update_ms = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    for step, host_covariance in enumerate(covariances):
        covariance = host_covariance.to(device)
        if device.type == "cuda":
            update_start = torch.cuda.Event(enable_timing=True)
            update_end = torch.cuda.Event(enable_timing=True)
            update_start.record()
        else:
            update_start = time.perf_counter()
        reset = (
            q is None
            or method == "full_eigh"
            or (reset_every > 0 and step > 0 and step % reset_every == 0)
        )
        if reset:
            q = SOAP._eigh_basis(covariance)
            resets += 1
        else:
            b = _sym(q.T @ covariance.float() @ q)
            if method == "held_reset":
                pass
            elif method == "jacobi_pairs":
                rotations = _select_block_rotations(
                    b,
                    pairs_per_refresh,
                    math.pi / 8,
                    0.0,
                    8,
                    "round_robin",
                    step - 1,
                )
                q = _rotate_axis(q, 1, rotations)
            elif method == "cayley_blocks":
                blocks = _select_cayley_blocks(
                    b,
                    block_size,
                    basis_lr,
                    damping,
                    rotation_cap,
                    0.0,
                    unbiased_scale,
                    step - 1,
                    seed,
                    stall_threshold,
                )
                q = _apply_block_axis(q, 1, blocks)
            else:
                raise ValueError(f"unknown method: {method!r}")
            refreshes += method != "held_reset"
        if device.type == "cuda":
            update_end.record()
            event_pairs.append((update_start, update_end))
        else:
            cpu_update_ms += 1000.0 * (time.perf_counter() - update_start)
        errors.append(_basis_errors(covariance, q))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        update_elapsed_ms = sum(
            start.elapsed_time(end) for start, end in event_pairs
        )
    else:
        update_elapsed_ms = cpu_update_ms
    wall_elapsed_ms = 1000.0 * (time.perf_counter() - wall_start)
    error_tensor = torch.stack(errors).cpu()
    labels = ("offdiagonal_error", "eigen_residual", "orthogonality_error")
    result = {
        "method": method,
        "update_elapsed_ms": update_elapsed_ms,
        "update_milliseconds_per_step": update_elapsed_ms / len(covariances),
        "wall_elapsed_ms": wall_elapsed_ms,
        "wall_milliseconds_per_step": wall_elapsed_ms / len(covariances),
        "dense_resets": resets,
        "warm_refreshes": int(refreshes),
    }
    for index, label in enumerate(labels):
        result[f"mean_{label}"] = float(error_tensor[:, index].mean())
        result[f"max_{label}"] = float(error_tensor[:, index].max())
        result[f"final_{label}"] = float(error_tensor[-1, index])
    return result


def replay(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    kinds = (
        ("rotating", "shock", "equal_diagonal")
        if args.trajectory == "all"
        else (args.trajectory,)
    )
    output = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "trajectories": [],
    }
    for kind in kinds:
        covariances = build_covariance_trajectory(
            kind, args.size, args.steps, args.seed
        )
        if device.type == "cuda":
            covariance = covariances[0].to(device)
            q = SOAP._eigh_basis(covariance)
            b = _sym(q.T @ covariance @ q)
            rotations = _select_block_rotations(
                b, args.pairs_per_refresh, math.pi / 8, 0.0, 8,
                "round_robin", 0,
            )
            _rotate_axis(q, 1, rotations)
            blocks = _select_cayley_blocks(
                b,
                args.block_size,
                args.basis_lr,
                args.damping,
                args.rotation_cap,
                0.0,
                args.unbiased_scale,
                0,
                args.seed,
                args.stall_threshold,
            )
            _apply_block_axis(q, 1, blocks)
            _basis_errors(covariance, q)
            torch.cuda.synchronize(device)
        methods = [
            run_method(
                covariances,
                method,
                device,
                args.reset_every,
                args.pairs_per_refresh,
                args.block_size,
                args.basis_lr,
                args.damping,
                args.rotation_cap,
                args.stall_threshold,
                args.unbiased_scale,
                args.seed,
            )
            for method in ("full_eigh", "held_reset", "jacobi_pairs", "cayley_blocks")
        ]
        output["trajectories"].append({"name": kind, "methods": methods})
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--trajectory",
        choices=("all", "rotating", "shock", "equal_diagonal"),
        default="all",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--reset-every", type=int, default=16)
    parser.add_argument("--pairs-per-refresh", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--basis-lr", type=float, default=0.5)
    parser.add_argument("--damping", type=float, default=1e-2)
    parser.add_argument("--rotation-cap", type=float, default=0.25)
    parser.add_argument("--stall-threshold", type=float, default=1e-6)
    parser.add_argument(
        "--unbiased-scale", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = replay(arguments)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n")
