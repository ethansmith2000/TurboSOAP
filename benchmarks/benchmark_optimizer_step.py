"""Benchmark complete dense/block and one-/two-sided SOAP optimizer steps."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soap import SOAP  # noqa: E402
from soap_block import BlockSOAP  # noqa: E402


def _state_bytes(value, seen: set[int]) -> int:
    if isinstance(value, torch.Tensor):
        pointer = value.untyped_storage().data_ptr()
        if pointer in seen:
            return 0
        seen.add(pointer)
        return value.untyped_storage().nbytes()
    if isinstance(value, dict):
        return sum(_state_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_state_bytes(item, seen) for item in value)
    return 0


def benchmark_variant(
    name: str,
    optimizer_class,
    optimizer_kwargs: dict,
    initial_parameter: torch.Tensor,
    gradients: list[torch.Tensor],
    warmup_steps: int,
    measured_steps: int,
) -> dict:
    device = initial_parameter.device
    parameter = torch.nn.Parameter(initial_parameter.clone())
    optimizer = optimizer_class([parameter], **optimizer_kwargs)

    # The first observation creates covariance state and eigenbases without a
    # parameter update. Later warmup steps initialize kernels and steady state.
    parameter.grad = gradients[0]
    optimizer.step()
    for index in range(warmup_steps):
        parameter.grad = gradients[1 + index]
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)

    events = []
    cpu_update_ms = 0.0
    wall_start = time.perf_counter()
    start_index = 1 + warmup_steps
    for offset in range(measured_steps):
        parameter.grad = gradients[start_index + offset]
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            optimizer.step()
            end.record()
            events.append((start, end))
        else:
            start = time.perf_counter()
            optimizer.step()
            cpu_update_ms += 1000.0 * (time.perf_counter() - start)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        update_ms = sum(start.elapsed_time(end) for start, end in events)
        peak_extra_bytes = max(
            0, torch.cuda.max_memory_allocated(device) - baseline_memory
        )
    else:
        update_ms = cpu_update_ms
        peak_extra_bytes = 0
    wall_ms = 1000.0 * (time.perf_counter() - wall_start)
    state = optimizer.state[parameter]
    delta = parameter.detach().float() - initial_parameter.float()
    active_factors = sum(value is not None for value in state["Q"])
    return {
        "variant": name,
        "update_milliseconds_per_step": update_ms / measured_steps,
        "wall_milliseconds_per_step": wall_ms / measured_steps,
        "state_megabytes": _state_bytes(state, set()) / 2**20,
        "peak_extra_megabytes": peak_extra_bytes / 2**20,
        "active_factors": active_factors,
        "basis_refreshes": int(state["basis_refreshes"]),
        "parameter_delta_rms": float(delta.square().mean().sqrt()),
        "finite": bool(parameter.isfinite().all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--columns", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--precondition-frequency", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--normalize-grads", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup_steps < 0:
        raise ValueError("steps must be positive and warmup_steps non-negative")

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial = torch.randn(
        args.rows,
        args.columns,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        generator=generator,
    )
    gradient_count = 1 + args.warmup_steps + args.steps
    base = torch.randn(
        initial.shape, device=device, dtype=initial.dtype, generator=generator
    )
    gradients = [
        base
        + (0.03 * index)
        * torch.randn(
            initial.shape,
            device=device,
            dtype=initial.dtype,
            generator=generator,
        )
        for index in range(gradient_count)
    ]
    common = {
        "lr": 3e-3,
        "betas": (0.95, 0.99),
        "shampoo_beta": 0.999,
        "weight_decay": 0.0,
        "precondition_frequency": args.precondition_frequency,
        "basis_reset_frequency": 0,
        "basis_track_stats": False,
        "normalize_grads": args.normalize_grads,
        "covariance_compute_dtype": "bfloat16" if device.type == "cuda" else "float32",
    }
    variants = (
        ("dense_two_sided", SOAP, {**common, "precondition_mode": "all"}),
        (
            "dense_one_sided",
            SOAP,
            {**common, "precondition_mode": "smaller_side"},
        ),
        (
            "block_two_sided",
            BlockSOAP,
            {
                **common,
                "precondition_mode": "all",
                "basis_block_size": args.block_size,
                "basis_block_unbiased_scale": False,
            },
        ),
        (
            "block_one_sided",
            BlockSOAP,
            {
                **common,
                "precondition_mode": "smaller_side",
                "basis_block_size": args.block_size,
                "basis_block_unbiased_scale": False,
            },
        ),
    )
    results = []
    for name, optimizer_class, kwargs in variants:
        results.append(
            benchmark_variant(
                name,
                optimizer_class,
                kwargs,
                initial,
                gradients,
                args.warmup_steps,
                args.steps,
            )
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "results": results,
    }
    serialized = json.dumps(output, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
