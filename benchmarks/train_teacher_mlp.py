"""Short teacher-student training validation for one-sided TurboSOAP."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soap import SOAP  # noqa: E402


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


def _evaluate(model, inputs, targets, batch_size: int) -> float:
    total = torch.zeros((), device=inputs.device)
    count = 0
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            prediction = model(inputs[start : start + batch_size]).float()
            target = targets[start : start + batch_size]
            total += (prediction - target).square().sum()
            count += prediction.numel()
    return float(total / count)


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


def _optimizer_diagnostics(optimizer: SOAP) -> dict:
    state_bytes = 0
    active_factors = 0
    basis_refreshes = 0
    hard_resets = 0
    for group in optimizer.param_groups:
        reset_frequency = int(group["basis_reset_frequency"])
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            state_bytes += _state_bytes(state, set())
            active_factors += sum(
                factor is not None for factor in state.get("Q", ())
            )
            refreshes = int(state.get("basis_refreshes", 0))
            basis_refreshes += refreshes
            if reset_frequency > 0:
                hard_resets += refreshes // reset_frequency
    return {
        "state_megabytes": state_bytes / 2**20,
        "active_factors": active_factors,
        "basis_refreshes": basis_refreshes,
        "hard_resets": hard_resets,
    }


def train_variant(
    name: str,
    optimizer_options: dict,
    initial_state: dict,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    validation_inputs: torch.Tensor,
    validation_targets: torch.Tensor,
    args: argparse.Namespace,
) -> dict:
    device = train_inputs.device
    model = MLP(args.input_dim, args.hidden_dim, args.output_dim).to(
        device=device, dtype=train_inputs.dtype
    )
    model.load_state_dict(initial_state)
    optimizer_options = dict(optimizer_options)
    beta2 = optimizer_options.pop("beta2")
    learning_rate = optimizer_options.pop("lr", args.lr)
    precondition_mode = optimizer_options["precondition_mode"]
    reset_frequency = optimizer_options.pop(
        "basis_reset_frequency", args.basis_reset_frequency
    )
    reset_method = optimizer_options.pop("basis_reset_method", "qr")
    optimizer = SOAP(
        model.parameters(),
        lr=learning_rate,
        betas=(args.beta1, beta2),
        shampoo_beta=args.shampoo_beta,
        eps=1e-8,
        weight_decay=0.0,
        precondition_frequency=args.precondition_frequency,
        precondition_1d=False,
        normalize_grads=True,
        basis_reset_frequency=reset_frequency,
        basis_reset_method=reset_method,
        basis_track_stats=False,
        covariance_compute_dtype=(
            "bfloat16" if device.type == "cuda" else "float32"
        ),
        **optimizer_options,
    )
    checkpoints = {
        0,
        args.steps // 4,
        args.steps // 2,
        3 * args.steps // 4,
        args.steps - 1,
    }
    curve = []
    step_events = []
    optimizer_events = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    for step in range(args.steps):
        batch_start = (step * args.batch_size) % (
            train_inputs.shape[0] - args.batch_size + 1
        )
        inputs = train_inputs[batch_start : batch_start + args.batch_size]
        targets = train_targets[batch_start : batch_start + args.batch_size]
        timed = step >= args.timing_warmup
        if timed and device.type == "cuda":
            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)
            optimizer_start = torch.cuda.Event(enable_timing=True)
            optimizer_end = torch.cuda.Event(enable_timing=True)
            step_start.record()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs).float()
        loss = (prediction - targets).square().mean()
        loss.backward()
        if timed and device.type == "cuda":
            optimizer_start.record()
        optimizer.step()
        if timed and device.type == "cuda":
            optimizer_end.record()
            step_end.record()
            step_events.append((step_start, step_end))
            optimizer_events.append((optimizer_start, optimizer_end))
        if step in checkpoints:
            curve.append(
                {"step": step + 1, "train_loss": float(loss.detach())}
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        training_ms = sum(start.elapsed_time(end) for start, end in step_events)
        optimizer_ms = sum(
            start.elapsed_time(end) for start, end in optimizer_events
        )
    else:
        training_ms = math.nan
        optimizer_ms = math.nan
    wall_ms = 1000.0 * (time.perf_counter() - wall_start)
    measured = max(1, args.steps - args.timing_warmup)
    return {
        "variant": name,
        "learning_rate": learning_rate,
        "beta2": beta2,
        "precondition_mode": precondition_mode,
        "basis_reset_frequency": reset_frequency,
        "basis_reset_method": reset_method,
        "validation_loss": _evaluate(
            model, validation_inputs, validation_targets, args.batch_size
        ),
        "final_train_loss": curve[-1]["train_loss"],
        "loss_curve": curve,
        "training_milliseconds_per_step": training_ms / measured,
        "optimizer_milliseconds_per_step": optimizer_ms / measured,
        "wall_milliseconds_per_step": wall_ms / args.steps,
        **_optimizer_diagnostics(optimizer),
    }


def run_seed(seed: int, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(seed)
    teacher = MLP(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    teacher.eval()
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    total_examples = args.train_examples + args.validation_examples
    inputs_float = torch.randn(
        total_examples,
        args.input_dim,
        device=device,
        generator=generator,
    )
    with torch.no_grad():
        targets = teacher(inputs_float).detach()
    inputs = inputs_float.to(dtype)
    del teacher, inputs_float

    torch.manual_seed(seed + 1000)
    initial_model = MLP(args.input_dim, args.hidden_dim, args.output_dim).to(
        device=device, dtype=dtype
    )
    initial_state = {
        key: value.detach().clone()
        for key, value in initial_model.state_dict().items()
    }
    del initial_model
    train_inputs = inputs[: args.train_examples]
    validation_inputs = inputs[args.train_examples :]
    train_targets = targets[: args.train_examples]
    validation_targets = targets[args.train_examples :]
    variants = (
        (
            "two_sided_beta099",
            {"precondition_mode": "all", "beta2": 0.99},
        ),
        (
            "two_sided_beta0999",
            {"precondition_mode": "all", "beta2": 0.999},
        ),
        (
            "two_sided_beta099_lr0025",
            {"precondition_mode": "all", "beta2": 0.99, "lr": 2.5e-3},
        ),
        (
            "two_sided_exact_beta099_lr0025",
            {
                "precondition_mode": "all",
                "beta2": 0.99,
                "lr": 2.5e-3,
                "basis_reset_frequency": 1,
                "basis_reset_method": "eigh",
            },
        ),
        (
            "two_sided_beta0999_lr0025",
            {"precondition_mode": "all", "beta2": 0.999, "lr": 2.5e-3},
        ),
        (
            "two_sided_beta099_lr0020",
            {"precondition_mode": "all", "beta2": 0.99, "lr": 2.0e-3},
        ),
        (
            "two_sided_beta099_lr0015",
            {"precondition_mode": "all", "beta2": 0.99, "lr": 1.5e-3},
        ),
        (
            "two_sided_beta099_lr0010",
            {"precondition_mode": "all", "beta2": 0.99, "lr": 1.0e-3},
        ),
        (
            "two_sided_beta099_lr00075",
            {"precondition_mode": "all", "beta2": 0.99, "lr": 7.5e-4},
        ),
        (
            "one_sided_beta095",
            {"precondition_mode": "smaller_side", "beta2": 0.95},
        ),
        (
            "one_sided_beta099",
            {"precondition_mode": "smaller_side", "beta2": 0.99},
        ),
        (
            "one_sided_beta0999",
            {"precondition_mode": "smaller_side", "beta2": 0.999},
        ),
        (
            "one_sided_beta0999_lr0025",
            {
                "precondition_mode": "smaller_side",
                "beta2": 0.999,
                "lr": 2.5e-3,
            },
        ),
        (
            "one_sided_beta099_lr0025",
            {
                "precondition_mode": "smaller_side",
                "beta2": 0.99,
                "lr": 2.5e-3,
            },
        ),
        (
            "one_sided_exact_beta099_lr0025",
            {
                "precondition_mode": "smaller_side",
                "beta2": 0.99,
                "lr": 2.5e-3,
                "basis_reset_frequency": 1,
                "basis_reset_method": "eigh",
            },
        ),
        (
            "one_sided_beta099_lr0020",
            {
                "precondition_mode": "smaller_side",
                "beta2": 0.99,
                "lr": 2.0e-3,
            },
        ),
        (
            "one_sided_beta099_lr0015",
            {
                "precondition_mode": "smaller_side",
                "beta2": 0.99,
                "lr": 1.5e-3,
            },
        ),
        (
            "one_sided_beta099_lr0010",
            {
                "precondition_mode": "smaller_side",
                "beta2": 0.99,
                "lr": 1.0e-3,
            },
        ),
        (
            "one_sided_beta099_lr00075",
            {
                "precondition_mode": "smaller_side",
                "beta2": 0.99,
                "lr": 7.5e-4,
            },
        ),
        (
            "one_sided_beta0999_lr0035",
            {
                "precondition_mode": "smaller_side",
                "beta2": 0.999,
                "lr": 3.5e-3,
            },
        ),
    )
    if args.variants:
        requested = set(args.variants)
        available = {name for name, _ in variants}
        unknown = requested - available
        if unknown:
            raise ValueError(f"unknown variants: {sorted(unknown)}")
        variants = tuple(item for item in variants if item[0] in requested)
    return {
        "seed": seed,
        "results": [
            train_variant(
                name,
                options,
                initial_state,
                train_inputs,
                train_targets,
                validation_inputs,
                validation_targets,
                args,
            )
            for name, options in variants
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--timing-warmup", type=int, default=40)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--validation-examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--beta1", type=float, default=0.95)
    parser.add_argument("--shampoo-beta", type=float, default=0.999)
    parser.add_argument("--precondition-frequency", type=int, default=10)
    parser.add_argument("--basis-reset-frequency", type=int, default=20)
    parser.add_argument(
        "--variants",
        nargs="+",
        help="run only the named variants (default: all)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 2 or not 0 <= args.timing_warmup < args.steps:
        raise ValueError("steps must exceed one and timing_warmup must be smaller")
    if args.train_examples < args.batch_size:
        raise ValueError("train_examples must be at least batch_size")

    output = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device_name": (
            torch.cuda.get_device_name(args.device)
            if torch.device(args.device).type == "cuda"
            else "cpu"
        ),
        "runs": [run_seed(seed, args) for seed in args.seeds],
    }
    serialized = json.dumps(output, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
