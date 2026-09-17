"""Compare dense and block one-factor basis refreshes."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soap import SOAP, _offdiag, _sym
from soap_block import BlockSOAP


def _clone_state(state: dict) -> dict:
    return {
        "Q": [state["Q"][0].clone()],
        "GG": [state["GG"][0].clone()],
        "exp_avg": state["exp_avg"].clone(),
        "exp_avg_sq": state["exp_avg_sq"].clone(),
        "basis_refreshes": 0,
    }


def _elapsed_ms(function, repetitions: int, device: torch.device) -> float:
    for _ in range(2):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repetitions):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1e3 * (time.perf_counter() - start) / repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--rotation-cap", type=float, default=0.25)
    parser.add_argument(
        "--selection",
        choices=("round_robin", "top_correlation"),
        default="round_robin",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(123)

    for size in args.dimensions:
        sample = torch.randn(size, size, device=device)
        covariance = sample @ sample.T / size
        covariance.add_(torch.eye(size, device=device), alpha=0.01)
        q = torch.eye(size, device=device)
        b = _sym(q.T @ covariance @ q)
        exp_avg = torch.randn(size, device=device)
        exp_avg_sq = torch.rand(size, device=device).add_(0.1)

        dense = SOAP(
            [torch.nn.Parameter(torch.zeros(size, device=device))],
            precondition_1d=True,
            basis_track_stats=False,
            basis_reset_frequency=0,
            covariance_compute_dtype="float32",
        )
        block = BlockSOAP(
            [torch.nn.Parameter(torch.zeros(size, device=device))],
            precondition_1d=True,
            basis_track_stats=False,
            basis_reset_frequency=0,
            covariance_compute_dtype="float32",
            basis_pairs_per_refresh=args.pairs,
            basis_pair_selection=args.selection,
            basis_block_size=args.block_size,
            basis_block_rotation_cap=args.rotation_cap,
        )
        dense_template = {
            "Q": [q],
            "GG": [covariance],
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
            "basis_refreshes": 0,
        }
        block_template = {
            "Q": [q],
            "GG": [b],
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
            "basis_refreshes": 0,
        }
        dense_ms = _elapsed_ms(
            lambda: dense._refresh_basis(
                _clone_state(dense_template), dense.param_groups[0]
            ),
            args.repetitions,
            device,
        )
        block_ms = _elapsed_ms(
            lambda: block._refresh_basis(
                _clone_state(block_template), block.param_groups[0]
            ),
            args.repetitions,
            device,
        )
        dense_state = _clone_state(dense_template)
        block_state = _clone_state(block_template)
        dense._refresh_basis(dense_state, dense.param_groups[0])
        block._refresh_basis(block_state, block.param_groups[0])
        dense_b = dense_state["Q"][0].T @ covariance @ dense_state["Q"][0]

        print(
            json.dumps(
                {
                    "dimension": size,
                    "dense_ms": dense_ms,
                    "block_ms": block_ms,
                    "dense_over_block_speedup": dense_ms / block_ms,
                    "before_diag_error": float(_offdiag(b).norm() / b.norm()),
                    "dense_after_diag_error": float(
                        _offdiag(dense_b).norm() / dense_b.norm()
                    ),
                    "block_after_diag_error": float(
                        _offdiag(block_state["GG"][0]).norm()
                        / block_state["GG"][0].norm()
                    ),
                    "pairs": args.pairs,
                    "block_size": args.block_size,
                    "rotation_cap": args.rotation_cap,
                    "selection": args.selection,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
