"""Actual natural-cycle radius qualification; all policies remain opt-in."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import torch
import retraction_diagnostic as runner
from movement_retraction import POLICIES as BASE_POLICIES, warm
from warm_retraction import Retraction

POLICIES = {
    'fro_cubic2_cap05': BASE_POLICIES['fro_cubic2_cap05'],
    'fro_cubic2_cap06': Retraction('fro_cubic2_cap06', order=3, iterations=2,
                                 cap_mode='skew_frobenius', cap=.6),
    'fro_cubic3_cap06': Retraction('fro_cubic3_cap06', order=3, iterations=3,
                                 cap_mode='skew_frobenius', cap=.6),
}


def optimizer_hash(opt):
    """Hash every tensor/scalar in parameter-order optimizer state, offline."""
    h = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            h.update(str((value.dtype, tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                h.update(str(key).encode()); visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value: visit(item)
        else:
            h.update(json.dumps(value, sort_keys=True).encode())
    for group in opt.param_groups:
        for p in group['params']: visit(opt.state[p])
    return h.hexdigest()


def main():
    original_diagnostics = runner.diagnostics
    runner.VARIANTS = tuple(POLICIES.values())
    runner.warm = warm
    def diagnostics(opt, ema, step, attribution):
        result = original_diagnostics(opt, ema, step, attribution)
        if step == 420:
            result['optimizer_sha256'] = optimizer_hash(opt)
        return result
    runner.diagnostics = diagnostics
    runner.main()


if __name__ == '__main__':
    main()
