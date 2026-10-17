"""Opt-in arithmetic atom only; highest FP32 Gram, optional BF16 residual product."""
import torch

def residual_retract_two(y,correction_mixed=False):
    for _ in range(2):
        residual=y.T@y
        residual.diagonal().sub_(1.)
        correction=(y.bfloat16()@residual.bfloat16()).float() if correction_mixed else y@residual
        y=y-.5*correction
    return y
