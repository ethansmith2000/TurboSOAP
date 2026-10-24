"""Additive blocked-SOAP refresh primitives; not a complete reference port.

Inputs are stacked FP32 square factors. QR reproduces the pinned reference's
regular refresh and variance permutation. Warm applies the existing FastSOAP
policy independently per factor. Raw-coordinate first momentum is not an input.
"""
import math
import torch


def _bf16_bmm_fp32(a, b):
    a, b = a.bfloat16(), b.bfloat16()
    if a.is_cuda:
        return torch.bmm(a, b, out_dtype=torch.float32)
    return torch.bmm(a.double(), b.double()).float()


def batched_warm_refresh(covariance, q, variance, epsilon):
    """Same damping/cap/two corrections as FastSOAP, with per-factor reductions.

    `epsilon` is [batch], already scaled by the reference recipe. Persistent
    inputs are unchanged. Warm leaves variance in its current coordinates.
    """
    c = covariance + epsilon[:, None, None] * torch.eye(
        q.shape[-1], device=q.device, dtype=q.dtype)
    b = q.transpose(-1, -2) @ c @ q
    b = .5 * (b + b.transpose(-1, -2))
    diagonal = b.diagonal(dim1=-2, dim2=-1)
    scale = diagonal.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    gap = (diagonal.unsqueeze(-1) - diagonal.unsqueeze(-2)) / scale.unsqueeze(-1)
    rotation = -.5 * (b / scale.unsqueeze(-1)) * gap / (gap.square() + .01**2)
    bound = torch.linalg.vector_norm(rotation, dim=(-2, -1), keepdim=True) / math.sqrt(2)
    rotation = rotation * (.5 / bound.clamp_min(1e-12)).clamp(max=1)
    y = q + _bf16_bmm_fp32(q, rotation)
    for _ in range(2):
        residual = y.transpose(-1, -2) @ y
        residual.diagonal(dim1=-2, dim2=-1).sub_(1.)
        y = y - .5 * _bf16_bmm_fp32(y, residual)
    return y, variance


def batched_qr_refresh(covariance, q, variance, epsilon):
    """One reference QR power step and row permutation of projected variance.

    For a right factor, pass transposed variance and transpose it back outside
    this primitive. Sorting follows estimated eigenvalues in descending order.
    """
    c = covariance + epsilon[:, None, None] * torch.eye(
        q.shape[-1], device=q.device, dtype=q.dtype)
    product = c @ q
    estimate = (q * product).sum(-2)
    order = estimate.argsort(dim=-1, descending=True)
    reordered = product.gather(-1, order.unsqueeze(-2).expand_as(product))
    nu = variance.gather(-2, order.unsqueeze(-1).expand_as(variance))
    return torch.linalg.qr(reordered).Q, nu
