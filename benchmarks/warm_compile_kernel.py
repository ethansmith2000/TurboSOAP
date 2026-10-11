"""Opt-in fixed warm refresh; production optimizer remains unchanged."""
import math
import torch


def warm_kernel(cov, q, mixed=False):
    # FP32 EMA input and stored basis. BF16 is restricted to these three GEMMs.
    scale = (cov.trace().abs() / cov.shape[0]).clamp_min(1e-12)
    c = cov.float() / scale
    if mixed:
        qb = q.to(torch.bfloat16)
        cb = c.to(torch.bfloat16)
        b = (qb.T @ cb @ qb).float()
    else:
        b = q.float().T @ c @ q.float()
    b = .5 * (b + b.T)
    diagonal = b.diagonal()
    gap = diagonal[:, None] - diagonal[None, :]
    tau = diagonal.abs().amax().clamp_min(1e-12) * .01
    omega = -b * gap / (gap.square() + tau.square()).clamp_min(1e-24)
    raw = .5 * (.5 * (omega - omega.T))
    bound = raw.norm() / math.sqrt(2)
    a = raw * (.5 / bound.clamp_min(1e-12)).clamp(max=1)
    if mixed:
        delta = (qb @ a.to(torch.bfloat16)).float()
    else:
        delta = q.float() @ a
    # Add to the original FP32 basis, rather than a BF16-rounded copy.
    z = q.float() + delta
    gram = z.T @ z
    z = 1.5 * z - .5 * (z @ gram)
    gram = z.T @ z
    return 1.5 * z - .5 * (z @ gram)


LIMITS = dict(compiled_basis_relative=1e-4, mixed_basis_relative=1e-3,
              movement_relative=.10, movement_cosine=.995, gram=.005,
              signed_diagonal_mean=.001, world_m_relative=.01,
              projected_m_relative=.001, tracking_ratio=1.01)


def classify(metrics, mixed):
    checks = {
        'basis': metrics['basis_relative'] <= LIMITS['mixed_basis_relative' if mixed else 'compiled_basis_relative'],
        'movement': metrics['movement_relative'] <= LIMITS['movement_relative'],
        'alignment': metrics['movement_cosine'] >= LIMITS['movement_cosine'],
        'geometry': metrics['gram'] <= LIMITS['gram'],
        'signed_drift': abs(metrics['signed_diagonal_mean']) <= LIMITS['signed_diagonal_mean'],
        'physical_m': metrics['world_m_relative'] <= LIMITS['world_m_relative'],
        'projected_m': metrics['projected_m_relative'] <= LIMITS['projected_m_relative'],
        'tracking': metrics['tracking_ratio'] <= LIMITS['tracking_ratio'],
    }
    return dict(passed=all(checks.values()), checks=checks)
