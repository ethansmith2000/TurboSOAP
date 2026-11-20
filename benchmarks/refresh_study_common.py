"""Shared pieces for the refresh study (2026-11-20).

``warm_basis_param`` is a parametrized copy of ``bf16_soap.warm_basis``.  With
the defaults (damping 0.01, Frobenius cap 0.5, two Newton--Schulz corrections,
one warm step) it executes exactly the production operation sequence with the
same Python scalars, and the study verifies bit equality against production on
CUDA before using it.  ``cap=None`` removes the global Frobenius cap.
``warm_steps=2`` applies the whole warm step (skew motion + corrections) twice.
"""
import math

import torch

from bf16_soap import mm32, qr_basis

BF16, FP32 = torch.bfloat16, torch.float32
WARM_DEFAULTS = dict(damping=.01, cap=.5, ns_steps=2, warm_steps=1)


def _warm_once(cov, q, compute_dtype, basis_dtype, damping, cap, ns_steps):
    qb, cb = q.to(compute_dtype), cov.to(compute_dtype)
    b = mm32(torch.bmm(qb.transpose(-1, -2), cb), qb)
    b = .5 * (b + b.transpose(-1, -2))
    diagonal = b.diagonal(dim1=-2, dim2=-1)
    scale = diagonal.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    gap = (diagonal.unsqueeze(-1) - diagonal.unsqueeze(-2)) / scale.unsqueeze(-1)
    a = -.5 * (b / scale.unsqueeze(-1)) * gap / (gap.square() + damping ** 2)
    if cap is not None:
        bound = torch.linalg.vector_norm(a, dim=(-2, -1), keepdim=True) / math.sqrt(2)
        a = a * (cap / bound.clamp_min(1e-12)).clamp(max=1)
    y = (q.float() + mm32(qb, a.to(compute_dtype))).to(basis_dtype)
    for _ in range(ns_steps):
        if basis_dtype == torch.bfloat16:
            gram = mm32(y.transpose(-1, -2), y)
        else:
            gram = torch.bmm(y.transpose(-1, -2), y)
        eye = torch.eye(y.shape[-1], device=y.device, dtype=gram.dtype)
        residual = gram - eye
        y = (y.float() - .5 * mm32(y.to(compute_dtype), residual.to(compute_dtype))).to(basis_dtype)
    return y.contiguous()


def warm_basis_param(cov, q, compute_dtype=BF16, basis_dtype=BF16, *, damping=.01, cap=.5,
                     ns_steps=2, warm_steps=1):
    for _ in range(warm_steps):
        q = _warm_once(cov, q, compute_dtype, basis_dtype, damping, cap, ns_steps)
    return q


def eigh_basis(cov):
    """CompiledSOAP initialization path: FP32 eigh of sym(float(C)), descending, BF16."""
    matrix = cov.float()
    matrix = .5 * (matrix + matrix.transpose(-1, -2))
    return torch.linalg.eigh(matrix).eigenvectors.flip(-1).to(BF16).contiguous()


def cholqr_basis(cov, q):
    """Cholesky-QR comparator with qr_basis's CQ product and sort.  Returns
    (new BF16 basis, order, cholesky info).  No shift/regularization."""
    cf, old = cov.float(), q.float()
    z = torch.bmm(cf, old)
    order = (old * z).sum(-2).argsort(dim=-1, descending=True)
    ordered = z.gather(-1, order.unsqueeze(-2).expand_as(z))
    gram = torch.bmm(ordered.transpose(-1, -2), ordered)
    lower, info = torch.linalg.cholesky_ex(gram)
    new = torch.linalg.solve_triangular(lower.transpose(-1, -2), ordered, upper=True, left=False)
    return new.to(BF16).contiguous(), order, info


def qr_refresh(cov, q):
    return qr_basis(cov, q, BF16)[0]


class highest:
    def __enter__(self):
        self.prev = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision('highest')

    def __exit__(self, *a):
        torch.set_float32_matmul_precision(self.prev)


@torch.no_grad()
def coordinate_matrix(cov, q):
    """B = Q^T C Q in FP32 at 'highest' (diagnostic precision of paper Sec. 5.5)."""
    with highest():
        qf = q.float()
        return torch.bmm(torch.bmm(qf.transpose(-1, -2), cov.float()), qf)


@torch.no_grad()
def residuals(b, floor=1e-4):
    """Per-factor r_C (Frobenius) and the diagonally normalized variant
    r_C^w = ||offdiag(D^-1/2 B D^-1/2)||_F / ||D^-1/2 B D^-1/2||_F, D = diag(B),
    restricted to coordinates with B_jj >= floor * max_j B_jj (each such
    direction weighted by the inverse of its own variance, so weak directions
    count as much as strong ones; directions below the floor are dominated by
    BF16 storage noise of C).  Also the RMS off-diagonal correlation
    sqrt(sum_{i!=j} rho_ij^2 / n_eff) over the same coordinates."""
    d = b.diagonal(dim1=-2, dim2=-1)
    off = b - torch.diag_embed(d)
    rc = torch.linalg.matrix_norm(off) / torch.linalg.matrix_norm(b).clamp_min(1e-30)
    scale = d.abs().amax(-1, keepdim=True)
    s = d.clamp_min(1e-30).rsqrt() * (d >= floor * scale)
    rho = off * s.unsqueeze(-1) * s.unsqueeze(-2)
    off_w2 = rho.square().sum((-2, -1))
    n_eff = (s > 0).sum(-1).clamp_min(1)
    rcw = (off_w2 / (off_w2 + n_eff)).sqrt()
    corr_rms = (off_w2 / n_eff).sqrt()
    return rc, rcw, corr_rms


@torch.no_grad()
def gram_error(q):
    with highest():
        qf = q.float()
        eye = torch.eye(q.shape[-1], device=q.device)
        return torch.linalg.matrix_norm(torch.bmm(qf.transpose(-1, -2), qf) - eye) / math.sqrt(q.shape[-1])


@torch.no_grad()
def topk_overlap(b, q, evecs, ks):
    """Subspace overlap ||U_k^T Q_S||_F^2 / k between the top-k eigenvectors U_k
    of C (columns of ``evecs`` sorted descending) and the k stored columns with the
    largest Rayleigh quotients diag(B) (warm refreshes never reorder columns)."""
    d = b.diagonal(dim1=-2, dim2=-1)
    order = d.argsort(-1, descending=True)
    out = {}
    with highest():
        qf = q.float()
        for k in ks:
            idx = order[:, :k]
            qs = qf.gather(-1, idx.unsqueeze(-2).expand(-1, qf.shape[-2], -1))
            m = torch.bmm(evecs[:, :, :k].transpose(-1, -2), qs)
            out[k] = m.square().sum((-2, -1)) / k
    return out
