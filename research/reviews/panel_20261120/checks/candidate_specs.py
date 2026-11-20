"""Reference implementations of the CANDIDATES.md refresh functions, smoke-tested on CPU against trained snapshots.

These are SPECS for the refresh-study harness (to be ported there by the main session), not harness code.  Each
takes BF16 (b, n, n) C and Q like production `warm_basis` / `qr_basis` and returns BF16 Q (plus order / ok where
noted).  On CUDA, wrap the FP32 Gram products in `ieee_fp32()` so that "high" matmul precision does not run them in
TF32.  The smoke test (main) runs each on one trained factor group and reports orthogonality, finiteness and the
top-n/8 subspace distance to Householder QR (production `qr_basis` arithmetic) on the same input.
"""
import contextlib
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/workspace/TurboSOAP')
from bf16_soap import mm32, qr_basis, warm_basis  # noqa: E402

BF16, FP32 = torch.bfloat16, torch.float32


@contextlib.contextmanager
def ieee_fp32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


# ---------------------------------------------------------------- C1 / C2: Gram-shifted CholeskyQR3, no operator ridge
def cq3s_basis(cov, q, shift_rel=1e-6, unshifted_passes=2, power_iters=8):
    """Same target as qr_basis (QR of the sorted Z = C Q); returns (Q_bf16, order, ok[b])."""
    c, old = cov.float(), q.float()
    z = mm32(cov, q) if cov.dtype == BF16 else torch.bmm(c, old)       # BF16 operands are exact; FP32 out
    order = (old * z).sum(-2).argsort(dim=-1, descending=True)
    x = z.gather(-1, order.unsqueeze(-2).expand_as(z))
    n = x.shape[-1]
    eye = torch.eye(n, device=x.device, dtype=FP32)
    ok = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    with ieee_fp32():
        g = torch.bmm(x.transpose(-1, -2), x)
        v = torch.full((x.shape[0], n, 1), 1 / math.sqrt(n), device=x.device, dtype=FP32)
        for _ in range(power_iters):                                    # lambda_max(G) = ||X||_2^2
            v = torch.bmm(g, v)
            v = v / v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
        lam = torch.bmm(v.transpose(-1, -2), torch.bmm(g, v))            # (b,1,1)
        shifts = [shift_rel * lam] + [None] * unshifted_passes
        for k, s in enumerate(shifts):
            if k:
                g = torch.bmm(x.transpose(-1, -2), x)
            u, info = torch.linalg.cholesky_ex(g + s * eye if s is not None else g, upper=True)
            ok = ok & (info == 0)
            u = torch.where(ok.view(-1, 1, 1), u, eye)                   # keep the graph finite on failure
            x = torch.linalg.solve_triangular(u, x, upper=True, left=False)
    ok = ok & torch.isfinite(x).flatten(1).all(1)
    return x.to(BF16).contiguous(), order, ok


def cq3s_with_fallback(cov, q, **kw):
    """Replay/production wrapper: Householder (qr_basis) for the batch elements whose Cholesky failed."""
    new, order, ok = cq3s_basis(cov, q, **kw)
    if not bool(ok.all()):
        bad = (~ok).nonzero().flatten()
        hq, ho = qr_basis(cov[bad], q[bad], BF16)
        new, order = new.clone(), order.clone()
        new[bad], order[bad] = hq, ho
    return new, order, ok


# ---------------------------------------------------------------- C5: head-panel power step (Fable P2 E2 form)
def head_panel_basis(cov, q, frac=0.125, ns_steps=1):
    """Exact QR of the top-k ordered columns of C Q; tail completed by the Householder complement (complete QR of
    the n x k coordinate panel).  Returns (Q_bf16, order).  Diagonal B returns the old basis up to signs."""
    c, old = cov.float(), q.float()
    z = mm32(cov, q) if cov.dtype == BF16 else torch.bmm(c, old)
    order = (old * z).sum(-2).argsort(dim=-1, descending=True)
    idx = order.unsqueeze(-2).expand_as(z)
    qp, zp = old.gather(-1, idx), z.gather(-1, idx)
    n = qp.shape[-1]
    k = max(1, int(frac * n))
    with ieee_fp32():
        xk = torch.bmm(qp.transpose(-1, -2), zp[..., :k])               # (b, n, k) panel in basis coordinates
        h, r = torch.linalg.qr(xk, mode='complete')                     # (b, n, n)
    sgn = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))                # positive-diagonal convention on the head
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    h = torch.cat([h[..., :k] * sgn.unsqueeze(-2), h[..., k:]], -1)
    y = mm32(qp.to(BF16), h.to(BF16)).to(BF16)
    eye = torch.eye(n, device=y.device, dtype=FP32)
    for _ in range(ns_steps):
        e = mm32(y.transpose(-1, -2), y) - eye
        y = (y.float() - .5 * mm32(y, e.to(BF16))).to(BF16)
    return y.contiguous(), order


# ---------------------------------------------------------------- C4 additions: bounded QR tangent, three bounds
def qr_tangent_basis(cov, q, rho=1e-4, bound='spectral', cap=.5, ns_steps=2, stats=None):
    """PB1 generator A_ij = B_ij sgn(d_j - d_i)/max(d_i, d_j, rho s) (ties by index, entries clamped to +-1), scaled
    by one of: 'frob' (production norm), 'schatten4' ((sum theta^4)^(1/4), GPT-6.2-astra P2), 'spectral' (8 power
    iterations on A^T A, x1.15 safety), 'row_spectral' (Carbon/Neon: row-l2 cap 0.5 as D_w A D_w, then spectral).
    Retraction as production (Q + Q A, then ns_steps Newton-Schulz)."""
    qb, cb = q.to(BF16), cov.to(BF16)
    b = mm32(torch.bmm(qb.transpose(-1, -2), cb), qb)
    b = .5 * (b + b.transpose(-1, -2))
    d = b.diagonal(dim1=-2, dim2=-1)
    s = d.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    n = b.shape[-1]
    di, dj = d.unsqueeze(-1), d.unsqueeze(-2)
    sg = torch.sign(dj - di)
    ix = torch.arange(n, device=b.device)
    sg = torch.where(sg == 0, torch.sign(ix.unsqueeze(-1) - ix.unsqueeze(-2)).to(b.dtype), sg)
    den = torch.maximum(torch.maximum(di, dj), rho * s.unsqueeze(-1))
    a = (b * sg / den) * (1 - torch.eye(n, device=b.device, dtype=b.dtype))
    a = a.clamp(-1, 1)

    def spec(m):
        v = torch.full((m.shape[0], n, 1), 1 / math.sqrt(n), device=m.device, dtype=m.dtype)
        v = v + 1e-3 * torch.cos(torch.arange(n, device=m.device, dtype=m.dtype) * 1.7).view(1, n, 1)
        for _ in range(8):
            v = torch.bmm(m.transpose(-1, -2), torch.bmm(m, v))
            v = v / v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
        return torch.bmm(m, v).norm(dim=-2, keepdim=True)

    if bound == 'row_spectral':
        w = (.5 / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
        a = w * a * w.transpose(-1, -2)
        size = 1.15 * spec(a)
    elif bound == 'spectral':
        size = 1.15 * spec(a)
    elif bound == 'schatten4':
        ata = torch.bmm(a.transpose(-1, -2), a)
        size = (.5 * ata.square().sum((-2, -1), keepdim=True)).pow(.25)
    elif bound == 'frob':
        size = a.norm(dim=(-2, -1), keepdim=True) / math.sqrt(2)
    else:
        raise ValueError(bound)
    mult = (cap / size.clamp_min(1e-12)).clamp(max=1)
    if stats is not None:
        stats['generator_size'] = size.flatten().tolist()
        stats['mult'] = mult.flatten().tolist()
    a = a * mult
    y = (q.float() + mm32(qb, a.to(BF16))).to(BF16)
    eye = torch.eye(n, device=y.device, dtype=FP32)
    for _ in range(ns_steps):
        e = mm32(y.transpose(-1, -2), y) - eye
        y = (y.float() - .5 * mm32(y, e.to(BF16))).to(BF16)
    return y.contiguous()


# ---------------------------------------------------------------- smoke test on trained snapshots (CPU)
def _load(width, idx, slot):
    a = np.load(f'/workspace/data/turbosoap_refresh_study_20261120/d{width}/f{idx}_bf16.npy', mmap_mode='r')[slot]
    return torch.from_numpy(np.array(a).view(np.int16)).view(BF16)


def _metrics(c, q, ref):
    cf, qf = c.double(), q.double()
    bb = qf.transpose(-1, -2) @ cf @ qf
    off = bb - torch.diag_embed(torch.diagonal(bb, dim1=-2, dim2=-1))
    rc = (off.norm(dim=(-2, -1)) / bb.norm(dim=(-2, -1))).mean().item()
    n = q.shape[-1]
    gram = ((qf.transpose(-1, -2) @ qf - torch.eye(n, dtype=torch.float64)).norm(dim=(-2, -1)) / math.sqrt(n)).mean().item()
    k = max(8, n // 8)

    def top(qq):
        d = torch.diagonal(qq.transpose(-1, -2) @ cf @ qq, dim1=-2, dim2=-1)
        o = d.argsort(-1, descending=True)[..., :k]
        return qq.gather(-1, o.unsqueeze(-2).expand(-1, n, -1))
    dist = (1 - (top(qf).transpose(-1, -2) @ top(ref.double())).norm(dim=(-2, -1)) ** 2 / k).mean().item()
    dg = torch.diagonal(bb, dim1=-2, dim2=-1)
    keep = dg >= 1e-4 * dg.abs().amax(-1, keepdim=True)
    sc = torch.where(keep, dg.clamp_min(1e-300).rsqrt(), torch.zeros_like(dg))
    rho = off * sc.unsqueeze(-1) * sc.unsqueeze(-2)
    rho_rms = (rho.square().sum((-2, -1)) / keep.sum(-1).clamp_min(1)).sqrt().mean().item()
    return (f'rC={rc:.3f} rho_rms={rho_rms:.3f} gram={gram:.4f} top{k}_dist_to_QR={max(dist, 0):.1e} '
            f'finite={bool(torch.isfinite(q.float()).all())}')


if __name__ == '__main__':
    torch.set_num_threads(4)
    width = 256
    for slot, idxs, lag in [(2, [1, 3, 5], 1), (50, [1, 3, 5], 1), (50, [0], 1), (2, [1, 3, 5], 5), (50, [1, 3, 5], 5), (50, [0], 5)]:
        c = torch.stack([_load(width, i, slot) for i in idxs])
        cp = torch.stack([_load(width, i, slot - lag) for i in idxs])         # exact basis from 10*lag ages earlier
        cpf = cp.float(); cpf = .5 * (cpf + cpf.transpose(-1, -2))
        q = torch.linalg.eigh(cpf).eigenvectors.flip(-1).to(BF16)
        ref, _ = qr_basis(c, q, BF16)
        print(f'--- age {slot * 10}, basis {10 * lag} ages stale, factors {idxs}, n={c.shape[-1]}: input {_metrics(c, q, ref)}')
        print('  qr_basis        ', _metrics(c, ref, ref))
        new, order, ok = cq3s_with_fallback(c, q)
        print('  cq3s (1e-6,2)   ', _metrics(c, new, ref), 'ok', ok.tolist())
        new, _ = head_panel_basis(c, q)
        kk = max(1, int(.125 * c.shape[-1]))
        hd = (1 - (new[..., :kk].double().transpose(-1, -2) @ ref[..., :kk].double()).norm(dim=(-2, -1)) ** 2 / kk).mean().item()
        print('  head_panel 1/8  ', _metrics(c, new, ref), f'first-k-columns subspace dist to QR={max(hd, 0):.1e}')
        print('  warm_basis prod ', _metrics(c, warm_basis(c, q, BF16, BF16), ref))
        for bnd in ('frob', 'schatten4', 'spectral', 'row_spectral'):
            st = {}
            new = qr_tangent_basis(c, q, bound=bnd, stats=st)
            print(f'  tangent {bnd:12s}', _metrics(c, new, ref),
                  'size', [round(x, 2) for x in st['generator_size']], 'mult', [round(x, 3) for x in st['mult']])
