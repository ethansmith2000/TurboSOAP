"""Moderator check: compare proposed warm-step generators (CPU, float64, exact polar retraction unless noted).

Generators (all produce skew A, applied as Q+ = polar(Q (I + A)) unless 'retraction' says otherwise):
  prod      production: -1/2 (B/s) h/(h^2+0.01^2), Frobenius cap 0.5 (||A||_F/sqrt2), 2 NS   [production BF16 also shown]
  prod_nocap  production generator without the cap
  jac1      first-order Jacobi, factor 1 (moderator prior M1), damping 0.01, no cap
  qrtan     QR tangent (GPT-6.2-astra B1 / Fable I2,B3): A_ij = B_ij sgn(d_j - d_i)/max(d_i,d_j,rho*s), rho=1e-4,
            computed in the current column order (no sort), no cap
  qrtan_sort  same after sorting columns by d descending (astra B1 sorts; ties by index)
  neon_b1   Gemini Neon B1: sort, -1/2 sgn atan2(2B, |gap| + 0.05 (|d_i|+|d_j|+1e-5 dmax)), row-l2 cap 0.5 (D_w A D_w),
            spectral cap 0.9, retraction R4(A/2)^2  (we use exact expm-free polar for comparability AND report R4 variant)
Metrics after ONE refresh from the same (C, Q): r_C, whitened RMS correlation rho_rms, distance of the result to the
QR refresh of the same input measured as top-k subspace distance (k = n/8) and full sign/permutation-invariant
column match  1 - mean_i max_j |<q_i, qr_j>|.
Then a CHAINED open-loop toy (the paper's Fig. 2 setting: n=128, lambda ~ 1/i, skew drift 0.01 sqrt2 per step,
32 samples/step, beta_C=0.95, refresh every 10 steps, 'X10/QR50' = generator X at non-multiples of 50, QR at 50s),
mean r_C and mean rho_rms over steps 100-1000 (before-refresh values included, every step).
"""
import math
import sys

import torch

sys.path.insert(0, '/workspace/TurboSOAP')
from bf16_soap import warm_basis, qr_basis  # noqa: E402

torch.set_num_threads(4)
dt = torch.float64


def polar(y):
    u, _, vh = torch.linalg.svd(y)
    return u @ vh


def metrics(c, q):
    b = q.T @ c @ q
    d = torch.diagonal(b)
    off = b - torch.diag(d)
    rc = (off.norm() / b.norm()).item()
    keep = d > 1e-4 * d.max()
    s = torch.where(keep, d.clamp_min(1e-300).rsqrt(), torch.zeros_like(d))
    rho = off * s[:, None] * s[None, :]
    neff = keep.sum().item()
    return rc, math.sqrt((rho ** 2).sum().item() / max(neff, 1))


def qr_ref(c, q):
    z = c @ q
    order = (q * z).sum(0).argsort(descending=True)
    qq, r = torch.linalg.qr(z[:, order])
    return qq * torch.sign(torch.diagonal(r))


def subspace_dist(q1, q2, c, k):
    # top-k by Rayleigh quotient in each basis
    def top(q):
        d = torch.diagonal(q.T @ c @ q)
        return q[:, d.argsort(descending=True)[:k]]
    a, b = top(q1), top(q2)
    return (1 - (a.T @ b).norm() ** 2 / k).item()


def gen_prod(b, cap=True, factor=.5):
    d = torch.diagonal(b); s = d.abs().max()
    h = (d[:, None] - d[None, :]) / s
    a = -factor * (b / s) * h / (h ** 2 + 1e-4)
    if cap:
        a = a * min(1., .5 / max((a.norm() / math.sqrt(2)).item(), 1e-12))
    return a


def gen_qrtan(b, rho=1e-4):
    d = torch.diagonal(b); s = d.abs().max()
    n = len(d)
    idx = torch.arange(n)
    # sign: rotate column j toward i's content when d_j > d_i; ties broken by index (earlier column = 'larger')
    sg = torch.sign(d[None, :] - d[:, None])
    tie = sg == 0
    sg = torch.where(tie, torch.sign(idx[:, None] - idx[None, :]).to(dt), sg)
    den = torch.maximum(torch.maximum(d[:, None], d[None, :]), rho * s * torch.ones_like(b))
    a = b * sg / den
    a.fill_diagonal_(0)
    return a


def gen_neon(b):
    d = torch.diagonal(b); dmax = d.abs().max(); n = len(d)
    sij = d.abs()[:, None] + d.abs()[None, :] + 1e-5 * dmax
    gap = (d[:, None] - d[None, :]).abs()
    idx = torch.arange(n)
    sgn = torch.sign(idx[None, :] - idx[:, None]).to(dt)
    a = -.5 * sgn * torch.atan2(2 * b, gap + .05 * sij)
    a.fill_diagonal_(0)
    rn = a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w = (.5 / rn).clamp(max=1.)
    a = w * a * w.T
    sig = torch.linalg.matrix_norm(a, 2).item()
    a = a * min(1., .9 / (1.05 * sig))
    return a


def sort_cols(c, q):
    d = torch.diagonal(q.T @ c @ q)
    return q[:, d.argsort(descending=True)]


def r4sq(a):
    a1 = a / 2
    s = a1 @ a1
    r = torch.eye(len(a), dtype=dt) + a1 + s / 2 + (a1 / 6 + s / 24) @ s
    return r @ r


def refresh(kind, c, q):
    if kind == 'qr':
        return qr_ref(c, q)
    if kind == 'prod_bf16':
        return warm_basis(c.float().to(torch.bfloat16)[None], q.float().to(torch.bfloat16)[None],
                          torch.bfloat16, torch.bfloat16)[0].double()
    if kind in ('qrtan_sort', 'neon_b1', 'neon_b1_r4'):
        q = sort_cols(c, q)
    b = q.T @ c @ q
    b = .5 * (b + b.T)
    if kind == 'prod':
        a = gen_prod(b)
    elif kind == 'prod_nocap':
        a = gen_prod(b, cap=False)
    elif kind == 'jac1':
        a = gen_prod(b, cap=False, factor=1.)
    elif kind in ('qrtan', 'qrtan_sort'):
        a = gen_qrtan(b)
    elif kind in ('neon_b1', 'neon_b1_r4'):
        a = gen_neon(b)
    else:
        raise ValueError(kind)
    if kind == 'neon_b1_r4':
        return q @ r4sq(a)
    return polar(q + q @ a)


KINDS = ['prod_bf16', 'prod', 'prod_nocap', 'jac1', 'qrtan', 'qrtan_sort', 'neon_b1', 'neon_b1_r4', 'qr']


def make_inputs(n, rot, seed):
    g = torch.Generator().manual_seed(seed)
    lam = torch.arange(1, n + 1, dtype=dt).pow(-1.0)
    s = (lam / lam.sum()).sqrt()
    u, _ = torch.linalg.qr(torch.randn(n, n, dtype=dt, generator=g))
    c = torch.zeros(n, n, dtype=dt)
    for _ in range(40):
        gg = u @ (s[:, None] * torch.randn(n, max(n // 4, 64), dtype=dt, generator=g))
        c = .95 * c + .05 * gg @ gg.T
    a = torch.randn(n, n, dtype=dt, generator=g) * rot / math.sqrt(n)
    a = (a - a.T) / 2
    q, r = torch.linalg.qr(u @ (torch.eye(n, dtype=dt) + a))
    return c, q * torch.sign(torch.diagonal(r))


def one_shot():
    print('=== one refresh from the same input ===')
    for n, rot in [(256, .2), (256, .5), (512, .2)]:
        c, q = make_inputs(n, rot, seed=n)
        qq = qr_ref(c, q)
        r0, p0 = metrics(c, q)
        print(f'n={n} rot={rot}: input r_C={r0:.4f} rho_rms={p0:.4f}')
        for kind in KINDS:
            qn = refresh(kind, c, q)
            rc, rr = metrics(c, qn)
            gram = ((qn.T @ qn - torch.eye(n, dtype=dt)).norm() / math.sqrt(n)).item()
            print(f'   {kind:12s} r_C={rc:.4f} rho_rms={rr:.4f} topk_dist_to_QR={subspace_dist(qn, qq, c, n // 8):.4f} '
                  f'gram={gram:.1e}')


def chained(n=128, steps=1000, seed=0):
    print(f'=== chained open-loop toy (n={n}, refresh every 10, QR at 50s) ===')
    g = torch.Generator().manual_seed(seed)
    lam = torch.arange(1, n + 1, dtype=dt).pow(-1.0)
    s = (lam / lam.sum()).sqrt()
    u0, _ = torch.linalg.qr(torch.randn(n, n, dtype=dt, generator=g))
    w = torch.randn(n, n, dtype=dt, generator=g); w = (w - w.T) / 2
    w = w * (.01 * math.sqrt(2) / w.norm())
    step_rot = torch.linalg.matrix_exp(w)
    samples = []
    u = u0.clone(); c = torch.zeros(n, n, dtype=dt)
    cs = []
    for t in range(steps + 1):
        gg = u @ (s[:, None] * torch.randn(n, 32, dtype=dt, generator=g))
        c = .95 * c + .05 * gg @ gg.T
        cs.append(c.clone())
        u = step_rot @ u
    schedules = {'qr10': None, 'qr50': None}
    for k in ['prod_bf16', 'prod', 'prod_nocap', 'jac1', 'qrtan', 'qrtan_sort', 'neon_b1']:
        schedules[f'{k}10/qr50'] = k
    res = {}
    for name, kind in schedules.items():
        q = torch.linalg.eigh(cs[0]).eigenvectors.flip(-1)
        rcs, rhos = [], []
        for t in range(1, steps + 1):
            c = cs[t]
            if t % 10 == 0:
                if name == 'qr10' or t % 50 == 0:
                    q = qr_ref(c, q)
                elif name != 'qr50':
                    q = refresh(kind, c, q)
            if t >= 100:
                rc, rr = metrics(c, q)
                rcs.append(rc); rhos.append(rr)
        res[name] = (sum(rcs) / len(rcs), sum(rhos) / len(rhos))
        print(f'   {name:18s} mean r_C={res[name][0]:.4f} mean rho_rms={res[name][1]:.4f}', flush=True)


if __name__ == '__main__':
    one_shot()
    chained()
