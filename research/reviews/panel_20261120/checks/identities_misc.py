"""Moderator checks of Fable and Gemini-Neon P1 identities (CPU, float64).

(a) Fable A3: Y0 = I + A + A^2/2 (A skew) has Y0^T Y0 = I + A^4/4; one NS step then maps theta^4/4 -> ...
(b) Neon A6: R4 = I + A + A^2/2 + A^3/6 + A^4/24 has R4^T R4 = I + A^6/72 + A^8/576.
(c) Neon: polar(I + A) = exp(arctan-angle) i.e. a plane rotation by arctan(theta) instead of theta.
(d) Neon V-mixing counterexample: C = U diag(1e4, 1e-2) U^T, basis misaligned by 0.01; (R.R)^T V (R.R) doubles the
    small variance instead of restoring it; ratio rescaling d+/d restores it.
(e) Neon B3 Procrustes reflection H+: orthogonal, maps Q1 -> -U1 R_p^T-ish, leaves span(Q1,U1)^perp fixed.
(f) Fable I3 warm/QR angle ratio formula vs the code, on random pairs.
"""
import math

import torch

dt = torch.float64
torch.manual_seed(0)
n = 16
a = torch.randn(n, n, dtype=dt); a = (a - a.T) / 2
a = a * (.5 / torch.linalg.matrix_norm(a, 2))
I = torch.eye(n, dtype=dt)

y = I + a + a @ a / 2
print('(a) ||Y0^T Y0 - (I + A^4/4)|| =', (y.T @ y - (I + torch.linalg.matrix_power(a, 4) / 4)).norm().item())
r4 = I + a + a @ a / 2 + torch.linalg.matrix_power(a, 3) / 6 + torch.linalg.matrix_power(a, 4) / 24
pred = I + torch.linalg.matrix_power(a, 6) / 72 + torch.linalg.matrix_power(a, 8) / 576
print('(b) ||R4^T R4 - (I + A^6/72 + A^8/576)|| =', (r4.T @ r4 - pred).norm().item(),
      ' ||R4^T R4 - I||_2 at ||A||_2=0.5:', torch.linalg.matrix_norm(r4.T @ r4 - I, 2).item())
th = .5
rot = torch.tensor([[1., -th], [th, 1.]], dtype=dt)
u, _, vh = torch.linalg.svd(rot); p = u @ vh
print('(c) polar(I+A) angle for theta=0.5:', math.atan2(p[1, 0].item(), p[0, 0].item()), ' arctan(0.5)=', math.atan(.5))

# (d)
lam = torch.tensor([1e4, 1e-2], dtype=dt)
t = .01
uu = torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype=dt)  # true eigvecs vs old basis I
c = uu @ torch.diag(lam) @ uu.T
q_old = torch.eye(2, dtype=dt)
v_old = torch.diagonal(q_old.T @ c @ q_old)          # V tracks old-basis variances
q_new = uu                                          # refresh to the true eigenbasis
r = q_old.T @ q_new
v_mix = (r * r).T @ v_old
d_new = torch.diagonal(q_new.T @ c @ q_new)
v_ratio = v_old * (d_new / v_old)                   # Kronecker-ratio rescaling (one side)
print(f'(d) true new variances {d_new.tolist()}; V unchanged {v_old.tolist()}; (R.R)^T V {v_mix.tolist()}; '
      f'ratio-rescaled {v_ratio.tolist()}')

# (e) Neon B3 reflection
nn, k = 12, 3
q, _ = torch.linalg.qr(torch.randn(nn, nn, dtype=dt))
q1, q2 = q[:, :k], q[:, k:]
u1, _ = torch.linalg.qr(q1 + .7 * torch.randn(nn, k, dtype=dt))
m = q1.T @ u1
uu_, ss, vv = torch.linalg.svd(m); rp = uu_ @ vv            # polar(M)
ut = u1 @ rp.T
s_ = q1.T @ ut
s_ = (s_ + s_.T) / 2
vp = q1 + ut
h = torch.eye(nn, dtype=dt) - vp @ torch.linalg.solve(torch.eye(k, dtype=dt) + s_, vp.T)
qmid = torch.cat([u1, h @ q2], 1)
print('(e) ||H^T H - I|| =', (h.T @ h - torch.eye(nn, dtype=dt)).norm().item(),
      ' ||H Q1 + U~1|| =', (h @ q1 + ut).norm().item(),
      ' ||Qmid^T Qmid - I|| =', (qmid.T @ qmid - torch.eye(nn, dtype=dt)).norm().item(),
      ' symmetric Q1^T U~1:', (q1.T @ ut - (q1.T @ ut).T).norm().item())

# (f) warm/QR angle ratio
s = 1.
worst = 0.
for _ in range(1000):
    di, dj = sorted(torch.rand(2, dtype=dt).tolist(), reverse=True)
    bij = 1e-3
    h_ = (di - dj) / s
    warm = .5 * (bij / s) * h_ / (h_ ** 2 + 1e-4)          # |A_ij| before cap
    qrang = bij / max(di, dj)
    pred = .5 * max(di, dj) * abs(di - dj) / ((di - dj) ** 2 + (0.01 * s) ** 2)
    worst = max(worst, abs(warm / qrang - pred))
print('(f) max |warm/QR - Fable formula| over 1000 random pairs (cap a=1):', worst)

# (g) Carbon's spike example: old basis aligned with C_old = diag(1, 1e-6); C rotates by 0.2 rad, refresh follows it.
#     M1=1, M2=1e-3, V=(1,1e-6).  Compare V rules: unchanged, (T.T) mixing, ratio-rescale to the new diagonal.
th = .2
rr = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=dt)
c_new = rr @ torch.diag(torch.tensor([1., 1e-6], dtype=dt)) @ rr.T
m_old = torch.tensor([1., 1e-3], dtype=dt); v_old = torch.tensor([1., 1e-6], dtype=dt)
q_new = rr                              # new basis = new eigvecs; T = Q_old^T Q_new = rr
m_new = rr.T @ m_old                    # transport
d_old = torch.diagonal(torch.eye(2, dtype=dt).T @ c_new @ torch.eye(2, dtype=dt))
d_new = torch.diagonal(q_new.T @ c_new @ q_new)
for name, v in [('unchanged', v_old), ('T.T mixing', (rr * rr).T @ v_old),
                ('ratio-rescale (d_new/d_old_at_refresh)', v_old * d_new / d_old),
                ('ratio-rescale (d_new/V-implied old diag)', v_old * d_new / v_old)]:
    print(f'(g) {name:42s} V+={[f"{x:.3g}" for x in v.tolist()]}  |M/sqrtV|={[f"{x:.3g}" for x in (m_new.abs()/v.sqrt()).tolist()]}')

# (h) Same rotation, but with M and V CONSISTENT with C at the refresh (as equal betas beta1=beta2=beta_C suggest):
#     V_old = diag(Q_old^T C Q_old), M_old = mean gradient along the new top eigvec expressed in the old basis.
v_cons = d_old.clone()
m_cons = rr @ torch.tensor([1., 0.], dtype=dt)          # mean along new eigvec 1, in old coordinates
m_new2 = rr.T @ m_cons
for name, v in [('unchanged', v_cons), ('T.T mixing', (rr * rr).T @ v_cons), ('ratio-rescale', v_cons * d_new / d_old)]:
    print(f'(h) consistent: {name:14s} V+={[f"{x:.3g}" for x in v.tolist()]} (true {[f"{x:.3g}" for x in d_new.tolist()]}) '
          f'|M/sqrtV|={[f"{x:.3g}" for x in (m_new2.abs()/v.sqrt()).tolist()]}')
