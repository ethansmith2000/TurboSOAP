# TurboSOAP refresh panel: ranked candidates for the refresh-study harness (2026-11-20)

Eight candidates, ranked. They come from P1 (50 reviewer ideas pooled into PA1–PA14 and PB1–PB12), P2 (four
reviewers ranking the pool with the web on), and the moderator checks. The evidence is in
[P1_SYNTHESIS.md](P1_SYNTHESIS.md), [P2_SYNTHESIS.md](P2_SYNTHESIS.md) and `checks/`.

**Updated after the refresh-study Part 2b and Part 1b results** (P2_SYNTHESIS §9): C1's cost is now measured, C3's cost
gate is corrected, and C5 is scored against Part 2b.

> **Correction (2026-11-20, ~05:45, after refresh-study's final Part 1b IEEE results and a reconciliation check).**
> C1 and C3 were priced from TF32-Gram timings and their trained-factor failure rate came from n ≤ 2048 only.
> - **Cost.** With IEEE-FP32 Grams, the 109M CholQR3 event is ≈ 320 ms, not ≈ 198.
>   - C1 saves ≈ 5.5 ms/call at QR50, not ≈ 7.8.
>   - C3 costs ≈ 32.5 ms/call, not ≈ 20.3: about QR20's ≈ 30.3, not cheaper than today's 15.94.
> - **Size limit.** CholQR3 loses to Householder above n ≈ 4096 at batch 12 (≈ 8192 at batch 1).
> - **Failures.** On trained factors a small Gram shift fails 1/96 (1e-7) or 4/96 (1e-6) factor-ages. Every failure is
>   at age ≤ 20, and most are on large (n ≥ 2048) left factors. The Householder fallback is mandatory.
> - **Rejected variant.** "1 unshifted pass + NS" is inaccurate (worst column angle up to 0.82 rad).
>
> Details in C1 and C3 below; evidence in `checks/reconcile_cholqr3*.out`.

**This ranking departs from the reviewers' mean ranks.** They put PB1 (2.25) ahead of PB5 (2.5) and PB4 (3.75). The
moderator's order rests on CPU checks no reviewer saw (P2_SYNTHESIS §6 and §9).

**Implementation.** None of these is implemented in the harness. Reference code for C1, C4 and C5 is in
`checks/candidate_specs.py`; it ran on CPU against trained snapshots (`checks/candidate_specs.out`). Porting goes
through the main session.

**Where each piece plugs in:**
- **QR-type refreshes:** a new `kind` in `refresh_study_replay.py`, next to `'qr'`, `'cholqr'` and `'eigh'`, with a
  schedule name in `SCHEDULES` and `refresh_kind()`, and its failure count recorded like `cholqr_failures`.
- **Warm-type refreshes:** a new `gen=` branch or `VARIANTS` entry in `refresh_study_replay_pb.py`, chained as
  `<variant>|warm10_qr50` and `<variant>|warm10`.
- **Timing:** a `METHODS` entry in `refresh_study_microbench.py`, or in `_ext`.
- **Metrics:** the harness's own (r_C, r_C^w, top-1/8/n-8 overlap, e_Q, ms per factor) unless a candidate defines a
  new one.

**Comparisons common to every candidate:**
- "Late window" means ages 500–1000.
- "Gap" means the late-window interval metric of default warm10/QR50 minus that of QR10 (Householder), per width.
- "Closes X% of the gap" means (default − candidate) / gap ≥ X.

**TF32.** Every FP32 Gram product (ZᵀZ, XᵀX) must run in IEEE FP32. Matmul precision "high" sets
`allow_tf32 = True`.
- In the eager replay, wrap the Gram products in `allow_tf32 = False`, as `ieee_fp32()` does.
- In compiled timing, set `torch.backends.cuda.matmul.fp32_precision = 'ieee'` for the process or cell before
  compiling. Then confirm that the Gram error matches an eager IEEE run.

---

## C1. Gram-shifted CholeskyQR3 for QR events, with no operator ridge (PA1)

> **Correction (2026-11-20, ~05:45). Reconciliation with refresh-study's trained-factor result.** Refresh-study found
> CholQR3's later passes failing on 11 of 16 age-10 trained factors. The panel claimed 1 of 48.
>
> **The difference is the shift size, not the snapshots, Q, precision or failure criterion.** On exactly their 16
> factors (d = 256/512, age 10, Q = the production eigh initialization, IEEE FP32 on CPU, failure = any pass with
> `info ≠ 0`), `checks/reconcile_cholqr3.py` gives:
>
> | shift on pass 1 | failures | max column angle vs FP64 Householder, where it succeeds |
> |---|---|---|
> | Fukaya's 11(2n² + n)·u·‖X‖² (theirs; 0.1–100+ ‖X‖² for square n×n) | 9/16 (theirs 11/16) | — |
> | 1e-4·‖X‖² | 3/16 | — |
> | 1e-5·‖X‖² | 2/16 | — |
> | **1e-6·‖X‖²** | **0/16** | ≤ 7e-4 rad |
> | **1e-7·‖X‖²** | **0/16** | ≤ 1.5e-3 rad |
>
> A large shift leaves the first pass a useless preconditioner, so the unshifted passes fail. The indefiniteness of
> C does not matter by itself, because XᵀX = QᵀC²Q is PSD.
>
> **Wider failure rate** (96 trained factor-ages: d = 256/512 at ages 10, 20, 50, 500, 1000; d = 1024 at ages 10 and
> 500):
> - **1e-7:** 1/96 (d = 1024, age 10, n = 4096);
> - **1e-6:** 4/96 (ages 10–20; three are the n = 2048, 3072 and 4096 left factors at d = 1024, age 10);
> - **Fukaya:** 34/96;
> - **no small-shift failures at ages 500 and 1000.**
>
> The small-shift failures break at the **last pivot** of pass 2: the early large BF16 factors are numerically rank
> deficient, with eigenvalues crossing zero. So no fixed shift is reliable on them, and the **Householder fallback for
> failed elements is required**, not optional. Use `shift_rel = 1e-7`. d = 2048 (n up to 8192) was not checked.
>
> **Cost** (refresh-study `microbench_ext_ieee.json`: Fukaya shift, same operation count as this spec apart from
> 20 vs 8 power iterations):
> - IEEE-Gram CholQR3, batch 12, is 7.8 / 28.9 / 78.8 / 173.8 ms per call at n = 768 / 1536 / 2304 / 3072. That is
>   **0.26 / 0.36 / 0.56 / 0.81× Householder**, but **1.07× at n = 4096 and 1.93× at 8192**. At batch 1 it is
>   0.78× at 4096 and 1.17× at 8192.
> - The 109M event is **≈ 320 ms vs 594**, so C1 saves **≈ 5.5 ms/call at QR50** (not 7.8), less the Householder
>   fallbacks at early ages.
> - The earlier "≈ 198 ms" used TF32-Gram times and is withdrawn.
> - **Use C1 only for factors with n ≤ 3072.** Above that, keep Householder, with C7's concurrency if it helps.
>
> **The "1 unshifted pass + BF16 NS" variant is rejected.** It never fails at d ≤ 512, but its worst column angle
> reaches 0.72–0.82 rad, and it fails 3/16 at d = 1024.

**Why.**
- The QR increment is 11.82 of 26.08 ms/call, and Householder runs at 0.8–23 TFLOP/s against warm's 165–174.
- PA1 is ranked #1 for speed by Fable, Neon and Carbon. astra ranked it #9 pending reliability on trained factors.
  That reliability is now checked on CPU (P2_SYNTHESIS §6.1):
  - on trained snapshots, a small Gram shift (1e-6·‖Z‖₂²) with two unshifted passes failed on 1 of 48 early
    factor-ages in FP32 at d ≤ 512 (n ≤ 2048). *Corrected above: 4/96 across d ≤ 1024; 1e-7 gives 1/96.*
  - it reproduced Householder exactly (top-n/8 distance 0);
  - unshifted CholQR2 failed 39/48, and Fukaya's theoretical shift failed 25/48.
- **Do not add the operator ridge** that every reviewer spec carries. At τ = 1e-4·d_max it moves the top-n/8
  subspace by up to 0.1 on early factors, and it can create failures near negative eigenvalues.

**Spec** (from `checks/candidate_specs.py`):

```python
def cq3s_basis(cov, q, shift_rel=1e-6, unshifted_passes=2, power_iters=8):
    c, old = cov.float(), q.float()
    z = mm32(cov, q)                                   # BF16 operands exact in FP32 accumulation
    order = (old * z).sum(-2).argsort(dim=-1, descending=True)
    x = z.gather(-1, order.unsqueeze(-2).expand_as(z))
    n = x.shape[-1]; eye = torch.eye(n, device=x.device)
    ok = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    with ieee_fp32():
        g = torch.bmm(x.mT, x)
        v = torch.full((x.shape[0], n, 1), n ** -.5, device=x.device)
        for _ in range(power_iters):                   # lambda_max(G) = ||X||_2^2
            v = torch.bmm(g, v); v = v / v.norm(dim=-2, keepdim=True).clamp_min(1e-30)
        lam = torch.bmm(v.mT, torch.bmm(g, v))
        for k, s in enumerate([shift_rel * lam] + [None] * unshifted_passes):
            if k: g = torch.bmm(x.mT, x)
            u, info = torch.linalg.cholesky_ex(g + s * eye if s is not None else g, upper=True)
            ok = ok & (info == 0)
            u = torch.where(ok.view(-1, 1, 1), u, eye)
            x = torch.linalg.solve_triangular(u, x, upper=True, left=False)
    ok = ok & torch.isfinite(x).flatten(1).all(1)
    return x.to(torch.bfloat16).contiguous(), order, ok
# Fallback: recompute the batch elements with ~ok via production qr_basis (one host check per QR event).
# Permute V by `order` exactly as the QR path does.
```

**Variants:**
- `shift_rel` ∈ {1e-7, 1e-6} (corrected: 1e-5 and larger fail more often);
- `unshifted_passes` = 2. The "1 pass + NS" option is rejected (correction above);
- add this small-shift variant to the queued Part 1b next to its Fukaya-shift CholeskyQR3.

**Replay.** Two chains: `warm10_cq3s50` (production warm, C1 at ages divisible by 50) and `cq3s10`. Compare them with
`warm10_qr50` and `qr10`. Also branch C1 and `qr_basis` from the same input at every refresh age.

**Decision.**
- Fallback rate ≤ 1% of factor-refresh events after age 50, and no non-finite outputs. Report early ages
  separately; they will fall back more often.
- Same-input top-n/8 subspace distance to Householder: median ≤ 1e-5, max ≤ 1e-3.
- Late-window r_C and r_C^w interval within 0.003 of the Householder chain at every width.
- e_Q ≤ 2.5e-3.
- Microbenchmark (compiled, IEEE Grams): the 109M QR event ≤ 330 ms including fallbacks (against 594), i.e.
  ≥ 5.3 ms/call saved at QR50. Report per-factor ms at n = 256–8192, batch 1 and 12. *(Corrected from ≤ 200 ms /
  7.9 ms/call.)*
- **Already measured.** ~~≈ 198 ms with TF32 Grams~~ → **≈ 320 ms with IEEE Grams** (refresh-study, Fukaya shift).
  The small-shift form has the same operation count; its GPU timing and fallback overhead are still unmeasured.

**Proposers:** Fable (A1), Neon (A5), Carbon (A4), astra (B4). The no-ridge form is the moderator's modification.

---

## C2. One Newton–Schulz correction in the warm step (PA6a)

**Why.**
- Part 2 already shows `ns=1` gives the same late-window r_C and r_C^w at every width, with e_Q 2.1–2.2e-3 and 0.64
  vs 0.82 ms at n = 768, batch 12.
- All four reviewers accept it; astra ranks it #1 for speed.
- Expected saving: about 14 ms per warm event, about 1.1 ms/call. At large n it removes 2 of 7 products, where warm
  is at the tensor-core limit.

**Spec.** Production `warm_basis` with `range(2)` replaced by `range(1)` (`warm_basis_param(..., ns_steps=1)`).

**Remaining check** (astra): over all chained factor-ages of `warm10_qr50|ns=1`, the **spectral** Gram error
max‖QᵀQ − I‖₂, including the 4th consecutive warm refresh.

**Decision.**
- max spectral Gram error < 0.07, all outputs finite;
- late interval metrics within 0.003 of default (already true);
- compiled 109M warm event ≥ 8 ms faster than 61.6 ms.

If it passes, make it the production default. The 6-product seed I + A + ½A² + 1 NS (PA6b) is needed only if a
larger cap is adopted (C5).

**Proposers:** astra (A4), Carbon (A5), moderator prior 3.

---

## C3. Exact QR at every refresh via C1, no warm step (PB4)

> **Correction (2026-11-20, ~05:45).** The "≈ 20.3 ms/call" and "hybrid ≈ 16.6" estimates below used TF32-Gram
> CholQR3 times and **do not hold**. With refresh-study's IEEE-Gram times (109M event ≈ 320 ms plus ≈ 5 ms of
> transport), refresh cost per call is:
>
> | schedule | refresh ms/call |
> |---|---|
> | today, warm10/QR50 | 15.94 |
> | warm10 / C1-50 (C1 replaces only the QR repairs) | ≈ 10.6 |
> | **C1 every 20 ages, no warm** (QR20 quality) | **≈ 16.3** |
> | **C3, C1 every 10 ages** (QR10 quality) | **≈ 32.5** |
> | Householder QR20 (measured) | ≈ 30.3 |
> | Householder QR10 | ≈ 59.9 |
> | size-hybrid (C1 every 10 on the n ≥ 2304 left factors, warm10/C1-50 elsewhere) | ≈ 27.8 |
>
> - **C3 is therefore not cheaper than today:** it is +16.6 ms/call, about +5.4% of the 309.5 ms step. It buys QR10
>   quality at roughly QR20's Householder cost.
> - **The new break-even option is C1 every 20 ages.** It costs about the same as today, and Part 2 shows QR20 beats
>   warm10/QR50 at every width (interval r_C 0.255–0.371 vs 0.290–0.419).
> - **Early-age failures:** C3 runs C1 at every refresh, so the early-age Householder fallbacks (C1 correction) add
>   cost during the first ~20 ages.
>
> Revised decision: put **C1-20 and C1-10 (C3)** in the 2,000-call training run alongside warm10/C1-50, against
> BF16 QR10 and warm10/QR50.

**Why.**
- It is QR10 by construction, so it removes the warm/QR quality gap entirely.
- Ranked #1 for tracking by Fable and Carbon (Neon #4; astra #9, because of the ridge, which C1 removes).
- The trained generator-size check (P2_SYNTHESIS §6.2) suggests no bounded first-order step can follow the 10-age
  drift on the 3d/4d left factors, where warm sits at r_C 0.64–0.80 against QR10's 0.35–0.54.

**Spec.** Chain `cq3s10`: C1 at every refresh age, Householder fallback, V permuted by `order`, M transported as now.

**Decision.**
- **Quality:** identical to `qr10` within numerics, i.e. as C1.
- **Cost.** Break-even with today's 15.94 ms/call is about 159 ms per event (96 factors plus 4 transports).
  - ~~From Part 1b's 3-pass times, the event is about 203 ms, i.e. ≈ 20.3 ms/call~~ *(TF32 times; superseded by
    the correction above: ≈ 32.5 ms/call with IEEE Grams).*
  - ~~Size-hybrid … about 16.6 ms/call~~ *(superseded: ≈ 27.8 ms/call).*
  - Decide between pure C3, the hybrid and C1 alone on the loss run below, not on cost alone.
- **Then** a 2,000-call training run on seed 204 against the pending compiled BF16 QR10 control and warm10/QR50. It
  should match BF16 QR10's CE at 500 and 2000 to within seed noise.

**Proposers:** Fable (B1), Neon (B6b), Carbon (B3/E1).

---

## C4. Head-panel power step at warm ages (PB5)

**Why.**
- Second-best mean tracking rank (2.5, after PB1's 2.25), and the only candidate every reviewer ranks #2 or #3.
- In the CPU smoke test (bases 50 ages stale, trained snapshots) it reached QR's r_C to within 0.002 in all three
  cases. Every bounded first-order variant stayed at 0.81–0.89 against QR's 0.44 and 0.087.
- It needs no ridge: head columns are well conditioned.
- **Smoke-test rerun** (10-age-stale basis, whitened metric):
  - its first k columns match QR's (subspace distance ≤ 1.3e-4), and r_C matches QR's;
  - **but ρ_rms stays well above QR's** (0.42 vs 0.24 at n = 256; 0.95 vs 0.35 at n = 768), because the tail is not
    power-stepped;
  - so it fixes the head, not the weak directions. Test it alone, then combined with C5's tangent on the remaining
    columns (Carbon's E3).

**Spec** (from `checks/candidate_specs.py`, the form of Fable's P2 E2):

```python
def head_panel_basis(cov, q, frac=0.125, ns_steps=1):
    c, old = cov.float(), q.float()
    z = mm32(cov, q)
    order = (old * z).sum(-2).argsort(dim=-1, descending=True)
    idx = order.unsqueeze(-2).expand_as(z)
    qp, zp = old.gather(-1, idx), z.gather(-1, idx)
    n = qp.shape[-1]; k = max(1, int(frac * n))
    with ieee_fp32():
        xk = torch.bmm(qp.mT, zp[..., :k])                       # (b, n, k) panel in basis coordinates
        h, r = torch.linalg.qr(xk, mode='complete')              # (b, n, n)
    sgn = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)); sgn = torch.where(sgn == 0, 1., sgn)
    h = torch.cat([h[..., :k] * sgn.unsqueeze(-2), h[..., k:]], -1)
    y = mm32(qp.to(torch.bfloat16), h.to(torch.bfloat16)).to(torch.bfloat16)
    eye = torch.eye(n, device=y.device)
    for _ in range(ns_steps):
        e = mm32(y.mT, y) - eye
        y = (y.float() - .5 * mm32(y, e.to(torch.bfloat16))).to(torch.bfloat16)
    return y.contiguous(), order                                 # permute V by `order`
```

- **Variants:** `frac` ∈ {1/16, 1/8, 1/4}.
- **Cheaper completion:** a Procrustes completion of the tail, H₊ = I − V₊(I + S)⁻¹V₊ᵀ with O(n²k) work (Neon B3 and
  Carbon E3; the identity was checked in M1). It needs no complete n×n QR and keeps the tail minimally moved.
- **Replay:** chains `panel10_qr50` and `panel10_cq3s50`. Report the 3d/4d left factors separately.

**Decision.**
- Late-window r_C interval closes ≥ 50% of the gap at every width.
- r_C^w interval no worse than default warm (the tail is not power-stepped).
- Top-1 and top-n/8 overlap after refresh within 0.02 of QR10.
- e_Q ≤ 2.6e-3.
- ≤ 2× warm per factor at n ≥ 1536 for the completion actually used.

**Proposers:** astra (B2), Fable (B4/E2), Neon (B3), Carbon (E3).

---

## C5. Bounded QR-tangent generator, with the bound as the variable (PB1 + PB2)

**Why.**
- It is the cheapest tracking change (same product count), first for astra and Neon.
- The refresh study is already replaying `pb1` with Frobenius and spectral caps at 0.5/1.0, ρ ∈ {1e-4, 1e-3}, and
  `ns=1`.
- **Moderator warning (P2_SYNTHESIS §6.2).** On trained factors the generator's ‖A‖₂ is 1–116, with no entry above
  the ±1 clamp, so any cap of about 0.5 scales the step by 0.005–0.5. In the smoke test, row-ℓ₂ + spectral moved
  furthest. Expected: worthwhile on the d/2d factors, weak on the 3d/4d left factors.

**Part 2b results (d = 256–1024; tables_pb.md):**
- the best tested bound, spectral cap 0.5, closes 23–40% of the r_C^w gap and 9–25% of the r_C gap;
- its cap binds on 99–100% of refreshes (mean multiplier 0.06–0.23);
- Frobenius-capped PB1 is no better than production on r_C;
- relative-gap damping is about 0%, and factor-1 Jacobi is about production.

The remaining question is whether the two untested bounds below do better. In the moderator's smoke test,
row-ℓ₂ + spectral was the best tangent: ρ_rms 0.38 vs production 0.45 at n = 256, but no help on the n = 768 left
factor.

**Spec.** `qr_tangent_basis(cov, q, rho, bound, cap)` in `checks/candidate_specs.py`. It uses the same generator
convention as `refresh_study_replay_pb.warm_generic(gen='pb1')`. Add two bounds to the running grid:
- `bound='schatten4'`: (½‖AᵀA‖_F²)^¼ = (Σθ⁴)^¼ (astra E1), one extra product;
- `bound='row_spectral'`: D_wAD_w with w_i = min(1, 0.5/‖a_i‖), then a spectral cap (Carbon and Neon E1).

Use `cap` ∈ {0.5, 0.65} and `ns_steps` 2, or 1 with the seed I + A + ½A² (PA6b) when the cap is 0.65. Log the
generator size and the cap multiplier per factor and age.

**Decision.**
- r_C^w interval closes ≥ 35% of the gap at every width (astra's H_w ≥ 0.35), with no non-finite outputs.
- e_Q ≤ 3e-3; cost ≤ 1.15× warm at n = 3072.
- Report d/2d and 3d/4d separately.
- If it passes only on d/2d, keep it as the warm step for those factors, with C3 or C4 on the large left factors.

**Proposers:** astra (B1/E1), Fable (B3/E3), Neon (E1), Carbon (B1/E2).

---

## C6. Front-loaded schedule (PB7), with the V diagnostic (PB8)

**Why.**
- All four reviewers rank it, and moderator prior 11.
- The call-500 gap (+0.12) is 10× the endpoint gap and also appears with FP32 state.
- Two supporting papers:
  - Purifying Shampoo found tight-early / loose-late refresh thresholds better than the reverse;
  - Wang et al. (2026, arXiv 2610.18695) show refresh placement matters at a fixed budget, with the sign depending on
    LR.

**Spec.**
- **Replay:** `qr10_to300_then_warm10_qr50`, i.e. QR (or C1) every 10 ages for ages ≤ 300, then production. Report
  r_C and r_C^w for ages 0–300 and 300–1000.
- **Training:** 2,000 calls on seed 204, with three arms: compiled BF16 QR10 (the pending control), warm10/QR50, and
  front-loaded. Cost: +24 QR events, about +1.3 ms/call amortized over 10k calls.
- **Log in all arms** (Fable's settling test for D2): the p99.9 of |M_f/(√V_f + ε)| on the first call after each
  refresh against ordinary calls, and the row sums of V against diag(QᵀCQ).

**Decision.**
- Front-loading closes ≥ 50% of the call-500 CE gap to BF16 QR10, and does not widen the call-2000 gap.
- Separately: if the post-refresh p99.9 is within 2× of ordinary calls, PB8 is dropped.

---

## C7. Concurrent Householder QR (PA2), the exact-arithmetic fallback

> **Note (2026-11-20, ~05:45).** After the C1 correction, Householder stays the QR path for n ≥ 4096 (where
> IEEE CholQR3 is slower) and for C1's early-age fallbacks. C7's concurrency therefore matters at large n even if C1
> passes.

**Why.** It needs no numerics change. It is the backstop if C1 fails on GPU, and it ranks #2 for astra and Neon.

**Spec.** Production `qr_basis` with the per-matrix `torch.linalg.qr` calls dispatched round-robin over S ∈ {1, 2, 4,
8} CUDA streams: a common input-ready event, a join before use, and nonaliasing workspaces. Also time
`qr(mode='r')`, to measure the geqrf/orgqr split that PA3 (R-only + triangular solve) would exploit.

**Decision** (microbenchmark, 30 interleaved QR events):
- ≥ 20% reduction in the median 109M QR event;
- no p95 regression;
- outputs bit-identical, or within 1e-6, to stock.

**Proposers:** astra (E3), Neon (A1/E3), Carbon (A3), Fable (A4).

---

## C8. Ordinary-step and small-n engineering (PA9)

**Why.** It is the only lever on the ordinary step, 39% of optimizer time, which every call pays. All four reviewers
propose it, at exact arithmetic.

**Spec** (microbenchmark event regions; check states bit-equal to production):
- (a) coalesce the five n = 768 factor batches into one batch-60 warm call;
- (b) pre-stacked parameter and gradient banks (no `torch.stack` or `copy_` per call);
- (c) CUDA graphs (`mode="reduce-overhead"`) for the ordinary and warm regions.

**Decision.** (a) ≥ 1.5 ms per warm event; (b) ≥ 0.5 ms per ordinary event; (c) report separately.

---

## Not carried forward, with reasons

| Candidate | Reason |
|---|---|
| PA12, overlap the refresh with the next forward/backward | Needs an in-training timing harness. astra bounds the hidden time at ≤ 9.79 ms/call with current kernels; Carbon claims more. Next after C8 |
| PA13, size-aware schedule | Subsumed by C3's hybrid |
| PA3, R-only Householder | Variant inside C7 |
| PA8, PA10, PA11 | Later engineering; PA11 changes the statistics and needs a training run |
| PB3, full Jacobi | Dominated by the QR tangent, and has the same size problem |
| PB6, grouped block QR | Prior art (Milligan et al. 2026); worth it only with basis-frame storage |
| PB8, V transport | Diagnostic only, inside C6. T∘T mixing has a counterexample and a published adverse result |
| PB12, adaptive trigger | Known (FOAM, EShampoo); try static per-size rules first |
| Basis-frame (P) storage of C (Milligan et al.) | Not in the pool. Fable suggests it may reduce the BF16 indefiniteness and the BF16 loss penalty. A separate study; flagged for the main session |

**The single most valuable measurement** for all eight is still the matched, compiled BF16 QR10 run (all four
reviewers, both stages).
