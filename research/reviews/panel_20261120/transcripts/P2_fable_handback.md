## Provenance

- **Read:** the one prompt file, in full (seven paged reads, lines 1–2347). No other file was opened.
- **Web:** 14 WebSearch calls and 9 WebFetch calls. One fetch failed (the Fukaya PDF came back as binary; I used the ar5iv HTML instead and did not open the saved copy). One page (Nam & Ahn) was fetched twice to read further in.
- **Other:** one ToolSearch call to load the web tools, and one advisor consultation before writing.
- **Not done:** no commands, no code, no subagents.
- **Reading status.** "Read" below means I fetched the page and worked from WebFetch's summary of it, so I quote no equations from those sources. "Search only" means I saw it in search results and did not open it.

---

## 1. What changed for me

### Contradicted

- **I4, "the Frobenius cap is about √n too tight" (10× at n = 768).** M2 measures a Frobenius-to-spectral ratio of 2.5–5 with no trend in n. My unstructured-A estimate was wrong for the production generator, whose damping confines A to head-versus-all pairs. Only the conclusion that the cap binds survives (a = 0.46–0.79).
- **B2, "a spectral cap likely doubles the step".** The synthetic prediction held (0.20 → 0.10–0.11 predicted, 0.115–0.118 measured). It does not transfer to trained covariances:
  - cap = 1.0 moves interval r_C by only −0.009 to −0.011, against a QR50→QR10 gap of 0.11–0.12;
  - with no cap, 59–82% of factor-ages go non-finite.
  So the gain from the cap on trained C is about 10% of the gap, and a bound is mandatory.
- **A1, "κ ≳ 4100 predicts the NaN between 4096 and 6144".** M4 ran κ = 7.2e3 at n = 4096 (κ²u ≈ 3) with no failure. My κ²u explanation of the microbenchmark failure is no better supported than Reviewer Z's TF32 explanation.

### Weakened

- **B3 (pooled PB1), "most of the way from warm to QR10".**
  - The tangent is correct (M1), and the chained toy agrees: ρ_rms 0.243 against 0.240 for QR10.
  - At input r_C 0.45, the one-shot tangent is worse than the uncapped half step (ρ_rms 0.676 against 0.485).
  - Trained pre-refresh r_C is 0.29–0.47 on average and 0.64–0.80 on the 3d and 4d left factors. PB1 is therefore outside the first-order regime exactly where the deficit is largest.
- **A1 / B1 (pooled PA1 / PB4), the ridge guidance.**
  - Unshifted Cholesky-QR fails at 800/800 refreshes for d ≥ 512, and some factors are indefinite. I had specified a ridge and a `cholesky_ex` fallback, so the design survives, but two of my statements do not.
  - "A ridge near 1e-4·λ₁ costs no fidelity" is wrong: M4 shows column changes up to 3.1e-3 at 1e-4 and 2.6e-2 at 1e-3.
  - "86–100 ms per event, 1.7–2.0× warm" holds only for one pass plus Newton–Schulz (NS). A two-pass version is 115–140 ms, which is 2.3–2.8× the warm event, and it misses the per-factor 2× gate at n = 768 (0.171 ms against 0.138 ms, even for one pass).
- **B7 (pooled PB8b), rescaling V to the new diagonal.** It is exact when V is consistent with C, but M5's stale case gives M/√V = 3.9e4. It cannot ship without clipping.
- **A3, the six-product retraction.** It is dominated at the current cap by the simpler one-NS variant. The R5 ablation shows one correction gives identical r_C with e_Q 2.1–2.2e-3, at 0.64 ms against 0.82 ms.
- **Pitfall 3, BF16 dead-bands.** M6 gives 0.6–0.8% with sample noise (4.2% only for noiseless drift). I downgrade it for C.

### Confirmed

- **I1 and I2** (QR tangent), the NS scalar map, and the I + A + ½A² identity (M1).
- **I3, head-only motion.** r_C^w for warm10/QR50 is 0.62–0.82, close to QR50 (0.66–0.86) and far from QR10 (0.46–0.59). 40–99% of directions sit below 1% of λ_max.
- **Pitfall 4, the small-rotation premise.** Even eigh10 climbs from 0.002 back to 0.26–0.36 within 10 ages. A QR removes 38–49% of the current r_C; a warm refresh removes 3–8%.
- **"Leave V unchanged is first-order correct"** (M5, consistent row: no spike under any rule). The second-order error is large in relative terms, though: 0.0395 against a true 1e-6.

### Single biggest change

In stage 1 I treated the warm step's deficit as a calibration problem: a gain of ½, a loose cap, and absolute damping. Part 2 shows it is structural. The 10-age drift exceeds what any bounded first-order step removes, and the best single knob moves interval r_C by at most 0.016 of a 0.11–0.12 gap. I now treat a real power step (PB4, with PB5 as the robust variant) as the main line and generator fixes as a hedge. The Cholesky route also needs an explicit treatment of null and indefinite directions; a small ridge is not enough.

---

## 2. Prior work

### Refresh frequency and cost in SOAP and Shampoo

- **Nam & Ahn (2026), "FOAM: Frequency and Operator Error-Based Adaptive Damping Method for Reducing Staleness-Oriented Error for Shampoo"**, [arXiv 2606.02365](https://arxiv.org/abs/2606.02365), ICML 2026. Read.
  - It is a per-factor trigger for Shampoo. A whitened staleness proxy is checked every f steps; damping is raised first, and a full eigendecomposition runs only when damping would exceed a ceiling.
  - Reported: eigendecomposition calls fall to about 5–10% of the fixed-frequency count on ViT; the proxy costs 2–5% of optimizer time.
  - There is no warm-started or approximate basis update. SOAP is listed only as a future direction.
  - Its appendix argues that a small diagonalization residual can coexist with a large inverse-root error. That is an independent argument that r_C is not the right proxy.
  - It also reports that warm-started QR needed 8 iterations to reach a residual of 0.098 on saved ViT states, at 3.9× the eigendecomposition time. That fits R5: one power step removes only 38–49%.
  - **Status of pooled candidates:** PB12 (adaptive trigger) is known. A whitened residual, of which r_C^w is one form, is the established kind of metric.
- **Yu, Heo, Yun & Han (2026), "FOAM: Factored One-sided Adam-Moment for Practical and Scalable SOAP"**, [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/149122). Read (abstract page only); details [unverified].
  - Input-side-only preconditioning plus a factored second-moment estimator built from row and column statistics, claimed equivalent to SOAP under a K-FAC assumption.
  - It takes the one-sided shortcut that this project excludes. Its relevance is to PB8b: the row/column factorization of V is the identity that the "rescale V to d⁺/d" proposal relies on.
- **Milligan, Xu, Lacoste-Julien, Dangel & Lin (2026), "Reparametrizing Shampoo and SOAP for Subspace Basis Updates and BFloat16 Storage"**, [arXiv 2605.26327](https://arxiv.org/abs/2605.26327), preprint. Read.
  - They store P = QᵀSQ in place of the factor S. **This is the moderator's M7, so M7 is prior art.**
  - They update part of the basis by a QR of P[I, I] on a greedily chosen index set (the largest off-diagonal entry plus the indices most coupled to it), and leave the other columns alone. Random selection is reported as usually worse. **This is PB6 in the basis frame; PB6 is known.**
  - Reported FP32 → BF16 loss deltas for SOAP: +0.013 (nanoGPT 123M) and +0.014 (Llama3 119M) in the standard parametrization, and +0.005 / +0.006 in theirs.
  - **Why this matters here.** The standard-parametrization BF16 penalty is the size of the whole +0.010 gap in this project. Their setup differs (FineWeb, per-method hyperparameter search, an unstated dtype map), so it does not transfer directly. It does make the pending BF16 QR10 control the most valuable outstanding measurement.
  - The summary I read does not say how the second moment is remapped after a subspace update.
- **Eschenhagen et al. (2025), "Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner"**, [arXiv 2506.03595](https://arxiv.org/abs/2506.03595), NeurIPS 2025. Read. Co-authors not retrieved.
  - Their adaptive criterion is r_C ≤ τ, checked every F steps, with warm-started QR iterations until it holds (τ = 0.1, F = 50 in their AlgoPerf runs).
  - By that standard this project's QR10 (r_C ≈ 0.2) is itself unconverged.
  - They find basis updates matter most early: a tight threshold early and a loose one late gave a much better loss than the reverse. **This supports PB7.**
  - They leave open whether the eigenvalue correction needs basis-aware treatment when the basis changes. **PB8 is an open question, not a known result.**
- **Vyas et al. (2024), "SOAP: Improving and Stabilizing Shampoo using Adam"**, [arXiv 2409.11321](https://arxiv.org/abs/2409.11321), and the [reference code](https://github.com/nikhilvyas/SOAP). Paper seen in search results; `soap.py` read.
  - The reference sorts by the estimated eigenvalues, only index-permutes the second moment, refreshes after the parameter update, and runs QR in float32.
  - So PB9 (sort and permute at every refresh) is what the target does. PB10 (refresh before the update) and PB8a (T∘T mixing) leave the QR10 target.
- **KL-Shampoo / KL-SOAP, "Understanding and Improving Shampoo and SOAP"**, [arXiv 2509.03378](https://arxiv.org/abs/2509.03378). Search only; [unverified].

I found no prior work that replaces the power-plus-QR step with a first-order skew generator and a polar correction. The current warm step and PB1 look new as far as I could find [unverified].

### CholeskyQR2 and shifted CholeskyQR3

- **Fukaya, Kannan, Nakatsukasa, Yamamoto & Yanagisawa (2020), "Shifted CholeskyQR for computing the QR factorization of ill-conditioned matrices"**, [arXiv 1809.11085](https://arxiv.org/abs/1809.11085). Read via ar5iv.
  - **The shift goes on the Gram matrix only.** X = QR still holds, so the final Q is that of the original matrix. The first-pass Q is not orthonormal, which is why two more passes follow.
  - Recommended shift: s = 11(mn + n(n+1))·u·‖X‖₂². One shifted pass cuts the condition number by about √(s/‖X‖₂²).
  - The proven range requires s ≤ ‖X‖₂²/100, and it does not cover rank-deficient input.
  - **The guarantee is vacuous here.** In FP32 with m = n = 768, s/‖X‖₂² = 11 × 1.18e6 × 6e-8 ≈ 0.78, and 12.5 at n = 3072. Any use at these sizes is empirical.
- **Two different shifts are in the pool.**
  - The pooled "ridge" is on the operator: Z = (C + τI)Q. It changes Q (M4), but it has a sensible meaning: directions with d ≪ τ keep the old basis.
  - Fukaya's Gram shift keeps Q but cannot orthonormalize null directions.
  - Reviewer Y's warning about "short columns" applies to a Gram-only shift used without follow-up passes.
- **Indefiniteness is not what breaks the Gram.** ZᵀZ = QᵀC²Q is positive semidefinite even when C is indefinite. The failure in R5 is near-singularity. An operator ridge can make an indefinite C worse if τ lands near |λ_min|.
- **Column-based reduced shift:** Fan, Guan & Qiao, [arXiv 2408.06311](https://arxiv.org/abs/2408.06311). Search only. It supports equilibrating columns and using a smaller shift.
- **Rank-revealing randomized CholeskyQR:** [arXiv 2210.09953](https://arxiv.org/abs/2210.09953). Search only; authors not confirmed. Claims unconditional stability on rank-deficient input.
- **Mixed precision:** Yamazaki, Tomov & Dongarra (2015), "Mixed-Precision Cholesky QR Factorization and its Case Studies on Multicore CPU with Multiple GPUs", SISC 37(3), [ICL page](https://icl.utk.edu/node/856). Search only.
  - Forming the Gram and its Cholesky factor in doubled precision makes the orthogonality error linear in κ, at a reported 1.4× cost.
  - An FP64 Gram would be the direct fix here. Its cost on a consumer Blackwell GPU is [unverified].
- **Garrison, Chen & Ipsen (2026)**, mixed-precision randomized preconditioning, [arXiv 2606.18411](https://arxiv.org/abs/2606.18411). Search only.
- **Status:** PA1 is a known family, but nothing I found covers FP32 at n ≈ 10³ on near-singular input.

### First-order and Jacobi-type tracking

- One orthogonal-iteration step per covariance update is the standard online subspace tracker in signal processing. Convergence theory under drift is reported as partial ([arXiv 1810.09298](https://arxiv.org/abs/1810.09298), search only).
- Drift scales as change in statistics over eigengap (Davis–Kahan, as cited in Nam & Ahn's appendix).
- Classical parallel Jacobi rotates disjoint pairs. Applying all pairwise angles at once is not a convergent Jacobi sweep, which fits M2 (simultaneous exact 2×2 angles give 0.131, worse than the first-order factor-1 step at 0.055).
- The equivalence of subspace iteration and the QR algorithm is textbook material; I did not re-check a source [unverified].

### Batched or concurrent QR on GPUs

- cuBLAS has a batched `geqrf` aimed at small matrices and no batched `orgqr`/`ormqr` ([NVIDIA forum](https://forums.developer.nvidia.com/t/batched-qr/52378), search only).
- A PyTorch commit description says batched `orgqr` uses MAGMA only for very small inputs and cuSOLVER otherwise, and a [forum post](https://discuss.pytorch.org/t/scaling-linalg-ops/116096) reports that QR does not scale with batch. Both search only.
- That matches the measured batch-1 = batch-12 per-factor time, but the dispatch in PyTorch 2.11 is [unverified].
- MAGMA's batch QR (Abdelfattah, Tomov & Dongarra, [ICCS 2022](https://www.iccs-meeting.org/archive/iccs2022/papers/133500064.pdf), search only) targets small-to-medium matrices.
- I found no published throughput for n = 768–3072 at batch 12. The speedups claimed for PA2 and PA4 are unverified.

### Adam's second moment across a basis change

- The reference permutes only. Eschenhagen et al. call basis-aware correction open. Yu et al.'s factored estimator is the nearest published relative of PB8b.
- I found nothing on T∘T mixing (PB8a). Both PB8 variants are new and untested.

---

## 3. Ranking

**One Part 2 "fact" should not be used for ranking.** The no-cap survivors show interval r_C of 0.036–0.052, below eigh10's 0.133–0.180 on the same trajectory. An exact refresh cannot be beaten on the same population, so the survivors are a selected subset. On r_C^w they are also worse than QR10 at d = 512 and 1024 (0.544 and 0.584 against 0.513 and 0.536). The divergence is most likely a retraction failure (NS outside its basin once ‖A‖₂ > √2); I do not read it as evidence about the generator.

### A: speed

1. **PA1, stabilized Cholesky-QR for QR events.** It targets the largest slice (11.8 ms/call), but it must be redesigned for near-singular and indefinite factors. Expected saving is about 9 ms/call if it passes; I give that about 0.45.
2. **PA6a, one NS correction.** Already validated in replay (identical r_C, e_Q 2.1–2.2e-3). It saves about 14 ms per warm event, or about 1.1 ms/call, with certainty. Variants (b) and (d) matter only if the cap is raised. Variant (c) has no self-correction of stored Gram error.
3. **PA3, R-only Householder plus a triangular solve.** With the same operator ridge it is the robust fallback for PA1, since its error scales as κ·u. Its value depends on the unmeasured `geqrf`/`orgqr` split.
4. **PA9, contiguous banks and batch-60 coalescing.** The only item that touches the 39% ordinary base. About 1 ms/call plus 2.5–5 ms per warm event.
5. **PA2, concurrent QR.** Possibly large on the five n = 768 groups (154 ms per event), but cuSOLVER concurrency is unverified, and it is moot if PA1 passes.
6. **PA13, size-aware schedule.** The 588 against 560 ms arithmetic is confirmed. It matters only at n ≥ 6144, and rises if Part 1b shows a stable Cholesky-QR at 8192–16384.
7. **PA12, overlap with the next forward/backward.** Only the latency-bound QR can overlap; the warm step already runs at the tensor-core limit. It is superseded if PA1 makes the QR event matmul-bound.
8. **PA8, symmetric kernels.** Up to 1–1.8 ms/call, but only if a custom kernel reaches about 70% of cuBLAS on this GPU, which is uncertain.

Not ranked:
- **PA5:** do it; about 0.1 ms/call, folded into PA1.
- **PA11:** changes the statistics and needs a training run.
- **PA10:** large n only.
- **PA7:** at most 0.5 ms/call.
- **PA4:** high effort, dominated by PA1 and PA3.
- **PA14:** e_Q is already below the QR baseline.

### B: tracking

1. **PB4, a real power step at every refresh.**
   - It is the only candidate that is the target by construction.
   - With two passes it costs about 12–14.5 ms/call of refresh, against 15.9 today.
   - It misses the per-factor 2× gate at n = 768 (about 3–4×), though the absolute cost there is under 15 ms per event.
   - The 3d and 4d left factors are where the warm step is worst (0.64–0.80 against 0.35–0.54) and where the Cholesky-to-warm cost ratio is best (1.2–1.5×). A size hybrid is the fallback.
   - It moves down if the fallback rate exceeds 1%, or if the ridge leaves r_C^w near the warm value.
2. **PB5, head-panel power step with an orthogonal completion.**
   - Head columns are well conditioned, so it needs no ridge and is indifferent to indefiniteness.
   - It directly targets top-1 overlap (0.67–0.85 against 0.78–0.92) and top-n/8 overlap (0.78–0.84 against 0.90–0.92).
   - It does not help weak directions.
3. **PB7, front-loaded schedule.** At most +1.3 ms/call. Supported by Eschenhagen et al.'s early-versus-late finding. It needs a training run.
4. **PB1 + PB2 + PB9, bounded and sorted QR tangent.**
   - Cheapest if it works; I expect it to help the d and 2d factors and not the large left ones.
   - **R6 dependence.** It moves to first if the PB1 chain closes at least 50% of the warm→QR10 gap in r_C^w at all widths with no non-finite values. It moves to the bottom if it diverges or closes less than 20%.
5. **PB6, grouped block QR.** Known in the basis frame (Milligan et al.), with greedy selection beating random or contiguous groups. It needs the P-frame storage to be cheap.
6. **PB3, full-gain Jacobi variants.**
   - M3 shows (a) at 0.121 one-shot and 0.254 chained, behind the tangent.
   - For (b), two production warm steps buy only 0.012 for 1.8× the cost.
   - It moves above PB1 only if R6's factor-1 Jacobi beats the tangent on r_C^w while staying finite under a bound.
7. **PB12, adaptive trigger.** Known and sound, but it breaks static graphs. A static size-based rule captures most of it.
8. **PB8, V after a refresh.** Training-only and off-target. M5 shows T∘T double-counts (2.01 against a true 0.01) and that rescaling spikes when V is stale. Log V row sums against d first.

Not ranked:
- **PB9:** packaged with PB1; M3 shows no effect on the toy.
- **PB10:** leaves the reference.
- **PB11:** e_Q is fine.

**A candidate the pool lacks: P-frame storage (M7, Milligan et al.).** My rough estimate is that BF16 rounding of a dense world-frame C adds spectral noise of about 3.4e-3·‖C‖_F/√n, roughly 1e-4·λ₁ at n = 2048. That would explain the 4–15% negative eigenvalues in R5. In the P frame the error is relative per entry of a near-diagonal matrix, so tail diagonals keep their precision. This is speculation, but it would explain Milligan's halved BF16 penalty, and it bears on PA1's conditioning problem.

---

## 4. Top three experiments

All three use `mm32` from the brief. `gram_ieee` is used only in the eager replay:

```python
def gram_ieee(Z):                       # Gram with TF32 disabled
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.bmm(Z.transpose(-1, -2), Z)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
```

### E1. Ridged, equilibrated Cholesky-QR power step at every refresh (PB4 / PA1)

```python
def cholqr_refresh(C, Q, rho=1e-3, shifts=(1e-4, 1e-6)):
    Cf, Qf = C.float(), Q.float()
    Cf = 0.5 * (Cf + Cf.transpose(-1, -2))
    Z = torch.bmm(Cf, Qf)
    d = (Qf * Z).sum(-2)                                    # [b, n]
    dmax = d.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    order = d.argsort(-1, descending=True)
    Z = Z + (rho * dmax).unsqueeze(-1) * Qf                 # operator ridge
    Z = Z.gather(-1, order.unsqueeze(-2).expand_as(Z))
    Z = Z / torch.linalg.vector_norm(Z, dim=-2, keepdim=True).clamp_min(1e-30)
    eye = torch.eye(Z.shape[-1], device=Z.device, dtype=Z.dtype)
    ok = torch.ones(Z.shape[0], dtype=torch.bool, device=Z.device)
    for s in shifts:                                        # two passes
        U, info = torch.linalg.cholesky_ex(gram_ieee(Z) + s * eye, upper=True)
        ok = ok & (info == 0)
        Z = torch.linalg.solve_triangular(U, Z, upper=True, left=False)
    Y = Z.to(torch.bfloat16)
    E = mm32(Y.transpose(-1, -2), Y) - eye                  # one NS correction
    Y = (Y.float() - 0.5 * mm32(Y, E.to(torch.bfloat16))).to(torch.bfloat16)
    ok = (ok & torch.isfinite(Y.float()).all(-1).all(-1)).view(-1, 1, 1)
    ident = torch.arange(Z.shape[-1], device=Z.device).expand_as(order)
    return torch.where(ok, Y, Q), torch.where(ok.view(-1, 1), order, ident)
```

- **Constants.** ρ swept over {1e-4, 3e-4, 1e-3, 3e-3}; Gram shifts 1e-4 then 1e-6 on unit-norm columns; one NS step; on failure the old basis is kept with the identity order.
- **Where.** The replay harness, as a "cholqr10" chain with no separate QR, at all four widths. Then the microbenchmark at n = 256–8192 and the 109M event region.
- **Pass criteria, for one fixed ρ:**
  - fallback rate ≤ 1% of factor-refresh events, with no non-finite values;
  - late-window r_C^w interval within 0.02 of QR10 at every width;
  - late-window r_C interval within 0.015 of QR10 at every width;
  - e_Q ≤ 2.6e-3;
  - compiled 109M event ≤ 150 ms and ≤ 2× warm per factor at n ≥ 1536 (report n = 768 separately; I expect it to miss the gate there).
- **Judge on r_C^w and overlaps, not on column distance to QR10.** QR10's own null-space columns are arbitrary.
- **Probability of passing all gates:** about 0.40 (0.6 numerics × 0.75 quality given numerics × about 0.9 timing).

### E2. Head-panel power step with a Householder completion (PB5)

```python
def head_panel_refresh(C, Q, frac=0.125):
    Cf, Qf = C.float(), Q.float()
    Cf = 0.5 * (Cf + Cf.transpose(-1, -2))
    Z = torch.bmm(Cf, Qf)
    d = (Qf * Z).sum(-2)
    order = d.argsort(-1, descending=True)
    idx = order.unsqueeze(-2).expand_as(Qf)
    Qp, Zp = Qf.gather(-1, idx), Z.gather(-1, idx)
    n = Qf.shape[-1]
    k = max(1, int(frac * n))
    X = torch.bmm(Qp.transpose(-1, -2), Zp[..., :k])        # [b, n, k]
    H = torch.linalg.qr(X, mode='complete').Q               # [b, n, n]
    Y = torch.bmm(Qp, H).to(torch.bfloat16)
    eye = torch.eye(n, device=Y.device, dtype=torch.float32)
    E = mm32(Y.transpose(-1, -2), Y) - eye                  # one NS correction
    Y = (Y.float() - 0.5 * mm32(Y, E.to(torch.bfloat16))).to(torch.bfloat16)
    return Y, order
```

- **Why it is exact on the head.** With an orthogonal basis, the first k columns of a QR depend only on the first k input columns. Sign flips are harmless, because M is transported through world coordinates and V is sign-invariant. A diagonal B returns the old basis.
- **Constants.** frac swept over {1/16, 1/8, 1/4}; one NS step; no ridge or shift.
- **Where.** The replay harness as panel10/QR50, then the microbenchmark.
- **Pass criteria, at frac = 1/8:**
  - top-n/8 overlap after refresh within 0.02 of QR10 at all widths;
  - late-window r_C interval closes at least 50% of the default-warm→QR10 gap at all widths;
  - r_C^w interval no worse than the default warm step;
  - e_Q ≤ 2.6e-3;
  - ≤ 2× warm per factor at n ≥ 1536, and a 109M event ≤ 150 ms.
- **Probability:** about 0.35 (quality about 0.55; the panel factorization may be latency-bound at n = 768, timing about 0.6).

### E3. Bounded, sorted QR tangent (PB1 + PB2 + PB9, with a second-order seed)

This complements R6: it adds a spectral bound and the six-product retraction to the tangent.

```python
def tangent_refresh(C, Q, rho=1e-3, theta=0.5, iters=8):
    B = mm32(torch.bmm(Q.transpose(-1, -2), C), Q)
    B = 0.5 * (B + B.transpose(-1, -2))
    d = B.diagonal(dim1=-2, dim2=-1)
    order = d.argsort(-1, descending=True)
    B = B.gather(-2, order.unsqueeze(-1).expand_as(B))
    B = B.gather(-1, order.unsqueeze(-2).expand_as(B))
    Qp, d = Q.gather(-1, order.unsqueeze(-2).expand_as(Q)), d.gather(-1, order)
    piv = torch.maximum(d, rho * d.abs().amax(-1, keepdim=True)).clamp_min(1e-30)
    L = (B / piv.unsqueeze(-2)).clamp(-1.0, 1.0).tril(-1)   # L_ij = B_ij / d_j, i > j
    K = L - L.transpose(-1, -2)
    n = K.shape[-1]
    v = torch.cos(1.7 * torch.arange(n, device=K.device, dtype=K.dtype) + 0.3)
    v = (v / v.norm()).view(1, n, 1).expand(K.shape[0], n, 1)
    for _ in range(iters):                                  # power iteration on KᵀK
        w = torch.bmm(K.transpose(-1, -2), torch.bmm(K, v))
        sig2 = torch.linalg.vector_norm(w, dim=-2, keepdim=True)
        v = w / sig2.clamp_min(1e-30)
    a = (theta / (1.2 * sig2.sqrt()).clamp_min(1e-12)).clamp(max=1.0)
    Kb = (a * K).to(torch.bfloat16)
    S = (Kb.float() + 0.5 * mm32(Kb, Kb)).to(torch.bfloat16)
    Y = (Qp.float() + mm32(Qp, S)).to(torch.bfloat16)
    eye = torch.eye(n, device=Y.device, dtype=torch.float32)
    E = mm32(Y.transpose(-1, -2), Y) - eye
    Y = (Y.float() - 0.5 * mm32(Y, E.to(torch.bfloat16))).to(torch.bfloat16)
    return Y, order
```

- **Constants.** Pivot floor ρ = 1e-3; angle bound θ swept over {0.5, 0.7}; 8 power iterations with a 1.2 safety factor; one NS step. Six products.
- **Where.** The replay harness as tangent10/QR50, then the microbenchmark.
- **Pass criteria:**
  - no non-finite values in any factor-age;
  - e_Q ≤ 2.6e-3;
  - late-window r_C^w interval at or below the midpoint between default warm and QR10 at every width (0.54 / 0.61 / 0.64 / 0.71 for d = 256 … 2048);
  - r_C interval no worse than default;
  - cost ≤ 1.0× warm at n = 3072.
- Report results separately for the d/2d factors and the 3d/4d left factors.
- **Probability:** about 0.25.

**First training run after these:** PB7 (QR10 for ages ≤ 300, then the current schedule), 2,000 calls on seed 204, read against the pending BF16 QR10 control.

---

## 5. Disputes

- **D1, target generator.** If a first-order step is used, the QR tangent is the right target for the QR10 trajectory (M1; chained toy ρ_rms 0.243 against 0.240). On trained inputs neither first-order family is reliable without a bound. Full-gain Jacobi aims at the exact eigenbasis, which is a different target. eigh10 is better than QR10 on every replay metric, so that target is not obviously worse for loss, but that is untested.
- **D2, V after a refresh.** Permute only, as the reference does. T∘T double-counts when a refresh corrects a misalignment (2.01 against 0.01). Rescaling is exact only when V is consistent with C, and M5's exactness is a 2×2 result that does not carry to two-sided V. PB8b is a clipped training experiment and should not be the default.
- **D3, why the measured Cholesky-QR failed.** Neither explanation is confirmed. M4 gives 0.0043 under emulated TF32 (not 0.034) and no failure at κ = 7.2e3. My current hypothesis is the R5 mechanism: the benchmark's BF16-stored sample covariance is near-singular at n ≥ 6144. Measuring λ_min of sym(C) on the benchmark inputs, and rerunning with an FP64 Gram, would settle it.
- **D4, size of the engineering win.** Pure engineering on the ordinary and warm paths is still about 1–2 ms/call; R4 shows the warm step at the tensor-core limit from n ≈ 2048. Overlap can hide only the latency-bound QR, and it rests on unverified concurrency. The larger lever is algorithmic (PA1).
- **D5, loss predictions.** I agree with Reviewer Y. M3 contradicted both single-refresh predictions, the best trained single knob is 0.016 of the gap, and the BF16 share of the +0.010 is unknown. Milligan et al. report +0.013/+0.014 for SOAP in a different setup. No CE prediction is defensible yet.
- **D6, leaving V unchanged.** Acceptable for matching QR10, and M5's consistent case shows no spike. The error is conservative: V stays too large in a newly cleaned weak direction, and at β₂ = 0.95 it takes ln(0.0395/1e-6)/0.0513 ≈ 206 calls to decay. That may be a real loss mechanism, but QR10 shares it.

### Strongest disagreement: Reviewer X, on T∘T mixing

**The claim.** Rotating M while only permuting V is a top-three pitfall, producing a "199× step spike", and T∘T mixing plus pre-update refresh fixes it with "medium-high confidence" for 0.002–0.005 nats.

**Why I disagree.**
- The spike needs M and V to be inconsistent with C. With β₁ = β₂ = β_C they are naturally consistent, and M5's consistent row shows no spike under any rule.
- The example sets |M₁| = √V₁, a fully coherent gradient. With stochastic gradients, |M|/√V is typically far below 1.
- In the case that actually occurs every refresh (a rotation that corrects misalignment), T∘T inflates the weak direction's V about 200× above truth. That under-steps those directions for roughly 100 calls, and they are the directions the method already tracks worst.
- It also leaves the QR10 target, which permutes only.

**What would settle it.** In the existing run, log the 99.9th percentile of |M_f/(√V_f + ε)| on the first call after each warm and QR refresh, and compare it with ordinary calls. If the post-refresh value is within 2× of the ordinary value, the pitfall is absent. If it is not, run a 2,000-call A/B of T∘T against permute-only.

---

## 6. Errors and gaps in the moderator checks

- **M2 and M3 use the wrong regime for ranking.** Inputs have r_C ≈ 0.2, are full rank, and have κ ≈ n. Trained inputs are at 0.3–0.8 and near-singular. M3 itself shows the ordering flips at input 0.45. The one-shot table should not be used to rank generators.
- **M2 does not test Reviewer Z's conditional claim.** It measures the Frobenius-to-spectral ratio only for the production generator. Z's claim was that the Frobenius norm grows once the tail is unfrozen. The R5 result that damping 0.001 worsens r_C by 0.012–0.016 is weak indirect support for Z. The right check is the same ratio for the PB1 generator with ρ = 1e-4, on trained B.
- **M2 and M3 use an exact polar retraction.** They cannot show the NS basin failure that makes the no-cap production step diverge. "Factor 1, no cap: 0.055" is therefore not reachable by the production retraction.
- **M3 mixes cap treatments.** Production is capped, PB1 is uncapped, and PB3a has its row cap. A fair comparison uses one bound for all.
- **The no-cap row in R5 is survivorship.** See section 3.
- **M4 does not reproduce the observation it adjudicates.** It reproduces neither the 0.034 Gram error at n = 4096 nor the NaN at n ≥ 6144, so its inputs are not the microbenchmark's. D3 stays open. Whether rounding operands to a 10-bit mantissa matches real TF32 behaviour is also unverified.
- **M4's "column distance to Householder" is ill-conditioned.** In near-degenerate or null clusters, individual columns are arbitrary. The 2.6e-2 at ridge 1e-3 may be mostly harmless tail relabelling. Subspace overlap or r_C^w is the right measure.
- **M5 is 2×2 and one-sided.** The exactness of rescaling there does not establish it for a two-sided V, where V_ij = d_i·d_j is only an approximation.
- **M1's Jacobi row tests a claim nobody made.** "Jacobi as an approximation of QR, relative error 4.2–140" is correct, but Jacobi targets the eigenbasis. It is not evidence that Jacobi is worse for loss.
- **M7 is prior art.** It is Milligan et al.'s reparametrization. The product count is right, but the check omits the possible BF16 benefit of P-frame storage, which is the reason that paper proposes it.