# P1: Fable

Subagent (`general-purpose`, model fable), 2026-11-20, about 13.5 min. Prompt: `prompts/P1_fable.md` (wrapper plus the verbatim brief), copied alone into an empty scratch directory and read with one Read call. Tool calls: Read ×1 (the prompt file only), SubagentHandback ×1 (tallied from the transcript by script). **Not fully blind to the user's preferences:** the instance CLAUDE.md and the one-line memory index were auto-injected. Fable reports that three index lines (best-tuned vs best-tuned comparison, tune at target horizon, never drop an arm for being slow) shaped its pitfalls. It saw no project file about this method. Verbatim final hand-back (4,121 words).

---

## Provenance

- **Tools used:** one `Read` of the brief file, plus the mandatory hand-back call that delivers this text. No search, web, commands, other file reads, sub-agents or reviewer/advisor tool. Nothing below was run; every number is hand arithmetic on the brief's figures.
- **Other context that bears on the topic** (auto-injected, not opened further):
  - Project instructions describing the machine (unprivileged container, PyTorch image, Blackwell needs CUDA ≥ 12.8 wheels). Consistent with the brief; no technical influence.
  - One-line memory index entries. Three shaped my pitfalls: "best-tuned vs best-tuned" comparison, "tune at target horizon", and "never drop an arm for being slow; report cost". I saw only the index lines, not the notes.
- **Labels:** [verify] marks literature/API claims, with [certain] or [recalled]. "Derived" means my own algebra, shown so you can check it. "Speculative" is labeled.

---

## 0. Where the time goes (from §2)

**Per-call optimizer mean, 26.08 ms.**

| component | arithmetic | ms/call | share |
|---|---|---|---|
| ordinary base, every call | 10.14 | 10.14 | 38.9% |
| warm increment | 0.08 × 51.47 | 4.12 | 15.8% |
| QR increment | 0.02 × 591.04 | 11.82 | 45.3% |

- **Amdahl.** The optimizer is 26.08/309.5 = 8.4% of the step, and the refresh increments are 15.94 ms = 5.1%. Forward/backward is about 283 ms, eager. No refresh speedup can buy more than about 5% of step time; the larger prize is the +0.010 loss gap.
- **Inside a warm event (49.84 ms).** 12×2.46 = 29.5 ms (n=3072), 12×1.02 = 12.2 (2304), 12×0.331 = 4.0 (1536), 60×0.069 = 4.1 (five n=768 groups). The two tall left factors are 84%.
- **Inside a QR event (about 598 ms).** 219.4 (3072) + 143.5 (2304) + 81.4 (1536) + 5×30.8 = 154 (768).
- **Flop rates** (one n×n product = 2n³ flops):
  - Warm is 7 products: QᵀC, ·Q, Q·A, and 2×(YᵀY, Y·E). At n=3072 that is 14·2.90e10 = 4.06e11 flops in 2.46 ms, about 165 TFLOP/s. It is compute-bound for n ≥ 2048 (same rate at 8192).
  - Householder QR is (4/3 + 4/3)n³ for geqrf + orgqr, plus 2n³ for Z. At n=3072 that is 1.35e11 flops in 18.1 ms, about 7 TFLOP/s; at n=768 about 0.8 TFLOP/s.
  - So QR does about 3× fewer flops than warm at about 25–200× lower throughput, and its per-factor time does not improve with batching. QR is latency-bound at every size you run. That is the main speed lever.
- **Ordinary step.** Six mixed products per matrix (2 project, 2 back-project, 2 covariance). I count about 1.3e12 flops over the 48 matrices, about 7.9 ms at 165 TFLOP/s against 10.14 measured. It is already about 80% matmul-bound, so there is little engineering slack.
- **Attribution.** A compiled BF16 QR10 arm would cost about 10.14 + 0.1×591 = 69.2 ms/call. The same formula predicts 39.7 for QR20 (measured 40.46) and 22.0 for QR50 (measured 22.86). Of the 95.2 → 26.1 ms headline, about 26 ms is compile+BF16 engineering that the baseline could also have, and about 43 ms is the schedule change.

---

## 1. Four identities the candidates rest on (all derived)

**I1. QR repair is one unshifted QR-algorithm step on B = QᵀCQ.**
If QQᵀ = I then CQ = QB, so qf((CQ)P_π) = Q·qf(B P_π). (Textbook equivalence of subspace iteration and the QR algorithm [verify][certain].)

**I2. First-order form of that step.**
- Write B = D + E, sorted so d_j ≥ d_i for j < i, and qf(B) = I + K with K skew.
- (I+K)(D+R′) = D+E gives, on the strict lower triangle, K_ij d_j = E_ij, so K_ij = B_ij/d_j for i > j.
- The exact first-order diagonalizer in your convention (B⁺ = B + BA − AB) is A*_ij = B_ij/(d_j − d_i).
- So QR moves each pair a fraction 1 − d_min/d_max of the Jacobi angle:

  **A^QR_ij = B_ij · sgn(d_j − d_i) / max(d_i, d_j)**

- Properties: |A^QR_ij| ≤ √(d_min/d_max) ≤ 1 for PSD B; no gap singularity; scale-free; ties are broken by column order, not by a gap.

**I3. The warm generator, relative to QR.**
Your A_ij = A*_ij · ½ · Δ²/(Δ² + (0.01s)²), with Δ = d_i − d_j, times the cap a. The ratio of rotation angles is

  warm/QR = (a/2) · d_max·|Δ| / (Δ² + (0.01·s)²).

- Well-separated pair (Δ ≈ d_max ≫ 0.01s): ratio = a/2 ≤ 0.5.
- Tail pair (d_max = 10⁻³s, Δ ≈ d_max): ratio = (a/2)·0.0099, a 100–200× under-rotation.
- Near-degenerate head pair (d_max = s, Δ = 0.1s): ratio ≈ 5a.
- Consequence: the damping is absolute in units of λ_max. With λ_i ∝ 1/i, every pair with both i, j ≳ 100 is effectively frozen. On that block, warm10/QR50 equals QR50.

**I4. The Frobenius cap is about √n too tight.**
- For skew A, ‖A‖₂ ≤ ‖A‖_F/√2, tight only at rank 2.
- For unstructured A with entry std e: ‖A‖_F/√2 ≈ e·n/√2 and ‖A‖₂ ≈ 2e√n. The slack is √n/(2√2): about 10× at n=768, 20× at 3072, 32× at 8192. Real A is structured, so the true slack needs logging.
- Consistency check on §2.2: for the synthetic inputs I estimate ‖A‖_F ≈ 1.4 (head-versus-all pairs only), so a ≈ 0.5. With the ½ factor that is about ¼ of the Jacobi step, predicting r_C 0.20 → 0.15. You measured 0.150. So the weak single-step quality is explained by ½ × cap, not by first-order error.

**NS and retraction scalars** (σ = singular value of Y, E = σ² − 1):
- One Newton–Schulz step maps E ↦ −¾E² + ¼E³ exactly.
- Y0 = Q(I + aA) has E0 = θ² for a rotation plane with angle θ = a·σ(A). At the cap θ = 0.5: 0.25 → −0.043 → −1.4e-3. That is why two NS steps are needed today.
- Y0 = Q(I + aA + ½a²A²) has E0 = θ⁴/4, since (I − A + A²/2)(I + A + A²/2) = I + A⁴/4. At θ = 0.5: 0.0156 → 1.8e-4 in one step. At θ = 0.7: 0.060 → 2.7e-3 in one step.

**BF16 floor.** If entries carry relative rounding error of RMS σ_u, then ‖QᵀQ − I‖_F/√n ≈ √2·σ_u. Your 0.0024 implies σ_u ≈ 1.7e-3, matching an 8-bit significand (half-ulp between 2⁻⁹ and 2⁻⁸ relative). No amount of NS gets below this.

---

## 2. Question A: speed

### A1. Stabilized Cholesky-QR in place of Householder QR (same as B1)

1. **Update.**
   - Z = 𝓕(C,Q) + τ·Q, with τ = ρ·max_j d_j and d = diag(QᵀCQ) from the same contraction as now.
   - Sort columns by d; scale columns to unit norm (this does not change qf).
   - G = ZᵀZ in true FP32; L = cholesky_ex(G + s·I); Y = Z·L⁻ᵀ by triangular solve.
   - Round to BF16, then 1–2 of your existing NS corrections.
   - Fallback when `info ≠ 0`: `torch.where` to the warm result with the identity order (stays in-graph).
2. **Cost.** Z: 1 product (BF16 operands, FP32 accumulate). Gram: 1 FP32 product. Cholesky: n³/3 flops (1/6 product). Triangular solve: n³ flops (1/2 product). That is about 2.67 product-equivalents, plus 2 BF16 products per NS step. Your table already prices Z + one pass: 0.171 / 0.618 / 1.50 / 3.03 ms per factor, i.e. 60×0.171 + 12×(0.618+1.50+3.03) = 72 ms per event against 598.
3. **Effect.** A QR event drops from about 601 ms to about 95–110 ms with one NS step. At the QR50 cadence that saves about 10 ms/call, 38% of the optimizer mean and 3.2% of the step. Confidence: high on timing (your own table); medium on robustness on trained C.
4. **Failure modes.**
   - Unshifted, it breaks when κ(C)²·u ≳ 1, i.e. κ ≳ 4100 in FP32. With λ ∝ 1/i, κ ≈ n, which predicts failure between 4096 and 6144, exactly where you saw NaN.
   - The ridge τ bounds κ ≤ 1/ρ. ρ = 1e-3 gives κ²u ≈ 0.06, so a single pass plus NS is enough. ρ = 1e-4 needs a second Cholesky pass (roughly +60% cost).
   - The ridge slows rotation among directions with d < τ: the gain becomes 1 − (d_i+τ)/(d_j+τ).
   - If the Gram product runs in TF32 (your "high" setting permits it [verify][recalled]), it will fail far earlier. Force true FP32 for that one product.
5. **BF16 and β_C.** BF16-stored C already has a rounding-noise floor that I estimate at 1e-4·λ₁ or higher, so a ridge near that level costs no fidelity. BF16 inputs are exactly representable in TF32, so Z can use any kernel.
6. **Test.** Replay: Cholesky-QR10 with ρ ∈ {0, 1e-4, 1e-3, 1e-2} against QR10. Reject if top-k overlap or normalized r_C differs from QR10 by more than the QR10-versus-eigh10 difference, or if any factor falls back more than 1% of the time. Microbenchmark at all n, including 6144 and 8192.
   - Literature: CholeskyQR2 (Yamamoto et al. 2015) and shifted CholeskyQR3 (Fukaya et al. 2020) [verify][recalled]; near-optimality of column equilibration (van der Sluis 1969) [verify][recalled].

### A2. R-only Householder plus triangular solve (robust fallback for A1)

1. **Update.** R = `torch.linalg.qr(Z_π, mode='r').R`; Y = Z_π·R⁻¹ by `solve_triangular`; one NS step. Same sign convention as the full QR.
2. **Cost.** geqrf only (4n³/3) plus the solve (n³), instead of geqrf + orgqr (8n³/3). All FP32.
3. **Effect.** If orgqr is about half of the QR time, about −40% per QR event (about −235 ms, −4.7 ms/call). Medium confidence; the geqrf/orgqr split is unmeasured. `mode='r'` skipping Q formation is [verify][recalled].
4. **Failure.** Orthogonality loss scales as κ(Z)·u rather than κ²·u. It is fine to κ ≈ 1e5 with one NS step; use a ridge ρ ≈ 1e-5 beyond that.
5. **BF16.** Same as A1.
6. **Test.** Microbenchmark time. Replay Gram error and overlap against QR10; reject if the Gram error after one NS step exceeds 0.004.

### A3. Second-order retraction with one NS step (6 products instead of 7)

1. **Update.** K = aA + ½a²·𝓕(A,A); Y0 = R_b(Q + 𝓕(Q, R_b(K))); one NS step.
2. **Cost.** 2 (B) + 1 (A²) + 1 (Q·K) + 2 (NS) = 6 BF16-operand products.
3. **Effect.** −14% on compute-bound sizes: about −6.5 ms per warm event, −0.5 ms/call. At n=8192, 45.5 → about 39 ms per factor. Alternatively keep 7 products and raise the cap to θ ≈ 1 (0.25 → 0.043 → 1.4e-3, the same worst case as today). High confidence in the algebra.
4. **Failure.** A² rounded to BF16 is harmless (relative error). If the cap estimate is low, NS still converges for θ⁴/4 < 2.
5. **BF16.** The single NS step also repairs the stored basis's own 0.002 drift.
6. **Test.** Microbenchmark time. Replay Gram error ≤ 0.003, and r_C no worse than current at an equal cap.

### A4. Run the per-factor decompositions concurrently (speculative)

1. **Update.** Unchanged arithmetic. Dispatch the 96 FP32 decompositions over 4 host threads, since the batched call loops and each is latency-bound. Variant: overlap the QR of a C snapshot with the next forward/backward and swap bases 1–2 calls late.
2. **Cost.** Same flops.
3. **Effect.** Guess 1.5–3× on the n ≤ 1536 groups (235 of 598 ms). Low confidence.
4. **Failure.** Kernels serialize on one stream; the decomposition leaves the compiled region; the overlapped variant changes the math by a 1–2 call basis lag.
5. **BF16.** None.
6. **Test.** Microbenchmark only: thread pool over 12 factors at each n. Reject below 1.3×.

### A5. Remove pack/unpack traffic and launch latency in the ordinary step

1. **Update.** Hold each group's 12 weights as views of one stacked leaf, so gradients arrive stacked and `torch.stack` / `copy_` disappear. Batch refreshes by factor size (one batch-60 call for all n=768 factors). Try `mode="reduce-overhead"` for the small-n regions [verify][recalled].
2. **Cost.** Removes about 1.5 GB/call of copies and 4 of 5 latency-bound warm calls.
3. **Effect.** My guess is 1–2 ms on every call, 10–20% of the ordinary base. Low-medium confidence; it rests on the 10.14 − 7.9 ms non-matmul remainder.
4. **Failure.** A model-code change; CUDA graphs may reject in-place parameter writes.
5. **BF16.** None.
6. **Test.** Microbenchmark of the ordinary region with pre-stacked inputs. Reject below 0.5 ms.

### A6. Cheaper covariance accumulation (equation-level, risky)

1. **Update.** Accumulate every second call with C ← β²C + (1−β²)GGᵀ, or sketch with an n×r Gaussian Ω: C ← βC + (1−β)(GΩ)(GΩ)ᵀ/r.
2. **Cost.** Saves about 1/3 of ordinary-step flops on the affected calls. The sketch replaces m²n by mnr + m²r.
3. **Effect.** About 1–1.5 ms/call. Halves the effective sample count in C.
4. **Failure.** Noisier bulk eigenvectors.
5. **BF16.** With β² = 0.9025 the BF16 EMA is safer, not worse.
6. **Test.** Not testable in replay (needs per-step gradients). Smallest test is a 2,000-call run.

**Larger factors (n = 4096–8192).** Warm is flop-bound (14n³ at 165 TFLOP/s) and already costs 0.2–0.4 of a QR. The useful levers are A3 (6 products), A1 (5.3n³ flops; 42 ms measured against 45.5 for warm at 8192, once the NaN is fixed) and B5 (about 4.2 products).

---

## 3. Question B: tracking QR10

### B1. Use the real power step with Cholesky orthogonalization at every refresh

1. **Update.** A1's refresh every 10 ages, with the sort and V permutation exactly as in your QR path. No warm step.
2. **Cost.** As A1: about 86–100 ms per event with 1–2 NS steps, against 49.8 for warm. That is 1.7–2.0× warm for the 109M set and equal to warm for n ≥ 4096. At a 10-age cadence it is about 9–10.5 ms/call including transport, against 15.9 ms/call for today's warm10/QR50 mix.
3. **Effect.** In exact arithmetic this is the QR10 trajectory, since Y = Z·R⁻¹ is the same Q as Householder. It should remove the schedule part of the gap while being cheaper than the current schedule. Confidence: medium-high, conditional on the ridge and shift working on trained C.
4. **Failure.** As A1. A ridge that is too large freezes the tail and reintroduces the gap.
5. **BF16.** After BF16 storage the basis sits at the same 0.0024 Gram floor as QR.
6. **Test.** Replay as in A1, on all metrics. Then one 2,000-call run against BF16 QR10; reject if CE at 2,000 is more than about 0.002 (seed noise) above it.

### B2. Spectral-norm cap instead of the Frobenius cap

1. **Update.** Estimate σ = ‖A‖₂ with 8 power iterations on −A² (FP32 matvecs, fixed start vector). Set a = min(1, θ_cap/(1.15σ)).
2. **Cost.** 16 matvecs = 32n² flops; negligible. No new products.
3. **Effect.** If the cap binds (I4 says a ≈ 0.5 synthetically, smaller in training where r_C ≈ 0.39), the step roughly doubles or more. Predicted single-step r_C 0.20 → about 0.10–0.11. Confidence: medium; logging a settles it.
4. **Failure.** Power iteration underestimates σ. NS still converges for θ < √2, so the cost is a larger Gram error for one refresh.
5. **BF16.** None.
6. **Test.** Replay r_C and normalized r_C after each refresh. Reject if a was already 1 in more than 90% of refreshes.

### B3. QR-matched, scale-free generator

1. **Update.** A_ij = −B_ij·sgn(d_i − d_j)/max(d_i, d_j, ρ·s). This is I2 in your sign convention. At exact ties use sgn(j − i). Combine with B2 and A3.
2. **Cost.** Elementwise only; same 7 (or 6) products.
3. **Effect.** Matches QR's per-pair gain to first order. It removes the factor ½, unfreezes the tail block, and removes the equal-diagonal stall: your 2×2 example gives |A_12| = 0.5 instead of 0. I expect the diagonally normalized r_C to move most of the way from warm to QR10. Confidence: medium.
4. **Failure.** First order only. A dominant direction that no column captures needs a large rotation and gets capped (B5 covers that). The floor ρ must sit at or above C's noise level, or noise pairs rotate.
5. **BF16.** Choose ρ at or above 1e-4 for BF16-stored C.
6. **Test.** Replay distance of the chained basis to the QR10 chain, ‖Q_QR10ᵀ Q_test‖ in bands of eigen-index, plus normalized r_C. Reject if the tail bands are no closer to QR10 than the current warm.

### B4. Head-block power step with deflation

1. **Update.** Take the k = 64–128 columns with the largest d. Z_k = (C + τI)·Q[:, idx]; H = thin CholeskyQR2(Z_k); T = Q_rest − H(HᵀQ_rest); write H and T back to their slots; two NS steps on the whole Y.
2. **Cost.** About 4 + 3k/n full products alone, or about 7.2 when appended to the generator step.
3. **Effect.** Captures a dominant direction in one shot, as the power step does. It targets the factors at r_C ≈ 0.97. Confidence: medium on the mechanism; unknown size.
4. **Failure.** T's singular values are the cosines of the head-subspace rotation. A near-90° rotation leaves a small singular value that NS grows only 1.5× per step.
5. **BF16.** The thin step runs in FP32 on n×k, which is cheap.
6. **Test.** Replay top-k subspace overlap after the refresh. Reject if it is not within 0.02 of QR10 on the worst decile of factors.

### B5. Front-loaded schedule, optionally with a trigger

1. **Update.** QR, or B1, every 10 ages for ages ≤ 200–500, then the normal schedule. Optionally promote a factor to QR when a < 0.5.
2. **Cost.** 16 extra QR events for ages ≤ 200 is about 16×540 ms = 8.6 s once, about 0.3% of a 10k run.
3. **Effect.** It tests whether the call-500 gap is a cold-start effect. The basis starts from a single-gradient eigendecomposition and your first QR is at age 50, while the reference has had four. Confidence: medium that it removes most of the +0.12; low that it changes the +0.010 plateau.
4. **Failure.** None numerically. A data-dependent trigger breaks the static graphs.
5. **BF16.** None.
6. **Test.** Replay metrics over ages 0–200. The loss effect needs a 500–2,000-call run.

### B6. Deadband-free basis storage (minor)

1. **Update.** Stochastic rounding for Y, or an FP32 master basis with a BF16 shadow for the hot path.
2. **Cost.** O(n²) elementwise, or +0.96 GB.
3. **Effect.** With round-to-nearest, an entry moves only if its increment exceeds half an ulp. Column rotations below about 2⁻⁹–2⁻⁸ rad are dropped and never accumulate. I estimate only a few percent of rotation energy is lost at your caps, so this is small.
4. **Failure.** Stochastic rounding raises the Gram floor slightly.
5. **BF16.** This is the BF16 issue itself.
6. **Test.** Replay with your `basis_dtype=float32` switch. Reject if the metrics are indistinguishable.

### B7. Refit V's marginals after a refresh (speculative, goes beyond QR10)

1. **Update.** With β₂ = β_C and orthogonal bases, Σ_j V_ij = diag(Q_LᵀC_LQ_L)_i and Σ_i V_ij = diag(Q_RᵀC_RQ_R)_j, because Σ_j Ḡ_ij² = (Q_LᵀGGᵀQ_L)_ii. After a refresh, rescale V's rows and columns to the new diagonals, using d⁺_i ≈ d_i + 2a·Σ_k B_ik·A_ki.
2. **Cost.** O(mn + n²) elementwise.
3. **Effect.** Unknown.
4. **Failure.** Multiplicative rescaling amplifies noise in small rows.
5. **BF16.** Fine at β = 0.95.
6. **Test.** Not testable in replay; needs a training run. As a free diagnostic, log the row sums of V against d as a V-staleness measure.

### The three questions you asked me to address

- **Is r_C the right proxy?** No, by I3. r_C weights by λ², so it sees only the head. The update is Adam-normalized, so every rotated coordinate counts about equally.
  - Evidence: on the toy, warm closes (0.125 − 0.093)/(0.125 − 0.075) = 64% of the r_C gap. On loss it closes 12–37%.
  - Use instead: (a) the correlation form R = D^(−½)·B·D^(−½), reported as ‖R − I‖_F/√n and −log det R / n; (b) distance to the QR10 chain by eigen-index band; (c) in replay, the cosine between idealized updates U(Q) = Q_L[(Q_LᵀGQ_R) ⊘ √(d_L d_Rᵀ)]Q_Rᵀ under the test and QR10 bases, with synthetic G = C_L^(½)·X·C_R^(½).
  - Validate by checking that the metric orders QR10 < QR20 < warm < QR50 as the loss does.
- **The call-500 gap.** It appears with FP32 state too, so it is schedule-driven. At call 500 the loss is still falling steeply, so +0.12 is probably a delay of tens of calls. Report "calls behind" rather than ΔCE. The persistent +0.007 to +0.012 from call 2,000 is the steady-state tracking deficit and is the number to optimize. B5 separates the two.
- **V and column order.** Leaving V unchanged at a warm step is correct to first order: the diagonal second moment changes by Σ_k A_ki²(V_k − V_i), which is second order, while M changes at first order and is transported. Not reordering is also correct for a continuous rotation, since each slot keeps its V. Order only matters for triangular steps (QR, B1, B3's tie-break). The second-order V error is largest for head-to-tail rotations, and is inherited from SOAP, which only permutes V [verify][recalled].

---

## 4. Pitfalls

1. **The warm step is head-only and quarter-gain** (I3 and I4): a global, √n-loose cap times ½, and damping tied to λ_max. One badly placed pair also throttles the whole factor.
2. **Attribution is confounded.** The reference is eager FP32. LR 1.5e-3 was tuned on it only, so the gap is biased against the variant. The matched BF16 QR10 run is pending, so the BF16/schedule split of +0.010 is unknown. The short controls are one seed with differences (0.005) near seed spread (about 0.002–0.003). CE at call 2,000 of a 10k cosine schedule is not the target horizon.
3. **BF16 state has deadbands that the replay harness cannot see.**
   - The EMA update is rounded away when (1−β)|X − C| is under half an ulp, i.e. |X − C|/|C| below 3.9–7.8% at β = 0.95.
   - Pure decay stops entirely for β ≳ 0.996, so β₂ or β_C cannot be raised under BF16 without stochastic rounding.
   - Replay C comes from an FP32 run with one rounding. It contains no accumulated-BF16 effects, no closed loop, and no M or V.
4. **The small-rotation premise is weak at β_C = 0.95.** 1 − 0.95¹⁰ = 40% of C's mass is new at each refresh and 87% is new between QRs. B⁺ ≈ B + BA − AB is not a safe assumption in early training.
5. **Stall points are wider than the 2×2 example.** A rank-1-dominated C spread evenly over columns is also a stall or slow point. The 0.965–0.975 maxima suggest this happens in training.
6. **"FP32 island" may mean TF32.** Harmless for C·Q on BF16-exact inputs; not harmless for any Gram, r_C or Cholesky computation on true FP32 data.
7. **Timing method.** Per-call synchronization changes overlap. The QR time is host-latency-sensitive and you have 4 CPU threads. Timing compares across launches. Microbenchmark medians hide tails.
8. **Minor.** QR uses an unsymmetrized C. ε = 1e-8 should be checked against √V in tail coordinates late in training. M transport error from the 0.2% non-orthogonality is negligible.

---

## 5. Rankings

**Top 5 for A (speed)**
1. A1, stabilized Cholesky-QR: removes about 85% of the 45% QR share, using timings you already measured.
2. A2, R-only Householder plus solve: robust, about −40% of QR time, with almost no numerical risk.
3. A3, 6-product retraction: a free 14% on the compute-bound warm sizes and the main lever at n ≥ 4096.
4. A5, pre-stacked storage and size-batched refreshes: the only item that touches the 39% ordinary base.
5. A4, concurrent decompositions: possibly large for small n, but speculative.

**Top 5 for B (tracking)**
1. B1, Cholesky-QR at every refresh: it is the target arithmetic, inside the cost budget and cheaper than the current mix.
2. B3, QR-matched generator: fixes the gain profile and the stall with no extra products.
3. B2, spectral cap: a one-line change that likely doubles the step.
4. B5, front-loaded schedule: about 0.3% cost, and it isolates the call-500 gap.
5. B4, head-block power step: aimed at the r_C ≈ 0.97 factors.

**Top 3 pitfalls:** 1 (head-only, quarter-gain warm step, invisible to r_C), 2 (confounded attribution and tuning), 3 (BF16 deadbands outside the harness's view).

---

## 6. What I would want measured first

1. The matched BF16 QR10 run, with per-factor r_C distributions (not only means) for both arms, and which factors sit near 0.97.
2. Per refresh and per factor: the cap scalar a, ‖A‖₂ against ‖A‖_F/√2, and the realized rotation by eigen-index band for warm and for QR.
3. The scale-free metrics above in replay, and whether they rank the four schedules as the loss does.
4. The geqrf/orgqr time split; Cholesky failure rate and κ(C) on trained snapshots against the ridge; TF32 on and off.
5. A front-loaded run to 2,000 calls, and a small LR sweep for the variant at the full horizon.