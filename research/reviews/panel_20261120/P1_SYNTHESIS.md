# Stage P1 synthesis (2026-11-20)

**Sources.** All four received the identical brief (P1_BRIEF.md). None saw another reviewer's text, the moderator
priors or any refresh-study Part 2 result. No web.

| Answer | Words | Run time | Tool use |
|---|---|---|---|
| [P1_gpt-6.2-astra.md](P1_gpt-6.2-astra.md) | 5,337 | 7 min | none |
| [P1_fable.md](P1_fable.md) | 4,121 | 13.5 min | one Read of its prompt file; saw the instance CLAUDE.md and the one-line memory index |
| [P1_gemini-4-neon-xhigh.md](P1_gemini-4-neon-xhigh.md) | 9,902 | 20 min | none |
| [P1_gemini-4-carbon-high.md](P1_gemini-4-carbon-high.md) | 7,731 | 23 min | none |

Below, "astra", "Fable", "Neon" and "Carbon" name the reviewers, and their own candidate IDs are given in
parentheses (e.g. Neon A5). Pooled candidates are **PA·** (speed) and **PB·** (tracking).

**Status of the evidence.**
- All costs and gains are the reviewers' estimates unless marked **[checked]**. Checked items were verified by the
  moderator in `checks/` (§4).
- Nothing here has run on the refresh-study harness or on trained covariances. The checks use synthetic inputs on
  CPU, in float64 or float32.
- Every reviewer marked its literature claims [verify]; none is settled here (P2 has the web).

## 1. Where all four agree

1. **Where the time goes.** All four reproduce the same 26.08 ms/call split:
   - ordinary step 10.14 ms (38.9%);
   - warm increment 0.08 × 51.47 = 4.12 ms (15.8%);
   - QR increment 0.02 × 591.04 = 11.82 ms (45.3%).

   The whole optimizer is about 8.4% of the 309.5 ms step, and the refresh increments about 5.1% (Fable). Two
   further points are shared:
   - **The QR repair is the largest single cost.** It is latency-bound at every measured size (about 0.8 TFLOP/s at
     n = 768 and 7.5 at 3072), because the batched call loops over the batch.
   - **The ordinary step is 70–80% matmul-bound,** so engineering headroom there is roughly 1–2.5 ms/call.

   A matched compiled BF16 QR10 would cost about 69.2 ms/call (astra, Fable, Carbon). So of the 95 → 26 ms headline,
   about 26 ms is compile + BF16 and about 43 ms is the schedule change.
2. **QR's first-order rotation is not the warm step's.** Three reviewers derived it independently (astra B1, Fable I2,
   Carbon §B.1). Take B = QᵀCQ = D + E, with columns ordered by descending d. One QR refresh is Q·qr(B).Q, which to
   first order is the skew generator

   K_ij = B_ij / d_j for i > j (d_j ≥ d_i),

   which leaves B⁺_ij ≈ (d_i/d_j)·B_ij **[checked]**. The warm step instead uses ½·B_ij/(d_i − d_j), damped by an
   absolute gap of 0.01·d_max and scaled by a global Frobenius cap. Relative to QR it therefore:
   - under-rotates well-separated pairs (by the factor ½ and the cap);
   - nearly freezes every pair whose diagonals are both below about 1% of d_max (most of a power-law spectrum);
   - over-rotates nearly degenerate head pairs;
   - does nothing at exactly equal diagonals.

   Fable gives the ratio in closed form: warm/QR = (a/2)·d_max·|Δ|/(Δ² + (0.01·s)²) **[checked]**.
3. **The global Frobenius cap is the wrong norm.** Newton–Schulz convergence depends on ‖A‖₂, and for a skew matrix
   ‖A‖₂ ≤ ‖A‖_F/√2 **[checked]**. A rotation spread over many planes is throttled even when every single plane is
   small. Fable, Neon and Carbon each explain the measured single-refresh r_C of 0.20 → 0.150 as "½ × cap ≈ ¼ step".
   - The moderator check agrees that the cap binds on these inputs: a = 0.46–0.79 while ‖A‖₂ ≤ 0.30.
   - Removing it moves single-refresh r_C from 0.172 to 0.116 at n = 256 (§4).
4. **Cholesky-QR, stabilized, should replace Householder QR.** All four propose it (Fable A1/B1, Neon A5/B6, Carbon
   A4/B3, astra B4, astra only conditionally). The stabilization they describe:
   - shift C by a small ridge;
   - equilibrate the columns;
   - two passes (CholeskyQR2), or one pass plus a Newton–Schulz polish;
   - run the Gram product in true IEEE FP32.

   All quote the same anchor: the measured one-pass Cholesky-QR costs 72 ms over the 96 factors against 594 ms for
   QR. They project a QR event of about 95–140 ms and a saving of about 9–10 ms/call, 35–38% of the optimizer.
5. **TF32 inside "FP32 islands".** The brief's matmul precision "high" lets FP32 `bmm` run in TF32. On this build,
   `set_float32_matmul_precision('high')` sets `allow_tf32 = True` **[checked]**.
   - It is harmless for Z = C·Q, whose inputs are BF16-exact.
   - It is not harmless for a true-FP32 Gram such as ZᵀZ in Cholesky-QR, or the optional FP32-basis Gram in
     `warm_basis`.
6. **r_C is the wrong selection metric.** Under a power law it is dominated by the top ~10 eigenvalues (Neon: 94%),
   but Adam weights all rotated coordinates about equally. All four want a whitened measure:
   - the RMS off-diagonal correlation ρ_rms (already in the harness), or −½·log det of the correlation matrix (Neon,
     Fable);
   - distance to the QR10 chain by eigen-index band (astra, Fable, Carbon);
   - the whole r_C distribution, not its mean.

   Carbon derives that the whitened-update error equals ρ_rms up to a factor in [1/√2, 1] (not checked). The
   validation all four suggest: does the metric order QR10 < QR20 < warm < QR50 as the loss does?
7. **The call-500 gap is mostly an early-schedule effect.** Every reviewer proposes front-loading accurate refreshes
   (astra A6, Fable B5, Neon B4, Carbon B5). Its reasons:
   - the cold-start basis comes from one gradient (for 3072×768, rank ≤ 768);
   - the first QR comes only at age 50;
   - the FP32-state warm pilot shows the same gap, so storage cannot explain it.

   The persistent +0.007 to +0.012 from call 2000 on is a separate steady-state problem (all four).
8. **The matched BF16 QR10 run is the most-wanted measurement** (named first by astra, Neon and Carbon, second by
   Fable). Without it, the +0.011 endpoint gap cannot be split into BF16-state and refresh-approximation parts. Fable
   adds a second confound: the LR was tuned on FP32 QR10 only.

## 2. Pooled candidates (deduplicated)

50 reviewer-level candidates (astra 11, Fable 13, Neon 15, Carbon 11) reduce to 14 speed and 12 tracking candidates.

"Tool" means the cheapest test:
- **MB:** the microbenchmark;
- **RP:** the open-loop replay;
- **TR:** a short training run (the replay has no M, V, gradients or loss).

### A: speed

| ID | Candidate | Who | Claimed cost / benefit | Main risks | Tool |
|---|---|---|---|---|---|
| **PA1** | **Stabilized Cholesky-QR for QR events.** Z = C·Q (BF16 `mm32`); optional ridge Z + τ·d_max·Q; column equilibration; CholeskyQR2 with IEEE-FP32 Grams; `cholesky_ex` with in-graph fallback | Fable A1, Neon A5, Carbon A4, astra B4 | 2n³ BF16 + about 6.7n³ FP32 per factor. QR event 594 → about 95–140 ms, −9 to −10 ms/call. At n = 8192 about 60–70 ms against 112 | the ridge slows tail rotation; κ²u breakdown without the ridge; needs TF32 off for the Gram; trained κ may exceed the synthetic one. **[checked, partly]** | MB (NaN count, Gram error, ms vs n) + RP (chain vs QR10) |
| PA2 | **Concurrent QR** over CUDA streams or host threads; same arithmetic | Neon A1, Carbon A3, Fable A4 | 2.5–4× at n = 768, 1.5–2.5× at 1536 (Neon); 594 → 130–190 ms (Carbon) | cuSOLVER handle or stream serialization; leaves the compiled graph | MB only |
| PA3 | **R-only Householder plus a triangular solve** (`qr(mode='r')`, Z·R⁻¹, one NS) | Fable A2 | geqrf + trsm instead of geqrf + orgqr; about −40% if orgqr is half | orthogonality error ∝ κ·u; the geqrf/orgqr split is unmeasured | MB |
| PA4 | **Custom batched, blocked Householder QR** (panels, YTYᵀ) | astra A3 | target −50% of the QR increment (−5.9 ms/call) | engineering cost; the explicit-Q phase | MB |
| PA5 | **`mm32` for Z = C·Q in `qr_basis`** (BF16 tensor cores instead of TF32; same products) | Neon A1, Carbon A1 | about 4–6 ms per QR event (≈ 0.1 ms/call) | none material | MB |
| PA6 | **Fewer warm products.** (a) one NS correction (5 products); (b) seed I + A + ½A², then one NS (6 products); (c) degree-4 Taylor R₄(A) with no NS (5 products); (d) R₄(A/2)² (6 products) | (a) astra A4, Carbon A5; (b) Fable A3, Carbon B2; (c) Neon A6; (d) Neon B1 | −14% to −29% of warm FLOPs; −0.5 to −1.2 ms/call at 109M; the main lever at n ≥ 4096 | (a) one correction leaves up to 0.043 spectral error at the cap; (c) does not heal stored-Q drift across 4 warm steps. Identities for (a)–(c) **[checked]** | RP (Gram error over 4 chained warms) + MB |
| PA7 | **Cheaper M transport.** (a) M⁺ = R_LᵀMR_R with the known small rotation (2 products instead of 4); (b) short-side reassociation | (a) Neon A6/B1; (b) astra A5, Carbon A6 | (a) −50% of the transport FLOPs; (b) 7.5–8.3% fewer **[checked]**. Transport is only 5.06 ms per event (≈ 0.5 ms/call) | (a) exact only for orthogonal bases | MB |
| PA8 | **Symmetric (syrk) kernels** for GGᵀ, GᵀG and the NS Grams | astra A2, Neon A4, Carbon A2 | covariance FLOPs −46%; warm 7 → about 6 products; −1 to −1.8 ms/call | a Triton kernel below cuBLAS erases the gain | MB |
| PA9 | **Remove the pack/unpack traffic:** contiguous parameter/grad banks; coalesce the five n = 768 groups into one batch-60 call; CUDA graphs | astra A1, Neon A2, Carbon A1, Fable A5 | 0.8–1.1 ms/call on ordinary steps; 2.5–5 ms per warm event | model-code change; graphs vs in-place writes | MB (event regions) |
| PA10 | **Fused epilogues** (`baddbmm`) and dropping FP32 temporaries in `warm_basis`; sub-batching at large n | Carbon A1, Neon A3 | recovers the measured 22–23% batch-12 vs batch-1 regression at n ≥ 6144 | rounding order changes slightly | MB |
| PA11 | **Strided covariance EMA:** update C every 2nd call with β_C² | Fable A6, Neon A7, Carbon A6 | about 1.2–1.8 ms/call; also shrinks the BF16 EMA dead-band | halves effective samples | TR only |
| PA12 | **Overlap the refresh with the next forward/backward** (separate stream; or Fable's 1–2-call-late basis swap) | Carbon A3, Fable A4 | exposed optimizer 26 → about 12 ms/call (Carbon) | SM contention; the lagged variant changes the method | TR timing |
| PA13 | **Size-aware schedule.** For large n, drop warm and run PA1 at cadence 10–20. Warm10/QR50 loses to QR20 on cost once T_warm/T_QR ≥ 3/8, which n = 8192 has already crossed (588 vs 560 ms per 100 calls) **[checked arithmetic]** | Neon A8/B6a | at n = 8192 PA1-20 costs about 325 ms per 100 calls against 588 for the current schedule | inherits PA1's risks | MB + RP |
| PA14 | **FP16 basis storage** (same bytes, 10-bit mantissa); optional FP16-accumulate back-projection | Neon A9 | Gram floor 0.0024 → about 0.0003; −0.9 to −1.3 ms/call from FP16 accumulation (speculative) | FP16-accumulate noise is 2.7–4.4% of U | RP (storage) / TR (accumulate) |

### B: tracking

| ID | Candidate | Who | Claimed cost / benefit | Main risks | Tool |
|---|---|---|---|---|---|
| **PB1** | **QR-tangent generator.** A_ij = B_ij·sgn(d_j − d_i)/max(d_i, d_j, ρ·s) with ρ ≈ 1e-4 (in Carbon's form, lower(B/d_safe) − transpose, clamped to ±1); usually with a sort plus V permutation | astra B1, Fable B3, Carbon B1 | same 7 (or 6) products. Claimed: single-refresh r_C 0.15 → about 0.04; matches QR to first order; ends the equal-diagonal stall. **[checked]** chained toy: r_C 0.0767 / ρ_rms 0.243, against 0.0759 / 0.240 for QR10 and 0.0977 / 0.294 for production | first order only; a dominant direction spread over many columns still needs a large rotation; the floor ρ must sit above BF16 noise | RP |
| **PB2** | **Spectral (or row-ℓ₂) cap instead of the Frobenius cap:** ‖A‖₂ from 2–8 power iterations | Fable B2, Carbon B1, Neon B1, astra B1 (ablation) | 16–64 n² FLOPs. Claimed: about doubles the applied step. **[checked]** the cap binds (a = 0.46–0.79), and removing it lowers single-refresh r_C from 0.172 to 0.116 (n = 256) | the power estimate can undershoot ‖A‖₂ (NS still converges below √2) | RP |
| PB3 | **Full-gain Jacobi variants:** (a) relative-gap atan Jacobi + row cap + R₄; (b) two inner iterations; (c) local-scale full-gain Jacobi + seed I + A + ½A² + 1 NS | (a) Neon B1, (b) Neon B2, (c) Carbon B2 | Neon claims single-refresh r_C 0.03–0.05 for (a) and below QR for (b), at "high confidence". **[checked, contradicts]** (a) gives 0.12 single-refresh; chained 0.0785 / ρ 0.254. First-order Jacobi at factor 1 gives chained r_C 0.074 (below QR10) but ρ 0.253 (above) | over-rotates nearly degenerate pairs; goes beyond the QR10 target; churns tail columns, which hurts V (Neon's own warning) | RP (+ churn metric) |
| **PB4** | **Exact QR at every refresh via PA1** (no warm step) | Fable B1, Neon B6b, Carbon B3 (one-pass + 1 NS) | about 1.6–2× the warm cost per event, but the whole schedule is cheaper than now: 9–10.5 ms/call (Fable) or 13.0 ms/call (Neon), against 15.94 for warm10/QR50. QR10 trajectory by construction | as PA1; a ridge too large changes the target. **[checked]** ridge 1e-3 moves columns by about 1.5e-2 at n = 4096 | MB + RP |
| PB5 | **Head block or panel power step** for the top k columns, with an orthogonal completion of the tail: (a) leading QR panel + Householder completion; (b) head-block CholQR2 + deflation; (c) top-k power step + Procrustes reflection, then PB3(a) | (a) astra B2, (b) Fable B4, (c) Neon B3 | +3k/n of a product (Neon). Exact QR on the top k; aimed at the r_C ≈ 0.97 factors. Neon's reflection identities **[checked]** | the tail gets no power step; k-boundary turnover | RP (top-k overlap, worst decile) |
| PB6 | **Grouped block QR rotations** (exact QR within size-b coordinate groups) | astra B3 | one cubic product + 4n²b. Exact for block-diagonal B | misses the coupling between groups | RP |
| PB7 | **Front-loaded schedule:** QR10 (or PB4) for the first 250–1000 ages, optionally buying it back later | astra A6, Fable B5, Neon B4, Carbon B5 | +0 to +1.3 ms/call. Should remove most of the call-500 gap; little effect on the plateau | budget-neutral variants open 100-age QR gaps late | RP (ages 0–1000) + TR (500/2000 calls) |
| PB8 | **V after a refresh:** (a) mix V by T∘T; (b) rescale V's rows and columns to the new diagonal d⁺/d (Kronecker ratio) | (a) astra B5, Carbon B4; (b) Neon B5, Fable B7 | O(mn) for (b). Claimed: fewer post-refresh step spikes. **[checked]** see dispute D2 | (a) double-counts leaked variance on corrective rotations; (b) is a Kronecker approximation | TR only (or a gradient-stream replay) |
| PB9 | **Sort columns at warm refreshes** and permute V | Neon B1, Carbon B1, astra B1 | about 0.4 ms per event | none (makes no difference to PB1 in the check) | RP |
| PB10 | **Refresh before the update** (C, refresh and transport, then project G_t) | Neon P4, Carbon B4 | 10% of steps use a fresh basis | a different method; conflicts with PA12 | TR |
| PB11 | **Basis storage precision:** FP16 Q (Neon A9/P3) or an FP32 master / stochastic rounding (Fable B6) | Neon, Fable | Gram floor ÷8 with FP16. Fable expects only a few % of rotation is lost to the round-to-nearest dead-band | — | RP (`basis_dtype` switch) |
| PB12 | **Adaptive QR trigger** (promote a factor to QR when the cap scale a < 0.5, or by residual) | Fable B5 (option); moderator prior 9 | negligible cost | data-dependent control breaks the static graphs | RP |

Also proposed, not ranked as candidates:
- astra's suggestion to log the spectral Gram error, not only the normalized Frobenius;
- Neon's tail-churn guardrail χ;
- astra's gradient-stream replay: world-update cosine and norm ratio against BF16 QR10 over the 10 calls after a
  refresh;
- Carbon's logging of max |M/√V| just before and after QR events.

## 3. Disputes (kept open)

**D1. Target generator: QR-matched or full Jacobi?**
- **The positions.** astra, Fable and Carbon say match QR's tangent (PB1). The QR10 trajectory is the target, and a
  bigger rotation is not better. Neon (B1/B2), and Carbon's B2 as a baseline, want full-gain Jacobi ("quadratic
  convergence"; the −½ is "a factor-of-2 bug").
- **The moderator's toy.** Chained, n = 128, the paper's Fig. 2 setting (§4.3):
  - factor-1 first-order Jacobi reaches the lowest r_C (0.074, below QR10's 0.076) but a higher ρ_rms (0.253, against
    0.240);
  - the QR tangent matches QR10 on both (0.0767 / 0.243);
  - Neon's exact spec sits between (0.0785 / 0.254).
- **Reading.** Jacobi over-diagonalizes the head and under-serves the whitened tail, relative to QR. r_C alone would
  pick Jacobi; the whitened metric picks the QR tangent. This is exactly the metric dispute of §1.6, and the replay
  on trained C decides it.

**D2. What to do with V at a refresh.**
- **Mixing V by T∘T** (astra B5, Carbon B4).
  - Neon's counterexample **[checked]**: C = U·diag(1e4, 1e-2)·Uᵀ, with the old basis 0.01 rad off. Mixing moves V₂
    from 1.01 to 2.01, against a true 0.01; leaving V unchanged is better.
  - Carbon's counterexample: M transported into a low-V coordinate gives |M/√V| ≈ 199 when V is unchanged; mixing
    fixes it **[checked]**.
- **The moderator's resolution (§4.5).** The two examples assume different consistency between the moments and C:
  - Neon's assumes V matches the old-basis diagonal of the current C. That is the natural state when
    β₂ = β_C = 0.95, since both are EMAs of the same gradients (Fable B7: the row sums of V equal diag(Q_LᵀC_LQ_L)).
  - Carbon's assumes M and V are stale relative to C.
  - When M, V and C are consistent, rescaling V to the new diagonal is exact, mixing overestimates the small variance
    76,000-fold (a safe but slow step), and no spike arises.
  - So PB8(b) is the principled default under equal betas, and PB8(a) is not. Carbon's spike remains possible only
    where M lags C, for example after large corrective QR rotations that follow 40 calls of drift.
- **Not testable in the replay.** Needs Carbon's |M/√V| logging around QR events, or astra's gradient-stream replay.

**D3. Why the measured Cholesky-QR failed.**
- **Fable and Carbon:** unshifted Cholesky-QR breaks once κ(C)²·u_FP32 ≳ 1, which is n ≈ 4096 for λ ∝ 1/i,
  regardless of TF32.
- **Neon:** TF32 truncation in ZᵀZ "directly explains" both the 0.034 Gram error at n = 4096 and the NaN at
  n ≥ 6144.
- **Moderator CPU check** (float32, TF32 emulated, §4.4): at n = 4096 (κ ≈ 7.2e3), unshifted one-pass Cholesky-QR
  does **not** fail. Its Gram error is 0.0026 in FP32 and 0.0043 with emulated TF32, not 0.034.
- **Reading.** TF32 measurably hurts but does not reproduce the GPU number on these inputs. Neon's "directly
  explains" is not confirmed. n ≥ 6144 was not run on CPU. The GPU microbenchmark with IEEE FP32 forced settles it.

**D4. Size of the engineering win.**
- Carbon claims the exposed optimizer time can fall from 26 to about 12 ms/call by overlapping refreshes with the next
  forward/backward (PA12).
- Fable puts the ordinary step about 80% matmul-bound, so 1–2 ms is the realistic engineering slack.
- **Not resolved.** Needs an in-training timing run.

**D5. Expected loss gains.**
- **Neon and Carbon** claim "high confidence" that PB1 or PB3 closes 60–80% of the steady-state gap.
- **astra:** "no defensible CE prediction".
- **Fable:** medium confidence.
- **The moderator** treats every CE number in P1 as unsupported until the matched BF16 QR10 run exists.

**D6. Is "V unchanged at a warm step" acceptable?**
- **Fable:** correct to first order under a diagonal approximation; the change is Σ_k A_ki²(V_k − V_i).
- **Neon and Carbon:** it causes spikes, or keeps V wrong for 13.5-call half-lives.
- **Moderator.** Fable's argument drops the cross term E[ḡ_i ḡ_k], which is exactly what a corrective rotation
  removes. Neon's example is first-order wrong for "unchanged" (1.01 against 0.01). Leaning to Neon and Carbon on
  correctness; size unknown.

## 4. Moderator checks (`checks/`)

All run on CPU, float64 unless stated; outputs are in the matching `.out` files.

**4.1 Algebra** (`astra_identities.py`, `identities_misc.py`).

QR first-order tangent, K_ij = E_ij/d_j:
- relative error against the exact qr(D + E).Q is 2e-3 at n = 8 and 2e-2 at n = 64 with off-diagonal entries of
  about 1e-3, shrinking linearly with their size **[confirmed]**;
- the inverse-gap (Jacobi) generator's relative error against the same target is 4.2–140, dominated by
  near-degenerate pairs **[confirmed]**.

Retraction identities:

| Claim | Who | Result |
|---|---|---|
| One NS step maps E ↦ −¾E² + ¼E³; worst case at the cap 0.25 → −0.04297 → −1.40e-3 | astra, Carbon | confirmed |
| (I + A + ½A²)ᵀ(I + A + ½A²) = I + ¼A⁴ | Fable, Carbon | confirmed to 6e-16 |
| R₄ᵀR₄ = I + A⁶/72 + A⁸/576; ‖R₄ᵀR₄ − I‖₂ = 2.1e-4 at ‖A‖₂ = 0.5 | Neon | confirmed |
| polar(I + A) rotates by arctan θ | Neon | confirmed |
| Neon's Procrustes reflection H₊ is orthogonal and maps Q₁ ↦ −Ũ₁ | Neon | confirmed |
| Fable's warm/QR angle-ratio formula | Fable | confirmed to 4e-15 |
| ‖K‖₂ ≤ ‖K‖_F/√2 for skew K | all | confirmed: max ratio 0.51 over random skews |
| Transport reassociation saves 8.3% / 0% / 7.5% / 8.3% of MACs on the four 109M shapes | astra, Carbon | confirmed |
| Neon's 8T_warm + 2T_QR vs 5T_QR arithmetic at n = 8192 (588 vs 560 ms per 100 calls) | Neon | confirmed |

**4.2 The cap and the half step** (`m1_half_step.py`, `m1b_cap_binding.py`). Synthetic inputs as in the
microbenchmark; exact polar retraction to isolate the generator.

| n = 256, input r_C 0.22 | single-refresh r_C |
|---|---|
| production (½ step, Frobenius cap 0.5) | 0.175 |
| ½ step, no cap | 0.118 |
| factor 0.75, no cap | 0.075 |
| factor 1, no cap | 0.055 |
| simultaneous exact 2×2 Jacobi angles | 0.131 |
| production BF16 `warm_basis` | 0.175 |

- The production BF16 output equals the float64 analogue to three digits, so BF16 is not what limits the warm step.
- Pre-cap ‖A‖_F/√2 is 0.63–1.10 for n = 128–1024 with no clear trend in n, and the cap scale a is 0.46–0.79.
  - Fable's "√n too tight" and Neon's "explodes by √n" overstate the n-dependence on these power-law inputs. The
    measured Frobenius/spectral ratio is 2.5–5.
  - Their conclusion that the cap binds is right.
- Simultaneous exact Jacobi angles (moderator prior 6) are worse than first-order full Jacobi, because near-degenerate
  pairs saturate at π/4. Prior 6 is demoted.

**4.3 Generators, one-shot and chained** (`generator_compare.py`).

One-shot, n = 256, input r_C 0.217, ρ_rms 0.489:

| generator | r_C | ρ_rms | top-k distance to QR |
|---|---|---|---|
| production | 0.168 | 0.400 | 0.015 |
| no cap | 0.115 | 0.331 | 0.006 |
| Jacobi factor 1 | 0.052 | 0.278 | 0.004 |
| **QR tangent** | 0.081 | **0.260** | **0.002** |
| Neon B1 (exact spec) | 0.121 | 0.341 | 0.012 |
| QR | 0.038 | 0.212 | 0 |

Chained open-loop toy (the paper's Fig. 2 setting: n = 128, λ ∝ 1/i, skew drift 0.01√2 per step, 32 samples per
step, β_C = 0.95), mean over steps 100–1000:

| schedule | r_C | ρ_rms |
|---|---|---|
| QR10 | 0.0759 | 0.240 |
| **QR-tangent 10 / QR50** | **0.0767** | **0.243** |
| Jacobi factor 1, 10 / QR50 | 0.0742 | 0.253 |
| Neon B1, 10 / QR50 | 0.0785 | 0.254 |
| no cap, 10 / QR50 | 0.0841 | 0.271 |
| production 10 / QR50 | 0.0977 | 0.294 |
| QR50 | 0.1343 | 0.353 |

- The toy reproduces the paper's Fig. 2 values (0.075 / 0.093 / 0.125) to within seed noise.
- Sorting columns makes no difference to the QR tangent.
- At large misalignment (rot 0.5), the first-order QR tangent's one-shot ρ_rms (0.676) is worse than the uncapped
  production generator's (0.485). PB1 should keep a cap.

**4.4 Cholesky-QR numerics** (`cholqr_precision.py`). float32 CPU; TF32 emulated by rounding the Gram operands to a
10-bit mantissa. Synthetic C with κ(C) = 8e2, 1.7e3, 3.5e3 and 7.2e3 at n = 512, 1024, 2048 and 4096.

| variant | normalized Gram error after BF16 | max column distance to Householder |
|---|---|---|
| unshifted one pass, FP32 | 0.0023–0.0026 | 1.4e-5–2.8e-4 |
| unshifted one pass, TF32 | 0.0027–0.0043 | 1e-4–4e-4 |
| CholeskyQR2, FP32, no ridge | 0.0023–0.0024 | **≤ 4.2e-7** |
| ridge 1e-4 + CholQR2 | 0.0024 | 8e-5 (n = 512) to 3.1e-3 (n = 4096) |
| ridge 1e-3 + CholQR2 | 0.0024 | 3.9e-3 (n = 512) to 2.6e-2 (n = 4096) |

- No failures at n ≤ 4096.
- **The ridge is not free.** It changes the target basis by an amount that grows with n. Use CholQR2 with no ridge,
  and keep the shifted pass as a fallback only when `cholesky_ex` reports failure, or keep τ ≤ 1e-5.
- The ridge Neon and Carbon propose (2e-4 to 1e-3) would move the basis measurably.

**4.5 V after refresh** (`identities_misc.py` (d), (g), (h)). The 2×2 cases of D2:

| case | true V₂ | unchanged | T∘T mixing | ratio rescale |
|---|---|---|---|---|
| Neon (corrective rotation) | 0.01 | 1.01 | 2.01 | 0.01 (exact) |
| Carbon, with M and V stale relative to C | — | spike 198 | ≈ 1 | spike 3.9e4, because the ratio uses a C that V never saw |
| same rotation, M and V consistent with C | 1e-6 | 0.0395 | 0.0758 | 1e-6 (exact); no spike under any rule |

**4.6 BF16 covariance EMA dead-band** (`astra_identities.py` (5); astra, Fable, Neon, Carbon all raise it). A
noiseless slow drift leaves the BF16 EMA 4.2% away from the FP32 EMA, larger than the EMA's own lag of 3.7%. With
realistic sample noise the gap falls to 0.6–0.8%, because the noise dithers the rounding. So the dead-band is real
but mostly masked at β_C = 0.95 with noisy gradients. All four agree it becomes binding for β ≥ 0.99.

**4.7 Basis-frame EMA** (`m2_basis_frame_ema.py`, moderator prior 1).
- **The identity holds.** ḠḠᵀ = Q_LᵀGGᵀQ_L when Q_R is orthogonal. With a BF16 Q_R (Gram error 0.0024), the
  relative off-diagonal error after 10 steps is 1.8e-3.
- **No product is saved.** Keeping B costs a 2-product rotation of B at each basis change, so the warm step stays at
  7 products. No reviewer proposed this idea.

## 5. Against the moderator priors ([MODERATOR_PRIORS.md](MODERATOR_PRIORS.md))

| Prior | Reviewer counterpart | Verdict after P1 |
|---|---|---|
| 1 basis-frame B | none | no product saving (§4.7); keep only as a cheap residual source for PB12 |
| 2 transport with the small rotation | Neon A6/B1 (PA7a) | proposed; small payoff (transport ≈ 0.5 ms/call) |
| 3 one NS correction | astra A4, Carbon A5 (PA6a) | proposed; Fable's 6-product seed (PA6b) dominates at an equal cap |
| 4 skip factors with small residual | none directly (Fable's trigger is close) | still open; graph-static only per group |
| 5 syrk/fusion | astra, Neon, Carbon (PA8, PA10) | proposed by three |
| 6 exact 2×2 Jacobi angles | Neon B1 (atan, damped) | **demoted**: simultaneous exact angles do worse than first order (§4.2); the QR tangent does better |
| 7 Cayley / exponential retraction | Neon (R₄), Fable and Carbon (2nd-order seed) | proposed; identities checked |
| 8 two warm steps | Neon B2 (two Jacobi iterations) | proposed in a Jacobi form; the refresh-study ablation has `warm_steps=2` |
| 9 adaptive warm vs QR | Fable B5 trigger (PB12) | weakly proposed |
| 10 approximate V rotation | astra B5, Carbon B4 (mixing); Neon B5, Fable B7 (rescale) | **split**: mixing refuted for corrective rotations; rescale preferred (D2) |
| 11 front-loaded schedules | all four (PB7) | strongest agreement of any prior |
| M1 half step | Neon, Carbon ("−½ bug"), Fable (I3), astra (B1) | confirmed (§4.2), but the panel's better fix is to change the generator (PB1), not just the factor |
| M2 prior 1 saves nothing | — | confirmed (§4.7) |
| M3 QR repair is the largest refresh cost at 109M | all four | confirmed by all |

**New relative to the priors.** The panel's most important additions are:
- PB1 (QR tangent) and PB2 (spectral cap);
- PA1/PB4 (stabilized Cholesky-QR, possibly at every refresh);
- PA2 (concurrent QR);
- PB5 (head-block power step);
- the whitened-metric argument;
- the TF32 finding.

## 6. Reviewer errors and overclaims caught

1. **Neon B1's expected effect.** Neon claimed single-refresh r_C 0.150 → 0.03–0.05 at "high confidence". The exact
   spec gives 0.12–0.13 on the brief's synthetic inputs (§4.3). Neon's B2 claim ("beats QR10's 0.025") is untested
   but rests on the same generator.
1b. **Carbon B1's expected effect.** Carbon claimed single-refresh r_C 0.150 → 0.035–0.050 for the QR-tangent
   generator at "high confidence". The QR tangent gives 0.081 (n = 256) and 0.099 (n = 512) one-shot (§4.3); its
   gain shows in the chained, whitened metrics, not as a one-step drop to near QR. Fable's prediction for its B2
   (spectral cap: 0.20 → 0.10–0.11) matches the uncapped production generator's measured 0.115–0.118.
2. **Neon on TF32.** Neon said TF32 "directly explains" the 0.034 Cholesky-QR Gram error. CPU emulation gives 0.0043
   at n = 4096 (D3); not confirmed.
3. **The cap's n-dependence.** Fable said the cap is "√n too tight" (hedging that real A is structured), and Neon
   that it "explodes by √n". The measured ratio
   is 2.5–5 with no clear trend in n = 128–1024 (§4.2). The direction is right, the magnitude overstated.
4. **The V-mixing rule.** astra B5 and Carbon B4 proposed mixing V by T∘T. It is wrong for corrective rotations under
   consistent statistics (§4.5); astra did flag a permutation-only control.
5. **Ridge sizes.** Neon (A5, τ = 3e-4 to 1e-3) and Carbon (A4/B3, 2e-4 to 5e-4) described the ridge as
   "eigenvector-preserving". It preserves C's eigenvectors, but it changes the one-step QR target and slows tail
   rotation. At n = 4096 a ridge of 1e-3 moves columns by up to 2.6e-2 (§4.4).
6. **astra's update-equation pitfall.** astra listed the weight-decay and ε ordering as a possible mismatch with the
   reference. **[checked] Moot:** `soap_reference.py` lines 104–108 compute `m/(v.sqrt()+eps)` scaled by
   lr·√(1−β₂ᵗ)/(1−β₁ᵗ), then apply `p.add_(p, alpha=-lr*wd)` after the update. That is (W − ηcU)(1 − ηλ), the same
   form as the variant, and the module docstring says it was verified against pinned upstream.
7. **Neon's "double-projects G_t into M".** Neon described M's update as projecting G_t twice. G_t enters M once, in
   the old basis, and is then transported; the issue is extra rounding, not double counting. Minor.
8. **Neon's "SOAP permutes V because mixing is a trap".** Speculation presented as an explanation.

## 7. Moderator's provisional ranking for P2 (before Part 2 data)

**Speed.**
1. **PA1**, stabilized Cholesky-QR for QR events: −9 to −10 ms/call claimed, from timings already measured. Spec:
   CholQR2, IEEE FP32 Grams, no ridge by default, shifted fallback. *(Withdrawn after Part 2; see §9.1.)*
2. **PA2**, concurrent QR: no change in arithmetic; the fallback if PA1's numerics fail on trained C.
3. **PA6b**, the seed I + A + ½A² plus one NS (6 products). It keeps self-healing; PA6a and PA6c are ablations.
4. **PA9**, pack removal, coalescing the n = 768 groups, CUDA graphs: recurring every call.
5. **PA13**, the size-aware schedule for large n (a consequence of PA1).

**Tracking.**
1. **PB1 + PB2**, the QR tangent with a spectral cap: 7 (or 6) products, toy-supported.
2. **PB4**, Cholesky-QR at every refresh: QR10 by construction, and claimed cheaper than today's mix.
3. **PB7**, front-loaded schedule (training run).
4. **PB5**, head-block power step, for the r_C ≈ 0.97 factors.
5. **PB8b**, rescale V to the new diagonal (training run).

**What Part 2 can already answer.** The refresh-study Part 2 (replay on trained 1-layer factors, plus the ablation
grid) partly tests:
- PB2 (`cap=1.0` / `cap=None`);
- PA6a (`ns=1`);
- prior 8 (`warm_steps=2`);
- unshifted Cholesky-QR quality on trained C (`cholqr10`, with TF32 Grams under "high").

None of it tests PB1, PB4 with IEEE Grams, or anything involving M, V or loss.

## 8. Pooled "measure next" list

1. Matched compiled BF16 QR10, at 500, 2000 and 10k calls (all four).
2. Per factor and per refresh: the cap scale a, ‖A‖₂ against ‖A‖_F/√2, and the realized rotation by eigen band. Also
   which factors sit at r_C ≈ 0.97, with their spectra (all four).
3. Whitened metrics in the replay (ρ_rms, log-det, distance to the QR10 chain by band), checked against the loss
   ordering QR10 < QR20 < warm < QR50 (Fable, Neon, Carbon, astra).
4. QR split into geqrf and orgqr; Cholesky failure rate and κ(C) on trained snapshots; IEEE vs TF32 Grams (Fable,
   astra, Neon).
5. max |M/√V| and the update cosine around QR events, via a gradient-stream replay or a short run (astra, Carbon).

## 9. Addendum after refresh-study Part 2 (written 2026-11-20 ~03:40, after §1–§8)

§1–§8 were written blind to Part 2 and are left as they were. Part 2 ran on trained 1-layer factors (d = 256–2048,
ages 0–1000, open loop; `reports/refresh_study_20261120/REPORT.md`). It qualifies these statements:

1. **PA1 / PB4 (Cholesky-QR).**
   - **The trained regime differs from the moderator check's.** Unshifted FP32 Cholesky-QR10 is non-finite from the
     first refresh on 791–800 of 800 factor-refreshes per width. Many trained directions are near-null, and some
     BF16-stored factors are slightly indefinite: 4–15% negative eigenvalues in the right SwiGLU factor at d ≥ 512.
   - **So "no ridge by default, shifted fallback" (§4.4, §7) is withdrawn.** The fallback would fire almost always.
   - **A ridge (or shift) is required on trained C,** and it must exceed the most negative BF16 eigenvalue.
   - **"The ridge is not free" is untested where it matters.** It was measured only where every direction is resolved
     (full-rank synthetic C, κ ≈ n). On near-null directions the Householder QR tail is itself arbitrary, so the
     ridge's cost there is unknown.
   - **The required test** is a ridge sweep on the trained snapshots: τ ∈ {1e-6, 1e-5, 1e-4, 1e-3}·d_max, CholQR2,
     IEEE-FP32 Grams. It should report the failure count, ρ_rms / r_C^w, and band-wise distance to the QR10 chain.
   - PA1 and PB4 remain top by consensus, but conditional on that sweep.
2. **PB2 (cap).** On trained factors, **removing** the Frobenius cap makes the current generator diverge (non-finite
   in 59–82% of factor-ages), and cap 1.0 is slightly better than 0.5. The §4.2 "no cap" result does not transfer.
   PB2 means **replace** the Frobenius cap with a spectral (or row-ℓ₂) bound, never remove it.
   - Mechanism (moderator reading): with δ = 0.01 and no cap, near-degenerate pairs get |A_ij| up to 25·B_ij/s.
3. **PB1 (QR tangent).** It is supported only on full-rank PSD toys. Its denominator max(d_i, d_j, ρ·s) and sign rule
   were never exercised with d ≤ 0 (indefinite BF16 C) or with d ≪ ρ·s (near-null directions). The floor ρ must be
   checked against BF16 noise on trained snapshots before PB1 is called supported.
4. **What Part 2 confirms from P1.**
   - **The warm step barely moves weak directions.** r_C^w for warm10/QR50 stays near QR50's. Part 2 names the same
     mechanism as Fable, Neon and Carbon: the max-normalized gap with δ = 0.01.
   - **The ablation cannot close the gap.** No change of damping, cap or NS count moves the interval r_C by more than
     0.016, against a QR50–QR10 gap of 0.11–0.12. That supports changing the generator (PB1/PB3), not tuning it.
   - **One NS correction:** same quality, e_Q 2.1–2.2e-3, −22% warm time (PA6a).
   - **Two warm steps** (prior 8, Neon B2's spirit): the best ablation, −0.012 r_C and −0.019 r_C^w, at 1.8× warm cost.
   - **QR20 beats warm10/QR50 at every width,** but per millisecond of refresh work the warm refreshes buy 1.3–1.9×
     more r_C reduction.
     > **Correction (2026-11-20, ~05:45).** The "1.3–1.9×" quoted from refresh-study is superseded. It divided every
     > width's gain by the fixed 109M cost. Refresh-study's corrected width-matched figures (each width priced with
     > its own factor sizes, batch-12 compiled costs, transports excluded): the warm refreshes buy **2.9× / 2.3× /
     > 0.92× / 0.35×** the per-millisecond r_C reduction of QR20's extra QRs, for d = 256 / 512 / 1024 / 2048. That is
     > **≈ 1.4× at the 109M factor sizes.** The warm advantage per millisecond reverses near d = 1024, as warm/QR cost
     > rises with n. Still an r_C statement, not a loss statement.
