# Stage 2 reveal: pooled candidates, moderator checks, and new measurements

In this document, "you" means your own stage-1 answer, and Reviewers X, Y and Z are the other three. Candidate IDs in
parentheses are each answer's own IDs; for example, "Reviewer X (A5)" is candidate A5 in Reviewer X's answer.

## R1. Pooled candidates

The moderator merged the 50 stage-1 candidates into 14 speed candidates (PA) and 12 tracking candidates (PB). The
"Claimed" column gives the proposers' own estimates. **[checked]** marks a claim the moderator tested (see R3).

### A: speed

| ID | Candidate | Proposed by | Claimed cost / benefit |
|---|---|---|---|
| PA1 | **Stabilized Cholesky-QR for QR events.** Z = C·Q (BF16 operands, FP32 accumulation); optional ridge Z + τ·d_max·Q; column equilibration; CholeskyQR2 with IEEE-FP32 Gram products; `cholesky_ex` with an in-graph fallback | ⟦fable⟧ (A1), ⟦neon⟧ (A5), ⟦carbon⟧ (A4), ⟦astra⟧ (B4, conditional) | QR event 594 → about 95–140 ms; −9 to −10 ms/call |
| PA2 | **Concurrent QR** over CUDA streams or host threads; same arithmetic | ⟦neon⟧ (A1), ⟦carbon⟧ (A3), ⟦fable⟧ (A4) | 2.5–4× at n = 768 |
| PA3 | **R-only Householder plus a triangular solve,** then one Newton–Schulz (NS) step | ⟦fable⟧ (A2) | about −40% of QR time if orgqr is half of it |
| PA4 | **Custom batched, blocked Householder QR** | ⟦astra⟧ (A3) | target −50% of the QR increment |
| PA5 | **`mm32` (BF16 tensor cores) for Z = C·Q** inside the QR path | ⟦neon⟧ (A1), ⟦carbon⟧ (A1) | 4–6 ms per QR event |
| PA6 | **Fewer warm products:** (a) one NS correction (5 products); (b) seed I + A + ½A², then one NS (6); (c) degree-4 Taylor R₄(A), no NS (5); (d) R₄(A/2)² (6) | (a) ⟦astra⟧ (A4), ⟦carbon⟧ (A5); (b) ⟦fable⟧ (A3), ⟦carbon⟧ (B2); (c) ⟦neon⟧ (A6); (d) ⟦neon⟧ (B1) | −14% to −29% of warm FLOPs |
| PA7 | **Cheaper M transport:** (a) M⁺ = R_LᵀMR_R with the known rotation; (b) short-side reassociation | (a) ⟦neon⟧ (A6); (b) ⟦astra⟧ (A5), ⟦carbon⟧ (A6) | transport ≈ 0.5 ms/call in total |
| PA8 | **Symmetric (syrk) kernels** for GGᵀ, GᵀG and the NS Grams | ⟦astra⟧ (A2), ⟦neon⟧ (A4), ⟦carbon⟧ (A2) | −1 to −1.8 ms/call |
| PA9 | **Contiguous parameter/grad banks** (no stack or copy); coalesce the five n = 768 groups into batch 60; CUDA graphs | ⟦astra⟧ (A1), ⟦neon⟧ (A2), ⟦carbon⟧ (A1), ⟦fable⟧ (A5) | 0.8–1.1 ms/call + 2.5–5 ms per warm event |
| PA10 | **Fused epilogues and no FP32 temporaries in the warm step;** sub-batching at large n | ⟦carbon⟧ (A1), ⟦neon⟧ (A3) | recovers the batch-12 vs batch-1 regression at n ≥ 6144 |
| PA11 | **Strided covariance EMA** (update C every 2nd call with β_C²) | ⟦fable⟧ (A6), ⟦neon⟧ (A7), ⟦carbon⟧ (A6) | 1.2–1.8 ms/call; not testable in replay |
| PA12 | **Overlap the refresh with the next forward/backward** | ⟦carbon⟧ (A3), ⟦fable⟧ (A4 variant) | exposed optimizer time 26 → about 12 ms/call |
| PA13 | **Size-aware schedule:** at large n, drop warm and run PA1 every 10–20 ages | ⟦neon⟧ (A8, B6a) | n = 8192: about 325 vs 588 ms per 100 calls |
| PA14 | **FP16 basis storage** (optionally FP16-accumulate back-projection) | ⟦neon⟧ (A9) | Gram floor 0.0024 → about 0.0003 |

### B: tracking

| ID | Candidate | Proposed by | Claimed cost / benefit |
|---|---|---|---|
| PB1 | **QR-tangent generator.** A_ij = B_ij·sgn(d_j − d_i)/max(d_i, d_j, ρ·s), the first-order rotation of one QR power step; optional sort plus V permutation | ⟦astra⟧ (B1), ⟦fable⟧ (B3), ⟦carbon⟧ (B1) | same product count; matches QR to first order |
| PB2 | **Spectral (or row-ℓ₂) cap instead of the Frobenius cap** | ⟦fable⟧ (B2), ⟦carbon⟧ (B1), ⟦neon⟧ (B1), ⟦astra⟧ (B1 ablation) | about doubles the applied step |
| PB3 | **Full-gain Jacobi variants:** (a) relative-gap atan Jacobi + row cap + R₄; (b) two inner iterations; (c) local-scale full-gain Jacobi + seed I + A + ½A² + 1 NS | (a, b) ⟦neon⟧ (B1, B2); (c) ⟦carbon⟧ (B2) | claimed near-QR or better single-refresh r_C |
| PB4 | **Exact QR at every refresh via PA1** (no warm step) | ⟦fable⟧ (B1), ⟦neon⟧ (B6b), ⟦carbon⟧ (B3) | 9–13 ms/call of refresh, against 15.9 today; QR10 by construction |
| PB5 | **Head-block or panel power step** on the top k columns, with an orthogonal completion of the tail | ⟦astra⟧ (B2), ⟦fable⟧ (B4), ⟦neon⟧ (B3) | +3k/n of a product; exact on the top k |
| PB6 | **Grouped block QR rotations** | ⟦astra⟧ (B3) | one cubic product |
| PB7 | **Front-loaded schedule:** QR10 (or PB4) for the first 250–1000 ages | ⟦astra⟧ (A6), ⟦fable⟧ (B5), ⟦neon⟧ (B4), ⟦carbon⟧ (B5) | 0 to +1.3 ms/call; aimed at the call-500 gap |
| PB8 | **V after a refresh:** (a) mix V by T∘T; (b) rescale V's rows and columns to the new diagonal d⁺/d | (a) ⟦astra⟧ (B5), ⟦carbon⟧ (B4); (b) ⟦neon⟧ (B5), ⟦fable⟧ (B7) | fewer post-refresh step errors; training run only |
| PB9 | **Sort columns at warm refreshes** and permute V | ⟦neon⟧, ⟦carbon⟧, ⟦astra⟧ (all inside their B1) | about 0.4 ms per event |
| PB10 | **Refresh before the update** | ⟦neon⟧ (pitfall P4), ⟦carbon⟧ (B4) | fresh basis on 10% of steps; a different method |
| PB11 | **Basis storage precision:** FP16 Q, or an FP32 master / stochastic rounding | ⟦neon⟧ (A9), ⟦fable⟧ (B6) | Gram floor ÷8 (FP16) |
| PB12 | **Adaptive QR trigger** (e.g. when the cap scale a < 0.5) | ⟦fable⟧ (B5 option) | negligible cost; breaks static graphs |

## R2. Open disputes

- **D1. Target generator.** Match QR's first-order rotation (⟦astra⟧, ⟦fable⟧, ⟦carbon⟧), or apply full-gain Jacobi
  (⟦neon⟧; ⟦carbon⟧'s B2 as a baseline)?
- **D2. V after a refresh.** Mix V by T∘T (⟦astra⟧, ⟦carbon⟧), rescale V to the new diagonal (⟦neon⟧, ⟦fable⟧), or
  leave it (⟦fable⟧ argues this is correct to first order)?
- **D3. Why the measured Cholesky-QR failed.** κ(C)²·u_FP32 ≳ 1 (⟦fable⟧, ⟦carbon⟧), or TF32 truncation of the Gram
  product (⟦neon⟧)?
- **D4. Size of the engineering win.** About 1–2 ms/call of slack, since the ordinary step is mostly matmul-bound
  (⟦fable⟧), or 26 → 12 ms/call exposed via overlap (⟦carbon⟧)?
- **D5. Loss predictions.** "High confidence" that PB1 or PB3 closes 60–80% of the steady-state gap (⟦neon⟧,
  ⟦carbon⟧), or no defensible CE prediction (⟦astra⟧)?
- **D6. Is leaving V unchanged at a warm step acceptable?**

## R3. Moderator checks

All checks ran on CPU (float64, or float32 where stated), on **synthetic** inputs: full-rank PSD C, power-law
λ_i ∝ 1/i, κ ≈ n. They are not trained covariances; R5 shows how trained factors differ.

**M1. Algebra.**

| Claim | Result |
|---|---|
| QR first-order tangent K_ij = E_ij/d_j for i > j (d_j ≥ d_i) | confirmed: relative error against the exact qr(D + E).Q is 2e-3 (n = 8) and 2e-2 (n = 64) for off-diagonal entries of about 1e-3, shrinking linearly |
| The inverse-gap (Jacobi) generator as an approximation of QR | relative error 4.2–140 |
| One NS step maps E ↦ −¾E² + ¼E³ (0.25 → −0.04297 → −1.40e-3) | confirmed |
| (I + A + ½A²)ᵀ(I + A + ½A²) = I + ¼A⁴ | confirmed |
| R₄ᵀR₄ = I + A⁶/72 + A⁸/576 | confirmed |
| polar(I + A) rotates by arctan θ | confirmed |
| The top-k Procrustes reflection is orthogonal and maps Q₁ ↦ −Ũ₁ | confirmed |
| The closed-form warm/QR angle ratio | confirmed |
| ‖K‖₂ ≤ ‖K‖_F/√2 for skew K | confirmed |
| Transport reassociation saves 8.3% / 0% / 7.5% / 8.3% of multiply-adds on the four 109M shapes | confirmed |
| 8T_warm + 2T_QR = 588 vs 5T_QR = 560 ms per 100 calls at n = 8192 | confirmed |
| `set_float32_matmul_precision('high')` sets `allow_tf32 = True` (PyTorch 2.11) | confirmed |
| The reference SOAP applies (W − ηcU)(1 − ηλ) with c_t·m/(√v + ε), the same form as the variant | confirmed; the update-ordering concern in ⟦astra⟧'s pitfalls is moot |

**M2. The cap and the half step.** Synthetic inputs as in the microbenchmark, exact polar retraction.

| n = 256, input r_C 0.22 | single-refresh r_C |
|---|---|
| production (½ step, Frobenius cap 0.5) | 0.175 |
| ½ step, no cap | 0.118 |
| factor 0.75, no cap | 0.075 |
| factor 1, no cap | 0.055 |
| simultaneous exact 2×2 Jacobi angles | 0.131 |

- Production in BF16 equals its float64 analogue to three digits.
- Pre-cap ‖A‖_F/√2 is 0.63–1.10 for n = 128–1024, with no clear trend in n. The cap scale a is 0.46–0.79, and
  ‖A‖₂ ≤ 0.30.
- **This contradicts in magnitude** the "√n too tight" claim (⟦fable⟧, I4) and "explodes by √n" (⟦neon⟧). The
  measured Frobenius-to-spectral ratio is 2.5–5. The conclusion that the cap binds stands.

**M3. Generators, one-shot and chained.**

One-shot, n = 256, input r_C 0.217 and ρ_rms 0.489. ρ_rms is the RMS whitened off-diagonal correlation.

| generator | r_C | ρ_rms | top-k subspace distance to QR |
|---|---|---|---|
| production | 0.168 | 0.400 | 0.015 |
| ½ step, no cap | 0.115 | 0.331 | 0.006 |
| first-order Jacobi, factor 1 | 0.052 | 0.278 | 0.004 |
| QR tangent (PB1, ρ = 1e-4, no cap) | 0.081 | 0.260 | 0.002 |
| relative-gap atan Jacobi + row cap (PB3a, as specified) | 0.121 | 0.341 | 0.012 |
| QR | 0.038 | 0.212 | 0 |

At n = 512 the QR tangent gives r_C 0.099. At large misalignment (input r_C 0.45), the QR tangent's one-shot ρ_rms is
0.676, against 0.485 for the uncapped ½ step. A first-order generator therefore still needs a bound.

Chained open-loop toy: n = 128, λ ∝ 1/i, skew drift 0.01√2 per step, 32 samples per step, β_C = 0.95, refresh every
10, QR every 50. Means over steps 100–1000:

| schedule | r_C | ρ_rms |
|---|---|---|
| QR10 | 0.0759 | 0.240 |
| QR tangent, 10 / QR50 | 0.0767 | 0.243 |
| Jacobi factor 1, 10 / QR50 | 0.0742 | 0.253 |
| PB3a, 10 / QR50 | 0.0785 | 0.254 |
| ½ step, no cap, 10 / QR50 | 0.0841 | 0.271 |
| production | 0.0977 | 0.294 |
| QR50 | 0.1343 | 0.353 |

Sorting columns made no difference to the QR tangent.

**These contradict two expected-effect claims:**
- "Single-refresh r_C 0.150 → 0.03–0.05" for PB3a (⟦neon⟧, B1): measured 0.12.
- "0.150 → 0.035–0.050" for the QR tangent (⟦carbon⟧, B1): measured 0.08–0.10. Its advantage appears in the chained
  and whitened metrics.

The prediction 0.20 → 0.10–0.11 for removing the Frobenius cap (⟦fable⟧, B2) matches the measured 0.115–0.118.

**M4. Cholesky-QR numerics.** float32 CPU; TF32 emulated by rounding the Gram operands to a 10-bit mantissa.
Synthetic C with κ = 8e2 to 7.2e3 at n = 512–4096.

| variant | Gram error after BF16 storage | max column distance to Householder |
|---|---|---|
| unshifted one pass, FP32 | 0.0023–0.0026 | up to 2.8e-4 |
| unshifted one pass, TF32 | 0.0027–0.0043 | up to 3.8e-4 |
| CholeskyQR2, FP32, no ridge | 0.0023–0.0024 | ≤ 4.2e-7 |
| ridge 1e-4·d_max + CholeskyQR2 | 0.0024 | 8e-5 (n = 512) to 3.1e-3 (n = 4096) |
| ridge 1e-3·d_max + CholeskyQR2 | 0.0024 | 3.9e-3 to 2.6e-2 |

- No failures at n ≤ 4096.
- **This contradicts** the claim that TF32 "directly explains" the measured 0.034 Gram error at n = 4096 (⟦neon⟧,
  pitfall P2): emulated TF32 gives 0.0043. n ≥ 6144 was not run.
- **But see R5:** on trained covariances, unshifted Cholesky-QR fails almost always. The no-ridge result above holds
  only for full-rank synthetic C.

**M5. V after a refresh** (2×2 cases).

| case | true V₂ | V unchanged | T∘T mixing | rescale to d⁺/d |
|---|---|---|---|---|
| ⟦neon⟧'s case: C = U·diag(1e4, 1e-2)·Uᵀ, old basis 0.01 rad off, refresh corrects it | 0.01 | 1.01 | 2.01 | 0.01 (exact) |
| ⟦carbon⟧'s case, with M and V stale relative to C | — | spike: M/√V = 198 | M/√V ≈ 1 | spike: M/√V = 3.9e4 (the ratio uses a C that V never saw) |
| same rotation, with M and V consistent with C (as β₁ = β₂ = β_C makes natural) | 1e-6 | 0.0395 | 0.0758 | 1e-6 (exact); no spike under any rule |

**M6. BF16 covariance EMA dead-band.** A noiseless slow drift leaves the BF16 EMA 4.2% away from the FP32 EMA. With
sample noise the gap is 0.6–0.8%.

**M7. Basis-frame covariance** (B = QᵀCQ kept as an EMA of ḠḠᵀ). The identity holds: the off-diagonal error is
1.8e-3 with a BF16 Q_R. But B must be rotated at every basis change (2 products), so the warm step keeps 7 products.
No reviewer proposed this; it is listed for completeness.

## R4. Part 1 microbenchmark: roofline view

Compiled, per factor, from the stage-1 cost table (RTX 5090).

**Warm refresh at batch 12,** counting 14n³ FLOPs (7 products):

| n | 768 | 1024 | 1536–2048 | 2304–8192 |
|---|---|---|---|---|
| TFLOP/s | 92 | 147 | 153–164 | 165–174 |

At batch 1 it reaches 178 at n = 3072 and 207–213 at n = 6144–8192. **From n ≈ 2048 up, the warm step runs at the
BF16 tensor-core limit, so only fewer products help it at large n.**

**Production QR,** counting (2 + 8/3)n³:

| n | 768 | 1024 | 2048 | 3072 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| TFLOP/s | 0.8 | 1.3 | 4.1 | 7.5 | 11.6 | 22.9 |

**Warm/QR time per factor:**

| n | 768–1024 | 2048 | 3072 | 4096 | 6144 | 8192 |
|---|---|---|---|---|---|---|
| warm/QR | 0.027 | 0.075 | 0.136 | 0.20 | 0.30 | 0.41 |

**Planned, not run.** A measured extension to n = 16384 with a shifted CholeskyQR3 is queued (Part 1b). No results
exist yet.

## R5. Part 2: open-loop replay on trained covariances

⟦PART2⟧

## R6. Tests running in parallel (do not assume their outcome)

The refresh study is replay-testing three generators on the same trained covariances right now:
- PB1, the QR-tangent generator;
- a relative-gap damping variant of the current step;
- first-order Jacobi at factor 1.

Results may arrive after your answer. Your rankings should not assume any particular outcome; say instead what
result would change your ranking.
