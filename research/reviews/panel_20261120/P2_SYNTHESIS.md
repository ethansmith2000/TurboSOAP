# Stage P2 synthesis (2026-11-20)

**Setup.** Each reviewer received, in one bundle: the P1 brief, its own P1 answer, the other three P1 answers
anonymized as X, Y and Z (labels drawn with seed 20261120, `prompts/P2_assignment.json`), and the reveal
(P2_REVEAL_TEMPLATE.md). The reveal contained the pooled candidates, disputes D1–D6, moderator checks M1–M7, the
Part 1 roofline, the Part 2 replay results, and a note on the parallel PB1/Jacobi replay tests. Web search was on.

| Answer | Words | Run time | Tools |
|---|---|---|---|
| [P2_gpt-6.2-astra.md](P2_gpt-6.2-astra.md) | 5,002 | 15 min | 52 web searches; no commands |
| [P2_fable.md](P2_fable.md) | 4,977 | 12 min | 9 Reads of its prompt, 14 WebSearch, 9 WebFetch, **1 advisor call** (outside the allowed tools; disclosed by Fable itself; its answer may carry some influence from that model) |
| [P2_gemini-4-neon-xhigh.md](P2_gemini-4-neon-xhigh.md) | 7,032 | 16 min | 9 reads of `prompt.md`, 49 `search_web` |
| [P2_gemini-4-carbon-high.md](P2_gemini-4-carbon-high.md) | 6,590 | 20 min | 11 reads of `prompt.md`, 43 `search_web` |

**How the sources were read.** astra and Fable opened their sources; Neon and Carbon worked from search results
only, and say so. Citations were spot-checked by the moderator against OpenAlex/arXiv (§4).

**Method name leak.** The sanitized Part 2 text in every bundle kept one sentence naming the method ("closed loop TurboSOAP would see different C"); the build script's assert list missed it. None of the 52 + 14 + 49 + 43 web queries contains the name (checked in the event logs and Fable's transcript). Carbon repeats the name once in its answer.

**Moderator checks added after P2 was sent.** Two were run while the reviewers worked, so **no reviewer saw them**
(§6):
- Cholesky-QR variants on the trained snapshots;
- generator norms on the trained snapshots.

They change the PB4/PA1 spec and temper PB1.

## 1. What changed for the reviewers

All four retracted or softened claims that the checks and Part 2 contradicted:

- **Loss predictions are withdrawn by all four** (D5). Neon and Carbon each conceded that "Reviewer Y" (astra) was
  right that no CE number is defensible before the matched BF16 QR10 run.
- **One NS correction (PA6a) is accepted by all four,** since R5 shows identical r_C and r_C^w, e_Q of 2.1–2.2e-3,
  and −22% time. astra ranks it first for speed.
- **The QR tangent is the right first-order target (D1)** for all four; Neon was the convert from full Jacobi, after
  finding a self-inflicted bug in its own row-cap code (w·a·wᵀ squares the damping).
- **V stays permute-only by default (D2)** for all four. T∘T mixing is dropped by its own proposers (astra, Carbon).
- **The "√n cap slack" claims are withdrawn in magnitude** by Fable, Neon and Carbon. Neon and Carbon note that their
  claim was conditional on removing the damping, and that M2 did not test that (§5).
- **Biggest shifts toward a real power step.**
  - **Fable:** "the warm step's deficit is structural, not calibration". PB4 is now its main line, with PB5 as the
    robust variant.
  - **Carbon:** PB4 is promoted to #1 for tracking.
  - **Neon:** moved the other way. Indefiniteness makes Cholesky-QR hard, so PB1 is its #1.
  - **astra:** demoted PA1 to #9 until its reliability on trained factors is shown, and promoted PA6a and PA2.
- **R5's "no-cap survivors beat QR10" is survivorship** (Fable). Their interval r_C (0.036–0.052) is below eigh10's
  (0.133–0.180) on the same trajectory, which is impossible for the full population. astra and Carbon also caution
  against reading it. Moderator: agreed.

## 2. Rankings (1 = best; "nr" = not ranked; bundled ranks shown once)

**A: speed**

| | astra | Fable | Neon | Carbon | mean rank (nr = 12) |
|---|---|---|---|---|---|
| PA1 stabilized Cholesky-QR for QR events | 9 | **1** | **1** | **1** | 3.0 |
| PA6 fewer warm products (PA6a, one NS) | **1** | 2 | 4 | 5 | **3.0** |
| PA9 banks, coalescing, CUDA graphs | 5 | 4 | 6 | 4 | 4.75 |
| PA2 concurrent QR | 2 | 5 | 2 | 10 | 4.75 |
| PA12 overlap with the next forward/backward | 6 | 7 | 5 | 2 | 5.0 |
| PA3 R-only Householder + solve | 13 | 3 | 3 | 9 | 7.0 |
| PA13 size-aware schedule | 8 | 6 | 10 | 3 | 6.75 |
| PA8 syrk kernels | 3 | 8 | 9 | 6 | 6.5 |
| PA11 strided covariance EMA | 10 | nr | 7 | 7 | 9.0 |
| PA10 epilogues, sub-batching | 7 | nr | 8 | 8 | 8.75 |
| PA4 custom blocked Householder | 4 | nr | 11 | 13 | 10.0 |
| PA5, PA7, PA14 | 11–14 | nr | 12–14 | 11–14 | — |

**B: tracking**

| | astra | Fable | Neon | Carbon | mean rank |
|---|---|---|---|---|---|
| PB5 head-panel power step | 3 | 2 | 3 | 2 | 2.5 (second; the only one every reviewer puts at #2 or #3) |
| PB1 QR tangent (+ PB2 bound, PB9 sort) | **1** | 4 | **1** | 3 | **2.25** (best) |
| PB4 Cholesky-QR at every refresh | 9 | **1** | 4 | **1** | 3.75 |
| PB2 spectral / row cap | 2 | (with PB1) | 2 | 4 | — |
| PB7 front-loaded schedule | 6 | 3 | 6 | 6 | 5.25 |
| PB3 full-gain Jacobi | 4 | 6 | 5 | 5 | 5.0 |
| PB6 grouped block QR | 5 | 5 | 8 | 10 | 7.0 |
| PB12 adaptive trigger | 7 | 7 | 9 | 9 | 8.0 |
| PB8 V after refresh | 10 | 8 | 10 | 7 | 8.75 |
| PB9, PB10, PB11 | 8, 12, 11 | nr | 7, 12, 11 | 8, 11, 12 | — |

**Everyone's top-3 experiments.** P = the reviewer's own probability of passing its stated gates.

| Reviewer | E1 | E2 | E3 |
|---|---|---|---|
| astra | PB1 with a **Schatten-4 bound** (Σθ⁴)^¼ vs Frobenius, 8 products; P 0.40 | PA6a one NS, qualified by the max spectral Gram error; P 0.80 | PA2 concurrent Householder, 1/2/4/8 streams; P 0.45 |
| Fable | PB4: ridged, equilibrated CholQR2 every 10 ages, ρ ∈ {1e-4…3e-3}; P 0.40 | PB5: head panel by complete QR of the n×k coordinate panel; P 0.35 | PB1 + spectral bound + I + A + ½A² seed + 1 NS; P 0.25 |
| Neon | PB1 + spectral cap 0.65 + order-2 seed + 1 NS, d_j floored at 0; P 0.65 | PA1/PB4: adaptive-ridge (τ ≥ 2·|min d|) shifted CholQR3; P 0.55 | PA2 + PA3: multi-stream R-only Householder + ridged trsm + NS; P 0.75 |
| Carbon | PB4: hardened sCholQR2 (ridge 2e-4, columns with d ≤ 0 zeroed) every 10; P 0.85 | PB1 + row-ℓ₂ cap 0.5 + spectral cap 0.65 + order-2 seed + 1 NS; P 0.70 | PB5 (top-k Procrustes) + tangent warm, 6 + 2k/n products; P 0.80 |

**Where they converge.** A real power step (PB4 or PB5) appears in all four top-3 lists except astra's. astra puts
the bounded tangent first, and so do Neon and Carbon in one of their three; Fable's tangent comes third. Every
PB4/PA1 spec includes an **operator ridge** on C. §6 shows that this is unnecessary and changes the target.

## 3. Disputes after P2

- **D1, generator.** QR tangent for all four. The open question is whether any bounded first-order step can move far
  enough on trained factors. §6.2 says probably not, and that is now the main issue.
- **D2, V after a refresh.** Permute-only by default (all four).
  - Clamped rescaling (Neon's PB8b, with [0.25, 4] and α) remains a training-only option.
  - Carbon and Fable point out the "slow" side: V stays about 200× too large for about 206 calls in a newly cleaned
    weak direction, which QR10 shares.
  - astra adds a published adverse result for squared-overlap mixing (Purifying Shampoo, App. D.2).
- **D3, why the measured Cholesky-QR failed.**
  - Agreed: the GPU microbenchmark's 0.034 Gram error is not reproduced by M4, and on trained factors the failure is
    near-singularity and indefiniteness, not TF32.
  - Still open: the n ≥ 6144 synthetic NaN mechanism (Fable: possibly BF16-stored C becoming near-singular at large
    n).
  - **New moderator evidence (§6.1):** on trained factors, unshifted CholQR2 fails 39/48 early factor-ages even in
    FP32.
- **D4, engineering headroom.**
  - astra bounds overlap: with the current kernels it hides at most 0.08 × 51.47 + 0.02 × 283.4 = 9.79 ms/call.
  - Carbon still claims 26 → 9–11 ms/call exposed, combined with PA1.
  - Neon and Fable: the warm step is tensor-core-saturated, so only the latency-bound QR overlaps usefully.
  - Unresolved; it needs an in-training timing run.
- **D5, loss.** Settled: no CE prediction until BF16 QR10 exists. Fable adds Milligan et al.'s reported FP32 → BF16
  SOAP penalty (+0.013 / +0.014 in their setting) as a reason the BF16 share of +0.010 may be large.
- **D6, V unchanged at a warm step.** Acceptable for matching QR10 (all four), with the slow-decay caveat above.

**Strongest disagreements** (one each, required):

| Reviewer | Disagrees with | Claim | Settling measurement |
|---|---|---|---|
| astra | Neon (B6b) | Ridged CholQR reproduces QR10 "identically" | Same-input QR-prefix comparison on trained factors. Moderator §6.1 confirms the ridge changes the target |
| Fable | Carbon (B4) | T∘T mixing fixes a "199× spike" | Log the p99.9 of \|M/√V\| on the first call after each refresh vs ordinary calls |
| Neon | Carbon (B3) | One-pass TF32 CholQR + 1 NS gives 0.0018 orthogonality | Run Carbon's verbatim function on trained snapshots. Neon derives σ_min(Y₁) ≈ 0.17 after the 1e-4 Gram shift, which one NS cannot repair |
| Carbon | astra | Blocked Householder (PA4) as #1 speed and PA1 excluded | Microbenchmark of PA1 vs blocked Householder plus a trained-replay chain. astra had in fact ranked PA4 4th and PA1 9th, not "excluded" |

## 4. Prior work (web), with the moderator's citation checks

Verified by the moderator against arXiv/OpenAlex (title, first author, year):
- 2606.02365 (FOAM, Nam & Ahn);
- 2611.16905 (FOAM, Yu et al.);
- 2605.26327 (Milligan et al.);
- 2506.03595 (Purifying Shampoo);
- 1809.11085 / SISC 2020 (shifted CholeskyQR);
- 2610.18695 (refresh placement);
- 2609.24522 (rotation-optimizer convergence);
- 2408.06311 (column-based shifted CholQR);
- 2509.03378 (KL-Shampoo);
- 2410.16103 (LDAdam).

**Citation error.** Neon's arXiv:2604.14816 for "Convergence Rate Analysis of SOAP with Arbitrary Orthogonal
Projection Matrices" resolves to an unrelated NTIRE 2026 video-saliency report. The paper exists as
arXiv:2604.21616.

What the literature means for the pool:
- **Milligan et al. (2026)** store P = QᵀSQ in basis coordinates (this is M7) and do restricted-coordinate QR updates
  of greedily chosen subsets (PB6, and a relative of PB5). **M7 and PB6 are prior art; PB5 is close to it.** They
  also report the FP32 → BF16 penalty for SOAP falling from +0.013/+0.014 to +0.005/+0.006 under their
  reparametrization (Fable read; astra read). That suggests world-frame BF16 C is part of the BF16 loss, and possibly
  of the indefiniteness seen in R5 (Fable's speculation: rounding a dense C gives spectral noise ~1e-4·λ₁ at
  n = 2048).
- **FOAM (Nam & Ahn).** An adaptive refresh trigger for Shampoo driven by a whitened staleness proxy (PB12 is known).
  It also gives an independent argument that a small diagonalization residual can coexist with a large inverse-root
  error (r_C is not the proxy). A warm-started QR needed about 8 iterations to reach residual 0.1 on ViT states
  (3.9× EVD).
- **FOAM (Yu et al.).** One-sided preconditioning plus a factored row/column V. It is excluded by this project's
  coverage rule, but it is the nearest relative of PB8b.
- **Purifying Shampoo (EShampoo).**
  - It uses an adaptive r_C ≤ τ criterion with warm-started QR, so by that standard QR10 (r_C ≈ 0.2) is unconverged.
  - Tight-early / loose-late thresholds beat the reverse, which supports PB7.
  - Its App. D.2 squared-overlap V correction "generally hurt" (astra read Fig. 6). Fable reads the same paper as
    leaving basis-aware correction open. Moderator: both can hold; the specific T∘T correction did not help there.
- **Wang et al. (2026, 2610.18695).** At a fixed refresh budget, placement matters and its sign depends on the LR:
  higher LR favours early refresh. This supports PB7, with the caveat that it must be tested at the target LR.
- **Shifted CholeskyQR (Fukaya et al.).**
  - The Gram shift preserves the target Q; an operator ridge changes it (astra, Fable, Carbon).
  - Fukaya's theoretical shift 11(mn + n(n+1))u‖X‖² is vacuous for square n×n: about 0.78‖X‖² at n = 768 and 12.5 at
    3072 (Fable). **The queued Part 1b script uses exactly this shift.**
- **cuSOLVER and PyTorch QR.** There is no batched orgqr; PyTorch loops over the batch (consistent with batch-1 =
  batch-12 time per factor). No published throughput exists for n = 768–3072 at batch 12, so the PA2 and PA4 gains
  are unverified.
- **Novelty.** None found for a first-order skew generator plus polar/NS retraction replacing SOAP's power-QR step,
  i.e. the current warm step and PB1 (Fable, Neon, Carbon). Search was not exhaustive.

## 5. Reviewer critiques of the moderator checks

| Check | Critique | By | Moderator verdict |
|---|---|---|---|
| M5 | Tested **unclipped** rescaling, not Neon's specified rule (clip to [0.25, 4], floor, α). Under that rule the spike is bounded (≤ 2–4× amplification; at most M/√V = 396 in that row). Row 2's staleness can only arise from non-Kronecker structure when β₂ = β_C | astra, Neon | **Accepted.** M5's rescale column applies to unclipped marginal rescaling only; Fable's B7, which was unclipped, is the rule actually tested |
| M5 | Row 3 hides a slow-down: V about 200× too large decays over about 206 calls | Carbon, Fable | **Accepted** as a real effect, shared with QR10 |
| M2 | Measured the **damped** production generator; the √n claim was conditional on removing the 0.01·s damping | Neon, Carbon, Fable | **Accepted.** Re-tested on trained data with the undamped PB1 generator (§6.2): Frobenius/spectral is 1.1–3.5, so the √n claim fails there too, for a different reason: the generator is huge and concentrated |
| M3 | Mixes cap treatments (PB1 uncapped, PB3a row-capped, production Frobenius-capped); the sorting result is vacuous because PB1 is permutation-equivariant | Neon, Carbon, Fable | **Accepted** |
| M4 | Does not reproduce the GPU's 0.034 or the n ≥ 6144 NaN; column distance is ill-conditioned in near-null clusters; it ran on synthetic, not trained, C | astra, Fable, Neon, Carbon | **Accepted.** Superseded by §6.1, which runs on trained snapshots and also reports top-n/8 subspace distance |
| M1 | The "Jacobi vs QR" row compares different targets, not an error; the ordered tie convention needs stating | Fable, Neon, astra | **Accepted** |
| M7 | Prior art (Milligan et al.); with structured or block rotations the B rotation is cheap; after a Householder QR it costs 3 products | Fable, astra, Neon | **Accepted** |
| M2/M3 | Exact polar retraction cannot show the NS-basin divergence behind R5's no-cap failure | Fable, astra | **Accepted** |

## 6. New moderator checks, run after P2 was sent (not seen by reviewers)

### 6.1 Cholesky-QR on the trained snapshots

`checks/trained_spectra_cholqr.py`; outputs `trained_early_fp32.out` and `trained_early_tf32.out`. CPU, read-only.
d = 256 and 512, ages 10/20/50 (plus 500/1000), all 8 factors. Q is the eigenbasis of the snapshot 10 ages earlier.

- **Early trained factors are indefinite in BF16.** λ_min/λ_max is −1e-4 to −4e-4, and 2–21% of eigenvalues are
  negative. Late factors (d = 256, ages 500 and 1000) are PSD with λ_min/λ_max ≈ 1e-5 to 2e-3.

| variant (Q unchanged as the target unless noted) | early fails, FP32 (of 48) | early / late fails, TF32 Gram emulated | agreement with Householder when it succeeds |
|---|---|---|---|
| unshifted CholQR2 | 39 | 40/48, 2/16 | exact (column distances in this table are means over columns) |
| **Gram-shift CholQR3, s = 1e-6·‖Z‖₂², no operator ridge** (d ≤ 512 only; see §10.3 for d = 1024) | **1** | **2/48, 0/16** | **exact (column distance ≤ 2e-6, top-k distance 0)** |
| Fukaya shift 11(n² + n(n+1))u‖Z‖² (as in Part 1b) | 25 | 30/48, 1/16 | exact |
| operator ridge (C + τ·d_max·I)Q, τ = 1e-6 | 1 | 1/48 | mean column distance up to 3e-2 (top-k up to 4e-5) |
| operator ridge, τ = 1e-5 | 2 | 1/48 | top-k distance up to 3e-3 |
| operator ridge, τ = 1e-4 | 1 | 0/48 | top-k distance up to **0.1** |

**Conclusions.**
- A small **Gram** shift is enough on trained factors. ZᵀZ = QᵀC²Q is PSD even for indefinite C (Fable's point), and
  the early factors are indefinite but not singular.
- The **operator ridge** that all three reviewer PB4 specs carry (Fable ρ 1e-4 to 3e-3; Neon τ ≥ max(3e-4, 2|d_min|);
  Carbon 2e-4) changes even the head subspace. At 1e-4 on early factors, the top-n/8 subspace moves by up to 0.1. A
  ridge can also *cause* failures, when τ lands near a negative eigenvalue.
- **This contradicts** Neon's literature-based claim that "Gram-only shifting fails on R5's rank-deficient factors".
  They are indefinite, not rank-deficient.
- It also shows that Fukaya's theoretical shift (Part 1b) fails on 25–30/48 trained early factors.

`checks/candidate_specs.py` smoke-tests the Gram-shift CholQR3 spec (C1 in CANDIDATES.md) on trained groups with a
basis 50 ages stale. It matches `qr_basis` exactly (top-k distance 0, no fallback).

### 6.2 Generator size on the trained snapshots

`checks/trained_cap_norms.py`, `trained_cap_norms.out`. Same inputs (an exact basis 10 ages stale; input r_C
0.02–0.70).

| generator | ‖A‖₂ | Frobenius/spectral |
|---|---|---|
| production | 0.1–8.6 | — |
| QR tangent (PB1), ρ = 1e-4 | **1–116** | 1.1–3.5 |
| QR tangent (PB1), ρ = 1e-3 | **0.8–71** | 1.1–3.5 |
| Schatten-4 bound | within 25% of spectral | — |

No PB1 entry exceeds the ±1 clamp. The generator is large because the near-degenerate tail has whitened correlations
of order 0.1–1 over large blocks: K_ij ≈ ρ_ij there, and the step is not small.

**Consequences.**
- Any cap of about 0.5–0.65 scales PB1 by about 0.005–0.5 on these factors.
- In the smoke test (basis 50 ages stale), every bounded tangent variant leaves r_C at 0.81–0.89 (n = 768, age 500;
  n = 256, age 20), against 0.44 and 0.087 for QR. Row-ℓ₂ + spectral moves furthest (cap multipliers 0.23–1.0).
- At age 500 with a milder input (r_C 0.096), production warm (0.060) beats every tangent variant (0.076–0.090).
- **The head-panel step (PB5, k = n/8) reaches QR's r_C to within 0.002** in all three cases.

**Prediction from this check** (not pre-registered: Part 2b's results had already been written at 04:44 when this was drafted, though the moderator had not read them; scored in §9): capped PB1 will close well under half of the
warm → QR10 r_C^w gap on trained factors. The row-scaled variant will do best among the bounded tangents. PB4 and
PB5 will dominate on r_C.

This supports Fable's and Carbon's "non-perturbative drift" view (Fable: "a real power step as the main line"), and
lowers the probabilities that Neon (0.65) and Carbon (0.70) gave their bounded-tangent experiments.

## 7. Errors and overclaims caught in P2

1. **Neon's citation ID** for the SOAP convergence paper is wrong (§4). The argument it supports, that refresh before
   the update breaks the conditional independence of Q_t from G_t, is unaffected but unverified.
2. **The operator ridge is not needed, and is harmful** (§6.1). Every reviewer PB4/PA1 spec includes it, and Neon and
   Carbon present it as structurally necessary. Carbon gives P = 0.85 for a spec that also zeros the columns with
   d ≤ 0, which changes the target.
3. **The bounded-tangent pass probabilities** of Neon (0.65) and Carbon (0.70) assume the cap rarely binds. On
   trained factors it binds by a factor of 2–200 (§6.2).
4. **Carbon misstates astra's ranking** in its strongest disagreement: astra ranked PA4 4th and PA1 9th, and did
   not exclude PA1.
5. **Carbon credits Distributed Shampoo with PA11 (strided accumulation) and PA2 (multi-stream factor concurrency)**
   from search results only. Unverified.
6. **Carbon's PA12 claim** ("hides up to ~14 ms/call with zero lag") exceeds astra's bound of 9.79 ms/call at the
   current kernel costs.

## 8. Moderator ranking for CANDIDATES.md

Combining the reviewer ranks, their top-3 lists, the checks, and §6:

1. **C1 = PA1, Gram-shift CholQR3 (no ridge) at QR events.** The largest speed lever (11.8 of 26.1 ms/call). The
   numerics are now supported on trained factors. *(Corrected in §10: ≈ 5.5 ms/call saved, only for n ≤ 3072, with a
   mandatory Householder fallback.)*
2. **C2 = PA6a, one NS correction.** Near-certain, about 1.1 ms/call; only a spectral-Gram check remains.
3. **C3 = PB4, the C1 primitive at every refresh.** QR10 by construction. Its cost relative to today's mix depends on
   2 vs 3 passes; a size-hybrid is the fallback. *(Corrected in §10: ≈ 32.5 ms/call. C1 every 20 ages at ≈ 16.3 ms/call is the
   cost-neutral option.)*
4. **C4 = PB5, head-panel power step.** Second-best mean rank (2.5, after PB1's 2.25); matches QR's r_C in the smoke
   test; cheap if the completion is implicit.
5. **C5 = PB1 + bounded variants.** Already in the refresh-study replay. Adds the Schatten-4 and row-ℓ₂ + spectral
   bounds and logging of the generator size. Expected to disappoint on 3d/4d factors (§6.2); kept as the cheap hedge
   for d/2d factors.
6. **C6 = PB7, front-loaded schedule.** Training run; carries the PB8 diagnostic.
7. **C7 = PA2, concurrent Householder QR.** The exact-arithmetic fallback if C1 fails on GPU.
8. **C8 = PA9, banks, coalescing, CUDA graphs.** The only lever on the 39% ordinary base.

Not carried forward:
- **PA12** overlap: needs an in-training timing harness; next after C8.
- **PA13:** subsumed by C3's hybrid.
- **PA3:** keep as a C7 variant.
- **PB3:** dominated by the QR tangent and by the same size problem.
- **PB6:** prior art; consider only with P-frame storage.
- **PB8:** diagnostic only.
- **PB12:** known; static per-size rules first.
- **PA8, PA10, PA11:** later engineering.

**This ordering departs from the reviewers' mean ranks.** They put PB1 (2.25) ahead of PB5 (2.5) and PB4 (3.75). The
moderator's C3 > C4 > C5 rests on the CPU checks of §6 and §9, which no reviewer saw.

## 9. Addendum: refresh-study Part 2b and Part 1b (read at ~05:00, after §1–8 were drafted)

The refresh study finished two runs while P2 was running:
- **Part 2b:** a trained-replay test of PB1, relative-gap damping and factor-1 Jacobi, written at 04:44
  (`reports/refresh_study_20261120/tables_pb.md`, d = 256–1024; d = 2048 still running).
- **Part 1b:** larger n and a shifted CholeskyQR3, written at 04:40.

### 9.1 Part 2b against the reviewers' and the moderator's expectations

Late-window gap closed (r_C / r_C^w) for warm10/QR50 chains:

| variant | gap closed (r_C / r_C^w) | how often the cap binds (mean multiplier) |
|---|---|---|
| production | 16–22% / 14–19% | 65–72% (0.41–0.47) |
| PB1, Frobenius cap 0.5 | 3–11% / 11–21% | 100% (0.02–0.09) |
| PB1, Frobenius cap 1.0 | 7–20% / 20–35% | 100% (0.04–0.19) |
| **PB1, spectral cap 0.5** | **9–25% / 23–40%** (d = 1024 → 256) | 99–100% (0.06–0.23) |
| relative-gap damping | 0–2% / 2–4% | multiplier ≈ 0.01 |
| factor-1 Jacobi, Frobenius 0.5 | 18–24% / 15–20% | about production |

Without QR repairs (warm10 alone), every variant drifts above QR50 on r_C.

**Scoring the expectations.**
- **astra** (PB1 E1 gate H_w ≥ 0.35 at every width): met only at d = 256 by spectral 0.5 (40%); 29% at d = 512, 23%
  at d = 1024.
- **Neon** (≥ 60% of r_C and r_C^w, P 0.65) and **Carbon** (≥ 50% of r_C^w, P 0.70): **not met** by any tested
  bound.
- **Fable** (E3 midpoint gate, P 0.25): not met.
- **Moderator §6.2** ("well under half of the r_C^w gap"): consistent.
- Row-ℓ₂ + spectral and Schatten-4 are untested in Part 2b; in the moderator's smoke test row-ℓ₂ + spectral was the
  best tangent.

### 9.2 The smoke test rerun with the whitened metric and a 10-age-stale basis

`checks/candidate_specs.out`, as the advisor asked; trained snapshots at d = 256.

| case | QR ρ_rms | head panel (C4) | bounded tangents | production |
|---|---|---|---|---|
| n = 256, age 500 | 0.24 | r_C equals QR's, ρ_rms 0.42 | **0.38** (spectral / row) | 0.45 |
| n = 768 left factor, age 500 | 0.35 | ρ_rms 0.95 | 1.7–2.4 | 2.4 |

- **Shifted CholeskyQR3 (C1)** equals QR on r_C and ρ_rms in all six cases.
- **The head panel** matches QR's first k columns directly (subspace distance ≤ 1.3e-4 with the 10-age basis, 0 with
  the 50-age basis) and matches QR's r_C. Its whitened ρ_rms is clearly worse than QR's, because the tail is not
  power-stepped.
- **Bounded tangents** help the whitened metric on mild factors, not on the large left factors.
- **Reading.** PB5 fixes the head, PB1 helps the tail of mild factors, and only PB4 does both. **Carbon's E3**
  (head panel plus tangent on the rest) is the natural combination to test after C4 and C5 individually.

### 9.3 Part 1b prices C1 and C3

*Superseded by §10.2: the times below are TF32-Gram times. With IEEE Grams the event is ≈ 320 ms and C3 ≈ 32.5 ms/call.*

Shifted CholeskyQR3 (Fukaya shift, TF32 Grams under "high"), batch 12, ms per factor:

| n | CholQR3 | QR |
|---|---|---|
| 768 | 0.569 | 2.537 |
| 1536 | 1.671 | 6.641 |
| 2304 | 4.012 | 11.682 |
| 3072 | 7.978 | 17.804 |

- **The 109M set:** 60 × 0.569 + 12 × (1.671 + 4.012 + 7.978) ≈ **198 ms** for C1, against ≈ 586 ms for QR on the
  same sum.
- **C1 at QR events:** saves about 7.8 ms/call at QR50.
- **C3 (every 10 ages):** about 203 ms per event with transport, i.e. **≈ 20.3 ms/call**. That is ≈ +4.4 ms/call
  (+1.4% of the step) over today's 15.94, but cheaper than QR20's ≈ 30.3 ms/call, and it gives QR10 quality.
- **The reviewers' estimates of PB4's cost** (Fable 9–14.5, Carbon 11–13 ms/call) **were too low**; Neon's 3-pass
  estimate (170–200 ms) was right.
- **Untested, on GPU or trained factors:** the IEEE-Gram and small-shift variants (C1 spec) and the cheaper
  "1 unshifted pass + BF16 NS" variant.
- **Part 1b confirms the Fukaya-shift failure on synthetic inputs too:** every factor at n ≥ 6144, failing in passes
  2–3. This matches §6.1's 25/48 failures on trained early factors.

## 10. Corrections (2026-11-20, ~05:45, after refresh-study's final Part 1b/Part 2 revisions)

**10.1 The per-millisecond figure the reviewers saw was superseded.**
- Every P2 bundle's R5 (the Part 2 report text as of 04:2x) said the warm refreshes buy "1.3–1.9×" the
  per-millisecond r_C reduction of QR20's extra QRs. P2_BRIEF.md §3 told reviewers that "QR20 beats warm10/QR50 per
  refresh, but not per millisecond".
- Refresh-study's corrected, width-matched figures are **2.9 / 2.3 / 0.92 / 0.35×** for d = 256 / 512 / 1024 / 2048
  (≈ 1.4× at 109M sizes). So the brief's statement holds only up to about d = 512, and **reverses at d ≥ 1024**.
- astra's §1 ("tracking per millisecond") and any rank that leaned on "warm wins per ms" were made with the
  superseded figure.
- Under the corrected figure, QR-type schedules (and C1-based ones, 10.2) look relatively better at larger widths.
  This strengthens, rather than weakens, the case for C1/C3 over warm-step tuning.

**10.2 C1/C3 pricing.**
- §9.3 priced C1 at ≈ 198 ms and C3 at ≈ 20.3 ms/call from **TF32-Gram** CholQR3 times. With IEEE-FP32 Grams
  (refresh-study `microbench_ext_ieee.json`), the 109M event is ≈ 320 ms vs 594 for Householder. So:
  - C1 saves ≈ 5.5 ms/call at QR50;
  - C3 costs ≈ 32.5 ms/call, ≈ QR20's Householder cost, and is not cheaper than today's 15.94;
  - C1 every 20 ages costs ≈ 16.3 ms/call, about today's cost, at QR20 quality.
- IEEE CholQR3 is slower than Householder from n ≈ 4096 at batch 12 (1.07× at 4096, 1.93× at 8192) and from
  n ≈ 8192 at batch 1.
- The reviewers' PB4 cost estimates (Fable 9–14.5, Carbon 11–13, Neon 13 ms/call) are all too low. Neon's 3-pass
  event estimate of 170–200 ms is also below the measured 320.

**10.3 C1 failure rate, reconciled with refresh-study's 11/16.**
- On the same 16 age-10 trained factors and the same criterion (IEEE FP32 CPU, Q = production eigh init, any pass
  with `info ≠ 0`), Fukaya's shift fails 9/16 here (11/16 there). A Gram shift of 1e-6 or 1e-7·‖X‖² fails 0/16 and
  matches FP64 Householder to ≤ 1.5e-3 rad, below the BF16 floor. **The disagreement is the shift size.**
- §6.1's "1/48" held only for d ≤ 512 (n ≤ 2048). Across 96 factor-ages including d = 1024 (n up to 4096):
  - 1e-6 fails 4/96 and 1e-7 fails 1/96. All failures are at ages ≤ 20, and most are on the n ≥ 2048 left factors.
    They break at the last pivot of pass 2: the early factors are numerically rank deficient.
  - So a Householder fallback is required.
  - The cheaper "1 unshifted pass + NS" variant is inaccurate (worst column angle 0.72–0.82 rad) and is rejected.
- Files: `checks/reconcile_cholqr3.py`, `checks/reconcile_cholqr3_age10.out`, `checks/reconcile_cholqr3_more.out`,
  `checks/reconcile_cholqr3_d1024.out`.
