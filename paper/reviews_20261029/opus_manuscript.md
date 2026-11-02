# Independent manuscript review: TurboSOAP working draft (snapshot 2026-10-29)

Reviewer: Claude Opus 5.5 (review sub-agent, in-project review; not an external referee).
Reviewed: `paper/reviews_20261029/snapshot/DRAFT.md` (sha256 `56f71b53…`), with CLAIMS.md,
REPRODUCIBILITY.md, evidence.json, configuration.json, tables.md, references.bib and both figures.
Verified against: `bf16_soap.py`, `soap_reference.py`, `fast_soap.py`, `benchmarks/bf16_storage_screen.py`,
`benchmarks/owt_restart.py`, and saved scalar reports under `reports/`.
Method: read-only CPU inspection with standard-library Python over the saved JSON. I ran no GPU work,
training, or code changes. Two bibliography lookups (OpenAlex/arXiv, read-only) were used to verify the
suggested citations. I did not open other reviewers' reports from 2026-10-29. I used `reports/review_20261026/`
only as context.

Labels used below. **[F]** factual error or misleading statement. **[O]** essential detail that is missing
but available in existing evidence. **[C]** conceptual clarity or explanation. **[A]** limitation the
draft already acknowledges (I add no penalty, only wording). Severity: **Major**, **Moderate**, **Minor**.
None of the recommendations needs a new experiment.

---

## 1. Overall disposition

**Ready after a focused text revision, as a narrow systems/optimizer case study.** Every number I checked
matches the saved evidence (§6). The draft is unusually careful about scope. Its main weakness is
*interpretation*, not correctness. The draft treats the result as "a BF16-state approximation with a loss
gap". The existing evidence supports a sharper and more useful account:

1. **The loss gap goes with the refresh schedule (warm10/QR50 vs QR10), not with BF16 storage.**
   An FP32-state warm arm in the pilot shows the same early gap and a similar call-2000 deficit.
2. **Most of the time saving also comes from the schedule.** FP32 QR events cost about 600–650 ms
   in *both* implementations. Doing them 5× less often accounts for about 75% of the optimizer-time saving.
   Compilation, batching and BF16 make the ordinary step about 3.4× cheaper, which accounts for the rest.
   Warm events give back about 7%.
3. **BF16 storage is close to free here.** It halves matrix state and is within the warm-family spread on
   loss (single seed, 2k calls).

So the paper's real tradeoff is "fewer expensive refreshes, partly compensated by cheap tracking", with
BF16 contributing memory. Stating this openly makes the contribution clearer and harder to misread. It uses
only saved scalars. The draft has the needed numbers but leaves them in Appendix A or omits them.

**Strongest defensible contribution (suggested wording):**
> A fully specified, compiled, BF16-state dense SOAP implementation whose ordinary step is ~3.4× cheaper
> than the original eager implementation and whose persistent matrix state is halved. Combined with a
> warm10/QR50 refresh schedule, it cuts optimizer time 3.65× and step time ~18.7% on one 109M-parameter LM,
> at a ~1% perplexity cost over 10k calls. Existing development and short-control evidence associate that
> cost mainly with the reduced QR cadence rather than with BF16 storage. Cheap warm tracking recovers part,
> not all, of the cadence loss.

---

## 2. Major findings

### M1 [O/C] The loss gap should be attributed, descriptively, to the refresh schedule (Abstract, §1, §5.1, §6, App. A)

*Evidence.* At call 500 (seed 204, identical init/data, `reports/bf16_learning_20261026/comparison.json`
and `reports/bf16_refresh_controls_20261027/comparison.json`):

| Arm (seed 204) | State | CE @500 | Δ vs QR10 @500 | CE @2000 | Δ vs QR10 @2000 |
|---|---|---:|---:|---:|---:|
| Stock QR10 (original) | FP32 | 4.87760 | — | 3.98591 | — |
| BF16 QR20 (control) | BF16 | 4.93685 | +0.059 | 3.99038 | +0.0045 |
| Original warm10/QR50 | FP32 | 4.99595 | +0.118 | 3.99575 | +0.0098 |
| Full BF16 warm10/QR50 | BF16 | 5.00151 | +0.124 | 3.99394 | +0.0080 |
| BF16 M/V/C, FP32 Q | mixed | 4.99919 | +0.122 | 3.99561 | +0.0097 |
| Compiled FP32 storage | FP32 | 5.00300 | +0.125 | 3.99156 | +0.0057 |
| BF16 QR50 (control) | BF16 | 5.01805 | +0.140 | 3.99868 | +0.0128 |

At call 500 the early gap tracks the full-refresh frequency (QR10 < QR20 < warm/QR50 variants < QR50).
At fixed schedule it is essentially the same whether state is FP32 or BF16. At call 2000 the full-BF16 arm
is *ahead of* the FP32-state original warm arm. (The stock QR10 pilot arm is identical through call 2000 to
the cached seed-204 full reference prefix, e.g. 4.87760 @500 and 3.98591 @2000 in both.)

*Problem.* The title, abstract and §5.1 put "BF16 state" first and report the gap without attribution, so
readers will assume BF16 causes it. The draft has the evidence to say otherwise (descriptively) but leaves
it in Appendix A, framed only as "within-family" context.

*Recommended text (§5.1, after the call-500 sentence):*
> The development pilot (Appendix A; same seed, initialization and data) shows a gap of the same size at
> call 500 for an FP32-state warm10/QR50 arm (+0.118 CE) and a larger one for BF16 QR50 without warm events
> (+0.140), against +0.059 for BF16 QR20. Within this single-seed, 2,000-call evidence, the early gap is
> associated with the frequency of full QR refreshes rather than with BF16 storage. This is a descriptive
> association, not a factorial attribution.

*Abstract:* add one clause, e.g. "…a development pilot associates the gap mainly with the reduced QR
cadence rather than BF16 storage." Consider retitling so the schedule appears, e.g. "TurboSOAP: compiled
BF16-state SOAP with sparse QR refresh and dense warm tracking".

### M2 [O] Optimizer-only time and per-event accounting are missing from the main text (§5.2, Table timing)

*Evidence.* Seed 205 (`reports/bf16_full_horizon_20261026_seed205/comparison.json`,
`results[*].mean_optimizer_ms` and `optimizer_by_kind`; clean window calls 101–10000):

- Optimizer ms (matrix + auxiliary step, as timed in `bf16_storage_screen.py`): **95.23 → 26.08**
  (−72.6%, 3.65×). Non-optimizer (forward/backward) time: 285.3 vs 283.5 ms, so essentially all of the
  71.0 ms step saving is optimizer time.
- Per-event means: reference ordinary 34.44 ms, QR 642.37 ms. Candidate ordinary 10.14, warm 61.61,
  QR 601.18 ms.
- Exact accounting of the means (event frequencies over complete cycles: reference 9 ordinary + 1 QR per 10;
  candidate 45 ordinary + 4 warm + 1 QR per 50):
  - Reference: 0.9×34.44 + 0.1×642.37 = 31.00 + 64.24 = 95.23 ms.
  - Candidate: 0.9×10.14 + 0.08×61.61 + 0.02×601.18 = 9.12 + 4.93 + 12.02 = 26.08 ms.
  - Saving 69.15 ms = **QR amortization 52.2 (≈75%)** + ordinary step 21.9 (≈32%) − warm cost 4.9 (≈−7%).
- QR events cost about the same in both implementations because the FP32 solver dominates. The §5.4
  batch-12 CQ-reuse primitive times, summed over the eight factor groups actually present
  (2304, 768 | 768, 768 | 3072, 768 | 768, 1536), give ≈592 ms. That matches the 601–616 ms measured QR
  events. This is a good internal consistency check that the draft does not use.

*Problem.* §5.2 reports only complete-step percentages, which depend on this model's forward/backward cost
and do not transfer to other workloads. The draft says the comparison "does not isolate warm tracking as the
cause of all time savings". That is true but understated: the accounting shows the cadence is the dominant
lever and warm events are a net *cost*.

*Recommendation.* Add optimizer-ms columns to Table timing (seed 205: 95.23/26.08; seed 204 prefix
values if paired, otherwise mark them cross-launch). Add a short paragraph or table with the event
accounting above, labeled "accounting identity over measured means, not a causal ablation". Suggested
sentence:
> Because FP32 QR events cost ~600–650 ms in both implementations, about three quarters of the optimizer-time
> reduction comes from performing them every 50 rather than every 10 calls. The compiled, batched BF16
> ordinary step (34.4 → 10.1 ms) supplies most of the rest, and warm events add back ~4.9 ms per call.

### M3 [F, by omission] "The gap then narrows substantially" misdescribes the late curve (§5.1, Fig. 1)

*Evidence.* Paired CE gaps from the saved validation arrays:

| Call | 500 | 1000 | 1500 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 | 10000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Seed 204 | .1239 | .0528 | .0140 | .0080 | .0072 | .0100 | .0116 | .0116 | .0088 | .0103 | .0104 | .0112 |
| Seed 205 | .1184 | .0447 | .0123 | .0074 | .0081 | .0097 | .0085 | .0090 | .0091 | .0088 | .0093 | .0101 |

The gap narrows sharply from 500 to about 2000. After that it stays in a band of about 0.007–0.012 CE and
ends near the top of that band in both seeds. It does not continue to close.

*Recommended text:*
> The gap narrows sharply between calls 500 and 2,000, to 0.007–0.008 CE. Thereafter it remains between
> about 0.007 and 0.012 CE and ends at 0.010–0.011. The observations show no continued late closure, so we
> do not extrapolate the endpoint gap to longer horizons.

### M4 [C] The warm step needs a conceptual derivation, and its suppressed-motion regime is broader than the single counterexample (§3.2, §3.3, §5.5, §6)

*What the code computes* (`bf16_soap.warm_basis`; `fast_soap.fast_warm_refresh`). Write B = QᵀCQ = D + O,
where D is the diagonal and O the off-diagonal part. For Q⁺ = Q(I + A) with skew A,
(Q⁺)ᵀCQ⁺ ≈ B + BA − AB, so the (i,j) off-diagonal becomes b_ij + (d_i − d_j)A_ij. The first-order (Jacobi /
eigenvector-perturbation) choice is A_ij = −b_ij/(d_i − d_j). The implemented
A_ij = −½(b_ij/s)·h_ij/(h_ij² + δ²) is therefore:
- a **half** first-order step;
- with **Tikhonov-damped inverse gaps**, h/(h²+δ²) in place of 1/h, which bounds |A_ij| ≤ |b_ij|/(4δs) = 25|b_ij|/s;
- followed by a **global cap** on ‖A‖_F/√2, the ℓ₂ norm of the independent rotation angles, at 0.5;
- an additive first-order retraction Y₀ = Q(I + aA);
- and two **Björck–Bowie / Newton–Schulz orthonormalization** steps, Y ← Y − ½Y(YᵀY − I) = ½Y(3I − YᵀY).

The draft gives the formulas but not this interpretation. It describes the corrections only as "an
approximate orthogonality correction". Readers need the interpretation to judge the constants.

*Analytic consequence the draft does not state.* δ = 0.01 is measured relative to s = max|d_i|. For any
pair with |d_i − d_j| ≪ 0.01·s, A_ij ≈ −½ b_ij (d_i − d_j)/(s δ)². That is close to zero, and smaller the
closer the pair. So the equal-diagonal stall in §3.3 is not an isolated curiosity. It is the limit of a
broad regime in which directions whose eigenvalues lie below ~1% of the factor's largest Rayleigh quotient
barely rotate between QR repairs. Warm tracking then mostly follows the leading part of each spectrum. How
many pairs fall in this regime is **not measured**: spectra are not in the saved evidence, so it remains an
unresolved question (§8). The statement is algebraic, not an incidence claim.

*Recommended text (§3.2, after the formulas):*
> Interpretation. For B = D + O, the rotation that cancels O to first order is A_ij = −b_ij/(d_i − d_j). The
> warm step takes half of this step, replaces 1/h by the damped h/(h²+δ²), caps the total rotation, and
> re-orthonormalizes with two Björck/Newton–Schulz iterations [Björck & Bowie 1971]. Because the damping is
> relative to the largest diagonal entry, pairs whose gap is far below 1% of it receive almost no motion.
> Such directions are effectively updated only by QR repair. The equal-diagonal example in §3.3 is the
> limiting case.

Also state in §3.3 that "QR repair" is **one step of orthogonal (subspace) power iteration** from the stored
basis, the same primitive as the reference's QR10 refresh. It is not an exact eigendecomposition. The
reference also does this, only 5× more often.

---

## 3. Moderate findings

### Mo1 [O] Omitted diagonalization diagnostic (§5.5)
`runs/*/geometry.json` (`coordinate_residuals`, defined in `benchmarks/bf16_storage_screen.py:62–73`)
records ‖offdiag(QᵀCQ)‖_F / ‖QᵀCQ‖_F in FP32 "highest", measured after the step. This is the most direct
measure of what tracking is meant to achieve, yet the manuscript says only that Gram/M "neither alone
certifies diagonalization". Seed 205:
- Post-QR observations (ages 500–7500): candidate mean 0.22–0.34 vs QR10 0.19–0.25.
- Four warm steps after a QR (ages 1990 and 9990): candidate mean 0.39/0.38, max 0.96/0.98, vs QR10
  0.20/0.21, max 0.64/0.66.
- The pilot review reports the same pattern for the FP32-state warm arm.

Report this descriptively. It matches M1 and M4: between repairs the warm basis diagonalizes the covariance
less well than QR10. Even QR10 retains ~0.2, consistent with one power step not being exact. Note that the
Frobenius weighting is dominated by the leading block, so this metric is largely blind to tail
misalignment.

### Mo2 [F/C] "Screen crossings" are a BF16 representation floor, and their definitions are missing (§4, §5.5)
- Definitions (code): absolute Gram = ‖QᵀQ − I‖_F; normalized = absolute/√d; physical-M error =
  ‖Q_L⁺M⁺Q_R⁺ᵀ − Q_L M̂ Q_Rᵀ‖_F / ‖Q_L M̂ Q_Rᵀ‖_F. The screen values (absolute Gram ≤ 0.05, normalized
  ≤ 0.005, M ≤ 1%; `bf16_storage_screen.py` result block) are not stated in the manuscript, which still
  speaks of "crossings".
- An orthogonal FP32 basis rounded to BF16 has normalized Gram ≈ 0.00236 (seed-205 QR10 run,
  `fresh_bf16_rounded_qr_normalized_gram` at call 1). That implies absolute ≈ 0.065 at d = 768 and ≈ 0.131
  at d = 3072. **No BF16-stored basis with d ≳ 450 can pass the 0.05 absolute screen.** The observed
  maximum of 0.002367 occurs at post-QR observations, i.e. at the rounding floor. After warm steps the value
  is about 0.0018, because the Newton–Schulz steps act after rounding.
- The 1% M crossing (seed 205, 1.0356%) happens only at age 10, the first warm refresh after cold start.
  All later samples are 0.43–0.54%, against 0.06–0.07% for the reference.

*Recommendation:* define the metrics in §4. Replace the "crossing" sentences with the facts above.
Move screen history to the reproducibility notes. This fits the user's position that these are diagnostics,
not acceptance criteria.

### Mo3 [O] Reference arithmetic and global numerics flags are not stated (§4, §5.4, REPRODUCIBILITY)
The harness sets `torch.set_float32_matmul_precision('high')` (`bf16_storage_screen.py:186`;
`owt_restart.py:149`), `torch.use_deterministic_algorithms(True)` (`:185`) and
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False` (`:187`). So:
- The "FP32 state" reference uses **TF32** tensor-core matmuls for projections, covariance and products.
  The decompositions are FP32. "FP32" should be qualified, and the §5.4 phrase "FP32 high matmul precision"
  should read "TF32 (`'high'`) matmul precision".
- The claim of "FP32 accumulation" for BF16 GEMMs (§3.1) depends on disabling reduced-precision reduction.
  The REPRODUCIBILITY entry-point snippet should list all three global settings, since users of
  `CompiledSOAP` do not get them automatically.
- Deterministic algorithms are on in both arms. State this because it can affect absolute timings.

### Mo4 [O] Timing protocol details that affect magnitude (§5.2)
The step time is the sum of host wall time for forward/backward and for `opt.step()`, with
`torch.cuda.synchronize()` at each phase boundary (`bf16_storage_screen.py:258–269`). This prevents CPU
launch overhead from overlapping preceding GPU work. That likely penalizes the eager, launch-heavy reference
more than the compiled candidate. The size of this effect is unmeasured. Say so in one sentence.
Cross-launch variability is also available and worth one sentence. The identical warm prefix measured
313.162 ms (pilot) and 314.142 ms (replication). Reference means across launches range 380.06–382.77 ms.
This gives a ≈1 ms scale for interpreting the 0.59 ms warm-vs-QR50 difference in §5.3, which the draft
already calls contextual.

### Mo5 [C] State the efficiency question explicitly as a break-even (§5.2, §6)
The draft rightly disclaims time-to-quality. For an efficiency paper, the open question can still be stated
precisely without new runs. For stock QR10 to match the candidate's step-time cost while reaching its final
CE, it would have to reach that CE within ≈8,134 calls (seed 205: 1 − 0.1866), i.e. under its own
appropriately shortened schedule. The 10k-schedule reference curve passes the candidate's final CE only at
about call 9,350 (seed 204, 500-call grid) and about 9,480 (seed 205, coarse 1000-call grid). A reference
scheduled to end earlier would likely do better at its end, so these crossings do not decide the question
either way.

*Suggested sentence (§6):*
> Whether the 18.7% step-time saving outweighs the ~0.01 CE gap depends on whether stock QR10 with a schedule
> of about 8,100 calls would reach 3.42 CE. Our data do not answer this.

I advise against a wall-clock-axis loss plot unless it carries this schedule caveat.

### Mo6 [O] Controls table should include the stock reference and call-500 values (§5.3, Fig. 2 left)
Add a row "Stock QR10, original FP32/TF32 (cached seed-204 prefix)" with CE 3.985905 @2000 and
4.87760 @500, step 382.773 ms and optimizer 96.36 ms from the pilot (labeled cross-launch). Add a CE@500
column (QR20 4.93685, warm 5.00151, QR50 5.01805). The table then reads as a frontier. At call 2000, warm
recovers about 37% of the QR50-to-QR10 gap (0.00474 of 0.01277) for +3.55 ms optimizer time. At call 500 it
recovers about 12%. Label this as a mixed-implementation comparison, because QR10 is the original
implementation.

---

## 4. Minor findings

- **m1 [C] §3.4/§5.2 memory composition.** 1,245,708,320 B is exactly half of 2,491,416,576 B
  (= 1,245,708,288) plus 32 B of FP32 age/LR scalars (4 groups × 2 × 4 B). A footnote avoids confusion.
  Also note that factors and bases are 77% of matrix state (m² + n² terms) and moments 23%. The bases alone
  are 0.896 GiB in FP32, which explains the 1.608 GiB FP32-basis arm.
- **m2 [O] §3.3/§4 event counts.** State that over 9,999 matrix updates the candidate performs 999
  refreshes (199 QR, 800 warm) and the reference 999 QR. Also state that matrix updates number 9,999 while
  AdamW updates number 10,000 (configuration.json records this).
- **m3 [O] §3.4 "Compilation preserves the explicit BF16 rounding policy."** Cite the evidence:
  `runs/bf16/cuda_parity.json` shows bitwise eager/compiled equality for ordinary, warm and QR events on a
  small synthetic CUDA state. Also mention `emulate_precision_casts=True`. Keep the claim scoped to that probe.
- **m4 [C] Terminology.** "Refresh", "repair", "QR refinement" and "hard reset" (code) are used for related
  events. Define "refresh" = any basis change at age ≡ 0 mod 10 and "QR repair" = the age ≡ 0 mod 50 case,
  once, in §3.3.
- **m5 [C] §5.4 placement.** CQ reuse saves 0.95–4.63% of one primitive and does not affect any headline
  result. Consider shortening §5.4 to a paragraph and moving the table to an appendix. Its most useful role
  is the M2 consistency check (≈592 ms predicted vs ≈601–616 ms measured QR events).
- **m6 [C] Related work.** Add:
  - Wang, Zhou, Li & Huang, *4-bit Shampoo for Memory-Efficient Network Training*, arXiv:2405.18144
    (2024; verified via arXiv). As I recall, it quantizes preconditioner eigenvector matrices and rectifies
    orthogonality with Björck orthonormalization, the closest prior for "low-precision basis + orthogonality
    correction". The authors should confirm the details against the paper before citing.
  - Björck & Bowie, SIAM J. Numer. Anal. 8(2), 1971, doi:10.1137/0708036 (verified via OpenAlex), for the
    correction iteration.
  - Mention the classical Jacobi eigenvalue method as the origin of the gap-based rotation. No novelty
    claim on these components is needed, and none is made.
- **m7 [A] Limitations already acknowledged.** These need no change beyond wording: two seeds, fixed data
  order, validation panel reused in development, no statistical equivalence, no efficient-FP32-QR10 or
  subspace/adaptive comparator, synthetic profile, no first-compile or time-to-quality accounting. §6 could
  be shortened by grouping them into one list.

---

## 5. Figure and table changes (existing evidence only)

1. **Fig. 1 left** carries almost no information: a 0.01 CE gap is invisible on a 3.4–7.2 axis. Either
   drop it or plot calls ≥1000 with a tight axis. In the right panel, add a light band or annotation for the
   0.007–0.012 plateau after call 2000 (M3).
2. **Replace Fig. 2 right** (CQ-reuse bars) with stacked bars of **amortized optimizer ms per call by event
   kind** (ordinary / warm / QR) for: stock QR10, original warm (FP32), compiled FP32 storage, full BF16,
   BF16 QR20 and BF16 QR50. All values are in `optimizer_by_kind` and the cadences. This one figure carries
   M2 and the controls tradeoff.
3. **Fig. 2 left:** add the stock QR10 prefix (Mo6) as a reference curve and keep warm as zero, or plot all
   arms relative to stock QR10 so that the early gap is shown on the same axis as Fig. 1.
4. **Table timing:** add optimizer-ms columns. **Table controls:** add the stock row and a CE@500 column.
   **App. A table:** add optimizer ms and CE@500 columns (values in §2/M1 above).

---

## 6. Verified statements (no change needed)

- Endpoint CEs, deltas (0.011237 / 0.010082), perplexity increases (exp(Δ)−1 = 1.1301% / 1.0133%) and mean
  0.0106598 (evidence.json, both comparison.json files).
- Timing 380.740/314.638 → 17.36% and 380.548/309.533 → 18.66%. State bytes; peak 9.397 → 8.223 GiB.
- Controls: CE values, +0.003557 / −0.004740 differences, "all seven" vs QR50 and "six of seven" for QR20
  (checked per evaluation).
- Profile: 0.95–4.63%; 18.165 → 7.724 ms products and 207.613 ms solver at d = 3072; 160/36,864 reorders;
  2.68×10⁻⁵ band; 0.0906% / 0.1116%; cosine ≥ 0.9999994 (`bf16_qr_attribution_20261027/result.json`).
- §5.5 maxima: 0.9924% / 1.0356%; 0.0023659 / 0.0023671.
- §3.1–3.3 formulas match `bf16_soap.py` line by line: BF16 projections, FP32 unrounded moment temporaries
  for the direction, the F-output back-projection, bias correction √(1−β₂ᵗ)/(1−β₁ᵗ) at t = k−1, decay applied
  multiplicatively after the update (the same as `soap_reference.py`), one-rounding `baddbmm` covariance,
  refresh after direction using the updated covariance, single-product QR scoring, two-axis V permutation,
  M-only transport, and cold eigh with no matrix update.
- Pilot step ratio 0.99%; 12 graphs (3 events × 4 shapes); 48 matrices / 96 factors, no blocking
  (reference `max_precond_dim=10000` exceeds 3072).

I found no outright numerical errors. The [F] items are misleading-by-omission descriptions (M3, Mo2).

---

## 7. Prioritized edit list

1. M1: attribute the gap descriptively to the cadence (abstract, §5.1, possibly the title).
2. M2: optimizer-ms columns and the event accounting (§5.2).
3. M3: correct the "narrows" description.
4. M4: conceptual derivation of the warm step and its suppressed-motion regime; QR repair = one power step.
5. Mo2/Mo1: define the diagnostics, retire the "crossing" language, add the coordinate off-diagonal values.
6. Mo3/Mo4: TF32, determinism and BF16-reduction flags; synchronized timing protocol.
7. Mo5/Mo6: explicit break-even sentence; stock row and CE@500 in the controls table.
8. Figures per §5; minor items.

---

## 8. Unresolved questions for the authors

1. How much of each factor's spectrum lies below δ·s (1% of the top Rayleigh quotient)? The answer decides
   how much warm tracking can do between repairs (M4). This is not in the saved evidence. If it cannot be
   answered from existing artifacts, state it as unknown.
2. Does the 1.04% M error at age 10 come from the cold-start basis (built from one gradient's covariance)
   being far from the age-10 covariance? Could the early gap be partly a cold-start effect, given that QR20
   is also +0.050 at call 50? This is answerable only descriptively from the existing curves. Do not claim a
   cause.
3. Are the seed-204 cross-launch full-run means (reference 380.06 ms, candidate 314.06 ms, ≈17.4%) worth
   stating as contextual corroboration of the 9,900-call seed-205 figure? They are not paired, so they would
   need that label.
4. Is "TurboSOAP" the right name, given that most of the speed comes from cadence and compilation rather than
   a new algorithmic kernel? This is an editorial choice. The title should at least not suggest that BF16
   is the source of the speedup.
5. For the reproducibility guide: should `CompiledSOAP` assert or document the global matmul flags
   (Mo3)? This is a documentation-only suggestion. No production code change is proposed here.
