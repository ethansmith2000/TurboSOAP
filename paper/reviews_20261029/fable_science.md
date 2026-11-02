# Independent scientific review: TurboSOAP working manuscript (snapshot 2026-10-29)

Reviewer role: skeptical outside scientific reviewer (Claude Fable 5.1).
Reviewed: `paper/reviews_20261029/snapshot/DRAFT.md` (sha256 `56f71b53…`) with its
companion `CLAIMS.md`, `REPRODUCIBILITY.md`, `evidence.json`, `configuration.json`,
`references.bib`, `tables.md`, and both SVG figures.
Scope of this review: contribution and novelty positioning against the three cited
primary papers, what the evidence actually demonstrates, fair reading of the controls
and approximation tradeoffs, and whether the narrative overstates. Read-only CPU
inspection; no code, data, or manuscript was modified. No other 2026-10-29 reviewer
report was consulted. The 2026-10-26 review synthesis was read as historical context.

## 0. What I verified and how

Verified against saved artifacts (all numbers recomputed from the JSON files, not
copied from the draft):

- Quality table (§5.1), timing table (§5.2), controls table (§5.3), profile table
  (§5.4), pilot table (Appendix A): every entry reproduces from
  `reports/bf16_full_horizon_20261026_seed205/comparison.json`,
  `reports/bf16_replicate_20261027_seed204/comparison.json`,
  `reports/bf16_refresh_controls_20261027/comparison.json`,
  `reports/bf16_qr_attribution_20261027/result.json`, and
  `reports/bf16_learning_20261026/comparison.json`. Perplexity percentages, the
  mean paired delta, the call-500 gaps, the "seven of seven / six of seven"
  control statements, the 160/36,864 reordered columns, the 2.68e-5 band, the
  0.0906% / 0.1116% / 0.9999994 probe values, and the 0.9924% / 1.0356% /
  0.002366 / 0.002367 numerical maxima all check.
- Memory arithmetic: the six-tensor element count for the four shape groups times
  12 layers gives exactly 2,491,416,576 bytes in FP32; the BF16 figure differs
  from half of that by 32 bytes (the four groups' scalar age/lr tensors). Peak
  allocation 10,090,324,992 / 8,829,772,800 bytes = 9.397 / 8.223 GiB. The
  109,396,992 parameter count is consistent with the stated architecture (48
  matrices, tied 50,257x768 embedding, two norms per layer plus 64-wide q/k norms,
  final norm).
- Method equations (§3.1–3.3) versus `bf16_soap.py`: `matrix_step` (lines
  77–111), `warm_basis` (35–55), `qr_basis` (58–65), `transport_moment` (68–74),
  `covariance_ema` (27–32), and the event schedule in `step` (line 214). Every
  rounding boundary written in the draft matches the code, including the
  unrounded FP32 moment temporaries used for the direction, the multiplicative
  decoupled decay, the bias-correction age convention, the BF16 output of the
  first-moment transport, and the permutation-only V handling at QR events.
- Reference claim (§3.3, "three dense products"): `soap_reference.py:54–55` and
  the pinned upstream fixture `tests/reference/soap_upstream.py:412–413`
  (`est_eig = diag(o.T @ m @ o)`, then `qr(m @ o[:, order])`). Confirmed.
- Seed-204 replay protocol (§4): `completion_audit.json` records
  `exactly_reproduced_validation_calls: [0, 50, 100, 200]` and
  `cached_reference_call50_100_200_reproduced: true`. Confirmed.
- Primary sources, fetched 2026-10-29 from arXiv: SOAP v2 (2409.11321v2, revised
  31 Jan 2025), Milligan et al. v1 (2605.26327v1, submitted 25 May 2026, marked
  "working in progress"), Eschenhagen et al. v2 (2506.03595v2, revised 29 Oct
  2025). Bibliography versions and dates are correct. Section attributions in §2
  (Milligan §§3.1–3.2; Eschenhagen §4 "Controlling the approximation error
  induced by stale eigenbases") are correct.

Items I could not verify and treat as uncertain: GPU kernel-level behaviour
(e.g., that cuBLAS `baddbmm` with BF16 operands performs one output rounding);
anything about thermal or clock drift between sequential arms; the SVG figures'
plotted values beyond their axis labels and legends.

## 1. Overall assessment

The manuscript is unusually careful about scope. Its stated contribution, "the
specified dense tracking/repair/precision implementation and its reproducible
case study" (§1), is supported by the evidence, and the explicit disclaimers in
the abstract, §5, and §6 are accurate. I found no fabricated or misreported
number. The main scientific weaknesses are interpretive rather than factual:

1. The narrative attaches both the loss gap and the speedup to "BF16 state" and
   "basis tracking," but the manuscript's own saved evidence shows the refresh
   cadence (QR every 50 calls instead of every 10) is the dominant cause of both.
   This decomposition can be stated with existing numbers and currently is not.
2. The method's two numerical ingredients (a damped Jacobi-type rotation and
   Newton–Schulz-style orthogonality correction) are presented without lineage.
3. Several positioning sentences could be made stronger and more honest at the
   same time by quoting what SOAP v2 itself says about low precision and what
   Eschenhagen et al. report about when basis accuracy matters.

Disposition: **minor-to-moderate revision; no new experiments needed.** Details
and recommended wording follow, ordered by severity.

## 2. Findings

Severity scale: **Major** = changes what a reader would conclude; **Moderate** =
needed for fair interpretation; **Minor** = precision/wording; **Note** = optional.

### F1. [Major, interpretive] The time saving is mostly a cadence effect, and the saved per-event timings already show this

**Where:** Abstract; §1 ¶4; §5.2 last paragraph; §6 ¶1; §7.

**Evidence (seed 205, `reports/bf16_full_horizon_20261026_seed205/comparison.json`,
field `optimizer_by_kind`):**

| Arm | Ordinary ms/event | Warm ms/event | QR ms/event | Mean optimizer ms/call | Mean step ms/call |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock QR10 | 34.44 | – | 642.37 | 95.23 | 380.55 |
| Full BF16 warm10/QR50 | 10.14 | 61.61 | 601.18 | 26.08 | 309.53 |

Amortizing by event frequency (90% ordinary; stock 10% QR; candidate 8% warm, 2% QR)
reproduces the recorded optimizer means to 0.01 ms, so the decomposition is exact
for these means:

| Component of the 69.2 ms/call optimizer saving | ms/call | Share |
| --- | ---: | ---: |
| Refresh work: stock amortized QR 64.24 vs candidate warm 4.93 + QR 12.02 | 47.3 | 68% |
| Ordinary step: 34.44 → 10.14 at 90% of calls | 21.9 | 32% |

The optimizer accounts for 69.2 of the 71.0 ms/call step-time saving. The
per-event QR cost is nearly unchanged (642 vs 601 ms, about 6%), which is exactly
what the §5.4 profile predicts (solver dominates). The pilot (Appendix A) tells the
same story from the other side: the original eager FP32 implementation with
warm10/QR50 already saves 11.9% (337.2 vs 382.8 ms); compilation/batching/BF16
arithmetic adds about 5.5 points (316.3 ms); BF16 storage adds about 1 point
(313.2 ms). The QR50 control (no warm events at all) runs at 313.6 ms, within
0.2% of the warm10/QR50 prefix.

**Why it matters:** The title ("Compiled Dense Basis Tracking with BF16 State") and
abstract lead a reader to credit tracking and precision for the 18% speedup. The
evidence says roughly two thirds comes from doing five times fewer QR solves,
something the reference implementation could do with no code change (at a loss
cost the controls partly measure). The draft's sentence "the combined
implementation comparison does not isolate warm tracking as the cause of all time
savings" (§5.2) is true but far weaker than what the data support.

**Recommended text (§5.2, new paragraph after the step-mean paragraph):**
"The recorded per-event optimizer times (seed 205) decompose the 69 ms/call
optimizer saving: about 47 ms/call (68%) comes from replacing an amortized QR
every 10 calls with a QR every 50 calls plus a cheap warm step every 10, and
about 22 ms/call (32%) from the compiled, batched BF16 ordinary step. The QR
event itself costs 601 ms in the candidate versus 642 ms in the reference. These
shares are exact for the recorded means but describe one seed on one device."
Add one sentence to the abstract: "Most of the step-time reduction reflects the
sparser QR cadence rather than BF16 storage, which chiefly halves memory."

### F2. [Major, interpretive] The loss gap is likewise mostly attributable to the refresh policy, not to BF16, on the manuscript's own pilot evidence

**Where:** Abstract ("BF16 ... cross-entropy increases of 0.011"); §5.1; §6 ¶1
("Storage and compute changes also alter numerical trajectories").

**Evidence (`reports/bf16_learning_20261026/comparison.json`, seed 204, 2,000
calls, CE minus stock QR10 at each validation call):**

| Arm | 500 | 1000 | 2000 |
| --- | ---: | ---: | ---: |
| Original warm10/QR50, FP32 state, eager | +0.1183 | +0.0607 | +0.0098 |
| Compiled FP32 storage / BF16 arithmetic | +0.1254 | +0.0617 | +0.0057 |
| Full BF16 (the method) | +0.1239 | +0.0528 | +0.0080 |
| BF16 M/V/C, FP32 bases | +0.1216 | +0.0540 | +0.0097 |

And from the controls (`reports/bf16_refresh_controls_20261027/comparison.json`,
same seed, BF16 throughout): QR20 is +0.059 at call 500 and +0.0045 at 2000
versus stock; QR50 is +0.140 and +0.0128.

All five pilot arms share the same early gap of roughly +0.12 CE at call 500
regardless of precision; the arm that varies cadence (QR20) halves it. So the
early gap, which the draft rightly highlights, is a cadence/warm-policy effect.
At 2,000 calls the precision-attributable share is on the order of +0.002 CE
(full BF16 vs compiled FP32 storage), which is within the spread among the arms.

**Why it matters:** The abstract and §5.1 present the gap as the cost of the BF16
approximation. A reader comparing with Milligan et al.'s Table 1 (SOAP BF16
storage degradation +0.013/+0.014 test loss with the original parametrization,
+0.005/+0.006 with theirs, as fetched from 2605.26327v1) would wrongly conclude
TurboSOAP's +0.010/+0.011 is a comparable precision penalty. It is not; most of
it is the refresh schedule.

**Recommended text (§5.1, after the call-500 sentence):** "The development pilot
(Appendix A) shows a comparable early gap for the FP32-state warm10/QR50 arm,
and the QR20 control (§5.3) roughly halves it, so the early deviation is
attributable primarily to the refresh policy rather than to BF16 storage. At
2,000 calls the full-BF16 and compiled-FP32-storage arms differ by about
0.002 CE. The pilot is single-seed development evidence and does not isolate
precision at the 10,000-call endpoint."

### F3. [Moderate] The warm10/QR50 operating point is not shown to dominate QR20; §5.3 should say so symmetrically

**Where:** §5.3 last sentences; §6 ¶2 ("Short QR20/QR50 controls cannot determine
a final-horizon ranking"); §7.

**Evidence:** At 2,000 calls QR20 has lower CE than warm10/QR50 (by 0.0036) at six
of seven observations and at 500 calls by 0.065; its step time is 329.6 vs
314.1 ms (+4.9%), still 13.4% below the 380.5 ms stock reference. The amortized
QR cost explains the time difference (QR20: 0.05 x 615.7 = 30.8 ms/call; warm:
17.0 ms/call). Given the user's stated priority on loss/efficiency, QR20 is a
live competitor on this evidence, and the draft currently only disclaims the
reverse direction ("does not establish final-horizon sparse-QR superiority").

**Recommended text (§5.3, replace the last sentence):** "This study establishes
neither that QR20 beats warm10/QR50 at the full horizon nor the converse: on
this 2,000-call prefix QR20 reaches lower loss at about 5% more step time, and
the warm step buys 0.0047 CE over QR50 at a step-time cost within measurement
noise. Both orderings remain open at 10,000 calls."

### F4. [Moderate] Method lineage is missing for the two numerical ingredients

**Where:** §2 last paragraph; §3.2.

The warm step `A_ij = -1/2 (B_ij/s) h_ij/(h_ij^2 + 0.01^2)` is a damped
first-order (small-angle) Jacobi rotation, i.e., the leading term of
`tan(2θ) = 2B_ij/(d_j - d_i)` with Tikhonov regularization of the gap. The two
residual corrections `Y <- Y - 1/2 Y (Y^T Y - I)` are the first-order
Newton–Schulz (Björck–Bowie) orthogonalization iteration. Neither is named or
cited. Milligan et al. explicitly frame their block updates against the classical
Jacobi literature (Forsythe & Henrici 1960; Van Loan 1985; Bečka et al. 2002;
Yamamoto et al. 2014, as listed in their reference list), so an informed reader
will notice the omission. I recall Björck & Bowie (1971, SIAM J. Numer. Anal.) as
the standard reference for the orthogonalization iteration and that the same
iteration family is used in recent optimizer work (Muon); please verify those
citations before adding them, as I did not fetch them.

**Recommended text (§3.2, after the damped-motion display):** "Equation (…) is a
damped first-order Jacobi rotation in the current basis coordinates; the residual
corrections are two steps of the Newton–Schulz orthogonalization iteration. The
contribution is the specific dense, batched, BF16-boundary formulation, not these
primitives." Add the corresponding citations to §2.

### F5. [Moderate] Positioning against SOAP v2 can be made both stronger and more precise

**Where:** §2 ¶1; §3.3; §4 "stock".

Verified from the fetched SOAP v2 text:

- SOAP's own basis refresh (Algorithm 4, §4.1) is one warm-started power
  iteration plus QR, with a full eigendecomposition only at initialization. So
  the "stock QR10" reference is already a warm-started basis tracker; the
  candidate differs by inserting a cheaper intermediate step and sparsifying the
  QR. The draft's phrase "periodic QR repair" versus the reference's "QR10"
  obscures that both arms use the same QR primitive family.
- SOAP §7.2.1 names low-precision preconditioners "the most direct way to reduce
  memory," §7.3.1 notes `torch.linalg.qr` does not support below FP32, and §8
  lists lower-precision preconditioners as future work. TurboSOAP's design (BF16
  persistent state with an FP32 QR island) is a direct response to that stated
  gap. Saying so is accurate and helps the reader; the draft currently does not.
- SOAP §7.1 describes one-sided preconditioning; the draft's "no one-sided
  approximation" (§3.4) is consistent and could cite that section.
- The default values cited as "stock" (0.95, 0.95, covariance 0.95) match SOAP
  v2 Appendix A / §5 defaults for the 2M-batch setting, with the caveat that the
  SOAP authors report 0.99 for β2 and β_shampoo in some runs. The draft's
  definition of "stock" as memory settings only is fair. Note the local
  `soap.py` defaults are (0.95, 0.999)/0.999, so readers inspecting the repository
  will see a different "default"; one clarifying sentence in §4 would help.

**Recommended text (§2 ¶1):** "SOAP refreshes bases with one warm-started power
iteration followed by QR (Algorithm 4), performs these operations in FP32 because
`torch.linalg.qr` does not support lower precision (§7.3.1), and lists
lower-precision preconditioner storage as future work (§§7.2.1, 8). Our
reference uses that refresh at period 10; our candidate keeps the FP32 QR
island, sparsifies it to period 50, and stores all persistent matrix state in
BF16."

### F6. [Moderate] Eschenhagen et al. supply a published explanation for the early gap and support for the V-permutation choice; both are unused

**Where:** §2 ¶3; §3.3 (V handling); §5.1; §6.

Verified from the fetched 2506.03595v2 text: (a) their §4 experiments find that
frequent, accurate eigenbasis computation is most valuable early in training
(tight tolerance early / loose late gives markedly better final loss than the
reverse); (b) Appendix D.2 reports that rotating the second-moment statistic into
the new basis "seems to hurt performance overall and especially for low β2," with
no satisfying explanation. Point (a) is exactly the pattern in Figure 1 (right)
and in the QR20 control; point (b) supports the draft's permutation-only V policy
at QR events, which §3.3 currently presents only as an approximation to be
apologized for. Neither use requires a new experiment. (The draft's §2 statement
that their analysis is "context, rather than a directly measured comparator" is
correct and should stay.)

**Recommended text (§5.1 or §6):** "The early-gap pattern matches Eschenhagen et
al.'s finding that basis accuracy matters most early in training (their §4), and
suggests, without testing here, that a denser QR cadence during the first few
hundred calls could remove much of it at modest cost." And in §3.3: "Rotating the
second-moment history into the new basis is not obviously beneficial; Eschenhagen
et al. (Appendix D.2) report it hurt in their setting."

### F7. [Minor, factual precision] "Total GPU allocation therefore falls less than matrix-state storage" is only true in relative terms

**Where:** §3.4 last sentence; §5.2 ¶3.

Peak allocated memory fell by 1.174 GiB (9.397 → 8.223), which is slightly more
than the 1.160 GiB matrix-state reduction, not less. The sentence is true for
the percentage (12.5% vs 50%). Recommend: "Total peak allocation therefore falls
by a smaller fraction (about 12.5%) than matrix-state storage (50%), although the
absolute reductions are similar."

### F8. [Minor] "The gap then narrows substantially" understates the late plateau

**Where:** §5.1 ¶1.

Recomputed paired gaps: seed 204 reaches +0.0072 at call 3000 and then drifts up
to +0.0116 at 5000–6000 and +0.0112 at 10,000; seed 205 reaches +0.0074 at 2000
and ends at +0.0101. The gap does not keep narrowing; it plateaus near +0.01 and
the endpoint is slightly above the minimum. Recommend: "The gap then narrows to a
plateau of roughly +0.007 to +0.012 CE from call 2,000 onward, with the endpoint
near the upper end of that band for both seeds."

### F9. [Minor] Seed-204 timing window sits inside LR warmup and spans two 50-call cycles only

**Where:** §5.2; CLAIMS row "Seed 204 17.36%".

Calls 101–200 contain exactly two QR events and eight warm events for the
candidate and ten QR events for the reference, so the event mix is representative,
and the cached full-run means (380.06 / 314.06 ms) are within 0.2% of the window
means (380.74 / 314.64 ms). The draft handles this correctly. One extra sentence
noting that the window is event-balanced (two complete 50-call cycles) would
pre-empt the obvious objection that 100 calls is too short.

### F10. [Note] Comparator fairness is acknowledged but could be quantified with the pilot

**Where:** §6 ¶3.

The draft says a fully optimized FP32 QR10 comparator was not trained, and the
user has set the original implementation as the primary comparator. Within that
constraint, Appendix A already bounds the implementation-only effect: the
"compiled FP32 storage / BF16 arithmetic" arm at the same cadence is 316.3 ms
versus full BF16 313.2 ms, and the eager FP32 warm arm is 337.2 ms. A reader can
therefore estimate that an eager FP32 QR10 reference with CQ reuse would gain at
most the §5.4 profile's 1–5% of the QR event (about 1–3% of step time at period
10). Stating this bound explicitly is possible with existing numbers.

### F11. [Note] Unaddressed selection question: which arm was the learning rate tuned on?

**Where:** §4 ¶3.

The draft says LR selection used the research validation panel but not whether
the 0.0015 peak was chosen for the reference, the candidate, or both. If tuned on
the reference only, part of the gap could be LR mismatch; if on the candidate,
the reverse. This is a one-sentence disclosure, not a new experiment.

### F12. [Note] Numerical-screen reporting is appropriate

§5.5 correctly reports the 1% M-screen crossing (seed 205) and the Gram-screen
crossings as descriptive diagnostics and does not use them as acceptance
criteria, consistent with the user's steering. The normalized Gram level
(~2.4e-3) is consistent with the BF16 unit roundoff (2^-8 ≈ 3.9e-3) acting on
stored basis entries, which supports the draft's "orthogonality floor" phrasing.
No change needed; optionally add that one-line explanation.

## 3. Checks that passed (no action needed)

- All five tables and every in-text number reproduce from the cited JSON files.
- Event counts: candidate 1 cold / 9000 ordinary / 800 warm / 199 QR per 10k run;
  QR20 99 QR events; QR50 39; cached warm prefix 38 QR / 152 warm / 1710 ordinary.
- Graph captures: 12 unique graphs = 3 event kinds × 4 shape groups, no breaks.
- §3 equations match `bf16_soap.py` at every rounding boundary I checked.
- §3.3 "three dense products" claim matches the pinned upstream SOAP fixture.
- Equal-diagonal counterexample reproduces the saved receipt
  (`reports/review_20261026/equal_diagonal_counterexample.json`) and is scoped
  correctly as algebraic, not observed.
- Bibliography versions, dates, and author lists are correct for all three works.
- Section references into Milligan (§§3.1–3.2) and Eschenhagen (§4) are correct.
- The abstract's final disclaimer sentence is accurate and should be retained.

## 4. Unresolved questions for the authors

1. Which arm's validation loss was the 0.0015 peak LR selected on (F11)?
2. Is there any saved per-event timing for the seed-204 candidate window or the
   pilot's "original warm" arm that would let F1's decomposition be reported on
   two seeds instead of one? (The seed-204 full-run `optimizer_by_kind` exists in
   `reports/bf16_replicate_20261027_seed204/comparison.json`; it shows the same
   pattern: ordinary 10.18, warm 61.62, QR 615.92 ms.)
3. Does the production `baddbmm` BF16 path on the RTX 5090 actually perform a
   single output rounding as §3.1 states? This is a cuBLAS implementation detail
   that I could not verify on CPU; the draft's hedge in §3.1 ("does not promise a
   single GPU kernel") may or may not be meant to cover it.
4. Were the two arms of each seed run back-to-back on the same device without
   any intervening workload, so that clock/thermal state is comparable? The
   campaign records show sequential execution but no clock telemetry.

## 5. Disposition

**Accept the scientific content with minor-to-moderate textual revision.** The
numbers are correct, the scope disclaimers are honest, and the reproducibility
package is exemplary. The one substantive change I would insist on is F1/F2: the
manuscript's own saved evidence shows that the refresh cadence, not BF16 state or
the warm kernel, is the dominant cause of both the speedup and the loss gap, and
the abstract, §5, and §6 should say so with the amortized per-event decomposition
and the pilot/control gap comparison. F3 through F6 are fairness and positioning
improvements achievable with existing evidence and the already-cited sources. No
new experiment, acceptance threshold, or equivalence claim is proposed.
