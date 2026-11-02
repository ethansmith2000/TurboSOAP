# Pre-freeze review synthesis — 2026-10-26

The user requested independent Claude/GPT brainstorming and investigation of
the two related papers. Claude Opus5.5 and Sonnet5.5 both failed before doing
work with only "Provider turn failed". They produced no scientific review.
The completed independent reviews are GPT6.1-sol
([methodology/novelty](gpt_review.md)) and GPT6-astra
([math/optimization](gpt_math_review.md)); the parent independently checked
papers, code, timing decomposition and one CPU counterexample
([parent findings](parent_findings.md)). Exact task identities are retained in
[review_tasks.json](review_tasks.json). No subagent edited production code or
ran GPU work. No new GPU experiment was launched by this review.

## Decision

Preserve the tested mathematical v1 and its results as an immutable reference.
Do not finalize its performance headline against an assumed optimal QR baseline.
One bounded implementation audit is worthwhile before spending the confirmation
seed budget. If one last policy improvement is desired, a single predeclared
initial QR phase is the best-motivated candidate. Broader redesign is optional
future work, not an obligation before writing a focused paper.

## Findings agreed by both reviewers and checked by the parent

1. The completed result is supported: stock warm reduces the recorded clean
   complete-step mean10.81%, with +.006976 endpoint CE on one training seed.
   Source/input hashes, event counts and timing aggregation are consistent.
   This is a controlled speed/quality tradeoff, not statistical equivalence.
2. The QR branch has avoidable dense products. It computes a full QᵀCQ for its
   diagonal and separately recomputes CQ for sorted columns. Let Z=CQ; the
   diagonal is columnwise sum(Q*Z), and QR can consume Z[:,order]. This changes
   three cubic products to one plus a quadratic contraction/gather in exact
   mathematics. Numerical ordering and trajectories may change; no GPU speed
   estimate has been measured. Relevant source: `soap_reference.py:54–55`.
3. The idea already exists in `benchmarks/qr_reuse_profile_kernel.py:qr_basis`
   and a related batched primitive. It is not a newly discovered algorithm.
   The older complete-refresh hook requires different transport policy, and
   the batched helper introduces a ridge. Adapt only the applicable algebra,
   qualify the CURRENT recipe, and apply it to both QR10 and warm repairs.
4. Passing Gram guards does not certify diagonalization, adaptive-update
   fidelity or late physical-M preservation. Current actual-M observations
   are at ages10/50 only. Future confirmations should add a few late warm and
   repair observations plus same-C tracking diagnostics.
5. Warm motion alone has an equal-diagonal stall: Q=I and
   C=[[1,.5],[.5,1]] produce zero generator despite offdiagonal residual .4472.
   The parent verified the actual function on CPU; receipt is
   `equal_diagonal_counterexample.json`. This is a counterexample to general
   diagonalization, not proof it harmed the completed learner. QR repair is
   substantive recovery, not merely numerical housekeeping.
6. Transpose-based first-moment transport is approximate with nonorthogonal
   factors, and a diagonal V cannot retain exact rotated history. The tested
   policy is explicit and plausible, but these limitations must remain in the
   paper. Overlap transport is not an automatic correction or new novelty.

## Prioritized optimization choices

### First: efficient QR and bounded precision attribution

Use current resident nonzero states and actual factor shapes for a small
original/reuse QR audit, plus selected/all-FP32 warm operand attribution and
only the eager/compiled comparisons needed to explain cost. Validate sorting,
near-ties, Gram, physical M, following update and short trajectories. Include
stacking/layout/state traffic and complete cadence costs. No profiled gain is
assumed. A materially changed implementation needs a new version and qualified
learning comparison; do not pool it with old timing as if the implementation
were identical. This audit is proposed, not run.

### Second: one early accurate phase, if changing the policy

The observed stock CE deficit is +.118346 at500, +.060651 at1000, +.009847 at2000
and +.006976 at10000. One QR10-only phase followed by warm10/QR50 is a targeted
hypothesis. The parent favors predeclaring1000 calls rather than searching a
family of switch points. The methodology reviewer suggested500 as another
plausible example; this is an unresolved design choice, not two approved trials.

Simple arithmetic using old average costs estimates that first1000 QR calls
would add~41seconds and retain~9.73% step saving versus the original QR control.
This is NOT measured candidate timing or evidence that quality improves; see
`parent_timing_analysis.json` for assumptions. An improved QR baseline changes
the estimate. This is a new policy requiring learning validation.

### Third: common fusion/batching only where a profile supports it

Common elementwise moment/denominator/update fusion, covariance accumulation
without temporary outer-product materialization, and repeated-shape batching
are credible engineering candidates. Apply shared optimizations to both arms,
retain independent per-factor caps and V permutations, and count packing and
memory. They are performance hypotheses, not free numerical equivalence.

The stock candidate's warm-only event surcharge accounts for about2.31% of its
complete-step time, conditional on all other costs remaining fixed. Even
eliminating that surcharge entirely would have limited end-to-end benefit on
this setup. This makes an unlimited retraction/cast/kernel grid low priority.

### Later: residual-driven repair or a coordinate/subspace version

Existing warm B can support a residual observation at scheduled refreshes.
An adaptive policy must count formation of B when otherwise absent, reductions,
per-factor host synchronization, and any warm work discarded before QR. The
criterion is not a learning objective or an error certificate for approximately
orthogonal factors. Investigate near-gap coupling before designing a stall
rescue; do not globally increase step size to address an unmeasured issue.

Maintaining P, partial rotations, BF16 persistent storage, new V transport,
logical projection splits or one-sided coverage are larger algorithmic changes.
Their exact equivalences often assume orthogonal factors. They belong in a
separate version rather than a silent optimization of tested v1.

## Close comparison with the requested papers

[Milligan et al., 2605.26327v1](https://arxiv.org/html/2605.26327v1), §§3.1–3.2,
store and evolve P=QᵀCQ and rotate selected basis blocks using small QR
factorizations. The paper studies persistent BF16 storage; its SOAP storage
experiment and predominantly KL-Shampoo subspace experiments are distinct.
Appendix C already suggests cheap subspace updates with occasional full updates.
Our code retains world C/FP32 state and performs dense damped full-basis motion
with selected BF16 operand products and two approximate corrections. This is a
specific distinction, not a head-to-head win or priority proof. Their P state
also makes some residual/block-selection work cheap; ours does not maintain it.

[Eschenhagen et al., 2506.03595v2](https://arxiv.org/html/2506.03595v2), §4,
uses relative offdiagonal factor error to schedule/refine basis computation.
Its timing experiments often favor adaptive eigh over iterative QR; practical
EShampoo also changes update ordering/momentum coordinates relative to SOAP.
The early/late accuracy study motivates the startup hypothesis but does not
validate it on our workload. Appendix D studies diagonal-surrogate moment
transport and reports cases where it hurts. A faithful comparator must identify
which policy is ported and preserve remaining SOAP math for an attribution test.

We checked the linked [authors' implementation at revision
2837ffc80e1b50f325821b53bd6d8ac8247dbae1](https://github.com/facebookresearch/optimizers/blob/2837ffc80e1b50f325821b53bd6d8ac8247dbae1/distributed_shampoo/preconditioner/matrix_functions.py#L905).
It refines while the criterion fails, unlike the apparent reversed loop
inequality in the paper's rendered Algorithm4. It is a current source check,
not a recovered historical execution revision.

Remaining defensible positioning is the concrete dense matmul-oriented
update/retraction/repair/state/precision method and its measured cost-quality
tradeoff. Warm starting, Jacobi ideas, cheap/expensive hybrids, adaptive refresh,
low precision and overlap-V rules are not broadly new. A general superiority
claim over subspace/adaptive methods would require a direct matched comparator.
A narrow report can explicitly state those comparisons are unmeasured.

## Methodology qualifications to carry into the paper

- Stock betas stay the primary reference. The auxiliary AdamW beta values and
  model/grouping recipe remain distinct from a full upstream recipe.
- Per-arm elapsed is measured within a campaign whose compiler cache has
  already been exercised by qualification and calibration. It includes that
  arm's startup/compilation when incurred, but not all first-ever compilation,
  data preparation or calibration. It is sampled before post-training cleanup.
  Report cold setup separately if making deployment-cost claims.
- Clean steps exclude logging/LR assignment/evaluation/guard overhead; per-arm
  elapsed includes the relevant work during that arm. Label both accurately.
- The same validation set selects LR and reports final CE. It is held-out
  validation, not an untouched test set. The500-call LR screen is an early
  bracket, not proof of the best endpoint rate.
- Hash-preservation covers named parameters, present gradients and listed
  optimizer tensors; it does not exhaustively serialize all buffers, Python
  counters/group scalars or CPU RNG. CUDA RNG and gradient identity are checked
  separately. The recorded49-test XML came from frozen v3 CPU checks, while v4
  added registered-release/library-workspace cleanup and was GPU-qualified.
- Future sparse late diagnostics should not be inserted silently into timing:
  mark their calls, charge elapsed cost, and retain complete-cycle aggregation.

## Reconciliation and stopping point

The math review's stronger phrasing that warm "must earn" its intermediate
work versus QR50 is interpreted as a condition for a BROADER efficiency or
superiority claim. The user's primary matched-interval QR10 comparison remains
primary. QR20/50 provide context; they do not create a retrospective pass/fail
gate that invalidates a demonstrated QR10 tradeoff. A competitive sparse result
would narrow the paper and should receive fair bounded tuning/replication before
a best-method ranking.

After the implementation/policy decision, the proposed two paired-seed
replications and QR20/50 controls remain a bounded closure package. If the
implementation or policy changes, re-plan the confirmation matrix explicitly:
the old seed cannot automatically count as a same-version replication, and the
old percentage cannot be carried forward. Do not promise an unchanged six-run
budget across a version change. No new training or profiling is started here.
