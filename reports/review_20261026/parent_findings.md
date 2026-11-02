# Parent review: independent checks and prioritization

2026-10-26. Read-only examination of the completed v4 campaign, current code,
the two requested full papers, and the authors' linked implementation. No
production code changed and no GPU job was launched. The only new arithmetic
check was a CPU two-by-two counterexample, recorded separately.

## What the papers contribute

Milligan et al., [2605.26327v1](https://arxiv.org/html/2605.26327v1), §§3.1–3.2,
maintain covariance in basis coordinates P=QᵀCQ, update selected basis blocks
using local orthogonal factors, and rotate P consistently. Their BF16-storage
study includes SOAP; most subspace studies use KL-Shampoo (§4). Appendix C
explicitly suggests combining cheaper subspace updates with occasional full
updates. Our dense world-C, full-basis damped motion, approximate retraction,
FP32 storage and selected BF16 operands are different. Neither method is
established as faster/better than the other by our data. Generic warm reuse,
Jacobi inspiration and cheap/expensive hybrid updates are not adequate standalone
novelty claims.

Eschenhagen et al., [2506.03595v2](https://arxiv.org/html/2506.03595v2), §4,
control relative off-diagonal factor defect to schedule decomposition and study
stage/factor differences. §4.4's early-accuracy result motivates, but does not
validate, an early accurate phase on our LLM workload. Their practical EShampoo
changes gradient/basis ordering and momentum coordinates relative to official
SOAP (Appendix B), so replacing our full optimizer with it would not isolate
basis-refresh policy. The criterion does not certify adaptive-update fidelity
or learning quality. Their exact norm identity assumes orthogonal Q.

The linked authors' current [implementation, pinned revision
2837ffc80e1b50f325821b53bd6d8ac8247dbae1](https://github.com/facebookresearch/optimizers/blob/2837ffc80e1b50f325821b53bd6d8ac8247dbae1/distributed_shampoo/preconditioner/matrix_functions.py#L905)
uses refinement while the criterion FAILS (lines970–979), resolving the apparent
reversed inequality in the paper's HTML Algorithm4. This is a current source
check, not the paper's historical execution revision. Do not reproduce the
rendered loop literally. Per-factor bool conversion also implies synchronization
cost that must be counted in any port.

## Concrete baseline audit before promotion

`soap_reference.py:54–55` materializes QᵀCQ only to take its diagonal, then
recomputes CQ after sorting columns. Let Z=CQ. In exact arithmetic,
`diag(QᵀCQ) = sum(Q*Z, dim=0)` and the QR input is `Z[:,order]`. This reduces
three cubic products to one plus a quadratic contraction/gather. This identity
does not require orthogonal Q, but its mixed-precision evaluation/reduction
order can change rankings, QR signs and trajectories.

This is already implemented experimentally in
`benchmarks/qr_reuse_profile_kernel.py:qr_basis`, as well as a related batched
helper. Its older complete-refresh hook is restricted to QR-only/highest
momentum transport; the current v4 policy uses inherited high transport and
warm repairs. Do not transplant that whole hook or the batched helper's ridge.
Recoverable source is evidence of an idea, not a recovered successful GPU
qualification: the old external result bundle is missing.

Next bounded engineering work should qualify this factor-level reuse under the
current recipe, apply it to BOTH QR control and warm repairs, and measure full
cadence costs. Preserve the existing reported10.81% as a result against the
original tested baseline; it cannot be relabeled as a result against a new
implementation without measurement. No numerical speed prediction is made here.

## Strongest bounded policy idea: an early accurate phase

Actual stock warm-minus-QR validation gaps are +.118346 at500, +.060651 at1000,
+.009847 at2000 and +.006976 at10000. These observations justify testing ONE
predeclared QR10 burn-in (e.g. first1000 calls), then the current warm10/QR50
policy. This is a new schedule, not an implementation-preserving optimization.
It changes training history and may or may not close the final loss gap.

The raw event decomposition is saved in `parent_timing_analysis.json`. Stock
ordinary optimizer calls average32.65ms, warm-refresh calls130.69ms and repair
calls654.15ms. With other costs fixed, eliminating even the entire warm-only
surcharge would remove only about2.31% of current complete-step cost. Thus
many tiny warm-kernel variants have limited end-to-end headroom on this setup.

Conditional arithmetic estimate:1000 calls at observed average QR cost followed
by9000 at observed warm cost adds about41seconds, retaining about9.73% complete-
step saving versus original QR. This excludes changed learning, execution/order,
compilation and startup effects; it is NOT a measured result for the proposed
schedule, and an improved QR baseline would change the calculation.

## Correctness boundary that QR repair must address

GPT math reviewer identified a true stall example. I independently exercised
`fast_warm_refresh` on CPU with Q=I, C=[[1,.5],[.5,1]]. The equal estimated
row/column eigenvalues make the damped gap generator identically zero, leaving
Q unchanged despite relative off-diagonal defect .4472. This shows that warm
motion alone is not a general diagonalizer. It does not show that training
visits this exact state. Fixed/adaptive QR recovery is substantive; pure-warm
or general convergence claims are unsupported. See
`equal_diagonal_counterexample.json`.

Late actual physical-M checks and same-C before/after diagonalization checks
should be attached sparsely to future confirmatory runs. Current actual-M
checks cover only ages10/50. A small Gram error does not imply small tracking
or adaptive-update error. Diagonal-V overlap transport remains a different
approximation and is not automatically an improvement.

## Proposed scope and decisions

1. Preserve v1 and all completed results as immutable reference evidence.
2. Audit efficient QR before finalizing headline performance or committing the
   full seed budget to a new implementation.
3. If changing policy, permit one evidence-led early-QR candidate, then decide;
   avoid a broad burn-in/cap/precision/repair grid.
4. Common pointwise fusion/covariance accumulation and repeated-shape batching
   are engineering candidates for both arms; measured complete costs and
   numerical checks decide whether they are worthwhile.
5. Adaptive per-factor repair using an existing warm B can be investigated
   after sparse diagnostics. Factor checks are not free when B is not otherwise
   computed; reductions, host synchronization and rejected warm work count.
6. Persistent P, block-local updates, BF16 state, logical projection splits,
   changed moment rules and low-rank coverage are separate algorithmic branches.
   Reparametrization identities relying on orthogonality cannot be applied
   blindly to the current approximately orthogonal factors.
7. QR20/50 contextual controls remain valuable. Their purpose is to determine
   the warm policy's position among simpler schedules; they do not retroactively
   replace the user's primary matched-interval QR10 question or create an
   undisclosed requirement to beat every sparse policy. Broader superiority
   claims would require fair tuning and stronger competitors.
8. After implementation/policy choice, the agreed bounded replication/control
   package and a narrowly framed paper are the closure path. Remaining novelty
   must be argued for the concrete dense update/repair/state/precision algorithm
   and measured tradeoff, not warm-started eigenspaces in general.

The independent model reviews may add or challenge these findings. The root
will reconcile their claims explicitly; no provider failure counts as a review.
