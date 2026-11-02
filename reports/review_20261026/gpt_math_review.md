# Independent GPT mathematical and implementation review

2026-10-26. Second independent GPT reviewer, replacing the unavailable Claude review. The two reported Claude provider failures produced no scientific findings. I did not read or alter the other review artifact. This review made no implementation changes, ran no training/profiling/GPU work, installed nothing, and launched no services or agents.

**Recommendation: freeze the current mathematical candidate, but do not freeze a claim of loss equivalence or superiority over efficient sparse SOAP.** Complete the bounded baseline/replication package, then write a focused technical report. There are plausible optimizations left; the most urgent one strengthens the QR baseline rather than the candidate. No additional experiment is authorized or launched by this review.

## Evidence and scope

I read `/workspace/AGENTS.md`, `/workspace/GPU_QUEUEING.md`, the current opening proposal in `RESEARCH.md`, the full restart protocol, and the completion/calibration/comparison/provenance records. I inspected the actual construction and inherited update path. The current `fast_soap.py`, `soap_reference.py`, `soap.py`, and `benchmarks/owt_restart.py` are byte-identical to their frozen v4 source counterparts. Code citations below therefore also identify the tested implementation, rather than an inferred constructor default.

Primary local evidence:

- [Completion audit](/workspace/TurboSOAP/reports/owt_restart_20261026/completion_audit.json), [comparison including all four results](/workspace/TurboSOAP/reports/owt_restart_20261026/comparison.json), [calibration](/workspace/TurboSOAP/reports/owt_restart_20261026/calibration.json), [provenance](/workspace/TurboSOAP/reports/owt_restart_20261026/provenance.json).
- [Protocol](/workspace/TurboSOAP/OWT_RESTART_20261026.md), [actual construction](/workspace/TurboSOAP/benchmarks/owt_restart.py:68), [frozen campaign source](/workspace/optimizer_replay_results/soap_owt_restart_20261026_v4/source/benchmarks/owt_restart.py:68).

**Measured:** stock warm costs 338.956 versus 380.056 ms per clean complete step (10.814% reduction), and 52.920 versus 94.556 ms per optimizer call. Endpoint CE is 3.418128 versus 3.411152: a +0.006976 penalty, approximately +0.700% perplexity. Elapsed times are 58.454 versus 65.281 minutes. The other profile also trades worse endpoint CE for lower cost; it is robustness context, not an isolated beta experiment.

The audit supports completion and its declared numerical/integrity checks: 40,000 calls, 17,280 factor observations, 384 actual-M observations, and 12 full evaluation preservation checks. One call initializes matrix state, so each arm has 9,999 matrix updates, despite 10,000 training calls. These records do not establish convergence, statistical equivalence, a universal time-to-quality win, or width scaling. Initial-gradient equality establishes a matched starting point, not equality of evolved gradients.

The 500-call baseline-only LR bracket is a defensible bounded calibration, but is an early-trajectory selection on a 10,000-call schedule. It is not a terminal-quality optimum. A sparse competitor that proves competitive deserves bounded calibration before being declared inferior. Repeated validation panels are not independent seeds.

## Full-paper comparison and remaining novelty

I read the full HTML papers, including appendices: [Milligan et al., 2605.26327v1](https://arxiv.org/html/2605.26327v1) and [Eschenhagen et al., Purifying Shampoo, 2506.03595v2](https://arxiv.org/html/2506.03595v2). I first read Purifying v1, then checked the current v2 changes, including its practical pseudocode and added language-model diagnostics. No priority claim follows from these two papers alone.

**Reparametrizing Shampoo/SOAP:** §3.1, Eqs. 1–4 stores and evolves P=QᵀSQ, with full-basis QR equivalence under orthogonality and a unique nonsingular QR convention. §3.2, Eqs. 5–6 uses selected principal subspaces, local orthogonal factors, and matching P rotations. §4 Experiment 1 includes SOAP; subsequent subspace experiments focus on KL-Shampoo. Appendix C explicitly proposes cheap subspace updates plus occasional full-basis updates. Its BF16 result concerns persistent storage, unlike TurboSOAP's FP32 state with selected BF16 operands. [Full paper, §§3–4 and Appendix C](https://arxiv.org/html/2605.26327v1#S3).

**Purifying Shampoo:** §4.1 Eq. 11 controls relative off-diagonal factor error; §§4.2–4.4 study adaptive decomposition, cost, and stage/factor dependence. Appendix D.2, Eqs. 24–27 explains inconsistent second-moment coordinates and derives squared-overlap transport under a diagonal approximation; Figure 6 reports that correction can hurt. Appendix B distinguishes practical EShampoo from SOAP ordering/momentum. Appendix E.2 includes Llama3/C4 factor diagnostics, so this is not exclusively vision evidence. [Full paper, §4 and Appendices B/D/E](https://arxiv.org/html/2506.03595v2#S4).

**My positioning judgment:** neither paper's displayed algorithm is the exact `fast_warm_refresh` kernel: a dense damped gap-based skew generator, global skew-Frobenius cap, additive step, two mixed-precision residual corrections, and fixed periodic QR with the inherited SOAP state policy. That concrete algorithm/implementation and its measured cost-quality tradeoff remain a defensible subject. They do not establish first use of warm eigenbasis tracking, Jacobi-inspired updates, mixed precision, adaptive refresh, or cheap updates between full refreshes. A claim of a generally superior optimizer would need a direct modern related-method comparator; a narrow report can explicitly leave relative performance against reparametrized subspace SOAP unmeasured.

The algebraic distinction is material: TurboSOAP retains world-coordinate C and recomputes B=QᵀCQ; it does not maintain P as optimizer state. It remains dense cubic work. Calling this a subspace method or a BF16-storage method would be incorrect. Adding those features would move toward the cited paper and require a new numerical/state contract.

I also inspected the authors' linked [facebookresearch/optimizers implementation at revision 2837ffc80e1b50f325821b53bd6d8ac8247dbae1](https://github.com/facebookresearch/optimizers/blob/2837ffc80e1b50f325821b53bd6d8ac8247dbae1/distributed_shampoo/preconditioner/matrix_functions.py#L905): lines 905–930 implement the criterion, 967–979 refine while it fails, and 1014–1028 gate `eigh`. This is a current implementation reference, not evidence of the exact software used for the paper's experiments. The paper's HTML QR-loop inequality should not be copied uncritically; its explanatory criterion and this implementation use refinement when the tolerance is violated.

## Mathematical correctness and approximation boundaries

### 1. The rotation sign is locally sensible; convergence is not guaranteed

**Direct code evidence:** [fast_soap.py:33–46](/workspace/TurboSOAP/fast_soap.py:33). Let B=QᵀCQ, d=diag(B), s=max|d|, δij=di−dj, and η be the post-cap step multiplier (at most 0.5). In exact arithmetic the generator is

`Aij = −η Bij δij / (δij² + (0.01 s)²)`.

Symmetric B makes A skew-symmetric. With Qnew≈Q(I+A), the first-order transformed factor is B+[B,A]. Near a diagonal B, the ij residual is approximately

`Bij × [1 − η δij²/(δij²+(0.01s)²)]`.

**Mathematical reasoning:** this has the intended local contraction sign. It is a local diagonal-dominance argument, not a global descent proof: neglected commutator terms, finite rotations, orthogonalization and rounded products matter. The cap limits the collective motion and can slow useful directions when many pairs compete for its budget.

**A concrete stall case:** take Q=I and C=[[1,c],[c,1]], 0<|c|<1. Both diagonal estimates are equal, so A=0 despite nonzero off-diagonal error and distinct eigenvalues. The residual corrections leave I unchanged in exact arithmetic. A warm refresh cannot escape this state. Thus periodic QR is substantive recovery, not merely housekeeping for roundoff. This is a mathematical counterexample, not a claim that current training visits this exact state. Log near-gap coupling before proposing a remedy.

### 2. Two residual corrections yield approximate orthogonality

For the ideal exact update Y=Q(I+A), orthogonal Q gives YᵀY−I=−A². The real skew cap `||A||F/√2 ≤ 0.5` implies `||A||2 ≤ 0.5` and initial spectral Gram defect at most 0.25. A correction Y←Y(I−E/2), E=YᵀY−I, gives

`Enew = −3 E²/4 + E³/4`.

**Mathematical reasoning:** at the extreme scalar defect 0.25, two exact corrections leave about 0.0014 defect, not zero. This argument assumes initially orthogonal Q; it is not a guarantee for repeated finite-precision calls. BF16 operand rounding adds further error, even with direct FP32 outputs. The measured stock-warm maximum absolute Frobenius Gram error is 0.017439 versus 0.00006941 for QR. Both pass the declared guard; passing does not make the bases equally orthogonal.

### 3. Physical first-moment preservation is approximate in this implementation

The shared transport reconstructs `Mworld=Qold,L M Qold,Rᵀ`, then stores `Mnew=Qnew,Lᵀ Mworld Qnew,R` ([soap.py:783–791](/workspace/TurboSOAP/soap.py:783)). Reconstructing the result gives

`Qnew,L Qnew,Lᵀ Mworld Qnew,R Qnew,Rᵀ`.

This equals Mworld for square orthogonal new factors in exact arithmetic. Otherwise, if the two projector defects have operator norms eL,eR, relative Frobenius error is bounded by `eL+eR+eL eR`, before arithmetic error. The rule is therefore approximate for the warm basis. The small observed errors support the tested regime, not exact preservation.

**Evidence gap:** [owt_restart.py:118–139](/workspace/TurboSOAP/benchmarks/owt_restart.py:118) checks actual M only at ages 10 and 50. The stock maximum is 0.001543, below the 0.01 guard. The many factor observations elsewhere measure Gram/finite state, not late physical-M error or eigenspace accuracy. Add a few late warm and repair checks to already proposed replications; do not replay completed runs just to obtain them.

Storing world-coordinate M could avoid refresh transport, but it also changes the numerical recurrence when Q is nonorthogonal. Treat it as a separate implementation/numerical candidate, not a guaranteed trajectory-preserving cleanup. The current path already avoids redundant ordinary-step moment round trips.

### 4. Diagonal V cannot encode exact historical covariance transport

[soap_reference.py:44–75](/workspace/TurboSOAP/soap_reference.py:44) permutes V on QR and leaves V unchanged on warm refresh. This is explicitly the tested method, not an accidental missing call. It follows the baseline's approximation policy but does not produce the same approximation error on different basis trajectories.

**Independent derivation:** for old-coordinate full second moment H and new-coordinate transition T, the required diagonal is `diag(T H Tᵀ)`. V alone contains diag(H); discarded off-diagonal entries can contribute after rotation. The available overlap rule, `T⊙T` applied to V, is exact for the diagonal surrogate H=diag(V), not for arbitrary historical H. For two sides it becomes `(TL⊙TL) V (TR⊙TR)ᵀ`. The optional [soap.py:793–807](/workspace/TurboSOAP/soap.py:793) implements that surrogate.

For T=I+A+O(A²), rotating a truly diagonal H changes its diagonal only at second order; unknown cross-covariances can contribute at first order. This explains why small warm motion can make holding V plausible without proving it accurate. Short stock variance memory helps forgetting: 0.95^50≈0.077 versus 0.999^50≈0.951. These scalar coefficients neither bound basis drift nor explain the observed profile difference causally.

Do not introduce overlap transport merely because it looks more principled. It adds work and repeatedly redistributes a diagonal surrogate; its effects on training need a separate controlled comparison.

### 5. Geometry is a diagnostic, not a convergence certificate

An orthogonal but stale basis has perfect Gram error. A small covariance Frobenius defect need not control the inverse-scaled update in weak directions. Nearly degenerate factor eigenspaces can also be unidentifiable while coordinatewise SOAP V remains anisotropic there. Report Gram, diagonalization defect, and actual update distortion separately. Applying the exact orthogonal identity behind an off-diagonal criterion to a nonorthogonal warm Q requires accounting for its Gram defect.

## Highest-value bounded optimizations

### A. First priority: remove redundant QR work, symmetrically

**High-confidence implementation finding:** [soap_reference.py:54–55](/workspace/TurboSOAP/soap_reference.py:54) uses three dense products before QR: two for QᵀCQ and one for CQsorted. Only the diagonal of the first result is consumed. Algebraically compute Z=CQ once, estimate each diagonal by the column dot product `sum(Q*Z, dim=0)`, sort, and QR the corresponding columns of Z. This replaces three cubic products with one plus a quadratic reduction/gather.

The repository already contains this pattern in [batched_soap_refresh.py:42–55](/workspace/TurboSOAP/batched_soap_refresh.py:42). That helper also adds an epsilon ridge for another recipe; do not transplant the ridge into the present experiment. The [pinned official SOAP fixture](/workspace/TurboSOAP/tests/reference/soap_upstream.py:412) uses the redundant pattern too: upstream parity certifies semantics, not efficiency.

This is low risk in exact mathematics but changes reduction order, potentially column ranking at ties, QR signs, and numerical trajectories. Validate permutations, reconstructed update/state and finite precision behavior, then profile complete cadence cost. Apply the efficient repair to both QR10 and warm/QR50. **No numerical speedup estimate is justified without measurement.** Because QR10 pays it five times as often, the current relative advantage could shrink.

### B. Low-risk engineering candidates after attribution

- Fuse common elementwise moment/denominator/update work and consider covariance `addmm` accumulation to avoid outer-product temporaries ([soap_reference.py:100–109](/workspace/TurboSOAP/soap_reference.py:100), [soap.py:669–674](/workspace/TurboSOAP/soap.py:669)). Preserve epsilon placement, bias correction, update-before-decay ordering, and cold-call behavior. Floating-point reassociation still needs validation. Apply common improvements to both arms.
- Group repeated factor shapes for batched refreshes, using bounded chunks. Packing, copying back and peak temporaries may erase benefits at dimensions 2304/3072. The existing blocked helper is a starting point, not evidence for dense-model speed. Measure packing and transport, not just `bmm`/QR kernels.
- Attribute warm compilation and BF16 operands separately with a fixed, small profiling set. Persistent FP32 state does not mean all FP32 products use highest precision: caller policy is `high`. Shared transport/covariance precision and candidate-only products should remain explicitly separated.

### C. Mathematical changes: bounded ideas, not freeze requirements

1. **Near-gap block rescue:** if diagnostics show large coupling where diagonal gaps suppress A, try a small disjoint block rotation at scheduled refreshes. It addresses the explicit stall case. It changes the basis map, selection cost and V mismatch; it overlaps conceptually with block-Jacobi/subspace work and should not be called a new general subspace idea.
2. **Early accurate phase:** one predeclared early QR10 phase followed by the existing warm policy could target early sensitivity. Only test if current curves or added diagnostics justify it. Do not search many switch points; it is a new schedule requiring learning validation.
3. **Precision allocation before deleting corrections:** if Gram or actual-M diagnostics become limiting, test highest-precision final correction or transport in a small isolated screen. Dropping the second residual correction is not a free speed improvement; the exact recurrence above explains why it can materially worsen orthogonality.
4. **Transport-aware motion budget (speculation):** monitor a few actual preconditioned directions under old/new coordinates at identical state, alongside off-diagonal improvement. This can reveal a rotation that improves factor geometry while worsening the applied adaptive update. An eventual cap based on this quantity would change optimizer mathematics and must pay for the diagnostic; it is not a proposed default.

Maintaining B as persistent state could remove its repeated formation, but it requires every covariance update and every basis change to be handled consistently. With nonorthogonal Q and current mixed precision, the exact reparametrization argument is insufficient. Defer it, BF16 storage, low-rank claims, one-sided factors, and new V rules to a separate method version.

## Minimal decision package and stopping rule

1. **One bounded engineering audit before headline freeze:** original versus efficient QR; current warm versus a matched all-FP32 warm operand path; eager versus compiled only where needed to identify compilation contribution. Use actual factor shapes, include ordinary/warm/QR cadence, transport and temporary memory. CPU correctness and short future GPU profiles suffice to decide whether a full requalification is necessary; kernel timings alone cannot revise training claims. No profiling was run in this review.
2. **The already proposed six full runs are a reasonable ceiling for the immediate closing campaign:** two additional stock paired QR10/warm replications (four runs), plus QR20 and QR50 on seed204 (two runs). Freeze LR/horizon/grouping/numerical policy, counterbalance pair order, and declare whether data order varies across seeds. If the engineering audit changes the timed implementation materially, qualify and name that version first; do not pool timings with the old version as if identical.
3. **Attach sparse observations to those runs:** late warm and QR actual-M checks, same-C before/after off-diagonal residual, cap activation and near-gap coupling, and a few update-direction discrepancies. Keep observations out of clean timing and charged to elapsed. No dense all-factor eigendecomposition oracle is necessary.
4. **Decision:** warm must earn its intermediate work against QR50 and remain useful versus QR20. Three paired seeds should report all differences and uncertainty; absence of significance is not equivalence. Declare any acceptable quality margin before using it for a pass/fail claim. If simpler sparse QR dominates, narrow the report or stop promotion rather than expand into an indefinite tuning grid.

Begin drafting now. The honest current result is a reproducible dense SOAP basis-update tradeoff on one 109M OpenWebText setup, with approximately 11% lower complete-step cost against the tested QR10 implementation and a small observed endpoint loss penalty. Finalize the empirical claim after the efficient-baseline and replication checks. A second scale is conditional on broader claims; historical H200 refresh-event speed, effective-rank advantages, and the unfinished TransportMuon extension should not carry this paper's conclusions.
