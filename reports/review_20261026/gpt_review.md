# Independent GPT review of TurboSOAP — 2026-10-26

**Recommendation: freeze the current optimizer mathematics, finish a small comparison package, and start writing a narrowly scoped paper now. Do not freeze the present speed/novelty claims as final.** The completed OpenWebText experiment supports a useful speed–quality tradeoff against this QR10 implementation. It does not yet show that warm tracking improves the best simple sparse-QR tradeoff, matches QR10 statistically, or beats contemporary basis-update methods.

This review emphasizes methodology, baseline strength, provenance, and claim scope. I read `/workspace/AGENTS.md`, `/workspace/GPU_QUEUEING.md`, the full HTML of both requested papers including their appendices, the actual trainer and inherited optimizer paths, and the frozen campaign's raw event logs. I independently recomputed timing aggregates and checked hashes with CPU-only file processing. I did not execute training, import the optimizer for testing, install anything, change services, edit research code, or read the other review artifacts in this directory. Only this review was written. Historical missing-machine reports were not reconstructed or treated as new evidence.

Evidence labels below distinguish **observations** from **mathematical deductions**, **engineering hypotheses**, and **unvalidated algorithm changes**. Proposed experiments are recommendations, not launched work.

## High-confidence findings

### The recovered experiment is internally coherent and unusually inspectable

I verified all **572 frozen source entries and four input files** against the campaign manifest. All extant corresponding working-tree sources also matched those frozen hashes at review time. The repository copies of `comparison.json`, `calibration.json`, `completion_audit.json`, `provenance.json`, and all four primary result files matched the campaign copies byte for byte. The upstream fixture's SHA256 is `9e9021335ad02371584622fa256e73fe75f3eb74446a8d8a216e956534148f6b`, matching its recorded pin.

Authoritative locations are [the frozen bundle](/workspace/optimizer_replay_results/soap_owt_restart_20261026_v4), [provenance](/workspace/TurboSOAP/reports/owt_restart_20261026/provenance.json), [completion audit](/workspace/TurboSOAP/reports/owt_restart_20261026/completion_audit.json), and [raw stock QR events](/workspace/optimizer_replay_results/soap_owt_restart_20261026_v4/runs/primary_stock_qr/events.jsonl). This independently confirms file consistency; it is not a rerun of the scientific experiment.

The actual construction is explicit: matrix betas/covariance are `.95/.95/.95` or `.9/.999/.99`, matrix refresh interval 10, repair interval 50, and separate auxiliary AdamW betas `.9/.999` with peak LR `.0006`. FastSOAP constructor defaults do **not** specify the campaign. See [optimizer construction](/workspace/TurboSOAP/benchmarks/owt_restart.py:68), [parameter grouping](/workspace/TurboSOAP/benchmarks/cifar_optimizers.py:28), and [FastSOAP defaults](/workspace/TurboSOAP/fast_soap.py:59).

The 48 hidden linear matrices get full two-sided SOAP; the tied embedding/head and norms get auxiliary AdamW. Tying happens in [transformer.py](/workspace/TurboSOAP/transformer.py:290), and grouping deduplicates parameter identities. Fused QKV and SwiGLU make this a specific factorization geometry, including large 2304/3072-dimensional factors. Splitting those matrices would change the optimizer, not merely optimize its implementation.

The data preparation selects training rows without replacement and preserves their randomized order, with validation from a separate document split. The trainer correctly uses 1023 next-token targets per 1024-token block, four physical microbatches per global batch, loss division by four, and global clipping after accumulation. Causal attention is explicit. See [preparation](/workspace/TurboSOAP/benchmarks/prepare_owt_restart.py:32), [backward](/workspace/TurboSOAP/benchmarks/owt_restart.py:178), and [attention](/workspace/TurboSOAP/transformer.py:134). The corpus manifest records the first 5% of source documents as validation; that is a declared split, not a random-document or semantic-deduplication guarantee.

Each primary run completed 10,000 optimizer **calls**, presenting 327.36M targets. The first call initializes matrix state without updating the matrices, while auxiliary AdamW steps; there are 9,999 matrix updates. Calling these 10,000 matrix updates would be inaccurate. [Update semantics](/workspace/TurboSOAP/soap_reference.py:94) and [protocol](/workspace/TurboSOAP/OWT_RESTART_20261026.md:84) agree.

### The measured advantage is real for the recorded comparison, with a persistent quality cost

**Observed, independently recomputed from raw events:**

- Stock profile, LR `.0015`: QR10 final CE **3.4111515321**, warm10/QR50 **3.4181275256**, gap **+.0069759935**. Perplexity rises approximately **.700%**, from 30.300 to 30.512. Mean complete step is **380.055978 vs 338.956372 ms**, a **10.814% time reduction** or **1.1213× throughput**. Optimizer means are **94.556205 vs 52.920323 ms**, a **44.033% reduction**.
- Previous profile, LR `.0005`: CE **3.4421970602 vs 3.4511003383**, gap **+.0089032780**; complete step **380.904833 vs 338.418891 ms**, an **11.154% reduction**. Optimizer means are **94.594463 vs 51.999928 ms**.
- Stock elapsed time is **65.2813 vs 58.4541 minutes**, a **10.458% reduction**. This differs appropriately from the steady-state step percentage because elapsed time includes work outside the step.

Sources: [comparison](/workspace/TurboSOAP/reports/owt_restart_20261026/comparison.json), [stock QR result](/workspace/TurboSOAP/reports/owt_restart_20261026/primary_stock_qr.json), and [stock warm result](/workspace/TurboSOAP/reports/owt_restart_20261026/primary_stock_warm.json).

Warm is worse at **every recorded nonzero matched-call stock validation point**. The early deficit is materially larger: +.118346 CE at call 500, +.060651 at 1000, then roughly +.006–.008 late in training. The endpoint should not conceal that trajectory. Conversely, faster execution matters: the first observed stock CE≤3.5 is at 43.8640 minutes for warm and 48.9798 for QR. Warm's observation even precedes QR's previous above-threshold observation at 45.7615 minutes. This supports favorable sampled target attainment on this run, without establishing exact continuous crossing or a universal time-to-quality improvement. The CE≤3.6 brackets overlap, and much earlier targets can tell a less favorable story.

A paper should show the full matched-token and measured-time curves, including early training, and choose primary quality/time criteria prospectively for confirmation runs. It should not select only flattering thresholds after observing the curves.

## Trainer and timing audit

### The reported mean is cadence-weighted correctly

[Timing boundaries](/workspace/TurboSOAP/benchmarks/owt_restart.py:234) synchronize before backward, between backward and optimizer, and after optimizer. Complete-step time includes CPU data materialization/H2D, all four backwards, loss/norm scalar extraction, clipping, matrix optimizer, and auxiliary optimizer. It excludes LR assignment before the timer, JSON logging after the timer, validation, geometry diagnostics, and budget checks. Thus “complete step” is a measured training-step boundary, not total application wall time.

[Aggregation](/workspace/TurboSOAP/benchmarks/owt_restart.py:261) uses calls 101–10,000: **9,900 observations, exactly 198 complete 50-call cycles**. Counts are:

- QR: 8,910 ordinary + 990 QR steps.
- Warm: 8,910 ordinary + 792 warm + 198 QR steps.

This avoids the common mistake of comparing event medians without weighting their frequencies. Across the full run the counts are 9,000 ordinary/999 QR/one cold initialization for QR, versus 9,000 ordinary/800 warm/199 QR/one cold initialization for warm. The actual-M diagnostics inside calls 11 and 51 are outside the clean window.

For stock, mean optimizer event times were **32.574 ms ordinary / 652.396 ms QR** for the control, and **32.647 / 130.691 / 654.150 ms ordinary/warm/QR** for the candidate. The weighted candidate mean is therefore `.90×32.647 + .08×130.691 + .02×654.150 ≈ 52.920 ms`. Those are whole optimizer-step event costs, including other optimizer work, not isolated QR or warm-kernel latencies.

Stock forward/backward means are **285.500 vs 286.036 ms**. The 41.636 ms optimizer saving accounts for essentially the whole 41.100 ms step saving; the slight backward difference opposes the candidate. This is stronger attribution than a comparison driven by large unrelated backward variation. Quarter-run stock step means are also stable: approximately 380.0–380.1 ms for QR and 338.6–339.1 ms for warm. These repeated steps establish timing consistency within a run, not independent training replications.

**Mathematical implication:** QR optimizer time is only about 24.9% of baseline step time. Eliminating *all* optimizer cost would cap the reduction at that fraction under this execution recipe. Warm retains 52.9 ms of optimizer work, so kernel-only speedups must be assessed against this ceiling. There is no basis for translating a historical refresh-event “10×” into a current training claim.

### Elapsed-time accounting is useful but needs precise labels

`PROCESS_STARTED` is set before NumPy/Torch imports, and each arm is a fresh subprocess. Per-arm elapsed includes its imports/setup, initial repeat backwards, evaluations and guards, but is sampled before post-training cleanup. The primary arms reuse a persistent compiler cache already exercised by qualification and calibration; their elapsed time does **not** include the full campaign's data preparation, qualification, six LR trials, or all first-ever compilation. See [worker lifecycle](/workspace/TurboSOAP/benchmarks/owt_restart.py:298) and [launcher](/workspace/optimizer_replay_results/soap_owt_restart_20261026_v4/learning_launcher.sh).

Call this “per-arm elapsed under the declared warmed-cache campaign,” alongside steady-state means. Cold compilation is a separate deployment metric. The existing cache policy does not invalidate the comparison, but “all compilation/setup charged” would be too broad. Exclusive GPU claims also do not isolate host CPU, storage, power/temperature history, or other devices' effects. Reversing order across beta profiles is helpful context; it does not counterbalance the stock comparison itself.

### Numerical guards passed, within their actual coverage

The audit records 180 geometry observations/17,280 factor observations, 384 actual-M checks, 12 hash-checked evaluations, zero live registered model/optimizer/gradient storages after release, and zero final allocated bytes after library-workspace clearing. Peak allocation is about 9.4 GiB for both methods; there is no demonstrated optimizer-state memory saving.

Important qualifications:

1. [Actual-M checks](/workspace/TurboSOAP/benchmarks/owt_restart.py:118) run at ages **10 and 50 only**: 48 matrices × two events × four arms. They do not certify later warm sequences or repairs. Stock warm's maximum recorded relative physical-M discrepancy is **.00154288**, not the earlier qualification maximum .001083. It is below the declared .01 guard.
2. [Geometry](/workspace/TurboSOAP/benchmarks/owt_restart.py:97) checks finite state and `||QᵀQ−I||F`, with maximum stock warm **.0174389**, versus about .0000694 for QR. These are Frobenius norms, not maximum entries or eigenbasis-approximation residuals. They establish approximate orthogonality at sampled times, not accurate diagonalization or every-step stability.
3. [State hashing](/workspace/TurboSOAP/benchmarks/owt_restart.py:84) covers parameters, present gradients, and listed optimizer tensors. It does not hash all Python counters, parameter-group scalars, model buffers, or CPU RNG. CUDA RNG and gradient identities are checked separately. The implementation is convincingly side-effect-conscious, but “full state” should identify this coverage rather than suggest an exhaustive state serialization comparison.
4. The CE execution audit is an initial-batch numerical screen: gradient relative difference .1385%, cosine .99999904, negligible CE difference. It is neither a trained-state check nor a proof of learning equivalence. The paired runs share the path, which is the relevant fairness fact. [Screen implementation](/workspace/TurboSOAP/benchmarks/qualify_owt_restart_loss.py:33).
5. The 49-test XML is recorded evidence from the v3 qualification source. I did not rerun it. Its small CPU parity tests, including stock betas, support semantics within tolerances; they are not a large-factor GPU trajectory equivalence theorem. [Parity test](/workspace/TurboSOAP/tests/test_soap_reference.py:18).

## Exact mechanism and novelty comparison

### What the local method actually changes

The timed code retains physical-coordinate covariance matrices `GG`. At a warm event it forms `B=sym(QᵀCQ)`, constructs a damped skew generator using normalized diagonal differences, caps its skew Frobenius norm, takes `Y=Q+QΩ`, and performs two corrections `Y←Y−½Y(YᵀY−I)`. Selected products round operands to BF16 but emit FP32 directly; persistent covariance, bases, moments, and parameters remain FP32. See [complete warm kernel](/workspace/TurboSOAP/fast_soap.py:14).

Both arms share [the ordinary update](/workspace/TurboSOAP/soap_reference.py:99), [covariance accumulation](/workspace/TurboSOAP/soap.py:652), and [physical-coordinate first-moment round trip](/workspace/TurboSOAP/soap.py:783). QR means a **single sorted power/QR basis iteration**, not exact eigendecomposition and not optimizer-state reset. Every fifth scheduled refresh replaces the warm event with that QR operation. V is permuted for QR column ordering and unchanged on warm events; it is not generally transported through arbitrary rotations. [Refresh dispatch](/workspace/TurboSOAP/soap_reference.py:44).

Dense warm work includes two square products to form B, one for the tangent candidate, and four across two corrections, per factor, before first-moment transport. It remains cubic in factor dimension. “Full basis” means no rank truncation; it does not establish superiority to blocked or subspace algorithms.

### Milligan et al., 2605.26327v1

Their state replaces physical covariance S with P=QᵀSQ. Full-basis updates use O=qr(P), Q←QO, and P←OᵀPO; exact equivalence requires orthogonal Q and consistent QR signs. Subspace updates rotate selected columns using QR of a principal block, retaining a complete basis. Selection includes a largest-coupling pair expanded greedily into a block. Their BF16 contribution concerns **persistent storage**, not merely operand precision. [§§3.1–3.2, Eqs.1–6](https://arxiv.org/html/2605.26327v1#S3).

Their full-basis storage experiment includes SOAP/KL-SOAP; later subspace experiments emphasize KL-Shampoo. Those numbers are not a measured SOAP head-to-head against TurboSOAP. Appendix C explicitly suggests occasional full updates among cheap subspace updates and adaptive cadence. [§4 and Appendices B–C](https://arxiv.org/html/2605.26327v1#S4).

**Comparison/inference:** TurboSOAP computes B transiently, moves the dense basis through a damped skew step and approximate polynomial retraction, and repairs with QR50. It does not implement their stored-coordinate reparametrization or partial-block QR. That distinction leaves a possible implementation/algorithm contribution, but cheaper incomplete diagonalization, complete-basis retention, Jacobi connections, BF16 robustness, and hybrid refresh motivation cannot individually support a new priority claim. FP32-state results do not answer their storage result. No linked author implementation for this paper was established in the bounded search; this comparison rests on its full primary text.

### Eschenhagen et al., 2506.03595v2

The paper separates eigenvalue correction from basis staleness. Its criterion is `||offdiag(QᵀCQ)||F / ||QᵀCQ||F ≤ τ`; it can skip a factor update or terminate warm-started QR. Actual timing experiments often favor criterion-triggered `eigh`, rather than repeated QR. EShampoo also differs from SOAP in update order and momentum coordinates. [§§3–4 and Appendix B](https://arxiv.org/html/2506.03595v2#S4).

Appendix D distinguishes optimal current-basis second moments from mixed-coordinate EMA. Its Eq.27 propagates a diagonal approximation with squared basis overlaps; Figure 6 reports that this correction can hurt. A changing-basis V rule is therefore already studied and is not automatically an improvement. [Appendix D.1–D.2](https://arxiv.org/html/2506.03595v2#A4).

**Comparison/inference:** TurboSOAP replaces selected QR iterations with a different dense, fixed-budget local motion. It does not implement adaptive per-factor scheduling or establish a covariance-error tolerance. Warm-starting, approximate basis maintenance, and overlap-diagonal moment correction are not remaining novelty by themselves. The paper's criterion assumes orthogonality, whereas TurboSOAP accepts measurable Gram error; its guarantee cannot be imported unchanged. Neither a small covariance residual nor a small Gram error alone proves a small SOAP update error. No empirical superiority to the paper's adaptive policy is established here.

I also inspected the authors' linked [facebookresearch/optimizers implementation, pinned at 2837ffc80e1b50f325821b53bd6d8ac8247dbae1](https://github.com/facebookresearch/optimizers/blob/2837ffc80e1b50f325821b53bd6d8ac8247dbae1/distributed_shampoo/preconditioner/matrix_functions.py#L905). Lines 905–930 implement the residual test; 967–979 iterate while it is **not** satisfied. This resolves an apparent reversed inequality in the rendered Algorithm 4 while-condition. A comparator should follow the criterion and implementation, not copy that condition literally. This is current pinned implementation evidence, not a claim that this revision produced the paper's historical tables.

### What novelty remains defensible

The candidate contribution is the **specific dense damped-skew tracking kernel, its approximate retraction/repair and state/precision policy, and a demonstrated workload-level cost–quality tradeoff**. This is narrower than inventing warm basis reuse, accelerated SOAP generally, BF16-state SOAP, or hybrid cheap/expensive basis updates. It is also narrower than a theorem that the transported optimizer equals QR-SOAP.

The two requested papers do not by themselves establish that this exact kernel has already been published. They also do not establish that it is new. A broader literature check, particularly iterative-whitening SOAP and classical Jacobi/polar/retraction methods, remains necessary before an exact priority assertion. This review establishes overlap and distinctions, not novelty certification or acceptance prospects.

## Serious methodological concerns before final claims

**Sparse QR is the most consequential missing comparator.** QR50 holds the basis between the same repair opportunities and directly asks whether the 800 intervening warm updates earn their cost. QR20 tests a simpler middle tradeoff. Against QR10 alone, part of the benefit may simply be tolerating a cheaper basis schedule. Old FineWeb sparse results are context, not substitutes for these OWT controls. If sparse QR matches warm quality at lower cost, narrow the claim rather than keep tuning until it loses.

**Semantic fidelity does not certify an efficient QR baseline.** The current reference avoids upstream's redundant non-refresh momentum round trips, which is a good common optimization. But its timed QR branch still forms a full `QᵀCQ` for the diagonal and then separately computes `C @ Q[:,order]`. An algebraically cheaper existing local primitive makes this a concrete concern, not hypothetical skepticism. Any improved QR implementation should also be used for candidate repairs. [Reference QR](/workspace/TurboSOAP/soap_reference.py:54), [existing efficient primitive](/workspace/TurboSOAP/batched_soap_refresh.py:42).

**One seed is one paired scientific observation.** Shared initial gradients validate construction; evolved gradients should differ after different basis updates. Repeated validation panels, 9,900 timings, and 17,280 factor checks cannot become training-seed sample size. The current result does not establish equivalence, and an arbitrary retrospectively chosen .01 CE allowance would not fix that. Report a practical tolerance separately from statistical uncertainty.

**LR selection is fair for a controlled modification but not best-method tuning.** The six trials use baseline-only call-500 CE on the full 10,000-call schedule; they do not run compressed schedules. Stock trial CE was 4.94236/4.87760/4.90449 at `.0005/.0015/.003` respectively. Previous trial CE was 4.90680/4.92736/4.95700. This is a declared early-trajectory bracket, not evidence that `.0015` is the best endpoint LR. Sharing the baseline-selected rate avoids candidate-specific tuning advantage; it does not prove fair optimizer Pareto dominance. The same validation sample selects LR and reports final CE, so “held-out validation” is accurate; “untouched test” is not. [Calibration logic](/workspace/TurboSOAP/benchmarks/owt_restart.py:318).

**Profile differences are confounded.** Stock and previous profiles change beta1, beta2, covariance memory, and selected LR. They provide secondary robustness of the speed–quality pattern, not a causal finding about beta2/covariance. Stock is the requested reproducible anchor, not a universal best recipe.

**The state approximation is material.** With approximate Q, transpose-based first-moment transport preserves the physical moment only approximately. V carries history across changed coordinates even if Q were exactly orthogonal. Those two errors are different and should not be described as solved by QR repair or a Gram guard. No weights/checkpoints were saved, so missing trained-state diagnostics cannot be recovered offline from these scalars; do not repeat completed experiments just to recover them.

## Bounded optimizations worth evaluating

### Low-risk implementation work: preserve mathematics, verify floating-point effects

**1. Strengthen QR using a single shared product — highest priority.** Compute `Z=C@Q`, Rayleigh estimates `(Q*Z).sum(dim=0)`, sort them, and QR `Z[:,order]`. In exact arithmetic this replaces three square GEMMs with one plus a reduction and gathers. The pattern already exists in [batched_soap_refresh.py](/workspace/TurboSOAP/batched_soap_refresh.py:50), but that helper adds its own epsilon and has a blocked API: copy the applicable algebra, not the whole policy. Keep current sorting, V permutations, update order, FP32 state and caller precision. Test sign/gauge-aware first-moment consistency, ties/near-ties, large rectangular parameter shapes, and short shared-input trajectory behavior. Different summation order can change nearly tied column ordering, so exact mathematical equivalence does not promise bitwise identity. Measure the entire 50-call cycle and both arms' repair paths. No speedup is assumed.

**2. Batch repeated factor shapes and fuse pointwise work — second priority, conditional on profiling.** There are 60 factors of dimension 768 and 12 each of 1536, 2304 and 3072. Grouping same-shape GEMMs/QR can reduce Python dispatch and small-kernel overhead. Start with the existing primitives as evidence of feasibility, not as evidence of measured OWT gains. Preserve each factor's independent cap/reduction and both-axis V permutations; account for stack/unstack copies, temporary memory and layout. Batched QR can be slower on a particular backend. Compile/fuse the common moment/denominator/decay operations for both arms if they are a measured cost; compilation must not silently change covariance precision or multiply LR decay twice.

**3. Profile before further warm micro-optimization.** Separate B construction, BF16 casts/products, correction Grams, first-moment transport, QR sorting/decomposition and common covariance work. The warm event is already only 8% of calls, and forward/backward dominates complete cost. Reusing casts or eliminating temporaries is an engineering hypothesis; the compiler may already do it. Keep eager-vs-compiled warm and FP32-vs-BF16 operand attribution to a small profile, with geometry and physical-M checks. Give the control comparable common-kernel optimization. Do not call a codegen or dtype gain evidence of a better basis algorithm.

Trainer execution changes such as larger physical batch, extra transformer compilation, asynchronous input delivery, or altered synchronization may improve absolute training throughput. They should be applied symmetrically and separately qualified; the fractional optimizer advantage can shrink when the cost balance changes. A fixed global batch is necessary to keep the mathematical comparison, and execution changes can still change rounding. They are not necessary diversions before this narrower paper.

### Changes to optimizer mathematics: possible, but not part of a silent freeze

**A. A short QR-only startup before warm tracking is the most motivated schedule alternative.** The stock gap peaks early and narrows substantially later. A single predeclared startup boundary, for example call 500 chosen from the already observed transient, is a plausible candidate if the frozen tradeoff fails the user's practical criterion. Charge its additional QR cost. This is a new policy requiring learning validation, not a free numerical fix; choose one boundary, not a sweep. It should not displace the sparse controls.

**B. Detect damped-generator stalls rather than globally increase step size.** Mathematical deduction from [fast_soap.py](/workspace/TurboSOAP/fast_soap.py:35): for `B=[[a,b],[b,a]]` with `a>|b|>0`, both diagonal differences are zero, so Ω=0 although B is not diagonal. This is a positive-definite counterexample to guaranteed diagonalization progress. It is not evidence that current training suffered this failure. The general SOAP path has a related [largest-pair safeguard](/workspace/TurboSOAP/soap.py:762), but FastSOAP's override bypasses it. On future sparse diagnostics, record residual/cap activity and identify high-residual, tiny-generator cases. Only if observed, consider a rare 2×2/block rotation or early QR repair with a strict budget. That changes the update rule and overlaps known Jacobi ideas; validate it as such.

**C. Better retraction or V handling should remain secondary research.** For exactly orthogonal Q and skew Ω, the initial tangent candidate has Gram `I−Ω²`. The cap gives `||Ω||2≤.5`, so its largest initial Gram excess is at most .25. One ideal correction maps a Gram residual E to `−¾E²+¼E³`. This explains why two corrections can be useful, but it is not a bound for BF16 products, inherited `high` arithmetic or already nonorthogonal Q. Adding a third correction, replacing transpose by an inverse/dual transform, or using coordinate-space identities is a mathematical/numerical-policy change, not a guaranteed free optimization. In particular, replacing `YᵀY` with a formula assuming exact Q can discard existing Gram error.

Leaving V unchanged is also not exact transport. Under a diagonal-covariance assumption and orthogonal bases, the local two-sided approximation would be `Vnew≈(AL⊙AL) Vold (AR⊙AR)`, with `AL=QL,newᵀ QL,old` and `AR=QR,oldᵀ QR,new`. The [existing overlap path](/workspace/TurboSOAP/soap.py:793) implements this type of approximation; missing cross-coordinate correlations prevent general exactness. Do not sell it as a new cure or add it without a targeted reason. Independently, old-coordinate history decays by `.95^50≈.077` for stock V versus `.999^50≈.951` for previous V. That makes memory dependence plausible mathematically, but the completed profile comparison does not isolate it empirically.

Persistent basis-coordinate covariance, partial blocks, per-factor adaptive cadence, fewer retraction steps, one-sided preconditioning and altered matrix grouping are larger research branches. They may be valuable, but they change the method, complicate attribution, and should not become prerequisites for closing this version.

## Minimal decision package and stopping rules

**First, one bounded implementation/precision profile**, not another learning sweep. Use the exact current model/factor shapes and resident nonzero optimizer state in a new prospectively permitted run; no saved trained state exists to replay. Compare current vs efficient QR, selected vs FP32 warm operands, and eager vs compiled warm. Report setup/cache policy, all common work, weighted cycles, memory, Gram/residual and physical-M behavior. If efficient QR removes the relevant speed advantage, narrow the paper or revise the implementation before spending on confirmation. Large numerical ordering changes require learning validation; a large efficiency change invalidates reuse of the old timing percentage as a new-baseline claim.

**Second, stock QR20 and QR50 full-horizon controls on seed 204**, keeping the same selected LR, data, schedule, grouping, precision and evaluation. Their first purpose is a controlled screen at the existing recipe. A strong sparse result should get the same bounded tuning/replication treatment before a best-method ranking; an unsuccessful screen does not prove universal inferiority. Predeclare whether the question is lower CE at equal tokens, observed attainment at fixed CE targets, or a practical CE/time tolerance. Report all three policies honestly if none dominates.

**Third, two additional stock paired seeds**, giving three paired observations with the existing seed. Freeze LR and horizon, pair initialization/data order within each seed, and document whether the new seeds vary initialization alone or also data order. Initialization-only replication is the clean minimal choice for this frozen cohort; label its scope. Use warm-first once and QR-first once among the new pairs, and record run order; three pairs cannot be perfectly balanced. Report each paired CE/time difference, mean and dispersion. Three pairs may still be too few to establish a tight equivalence interval: an inconclusive result should remain inconclusive.

For those already planned runs, add sparse late diagnostics at an actual warm event and a QR repair, on representative large/small and left/right factors, plus pre-repair observations. Include `||offdiag(QᵀCQ)||F`, Gram error, cap activity, and before/after physical-M error. Reuse existing diagnostic code where compatible, but [reference_basis_diagnostics.py](/workspace/TurboSOAP/reference_basis_diagnostics.py:14) targets a different optimizer layout and is not already wired into this trainer. Charge diagnostics to elapsed and exclude marked diagnostic steps from clean means using complete-cycle accounting. This fills a mechanism gap without a separate full run or an exhaustive eigensystem audit.

That is **six additional full runs at most for the current closing proposal**, plus one small profile. Sequence the work so a decisive sparse or baseline-efficiency result can stop or narrow later claims. None is launched by this review. A direct matched implementation of one subspace/adaptive prior method is required for a claim of superiority over that family; it is not required to publish the narrower measured QR10 result with an explicit non-comparison limitation. If broad competitiveness is essential, substitute a bounded comparator for optional algorithm tinkering rather than expanding into every paper's entire tuning budget.

## Freeze and paper decision

I would freeze the current mathematical version and its manifest now; keep all completed evidence and its early failures immutable. Begin drafting the methods and reproducibility sections immediately. Finalize quantitative claims only after deciding the strength of the QR implementation and sparse controls. The updated opening proposal in [RESEARCH.md](/workspace/TurboSOAP/RESEARCH.md:1) is substantially aligned with that approach.

A defensible present statement is: **On one 109.397M OpenWebText training seed with the declared stock-beta matrix/auxiliary recipe, warm10/QR50 reduced measured mean complete-step time by 10.81% relative to the tested QR10 implementation, with a final CE increase of .00698 and favorable sampled late target attainment.** This is already useful evidence, provided the quality cost, timing boundaries and single-seed limitation remain adjacent to it.

Do not claim quality equivalence, exact V transport, lower persistent-state memory, subcubic scaling, effective-rank or width advantages, 10× training acceleration, universal SOAP improvement, or a new general principle of warm basis updating. No second architecture or huge hyperparameter sweep is necessary for the narrow technical report. Broader generality needs broader evidence.

TransportMuon should remain separate. Its [current README direction](/workspace/TransportMuon/README.md:1) itself calls for efficient fresh-polar comparisons and matched warm qualification. The SOAP results do not validate those different state, rectangular-polar or normalization questions. A joint paper claim would inherit that unfinished evidence burden.

If warm remains a useful point on the measured quality–time frontier after efficient QR and sparse controls, the contribution merits a focused systems/optimizer report with precise related-work positioning. If it does not, preserve the kernel and numerical findings as a narrower technical contribution. There is no methodological reason to continue indefinite optimizer redesign before writing.
