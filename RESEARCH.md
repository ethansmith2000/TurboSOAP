# SOAP research: basis maintenance under a compute budget

Updated 2026-10-06. This document describes the current research motivation and
comparison protocol. Optimizer defaults are unchanged; proposed studies are not
claims of measured improvement.

## Current priority: qualify timing and separate radius from bound conservatism

The [movement-bound round](../optimizer_replay_results/soap_movement_round_20261006/README.md)
implements a skew-Frobenius bound with two cubic corrections. Common-state
movement and residual reduction improve substantially; natural-cycle cost stays
close to row-cubic2. Paired CIFAR fixed-update CE improves only modestly and still
trails QR/NS6. The 90-second training-wall comparison failed its predeclared
repeat criterion: identical QR work varied by 11.85% in wall time while optimizer
time varied only .84%. Exact numerical repeats do not establish timing stability.

Keep the candidate experimental. First qualify repeated/interleaved identical
work for wall-quality interpretation. Separately compare allowed movement radius
and correction work on common states, using a small fixed shortlist and declared
moment/geometry tolerances. Select on learning and total cost; better instantaneous
covariance residual is insufficient. Replication and longer horizons remain due.

The [movement note](benchmarks/MOVEMENT_BOUND.md) records the bound and scalar
Gram-error maps. The [earlier design](benchmarks/WARM_RETRACTION_DESIGN.md) preserves
hypotheses and failures. BF16, natural-cycle compiled ownership and exact H200
provenance remain open. Production/defaults are unchanged; historical dated next
steps below are superseded by this priority and the shared roadmap.

## Research question

At a fixed total optimizer-time budget, which allocation between infrequent QR
refreshes, frequent warm basis updates, and hybrid schedules gives the best
training progress while preserving useful Adam state?

SOAP applies elementwise Adam in covariance-derived coordinates. Maintaining
those coordinates costs time, and changing them changes the interpretation of
historical moment estimates. Better instantaneous diagonalization need not imply
a better optimization update. Conversely, cheaper basis updates need not produce
a cheaper optimizer once projection, state transport and retraction are included.
Neither sparse hard refreshes nor frequent warm updates is presumed superior.

## Baselines and attribution

The existing fixed-10 LLM reference is already smaller-side TurboSOAP with warm
updates and occasional QR resets. It is a cadence control for that method, not
standard SOAP. The completed 4,000-step comparison establishes an observed
14.3% optimizer-time and 1.11% step-time saving for sparse cadence; it does not
establish a speed advantage over a matched QR-based SOAP implementation.

The original SOAP implementation initializes with eigh, then performs one
covariance-times-basis power iteration followed by QR. That QR refresh is not
an exact eigensolve. Pin original source for semantic reference and remove
avoidable redundant work in a separately verified runtime baseline. An every-
reset configuration of this repository is not automatically upstream-equivalent:
alignment, v handling, normalization and update ordering must be checked.

Primary attribution holds active factors, precision, covariance accumulation,
learning rates, decay, moment policy and model/data fixed. Compare:

| Arm | Basis policy | Cost to include |
|---|---|---|
| QR | QR every K updates; unchanged basis between | QR, projections and moment handling |
| Warm | Warm updates every k with rare QR safeguards | basis motion, retraction, m/v handling and safeguards |
| Hybrid | Both operations on declared schedules | the full combined cost |

Calibrate schedules from measured cost, not equal refresh counts. Report both
quality at equal updates and quality at equal wall time. A tuned practical
comparison is separate from the matched-setting attribution experiment.

## One-sided is a separate algorithmic choice

Full factors of an m-by-n tensor have nominal decomposition work O(m^3+n^3)
and storage O(m^2+n^2). Smaller-side SOAP removes a potentially dominant factor,
as well as associated covariance and projection work. For shape 4d-by-d, the
nominal cubic factor work changes from 65d^3 to d^3; this is not a wall-time ratio.

Run primary tracker comparisons with matched two-sided factors, then selected
one-sided comparisons. Neither regime should stand in for the other. The dense
warm tracker still uses cubic GEMMs. Possible gains are hardware efficiency,
continuity and frequency, not an established asymptotic reduction. Measure
covariance accumulation, projection, basis update, moment handling, retraction,
and complete optimizer steps. Cold initialization is reported separately.

Large factors with meaningful but trackable drift are a plausible favorable
regime. Nearly stationary bases may favor simply infrequent QR; rapidly changing
bases may defeat small warm updates. Small factors can be dominated by launches
and state movement. These are hypotheses to measure, not preselected winners.

## Preserve first moments; test second-moment handling

For orthonormal bases, define U=QL_old^T QL_new and V=QR_old^T QR_new. Exact
first-moment coordinate reexpression is M_new=U^T M_old V. Ordinary controls
must preserve the physical first moment. Keeping it in original coordinates and
projecting it for updates is another representation, with a different cost
schedule. Finite precision and imperfect orthogonality still require auditing.

Only diagonal second moments are retained. Squared-overlap transport is
V2_new approximately (U elementwise-squared)^T V2_old (V elementwise-squared).
It discards unrecorded cross second moments and is path-dependent when repeatedly
applied. Column continuity, carry/reindex, and approximate transport each have
tradeoffs. Compare their update transients and history error against a small full
second-moment oracle before adding repeated transport to training defaults.

The original dense SOAP class uses exact-form first-moment reexpression at every basis
change; v stays with continuous columns on warm steps and receives squared-
overlap mixing on hard resets. Optional warm v transport remains experimental.
The paper emphasizes preserved momentum; the inspected sources do not establish
a clean isolated ranking of m-transport versus v-transport importance.

## Memory settings

Only covariance beta and Adam beta2 values **.99 and .999** are in the current
study. .9999 is excluded by user direction. Keep beta1 fixed initially. Vary the
two second-moment memories independently; slower geometry with faster denominator
forgetting is one hypothesis, not the preferred answer. Half-lives are about 69
and 693 updates. At 175 CIFAR updates/epoch these are .39 and 3.96 epochs.

Use matched settings for attribution and fair LR calibration for practical
comparisons. Normalization, epsilon, startup behavior and schedule must remain
explicit. Slow covariance evolution does not guarantee stable eigenvectors near
small eigengaps. Stable coordinates may help preserve long-lived Adam statistics;
long memory can also preserve coordinate-history errors.

Covariance beta interacts with refresh cadence, reset cadence, warm rotation
step/cap/retraction, Adam memory and learning rate. For an EMA updated every
optimizer step, the total weight on the next k observations is `1 - beta**k`.
At k=10/20 this is about 9.56%/18.21% for beta .99, versus .996%/1.981% for .999.
This is an EMA identity, not a rule equating eigenspace movement or proving a
beta/cadence compensation. Eigenvalue gaps, gradient statistics and the tracker
matter. One-factor studies establish conditional effects; they cannot select a
universal beta independently of the remaining optimizer settings.

## Separate learning and runtime measurements

Keep three distinct sources of evidence:

1. Real CIFAR/OpenWebText learning: paired inputs, validated loss, learning-rate
   calibration, beta/cadence interactions and replication at actual workloads.
2. Synthetic runtime scaling: matrix dimensions/aspect ratios, factor sides,
   activation workload, cold initialization, ordinary updates and refresh/reset
   events using the actual optimizer, including m/v handling and retraction.
3. Target-model confirmation: complete natural cadence cycles in the intended
   trainer and precision/compile setup before claiming sustained throughput.

The new [synthetic protocol](../optimizer_replay_results/soap_synthetic_scaling_v3_20261002/PROTOCOL.md)
uses a short square-plus-4x-rectangular mock block. Explicit event sampling avoids
missing rare QR resets in a few arbitrary steps. Cadence-weighted event estimates
remain cost estimates, not measured long-run schedules or equal-quality speedups.
Report one-sided/two-sided panels separately: removing a large factor changes
both the method and potentially most of its cost. Report factor sizes, not just
model parameter count. Dense warm work remains cubic; constants, launch overhead,
solver efficiency, precision and memory determine the measured crossover.

CIFAR optimizer-cost rankings are scoped to its matrix shapes and execution path.
Neither small-model timings nor a synthetic crossover establish the ranking on
larger models. Do not combine synthetic time with CIFAR loss into a time-to-quality
curve for a workload that was never run. Current CIFAR wall-time variability is
a separate measurement issue, not an explanation of every dimensional effect.

The [first completed stress map](../optimizer_replay_results/soap_synthetic_scaling_v3_20261002/README.md)
retains all 72 cases, including 18 warm geometry rejections. A focused natural
200-update mock check also rejects warm at update 10 (two-sided, 3072 factor)
and 110 (smaller side, 768 factor); QR two-sided passes. On the failing natural
3072 input, the existing two-iteration retraction gives Gram error .76476;
six shadow iterations give .000663. Shadows do not change the executed update.
This identifies inadequate retraction convergence on these inputs, not a validated
fix or a failure claim about real-data training. Revisit rotation/scaling and
retraction safeguards, include their full cost, and repeat natural-cycle checks
before advancing large-factor speed claims. Event-cost advantages alone are
insufficient, including for cases that passed the short forced-event probe.

### Width, precision and compilation

The [large-factor protocol](benchmarks/LARGE_PRECISION.md) adds depth-one square
projections at widths 2048/4096. Depth and width serve different purposes: reducing
depth controls aggregate memory and runtime while preserving each large factor.
The earlier width-1536 4x MLP already had 6144 factors; a larger model width does
not automatically mean a larger eigensolve. Always report active factor shapes.

Separate BF16 model operations, covariance products, warm-kernel products and
FP32 state storage. Test strict FP32 against reduced internal precision, and
give QR controls the same applicable optimizations. Compile pure tensor work
first, account for first-call compilation separately, and retain full-optimizer
event costs including m/v handling. Optimizer-only inference mode needs explicit
state-lifecycle testing because inference tensors restrict later mutation;
training forward/backward must remain outside that context. Timing gains require
geometry and eventual learning gates at the same precision. NS6 is an explicit
experimental retraction setting, not a production default change or a general fix.

### Rectangular numerical gate completed

The [rectangular follow-up](../optimizer_replay_results/soap_rectangular_precision_20261004/README.md)
reproduces the original NS2 failures at updates10/110 and measures their momentum
damage. NS6, its compiled kernel, and BF16 covariance products pass the declared
1% momentum/update budgets over200 updates on both factor settings. BF16 gauge
passes the coarse Gram check but fails the stronger budgets; defer that precision
setting. Spectral-bound clipping passes but moves3072-factor bases about61x less
than NS6/average cap on their respective trajectories, so keep it a separate
tracking-policy choice. No optimizer defaults changed.

Current schedules accumulate covariance and update Adam each step, refresh warm
bases/transport m every10, and replace warm with scheduled QR every200. Research
checks run after refreshes outside measured optimizer time, stop failed cases,
and do not implement automatic QR fallback. Any eventual online checks must have
their overhead included. Next isolate cadence and covariance beta (.99/.999) with
the numerically surviving NS6 candidate and sparse QR controls before real-data
quality confirmation. Numerical budget acceptance alone is not convergence or
equal-quality speedup evidence.

### Cadence and covariance screen completed

The [cadence/beta screen](../optimizer_replay_results/soap_cadence_beta_20261004/README.md)
passes all24 numerical cases: covariance .99/.999, QR or NS6 warm every10/20/40,
both factors active on the original rectangular workload. All six repeated .999
controls exactly match prior loss and diagnostic records. Scheduled QR200 stays
fixed for warm; this does not introduce adaptive checks/fallbacks into the optimizer.

Warm is cheaper at matched intervals, but warm20 costs about20% more than QR40;
warm10 costs27–32% more than QR20. Sparse QR remains a serious alternative to more
frequent warm refreshes. These are short synthetic costs, not quality rankings.
Next: bounded real-data quality screening of NS6 warm20/40 and QR20/40 with both
covariance betas, verifying adapter settings first. No default changes.

### Real CIFAR cadence quality screen completed

The [CIFAR NS6 screen](../optimizer_replay_results/cifar_soap_cadence_20261004/README.md)
completes nine 20-epoch arms on the original 200-epoch schedule. At fixed LR .0005,
QR20/40 beats NS6 warm20/40 in validation CE and optimizer cost at both covariance
betas. At .99, QR20/QR40/warm20/warm40 CE is 1.74838/1.75978/1.80780/1.85002.
Covariance .99 improves all four early endpoints relative to .999. The earlier
100-epoch study showed horizon dependence, so do not treat this as convergence.

All 1,227 per-refresh/initialization probes pass (maximum Gram .0013854); QR20
historical anchors and the repeated control match exactly at all observations.
The control optimizer-time range is .38%, but training wall varies 20.7% and
model/backward 43.8%; no equal-wall claim. CIFAR matrix sizes remain separate
from the synthetic large-factor runtime map. No defaults change.

That follow-up is now complete; see the tracking and LR result below.

### CIFAR tracking and matrix-LR screen completed (2026-10-04 UTC)

[Report](../optimizer_replay_results/cifar_soap_tracking_lr_20261004/README.md): seven arms, 24,500 updates, 10.20 minutes. QR20 and NS6
warm20 each test matrix LR .00025/.0005/.001 with covariance .99, auxiliary
LR .0005 and Adam(.9,.999) fixed. Both choose .0005 at the 20-epoch endpoint:
CE 1.74838 versus 1.80780, a warm deficit of .05941. The same original 200-epoch
schedule and seed139 are used; this is early sensitivity, not convergence or
new-seed replication. The bounded LR check does not remove the quality gap.

On warm-trajectory incoming states, mean local covariance residuals for factors
192/576/768 are .3467/.2852/.3095 after one NS6 step, .3102/.2318/.2576 after two,
and .1047/.1038/.1108 after QR. Every second warm shadow improves its sampled
residual, but dimension means still trail QR. The cap binds in 58.9/41.7/44.4%
of those samples. QR-trajectory shadows show the same qualitative pattern.
These are identical-state local alternatives; they are never executed in
training, and lower residual alone does not establish better learning. QR
permutes v and warm carries v; update differences do not isolate transport.

All 1,225 refresh/initialization guards pass (maximum Gram .00143845). Six
historical observations, three repeated-control observations and all repeated
tracking records match exactly. 147 prelaunch CPU tests pass. Scalar probes
cover 21 events/504 matrices/1,008 factors including the repeat. A different
physical RTX5090 was used; numerical anchors are verified, timing is not pooled.
Control optimizer range/mean .575%, training-wall 1.659%. Selected-rate optimizer
cost is 9.773 ms QR versus 10.778 ms warm on these small factors. No equal-wall
or large-factor speedup claim. No defaults changed or weights saved; GPU released
and private supervisor stopped.

Next: a bounded tracking-efficiency gate, with two inner warm steps and a modest
cap change tested separately against unchanged warm and QR controls. Check
geometry, world-m/update fidelity and complete refresh cost before any expanded
training comparison. Distinguish one final m transport from two sequential
transports. The two-step shadow has no measured cost or learning result yet.
Keep v handling as a separate ablation, retain sparse QR budget controls, and
preserve .99/.999 and longer-horizon replication on the roadmap. No further
training grid was launched in this stage.

### Local refresh efficiency and precision gate (2026-10-04 UTC)

[Report](../optimizer_replay_results/cifar_soap_refresh_efficiency_20261004/README.md): three unchanged CIFAR trajectories, 10,500 updates,
6.58 minutes. QR20, NS6 warm20 and the QR repeat reproduce all nine historical
anchors exactly. The 504 actual-refresh Q/m/v controls and repeated numerical
probes match exactly. All 525 training guards pass; 156 CPU tests pass.

On 144 warm-trajectory matrix records excluding QR resets, one warm step at
cap .1 gives mean residual .32972 at 2.930 ms per complete local refresh. Cap .2
improves this to .31530 at 2.926 ms, about 27.1% more residual reduction per ms.
Two inner steps with final m transport reach .28729 at 5.764 ms: 1.97× cost,
8.9% lower local reduction/cost efficiency. QR reaches .10611 at 2.258 ms.
These are eager small-factor event measurements including m/v handling, not
whole-optimizer speedups or learning progress. No variant changes training.

All 2,520 candidate records pass Gram and both world-m fidelity checks, but no
variant passes every 1% strict-FP32 implied-update agreement check. Rejections
per 504 records: QR493, warm1 49, warm2-final72, warm2-sequential187, cap-.2 53.
Totals include the exact QR repeat and are not independent observations.
QR's maximum update difference is 73.3%, despite world-m preservation error
below .067%. This exposes full-refresh precision sensitivity in the baseline
as well; it does not prove broken transport or worse QR learning. Strict FP32
is a numerical reference, not an established learning optimum. Sequential versus
final-only m transport changes the two-step implied update by up to 3.63%, with
identical bases/v; sequential transport also costs slightly more.

Next: hold candidate Q and v fixed to isolate transport arithmetic, then isolate
basis construction and QR ordering/permutation effects. Preserve the full
strict-refresh comparison; do not relax its threshold after observing failures.
Resolve that attribution before promoting cap .2 into a training intervention.
Defer more inner steps unless their measured cost can be justified. Retain sparse
QR, .99/.999 and larger-factor follow-ups, with costs and learning kept separate.
No defaults changed, no weights/checkpoints/tensor snapshots saved. GPU released,
private supervisor stopped, prior evidence preserved. No further GPU run launched.

### Precision attribution completed (2026-10-04 UTC)

[Report](../optimizer_replay_results/cifar_soap_precision_attribution_20261004/README.md): 10,500 unchanged training updates, 4.61 minutes,
504 matrix records and 2,520 precision decompositions. All nine historical
anchors, actual-refresh Q/m/v controls, repeated probes and all previous
full-refresh update differences reproduce exactly. All 525 training guards
pass; 164 CPU tests pass. No candidate changes training.

Holding the high candidate basis/v fixed and using strict FP32 m transport
resolves every sampled one-step warm full-refresh agreement failure: cap .1
49→0 and cap .2 53→0 per 504 records, including the QR repeat. Worst remaining
difference is .844%. On the warm trajectory's 144 actual warm-event records,
both caps go 3→0, maximum .534%. This is a sufficient local intervention, not
proof that transport alone caused every combined discrepancy or that learning
improves. Two inner steps retain 7/13 failures with final/sequential transport.

QR remains different: failures 493→492, maximum remaining difference 73.098%.
The strict-basis stage with high ordering held fixed still has mean magnitude
12.086% of the strict update norm. Ordering and v differ in every sampled QR
record; their separate stage magnitudes are not additive. Vector closure is
at most 3.33e-8. Tiny world-m differences can become much larger after the Adam
denominator; geometry/world-m checks alone do not bound the implied update.
No claim of broken QR learning or a proven spectral cause follows from this.

Next: an opt-in strict m-transport path, full refresh cost measurement, then a
bounded paired learning check for one-step warm cap .1/.2 against unchanged
controls if its targeted checks pass. Retain both fixed-basis transport checks
and the full strict-refresh comparison. QR resets remain explicitly unresolved
under the old full-refresh gate; do not relax that threshold retrospectively
or claim the hybrid policy now passes it. Keep .99/.999 and large-factor work
separate. No defaults changed, weights/checkpoints/tensor snapshots saved, or
further GPU job launched. GPU released and private supervisor stopped.

## Strict transport, shape costs and learning completed (2026-10-04 UTC)

[Report](../optimizer_replay_results/cifar_soap_strict_transport_20261004/README.md):
implemented opt-in `ResearchSOAP(transport_precision="highest")`, then completed
18 paired shape checks and fourteen 3,500-update CIFAR arms (49,000 updates).
Strict precision covers first-moment transport at refreshes, including QR resets;
basis construction, v policy and nonrefresh work retain their prior behavior.
The setting restores on exceptions. Production `soap.py` and defaults are unchanged.

At covariance .99/.999, strict transport lowers QR CE by .010363/.004975.
Warm responds conditionally: strict transport worsens cap .1 at both betas and
improves cap .2 at both. Raising the cap at .999 hurts inherited transport but
helps strict transport. Best tested QR/warm CE is 1.738019/1.788615 at .99 and
1.774667/1.843668 at .999. QR also costs less on this small trainer. These are
one-seed, fixed-LR, 20-epoch endpoints on the original 200-epoch schedule;
neither a default recommendation nor a convergence ranking follows.

Synthetic complete refresh costs reverse rank with dimension: strict QR/warm
cap .1 cost 21.17/15.74 ms at 768×3072, but 76.82/98.99 ms at 1536×6144.
Selective strict-transport overhead is small in these short samples. These eager
NS6 fixture costs are separate from sustained model throughput or learning.

All 176 CPU tests and 2,450 refresh guards pass. All 1,176 strict-transport
matrix probes match the selected-basis update oracle exactly; actual warm events
also stay below 1% full strict-refresh disagreement (maximum .675%). QR/reset
full-reference failures remain explicit: the whole hybrid policy does not pass
that separate gate. Twelve historical observations and both repeated controls,
including numerical probes, reproduce exactly. No online controller or fallback
was added. The process-wide precision setting is qualified only for sequential
eager use; compiled/concurrent optimizer execution remains untested.

Next: stop expanding this precision/cap grid. Replicate selected QR and cap-.2
warm comparisons on paired seeds 271/811, retaining inherited precision and
both covariance controls; then extend promising settings on the original
schedule. Separately qualify matched-precision compiled refreshes and natural
cadence at the 3072/6144-factor shapes. Keep strict transport opt-in. No further
job is launched this round; no weights/checkpoints/tensor snapshots were saved.
GPU released and private supervisor stopped; compact evidence is retained.

## Foundation execution funnel (completed stages and general method)

1. Audit actual LLM compiled/eager BF16 evaluation at identical trained weights.
   SNRAdam found a CIFAR-specific discrepancy; LLM parity is not inferred from it.
2. Exact small-moment replay: abrupt versus incremental rotations, correct m,
   carry/reindex versus squared-overlap v, and .99/.999 memories. Include analytic
   sign/permutation controls and a full second-moment oracle.
3. Measure refresh and state-handling cost by factor size and aspect ratio, both
   sides versus smaller side. Benchmark the actual dense warm implementation and
   QR, with instrumentation overhead separated.
4. Build a semantically audited QR baseline and a frozen CIFAR-100 adapter using
   SNRAdam's corrected eager evaluator, cached split and explicit parameter groups.
   Preserve group-specific matrix/auxiliary LR schedules and decay rules.
5. Compare budget-calibrated strategies using common-state branches and an
   unchanged-control repeat, then replicate selected seeds. Confirm useful
   candidates on the LLM. CIFAR's width-192 factors are a mechanism laboratory,
   not evidence of large-factor speedup.

Use one claimed GPU, no automatic sweep expansion and no default checkpoints.
Keep compact configs, hashes, seeds, metrics and logs. If a diagnostic needs
weights or moment snapshots, declare purpose, size and retention before launch.

## First foundation results (2026-10-01)

The current LLM parity gate passes at three trained states (maximum compiled/
eager BF16 CE difference 2.9e-5). The 24-case moment oracle confirms exact-form
m reexpression and path-dependent diagonal v handling. Twelve eager refresh
cases show warm+m slower at 192-square two-sided factors (1.943 vs 1.455 ms),
but cheaper at 1536-square (2.317 vs 13.099 ms). These include state movement;
the later CIFAR calibration below adds full-optimizer measurements. Warm and QR yield
different residual/orthogonality errors, so equal cost is not equal quality.

[Complete results and reproduction](../optimizer_replay_results/soap_budget_20261001/README.md).
The QR reference and frozen CIFAR adapter are now complete. ResearchSOAP uses
upstream update/epsilon/decay ordering, sorts/reindexes v on QR, and carries v on
warm steps. It reexpresses m at basis changes, eliminating redundant round trips
on unchanged-basis steps. CPU semantics are tested against a pinned upstream
fixture; GPU differences are measured, not assumed zero.

The [CIFAR report](../optimizer_replay_results/cifar_soap_calibration_20261001/README.md)
finds QR10/warm10 at 12.17/12.07 ms per optimizer update, with QR20 at 9.80 ms.
QR20 saves 19.5% optimizer time and 9.4% steady step time. Warm5 costs more and
has worse early CE. An unchanged QR10 repeat has identical final weights/loss.
Warm Gram error is about 7e-4 versus 7e-7 for QR. The endpoints are only 640
updates, still in warmup, with untuned matrix LR and one seed.

The [20-epoch LR screen](../optimizer_replay_results/cifar_soap_lr_screen_20261001/README.md)
now covers three rates for all three policies, with an unchanged control repeat.
Every method selects the lower bracket boundary .0005. At that rate QR10 and
QR20 reach CE 1.77703 and 1.77964, while warm10 reaches 1.84933. QR20 costs about
19.5% less optimizer time. The control repeats bitwise, but training wall time
varies 13.7% despite only .2% optimizer-time variation, limiting wall-time claims.

The [lower bracket](../optimizer_replay_results/cifar_soap_lower_lr_20261001/README.md)
now resolves that boundary: exact .0005 anchors allow combining five rates, with
.0005 best for all three policies. This is one-seed early-learning calibration,
not a converged optimum. Warm10 has higher CE at every tested rate.

Three bitwise controls show 14.76% training-wall range/mean, failing the predeclared
3% interpretation gate. Model/backward elapsed time varies 31.51%, optimizer .53%.
This localizes the measured variation without isolating its cause; CUDA elapsed
includes dispatch gaps. No wall-quality winner is promoted.

The [independent memory screen](../optimizer_replay_results/cifar_soap_memory_20261002/README.md)
holds both LRs at .0005 and auxiliary AdamW beta2 at .999. Changing covariance beta
alone from .999 to .99 improves 20-epoch validation CE by .05121/.03126/.07148 for
QR10/QR20/warm10. Changing matrix Adam beta2 alone to .99 changes CE by
+.02317/+.02109/-.00779. Faster covariance adaptation is the stronger follow-up
candidate here; neither slower nor faster second-moment memory is universally
better. Warm's improvement does not establish that it beats matched QR: at
covariance .99, QR10/QR20/warm10 CE is 1.72581/1.74838/1.77785.

These are controlled fixed-LR, one-seed interventions, not separately tuned beta
rankings or convergence results. No joint .99/.99 arm was run. This motivated the
completed longer covariance comparison below; additional paired seeds are next.
Hold v/factor-side choices fixed, track warm orthogonality, and investigate
execution timing separately. No method defaults were changed.
See [adapter instructions](benchmarks/CIFAR100.md) for freezing a fresh run.

## Longer covariance learning (2026-10-02)

The [100-epoch extension](../optimizer_replay_results/cifar_soap_covariance_long_20261002/README.md)
changes the early readout: covariance .99 improves QR10 CE by .03014 and warm10
by .01433, while QR20's earlier gain shrinks to .00052 (accuracy -.04 percentage
points). At covariance .99, QR10/QR20/warm10 CE is 1.26909/1.30446/1.31237;
at .999 it is 1.29923/1.30498/1.32670. Treat QR20's endpoint effect as essentially
flat on this seed, not evidence of a meaningful benefit. Warm remains worse than
both QR schedules at matched covariance settings; QR20 costs about 19% less
optimizer time than QR10/warm10 on these small-factor matrices.

All 18 early anchors exactly reproduce the previous screen. The unchanged QR10
control repeats weights/metrics at all seven observation points through 100
epochs. This is the same seed extended on the original 200-epoch schedule, not
new-seed replication or convergence. Matrix/auxiliary beta2 .999 and both LRs
.0005 were fixed; no beta-specific LR retuning or default changes.

Next: paired seed replication before another memory or transport-policy sweep.
Predeclare seeds 271 and 811 on the same split, staging one seed at a time under
the one-GPU rule, with all six covariance/policy settings and the same 100-epoch
endpoint. Preserve the .999 controls: the covariance effect is schedule- and
horizon-dependent here, and its mechanism is unresolved. No seed job was added
to this campaign. Wider-factor confirmation remains necessary before dismissing
or promoting the warm approach; small CIFAR results have not shown its advantage.

## Evidence and references

- [SOAP paper, sections 3–4](https://arxiv.org/html/2409.11321v2#S3).
- [Original SOAP implementation](https://github.com/nikhilvyas/SOAP/blob/main/soap.py).
- [Subspace basis updates preprint](https://arxiv.org/abs/2605.26327): abstract reviewed; full comparison pending.
- [Shared current roadmap](../manifold_opt/optimizer_research_roadmap.md).
- [Research journal](../manifold_opt/optimizer_research_journal.md).
- [Receiving-device results](../optimizer_replay_results/rtx5090_cu128_20260930/README.md).
- [Current diagnostic bundle](../optimizer_replay_results/soap_budget_20261001/).
