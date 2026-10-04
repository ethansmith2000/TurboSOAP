# SOAP research: basis maintenance under a compute budget

Updated 2026-10-02. This document describes the current research motivation and
comparison protocol. Optimizer defaults are unchanged; proposed studies are not
claims of measured improvement.

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

## Execution funnel

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
