# Warm basis maintenance: retraction, normalization and execution cost

Updated 2026-10-05 (America/New_York). The original design and staged hypotheses
below are preserved. The completed round described here updates their status;
optimizer defaults and earlier frozen protocols remain unchanged.

## Movement follow-up completed (2026-10-06)

The [movement note](MOVEMENT_BOUND.md) and
[round report](../../optimizer_replay_results/soap_movement_round_20261006/README.md)
advance the first round below: a skew-Frobenius cap restores movement at almost
unchanged cubic2 cost, with modest fixed-update CIFAR gains over row clipping but
no QR/NS6 quality win. The attempted 90-second comparison fails timing repeat
qualification. Next qualify timing and separate bound tightness from movement
radius/correction work. Earlier plans below remain historical evidence.

## Completed first round and revised decision

[Full evidence and interpretation](../../optimizer_replay_results/soap_retraction_round_20261005/README.md).
At the first 3072-factor warm refresh, average cap .1 allowed rotation spectral
norm 3.757: the candidate was not near orthogonal. Later states were close enough
that conservative scaling itself added recovery work. A maximum absolute row-sum
bound allowed cheap fixed corrections through synthetic reset cycles up to factor
6144, but reduced movement and covariance-residual improvement.

One cubic correction passed those synthetic tests but failed the CIFAR momentum
guard (1.492% original-coordinate M error against the declared 1% limit). The
failure and replay are retained. Two cubic corrections and one quintic correction
passed 1,750 refresh guards and 1,680 targeted probes across ten revised CIFAR
arms, yet both had worse fixed-update CE than QR and NS6 at .99 and .999.
Approximation remains permissible: this is a workload-specific criterion, not
a universal loss criterion or grounds for silently changing old gates.

Compilation reduced complete-refresh fixture costs; graph replay was confirmed
and was slightly faster at factor 3072 but slightly slower at 6144 in these short
samples. Ownership copies are included. Natural-cycle compiled performance,
explicit BF16 casting and equal-budget learning remain unqualified.

Next inspect real CIFAR movement at reconstructed common states, then test a
less conservative bound or scheduled accurate/cheap corrections. Include checks,
transport and fallbacks in costs. Follow with paired measured-budget learning,
retaining fixed-update anchors. No broad coefficient/cap/precision grid. The
earliest local one-pass implementation differs from today's generator and
controllers; its relationship to the reported H200 run is still unverified.

## Original design review (recorded before this round)

## Decision and motivation

Investigate whether a cheap local retraction plus periodic stronger correction
can preserve useful Adam coordinates at lower total cost. Six Newton–Schulz
iterations are an experimental repair, not an established minimum. Approximate
tracking is acceptable if its learning/cost tradeoff is useful and error remains
controlled across reset cycles. Matching a strict reference update at every
refresh is a diagnostic question, not the definition of optimization success.

First reconcile implementations and diagnose the source of retraction work.
Then qualify a small number of fixed policies before adding online adaptation.
Every runtime check, cast, transport and fallback must be included in the cost
of a deployed policy. Expensive offline measurements need not become training
operations.

The user reports that their original H200 implementation was reliably and
substantially faster, with a small loss penalty. Treat this as a reproduction
target. The exact H200 source/configuration has not yet been reconciled with
the current experiments. A repository default is not proof of what that run used.
Hardware dependence is plausible, but we have no matched H200/5090 comparison.

## Implemented algorithm and experimental differences

For a matrix gradient G, maintain FP32 covariance EMAs C_L and C_R of GG^T and
G^T G, active bases Q_L and Q_R, and Adam M/V in those coordinates. Covariances
update every gradient-bearing step, with beta .99 or .999 in the current study.
Basis refresh and moment transport happen only at scheduled basis changes.
One-sided configurations omit a factor; this is a distinct algorithmic choice.

The recent ResearchSOAP step projects G into the current bases, updates Adam,
projects the adaptive update back and updates the parameters. It then updates
the covariances and refreshes bases if due. Cold initialization uses eigh and
does not take a parameter update on that first call. Subsequent QR refreshes
sort columns by estimated variances and QR-factorize C times the ordered basis;
they are power-iteration-style refreshes, not exact eigendecompositions.

A warm refresh, independently for each active factor, computes:

```text
B = sym(Q^T normalized(C) Q)
tau = damping * max(abs(diag(B)))
Omega_ij = B_ij (B_jj - B_ii) / ((B_jj - B_ii)^2 + tau^2)
A = clip(basis_lr * skew(Omega))
X = Q + Q A
Q_new = polynomial_polar_retraction(X)
```

The current generator has numerical floors. The basis learning rate is .5,
damping .01, and recent rotation caps are .1/.2. These are not parameter LRs.
The average cap clips ||A||_F/sqrt(d); the optional spectral-bound cap clips the
maximum absolute row sum. Neither policy scales small rotations up to the cap.

First moments are reexpressed through original coordinates at every basis
change. Strict transport optionally uses highest FP32 matmul precision only
for that operation. Basis precision stays separate. Exact preservation relies
on orthogonal bases and exact arithmetic; numerical drift can affect transport.

| Choice | Original SOAP class | Recent controlled ResearchSOAP experiments |
|---|---|---|
| Retraction iterations | Default 2 | 6 in recent screens |
| Warm cadence | Configurable, including optional residual gating | Fixed cadence; latest warm20 |
| Hard correction | Configurable QR/eigh and scheduling | QR at update multiples of 200 |
| Warm V policy | Carry; overlap transport optional | Carry |
| Reset V policy | Squared-overlap diagonal approximation | Column-permutation reindexing |
| Online adaptive fallback | Several optional controls in original class | No error-triggered fallback; experiment guards abort |

For warm20, updates 20–180 at multiples of 20 use warm refreshes; update 200
uses QR instead, then warm resumes. M is transported at all of these events.
A reset repairs the current basis; it does not undo parameter updates or recover
discarded cross-moments. QR may repair orthogonality without exactly diagonalizing
the current covariance. Stronger retraction alone does not fix stale tracking.

Source: [warm kernels](../soap.py), [research ordering and refresh](../soap_reference.py),
[CIFAR policy adapter](cifar_optimizers.py).

## What the existing evidence establishes

- [Large-square screen](../../optimizer_replay_results/soap_large_precision_20261002/README.md):
  two-iteration retraction failed the geometry guard at the first warm refresh
  at width 4096. This is not solely accumulated drift. Six iterations passed
  that tested trajectory; necessity of six was not established.
- The same screen compiled the warm tensor function with default mode,
  fullgraph=True and static shapes. Total optimizer savings were about 3–4%.
  This was not an exhaustive compiler, graph-replay or batching comparison.
- [Strict transport screen](../../optimizer_replay_results/cifar_soap_strict_transport_20261004/README.md):
  strict QR/warm cap-.1 refresh costs were 21.17/15.74 ms for 768×3072 matrices,
  but 76.82/98.99 ms for 1536×6144. These eager synthetic events do not establish
  sustained trainer throughput or hardware causality. The learning screen was
  one seed on small CIFAR factors; cap, precision and covariance beta interacted.
- [Rectangular precision gate](../../optimizer_replay_results/soap_rectangular_precision_20261004/README.md):
  BF16 gauge passed a coarse geometry check yet failed update agreement checks.
  This is evidence to diagnose, not proof that every BF16 design is unusable.

Keep historical thresholds and rejected samples intact. In particular, the
previous 1% full strict-refresh update diagnostic remains reported, including QR
failures. Any new practical acceptance criteria must be declared before the new
run and distinguished from that diagnostic. Approximation tolerance is not a
license to relabel an old failed gate as passed.

## Geometry: separate new error from inherited drift

Let A^T = -A, X = Q(I+A), and E = Q^T Q-I. In exact arithmetic:

```text
X^T X - I = -A^2 + (I-A) E (I+A).
```

For E=0, the spectral Gram error is exactly ||A||_2^2. With inherited error:

```text
||X^T X-I||_2 <= ||A||_2^2 + (1+||A||_2^2) ||E||_2.
```

This suggests local retraction can be cheap when both rotation and inherited
error are small. Finite-precision errors add to this model. The average cap does
not provide a dimension-independent spectral bound: ||A||_2 can be as large as
sqrt(d) times the average cap. The row-sum bound is safe in exact arithmetic for
a skew matrix but may suppress useful motion more than necessary.

Do not infer actual singular-value ranges from the nominal .1/.2 average caps.
Measure representative ranges offline. Keep Frobenius-normalized Gram metrics
and spectral/worst-direction metrics labeled separately; their values and
thresholds are not interchangeable.

## Three normalization decisions

### Covariance scaling

The current code divides C by max(abs(trace(C))/d, 1e-12) before forming B.
Ignoring floors and rounding, scaling B by any positive s also scales tau and
each diagonal gap by s. The generator's numerator and denominator both scale
by s^2, so Omega is unchanged. The generator is already scale-invariant.

Removing explicit covariance normalization could eliminate a trace reduction
and full-matrix division without changing the ideal rotation. This is an
algebraic candidate, not a proven numerical optimization: range, overflow,
underflow, damping floors and GEMM rounding may change. Compare first in FP32
over a range of covariance scales, then under reduced precision. If range
management is needed, investigate where scaling is cheapest rather than simply
removing all scale protection.

Absolute covariance magnitude is intentionally removed by this generator.
Preserving the raw double-bracket magnitude would define another tracker with
different LR/gradient-scale dependence. Do not conflate that redesign with
eliminating a redundant explicit normalization.

### Rotation clipping and natural movement

Current clipping applies A <- A * min(1, cap/magnitude). Rotations below the cap
retain their natural magnitude. As off-diagonal covariance tends to zero, the
rotation tends to zero. Normalizing every rotation to a fixed length would erase
this information and could keep moving a settled basis. Conversely, a small
generator can also reflect near-degenerate diagonal gaps or a stalled tracker;
small movement alone does not certify successful diagonalization.

Treat clipping as a stability/movement budget, not a demand to use the full
budget. Study cap geometry and retraction jointly, recording how much clipping
changes the proposed rotation. Do not silently compare cap values under different
norm definitions as equivalent step sizes.

### Scaling the retraction input

Current retraction forms H=X^T X and sets
s=min(1, 1.25/sqrt(max_i sum_j abs(H_ij))), then Z=sX.
This bounds the largest singular value conservatively. Many small off-diagonal
entries can make the row-sum bound loose, shrinking otherwise useful singular
values below one. Subsequent iterations must recover them.

Hypothesis: some of the six-iteration requirement comes from this scaling,
rather than the unscaled candidate's intrinsic difficulty. Record the scale,
actual singular-value range and per-iteration errors before claiming causality.
No-scaling is a controlled candidate only when its input regime is qualified.
Exact spectral normalization by SVD is too costly to assume as a runtime fix;
power-iteration estimates also cost work and are not guaranteed upper bounds.

Muon-style Frobenius normalization is poorly matched to an already orthogonal
square basis: ||Q||_F=sqrt(d), so it moves singular values from 1 to 1/sqrt(d).
Avoid creating a difficult recovery problem merely to normalize the input.

## Polynomial choice, iteration count and precision

With H=Z^T Z, compare these local corrections:

| Correction | Polynomial | Dense GEMMs per iteration |
|---|---|---:|
| Standard cubic | Z(3I-H)/2 | 2 |
| Current quintic | Z(15I-10H+3H^2)/8 | 3 |

The current docstring says “cubic Newton–Schulz”; the implemented singular-value
polynomial is quintic, with cubic local convergence. Correct that terminology
when the kernel is next edited; this documentation round does not alter source.
The GEMM counts describe straightforward evaluation, excluding rotation,
transport, reductions and scaling. Increasing current NS2 to NS6 adds twelve
dense GEMMs per factor per refresh. Compare error reduction per complete-refresh
time, not iteration counts alone.

For an initially orthogonal Q, no candidate rescaling, exact arithmetic and a
spectral rotation bound a, the largest candidate singular value is sqrt(1+a^2).
Evaluating the scalar polynomials gives the following worst absolute Gram errors
over the corresponding small interval. These are analytic examples, not GPU data:

| a | Before correction | One cubic | One current quintic |
|---:|---:|---:|---:|
| .1 | .01 | 7.475e-5 | 6.2267e-7 |
| .2 | .04 | .001184 | 3.94144e-5 |

Muon uses BF16, Frobenius scaling and aggressive quintic coefficients designed
to lift small singular values rapidly; the reference explicitly tolerates an
approximate polar result. Its transient update direction and our persistent
basis have different error consequences. One successful Muon iteration does
not determine the retraction requirement here. See the
[Muon implementation](https://github.com/KellerJordan/Muon/blob/master/muon.py).
Pin its revision if used as an experimental baseline.

Start with one/two cubic or current-quintic corrections and NS6 as a control.
Defer coefficient searches until actual input spectra show why these fail.
Alternative rotation constructions can reduce off-manifold motion, but matrix
exponentials, Cayley solves or higher-order polynomials add their own costs;
compare the whole update, including inherited drift and transport.

Keep persistent covariance and Adam accumulators FP32 initially, especially
with beta .999. Evaluate TF32-capable FP32 computation and explicit BF16 temporary
paths separately. “high” permits reduced internal matmul precision without
changing FP32 output dtype; it does not identify the exact dispatched algorithm.
[PyTorch precision semantics](https://docs.pytorch.org/docs/2.11/generated/torch.set_float32_matmul_precision.html).

An autocast wrapper around a function containing .float() conversions is not
evidence of a well-designed BF16 path. Profile actual casts/materializations;
.float() on an already FP32 tensor is normally not a conversion. Consider a
deliberate low-precision temporary region and optional final FP32 correction.
Keep sensitive reductions/accumulation precision explicit. Quantization can
impose an error floor that additional iterations do not remove. FP16 is a
separate candidate with different range; do not group all half precision together.
Keep m-transport precision and V policy controlled while testing retraction.

## Measurement and compilation are part of the cost model

Offline diagnostics may compute spectra, strict shadows and full per-iteration
geometry. Reconstruct representative states in memory and discard them; saved
tensor snapshots require a separately recorded purpose, size and retention.
Do not put these measurements into production training by default.

Prefer a fixed cheap correction and scheduled hard refresh as the first deployable
policy. A later adaptive policy must beat that control including its measurement,
decision, synchronization and fallback costs. Reuse Gram matrices already needed
by NS where possible, but the last iteration's input Gram does not certify its
output. A fresh output check can require another cubic GEMM. Reductions on GPU
do not inherently synchronize the CPU; .item(), Python branches and logging can.
If both branches are evaluated and then selected, charge for both branches.

Record covariance trace, diagonal maximum, rotation norm, retraction row-sum
reductions and pointwise matrix passes. They cost memory traffic and dependencies
even when their asymptotic work is smaller than GEMMs. Compilation can fuse some
passes; it cannot automatically remove the algorithmic need for a reduction.

| Execution option | What to establish |
|---|---|
| Eager | Reference semantics and event costs |
| Compile default, fullgraph | Fusion, graph breaks/recompiles, steady latency |
| Compile with CUDA graphs where supported | Replay benefit after copies/bookkeeping; state lifetime correctness |
| Selected autotuning or factor batching | Benefit after compilation cost, memory and shape constraints |

Default mode balances compilation/runtime overhead; fullgraph requires capture
without graph breaks, not a single GPU kernel. Reduce-overhead targets Python
overhead using CUDA graphs where applicable. A graph launch need not beat one
kernel or a small fused workload: its benefit is replacing multiple launches.
Separate compilation, capture, first launch and steady replay. Inspect whether
graphs actually activate and whether persistent outputs are safely owned across
replays. Keep process-wide precision switches outside captured kernels unless
their behavior is explicitly qualified.
[PyTorch compiler API](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html),
[NVIDIA graph overhead guidance](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/performance-issues.html).

Time the actual refresh and natural full-optimizer cycles as well as components.
Report device elapsed and host-observed wall time, warmup, repeat variation,
memory and diagnostics overhead. CUDA elapsed intervals can include dispatch
gaps. Profiling runs explain costs; unprofiled repetitions establish timing.
Cost estimates amortized over cadence do not substitute for measured throughput.

## Ordered next work and exit criteria

These stages are planned. No new experiment is launched by this document.

| Stage | Work | Required output / decision |
|---|---|---|
| 1. Reconcile original behavior | Recover exact H200 source/config; compare baseline QR, factor sides, shapes, NS coefficients/count, scaling, precision, cadence, resets, M/V and compiler versions | A discrepancy table with unknowns explicit. Reproduce available configuration on 5090; label any reconstruction. Do not block independent algebra/diagnostics on unavailable H200 access. |
| 2. Diagnose local retraction | Use controlled near-orthogonal/skew fixtures plus actual warm trajectory states at early/late ages and around resets; attribute incoming drift, clipping, scaling and iteration error | Show whether failure exists before rescaling, is introduced/amplified by it, or accumulates across refreshes. Verify covariance-scale invariance with floors/range exceptions. |
| 3. Qualify cheap fixed policies | Compare current NS2/NS6 controls with one/two cubic/quintic corrections; isolate scale removal and cap changes before combining | A small shortlist with complete refresh costs, geometry, tracking, world-M and update diagnostics over multiple reset cycles. Predeclare practical criteria; preserve old full-reference diagnostics. |
| 4. Optimize execution | Test TF32/explicit BF16 and compiler-only/graph replay on shortlist; fair matched-precision QR; include 3072/6144 factors and relevant smaller shapes | Repeated unprofiled timings, cast/reduction/GEMM attribution, initialization and compile amortization, natural cadence. No broad cross-product sweep. |
| 5. Learning and hardware replication | Short paired CIFAR learning with unchanged QR/NS6 controls, then seeds 271/811 and longer original schedule for survivors; exact portable benchmark on H200 when available | Quantify acceptable quality/cost tradeoff across seeds. Keep beta .99/.999, moment policies and factor sides explicit. LLM confirmation remains downstream. |

The first implementation deliverable should be a bounded diagnostic harness,
not a replacement optimizer default. Before launch, freeze the shortlist, shapes,
seeds, reset cycles, timeout, numerical stop conditions and storage budget.
Retain extreme cases as diagnostics; do not equate them with the distribution
of trained states. Replaying common states attributes local changes; separate
closed-loop trajectories are needed to observe the candidate's accumulated drift.

For numerical evaluation, distinguish finite/range failures, orthogonality,
covariance tracking, world-momentum preservation, selected-basis implementation
agreement and full strict-refresh sensitivity. An accurate polar result can still
track poorly; a strict-reference difference can coexist with useful learning.
Report all dimensions, then assess the predeclared learning/cost tradeoff.

Use at most one claimed GPU without queuing or disturbing other jobs. Check disk
capacity before launch. No periodic/final weights by default. Preserve compact
configs, seeds, versions, hashes, logs, summaries and an artifact manifest. Do not
rewrite historical results or archives to reflect a newer acceptance criterion.
