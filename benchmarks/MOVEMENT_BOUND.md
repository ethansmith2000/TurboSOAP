# Retain basis movement with inexpensive retraction

2026-10-06 (America/New_York). Experimental follow-up to the
[retraction round](../../optimizer_replay_results/soap_retraction_round_20261005/README.md).
Production optimizer defaults are unchanged.

## A bound that uses the skew structure

The Jacobi generator A is explicitly antisymmetrized. A real skew matrix has
paired nonzero singular values. Consequently,

```text
||A||₂ <= ||A||F / sqrt(2)
A_clipped = A * min(1, cap / max(||A||F / sqrt(2), epsilon))
X = Q + Q A_clipped
Q_new = cubic_polar(cubic_polar(X))
cubic_polar(Z) = 1.5 Z - 0.5 Z (Zᵀ Z)
```

This bound is tight for a rank-two skew rotation. It needs one norm reduction
and a scalar rescaling, without an eigensolve, power iteration, synchronization,
or tensor-dependent host branch. Two cubic corrections use four GEMMs after
candidate construction. The full refresh still includes covariance projection,
candidate movement, both factors and moment handling. Measure that full cost.

This does not establish perfect orthogonality or transport in floating point.
The candidate inherits any incoming Gram error. The cap bounds the new skew
movement, not the accumulated error. Retain periodic QR resets and diagnostics
across multiple reset cycles. The original normalized-Gram and world-M criteria
are experimental qualification criteria, not definitions of useful learning.

In exact arithmetic, a cubic correction maps a Gram eigenvalue error delta to
`-.75 delta² + .25 delta³`. For orthogonal incoming Q and cap .5, candidate
errors lie in [0, .25]; two cubic corrections leave a worst scalar Gram error
of about .0014. One quintic correction instead maps delta to
`.625 delta³ - .234375 delta⁴ + .140625 delta⁵`. The shadow quintic policy uses
cap .4, giving delta <=.16 and a worst scalar Gram error about .0024. These
calculations motivate fixed work without a runtime eigensolve. They assume
orthogonal incoming Q and exact arithmetic; real trajectories still require
the inherited-drift and moment checks above.

The Frobenius bound is not always tighter than maximum absolute row sum. Its
conservatism depends on the rotation spectrum: the ratio to the exact spectral
norm is sqrt(effective_rank/2), where effective_rank = ||A||F²/||A||₂². In dense
high-rank motion it can remain overly restrictive. A fixed spectral cap also
differs from the existing dimension-normalized average cap; their numerical
values are not interchangeable.

## Completed common-state attribution

[Frozen experiment](../../optimizer_replay_results/cifar_soap_movement_attribution_20261006/README.md).
Reconstruct NS6 and row-cubic2 CIFAR trajectories at covariance .99/.999, with an
exact repeat. At seven ages around initial and QR-reset transitions, apply all
alternatives to the same incoming covariance, Q and M. Alternatives never feed
back into these trajectories. Both candidates use high basis arithmetic; offline
spectra and moment diagnostics use highest FP32 precision.

Across 1,344 nonrepeat factor states, Frobenius-cubic2 cap .5 increases mean
movement from .02217 to .03852 and makes mean residual change more negative,
from -.01394 to -.03051, compared with row-cubic2 cap .5. Maximum sampled
world-M error is .001255 (0.126%). The Frobenius bound is tighter in 1,343 of
1,344 sampled states; that empirical result is not a general bound ordering.

At age 20, median Frobenius/exact-spectral bound ratio is 2.18 versus row/exact
2.99. At age 400 the ratios are 1.59 and 5.92. The more conservative clipping is
therefore not confined to the first refresh. Even this cheaper improved bound
leaves room between measured movement and the offline exact-spectral oracle.
That oracle requires an eigensolve and is not proposed for deployment.

Frobenius-quintic1 cap .4 also passes sampled shadow checks but moves less. Only
cubic2 cap .5 advances in this round, preserving polynomial, cap value, reset
schedule and moment policy for a direct row-versus-Frobenius intervention.
Immediate covariance-residual improvement does not establish improved learning.

## Qualification and learning design

[Natural cycles](../../optimizer_replay_results/soap_movement_natural_20261006/README.md):
all 18 cases complete 420 updates, through QR resets at 200/400, widths 768/1536
(largest factors 3072/6144) and both betas. Per-refresh and whole-history world-M
errors remain below the prospective 1% gate. Both repeated model/loss/diagnostic
sequences match exactly. At width 1536/beta .999, mean optimizer costs are 18.23 ms
QR, 20.04 ms NS6, 14.25 ms row cubic2 and 14.23 ms Frobenius cubic2. The two cheap
policies have similar cost; the motivation is improved movement at that cost.
These are instrumented synthetic cycles, not uninstrumented model throughput.

The completed [paired CIFAR screen](../../optimizer_replay_results/cifar_soap_movement_budget_20261006/README.md) uses ten arms, fixed 3500-update anchors and a
predeclared 90-second measured-training-wall endpoint. The budget excludes
initialization, evaluation and offline diagnostics, while including model and
launch/dispatch work. It is not an equal cumulative optimizer-time budget.
Repeat qualification requires <=3% fixed-update training-wall variation and
<=2% optimizer variation for both repeated .99 controls. If these fail, retain
the budget endpoints as exploratory. Original update-indexed LR schedules are
unchanged, with no per-arm horizon or LR retuning. Longer horizons and seed
replication remain necessary before promotion.

The screen completed 40,675 updates. Frobenius improves fixed 3500 CE over row
cubic2 by .00344/.02721 at covariance .99/.999, but still trails QR and NS6.
The QR repeat has 11.85% training-wall variation with exact numerical results;
Frobenius has .13%. The prospective wall-budget gate therefore fails. Preserve
those endpoints as exploratory and qualify timing before any time-to-quality
claim. Increasing local residual reduction was not sufficient to close the
learning gap. The current roadmap separates timing qualification from further
radius/correction experiments.
