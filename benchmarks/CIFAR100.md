# CIFAR-100 optimizer calibration

`soap_reference.py` implements the research QR reference and a matched warm
variant. The original SOAP class and defaults are unchanged. Dense 2D matrix
parameters only; QR uses upstream ordering and v permutations, with m reexpression
at basis changes. Optional squared-overlap v handling is a separate ablation.
The pinned upstream fixture and provenance are in `tests/reference/`.

`cifar_optimizers.py` assigns transformer-block Linear weights to ResearchSOAP,
patch/head weights to decayed AdamW, and other parameters to no-decay AdamW.
Matrix and auxiliary peak LRs retain their ratio under the shared cosine schedule.
`cifar_support/` preserves the corrected frozen SNRAdam recipe and eager evaluator.

Prepare a new immutable run directory:

```bash
PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python benchmarks/prepare_cifar_benchmark.py \
  --bundle /workspace/optimizer_replay_results/cifar_soap_calibration_new
```

Run its frozen `source/benchmarks/cifar_optimizer_benchmark.py --bundle <directory>`
using one lifetime GPU claim and managed supervisor, with a 1800-second outer
limit. No wait queue or second research GPU. Preparation checks source/data hashes
and disk space; the runner checks frozen inputs before training. No checkpoints
or final weights are saved. Shared branch states are held in RAM. Defaults are
128 common-prefix updates followed by 512-update branches, not full training;
matrix LR .001 has not been tuned.

The [completed calibration report](../../optimizer_replay_results/cifar_soap_calibration_20261001/README.md)
records reproduction, checks and limits. QR10 and warm10 cost almost the same;
QR20 saves 19.5% optimizer time. Warm5 is slower and has worse early CE. Unchanged
QR10 repeats bitwise in this run. GPU upstream comparisons quantify small
finite-precision differences from avoiding redundant m round trips, not bitwise
upstream equivalence. Warm Gram error is about 7e-4, versus 7e-7 for QR. Do not
infer convergence from this short one-seed screen.

The [20-epoch LR screen](../../optimizer_replay_results/cifar_soap_lr_screen_20261001/README.md)
is complete. Freeze a repeat with `benchmarks/prepare_cifar_lr_screen.py --bundle
<new-directory>` and execute its frozen `cifar_lr_screen.py` under the same claim/
supervisor rules. This screen adds fresh optimizers from identical initial states,
three matrix rates per policy, fixed-update and 75-second observations, and a
bitwise unchanged-control check. Summarizer and plotter scripts require completed
evidence; summary tests reject incomplete or repeat-mismatched studies.

The [lower-bracket follow-up](../../optimizer_replay_results/cifar_soap_lower_lr_20261001/README.md)
uses `prepare_cifar_lr_screen.py --bracket lower --bundle <new-directory>` and
adds beginning/middle/end controls plus phase timing. `bridge_cifar_lr_screens.py`
combines only fixed-update evidence after exact shared-anchor and source/config
checks; `plot_cifar_lr_screen.py --combined` plots the verified five-rate result.
The combined bracket selects .0005 as an interior best tested rate for all methods.

The three controls repeat bitwise but fail the wall-interpretation gate. Most
timing variation is in the model/backward interval; optimizer timing stays stable.
Do not pool timing across instrumentation versions. One-sided factors and general
v transport remain separate axes.

The [memory screen](../../optimizer_replay_results/cifar_soap_memory_20261002/README.md)
compares baseline (.999,.999), covariance-only (.99,.999) and matrix denominator-
only (.999,.99), with matrix/auxiliary LR .0005. Auxiliary AdamW beta2 stays .999;
`matrix_betas` overrides only ResearchSOAP, falling back to `betas` when absent.
Covariance .99 improves early CE for all schedules; matrix beta2 .99 worsens QR
and only slightly improves warm. Follow with longer paired covariance trajectories
and seed replication; this fixed-LR screen does not retune each memory setting.

Freeze with `prepare_cifar_memory_screen.py --bundle <new-directory>`, then run
the frozen `source/benchmarks/cifar_lr_screen.py --bundle <directory>` using one
claimed GPU and private supervisor (2100-second outer cap). It runs eleven fresh
3500-update arms including beginning/middle/end controls, with no wall-target
extension or saved weights. `summarize_cifar_memory_screen.py --bundle <directory>
--prior <lower-LR-directory>` validates effective groups and prior baseline anchors;
`plot_cifar_memory_screen.py --bundle <directory>` plots within-policy CE changes.

The [100-epoch covariance extension](../../optimizer_replay_results/cifar_soap_covariance_long_20261002/README.md)
uses `prepare_cifar_covariance_long.py --bundle <new-directory>`, then the frozen
`cifar_lr_screen.py`. Seven 17500-update arms use a 4200-second internal and
4500-second outer cap. `summarize_cifar_covariance_long.py --bundle <directory>
--prior <memory-screen-directory>` checks all six early trajectories and the final
control; `plot_cifar_covariance_long.py --bundle <directory>` shows paired curves.
QR10/warm retain covariance-.99 gains; QR20 is essentially flat at 100 epochs.
Additional paired seeds precede further memory-policy sweeps. No weights saved.

Runtime extrapolation now has a separate [synthetic scaling benchmark](SYNTHETIC_SCALING.md).
Its larger-factor retraction guard failures take priority before advancing a
warm speed claim. CIFAR's loss and timing conclusions remain scoped to this
workload; mock costs must not be attached to its loss curves.
