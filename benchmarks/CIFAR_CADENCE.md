# CIFAR-100 NS6 cadence and covariance quality screen

Use the corrected cached SNRAdam strong ViT recipe, seed139 and frozen train/
validation/train-panel split. Nine arms: QR20/QR40/NS6-warm20/NS6-warm40 at
covariance beta .999/.99, plus a final exact QR20/.999 control repeat. Adam
matrix/auxiliary betas(.9,.999), both peak LRs.0005, weight decay, augmentation,
BF16 compiled model/eager validated evaluation and FP32 optimizer path remain
the established recipe. Explicitly record six NS iterations for warm; preserve
two as the unused QR/default setting. QR-only never invokes the warm retraction.

All arms start from identical in-RAM weights and restored RNG/augmentation state.
Run3500 updates (20 epochs), evaluating at875/1750/3500 (5/10/20 epochs) on the
original200-epoch LR schedule with five warmup epochs. Do not compress that
schedule to20 epochs. Primary outcome: fixed-update validation CE at3500; accuracy,
earlier curves and train-panel CE are secondary. No best-checkpoint selection,
new-seed/convergence claim, LR retuning, synthetic-loss ranking or extra grid.

Covariance/Adam run each step; basis updates and m transport only on refreshes.
Warm refresh at20/40, replaced by QR each200 optimizer updates. Initialization
consumes the first trainer step without a matrix update: at trainer step3500,
matrix age3499 gives174/87 total refreshes for intervals20/40, with17 QR resets
and157/70 warm events. Multiply those counts by actual matrix parameter count.
QR-only has174/87 QR events. Warm carries v; QR permutes v. No adaptive fallback.

Check finite matrix parameters/moments and normalized Gram<.05 after cold start
and every refresh, before later resets can conceal drift. These are offline
diagnostics and their synchronized time is excluded from training/component cost.
Retain all check steps and maxima. Evaluation uses the existing eager policy,
parity and no-mutation checks. Batch/augmentation/RNG hashes must match across
arms. Require exact final-control weights/metrics at every observation and exact
QR20/.999 and QR20/.99 anchors against the earlier memory study where comparable.

Record full optimizer CUDA intervals, augmentation/model-backward/iteration
intervals, train wall, peak memory, counters, groups and hashes. First50 updates
are excluded from the reported steady component means as in prior CIFAR screens.
These small-factor CIFAR times apply to this trainer; never attach large-matrix
synthetic times to its loss. Shared-host timing variability and absence of
equal-wall observations prevent an equal-wall quality ranking.

Budget: nine arms,31500 trainer updates plus the existing compile smoke;1800s
internal,2100s external. Expected duration roughly12–16 minutes on one RTX5090.
One lifetime GPU claim/private supervisor, no wait/queue/restart. Check5GiB free
before launch. No model weights, checkpoints or moment snapshots. Preserve compact
configs, source/data hashes, scalar metrics and logs within20MiB; cached data stays
in place. Initial in-RAM weights are discarded. No retention changes elsewhere.

Freeze with `PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python
benchmarks/prepare_cifar_cadence.py --bundle <new-directory>` and run the frozen
`source/benchmarks/cifar_lr_screen.py --bundle <directory>` under the claim.
