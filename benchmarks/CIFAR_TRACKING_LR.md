# CIFAR tracking diagnosis and bounded matrix-LR check

Seven 20-epoch arms: QR20 and NS6 warm20 at matrix peak LR .00025/.0005/.001,
plus a final QR20/.0005 repeat. Covariance beta .99, Adam(.9,.999), auxiliary
LR .0005, seed139, frozen strong CIFAR recipe and original 200-epoch schedule.
Evaluate at875/1750/3500 updates. Compare each family's best tested rate at the
predeclared3500 endpoint; no best-checkpoint or automatic bracket extension.
Only matrix LR changes: its associated decoupled weight-decay step also scales
with LR as in the existing optimizer. This is not globally retuned Adam/LR.

Both matrix factors active. Warm uses six retraction iterations, average cap.1,
basis LR.5, refresh20 and QR200. QR20 reindexes v; warm carries v. Every refresh
gets the existing finite-state/Gram<.05 offline check. Actual counts and effective
groups must match the previous cadence screen. No defaults or online fallback.

Attach an observer only to LR.0005 arms, including the repeated control. At matrix
ages20/180/200/220/860/1740/3480 (trainer step is age+1), retain references to the
incoming Q, m, v and accumulated covariance during the actual refresh. After the
timed update, score old/executed bases and three strict-FP32 alternatives on the
same covariance: QR, one NS6 warm step, two NS6 warm steps. Both warm steps use
the same covariance; they are not two trainer updates. Record normalized
off-diagonal covariance residual, Gram error, movement and rotation-cap binding.

For each full matrix, reexpress the same incoming world m into each candidate
basis; use QR's permutation-only v policy or warm's carried v as appropriate.
Record implied post-refresh update differences from executed state and world-m
preservation. Differences between algorithmic alternatives are not error against
ground truth. QR may substantially change a near-degenerate eigenspace and v
coordinates; lower off-diagonal residual alone does not prove better learning.
No alternative is executed or saved as a tensor. All 24 matrices/48 factors at
each selected age are observed; residual statistics stratify factor size and
actual warm-vs-reset events. The two actual training trajectories differ, but
each local comparison uses identical incoming state.

Diagnostics run after optimizer events and are excluded from component/training
timing. Snapshot references live only until that step's probe finishes; no disk
snapshots. Close the observer at the end of each arm. Tests and exact historical
LR.0005 trajectories check that observation does not change training. Use any
free claimed RTX5090; do not displace other jobs. Record its UUID, keep timing
separate across devices, and verify rather than assume old numerical anchors.

Budget:24500 trainer updates plus compile smoke,1800s internal/2100s external,
expected about9–12minutes. One lifetime GPU claim, private supervisor, no queue/
restart. Require5GiB free. No weights/checkpoints/moment snapshots. Keep compact
scalar/config/source evidence below20MiB. LR.0005 shadow probes and controls are
for attribution, not new-seed replication or a convergence claim. Do not infer
large-factor speedup from this CIFAR screen.

Freeze with `PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python
benchmarks/prepare_cifar_tracking_lr.py --bundle <new-directory>`; execute frozen
`source/benchmarks/cifar_lr_screen.py --bundle <directory>` under one GPU claim.
