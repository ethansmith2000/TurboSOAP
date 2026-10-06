# Covariance beta and natural refresh cadence screen

The next numerical/cost experiment keeps the original width768 rectangular MLP,
64 activation rows, four repeated synthetic batches and seed20261002. Both factors
remain active (dimensions768/3072). Initialization plus200 natural updates, two
identical-seed timing repeats, forward/reverse arm order. Covariance beta .999/.99,
QR or NS6 warm refresh every10/20/40:12 settings,24 cases. Warm replaces the event
at200 with QR. Keep Adam(.9,.999), LR.0005, decay.05, eps1e-8, basis LR.5, average
rotation cap.1, FP32 state/covariance products with `high` internal precision,
BF16 model operations and eager execution. No default or v-policy change.

Covariance EMAs update every step. Projection and Adam also run every step using
the current basis. Basis updates and m transport occur only at scheduled refresh;
warm carries v and QR permutes v. No error-triggered fallback or online monitor.
Warm10/20/40 has19/9/4 warm events plus one QR reset over200 updates. QR10/20/40
has20/10/5 QR events. Sparse warm and sparse QR are explicitly competing policies.

The quantity1-beta**frequency describes new covariance EMA weight between refreshes.
It does not quantify eigenspace movement or establish equivalent schedules. Keep
beta and cadence explicit; do not infer a preferred beta from numerical stability.

Reuse the rectangular screen's strict-FP32 diagnostics after every refresh:
finite state/Gram<.05, actual-gradient world-momentum EMA, per-refresh world m
preservation, same-state strict-FP32 implied-update shadow, and factor movement/
off-diagonal covariance residual. Predeclared limits: momentum relative error<=1%,
shadow update relative error<=1%, cosine>=.999, shadow Gram<.05. Stop geometry
failures; retain other diagnostic failures but do not promote them. Shadow
comparisons share the current covariance and v, so they cannot establish target
tracking correctness, diagonal-v accuracy or equivalent learning quality.

Time full optimizer events including covariance, projections, Adam, retraction,
m transport and QR v permutation. Cold initialization is separate. Diagnostic
and oracle-maintenance costs are outside per-step timings; raw first-event costs
are retained. All samples synchronize; one reset per warm cycle is sparse timing
evidence. Record repeat ranges, not only medians. All synthetic loss values are
sanity checks, not a task-quality ranking. No compilation or extra BF16 optimizer
products are mixed into this attribution study. Compare unchanged .999 QR10,
QR20 and warm10 numerical/loss records against the previous frozen run.

Use one lifetime GPU claim with private supervisor, no wait/queue/restart. Budget:
600seconds internal,750seconds external, at most4824 initialization/update calls.
Require5GiB free before launch. No weights, checkpoints or moment snapshots;
keep compact scalar/source evidence below20MiB. In-memory oracles are discarded.
Do not automatically expand into a real-data or larger numerical grid.

Prepare with `PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python
benchmarks/prepare_cadence_beta.py --bundle <new-directory>`. Execute the frozen
`source/benchmarks/rectangular_precision_gate.py --bundle <directory>` and summarize
with frozen `source/benchmarks/summarize_rectangular_precision.py --bundle <directory>`.
