# Precision attribution on unchanged CIFAR states

Predeclared 2026-10-04 UTC. Replay QR20, NS6 warm20 and a QR20 repeat on the
same seed139, covariance .99, Adam(.9,.999), both peak LRs .0005 and first20 of200
scheduled epochs. Observe all24 matrices at ages20/180/200/220/860/1740/3480.
No candidate changes training. Require all historical anchors, actual-refresh
Q/m/v controls and preceding full-refresh update differences to match exactly.

Keep the same five alternatives: QR; one NS6 step/cap.1; two NS6 steps/cap.1
with final or sequential m transport; one NS6 step/cap.2. Warm carries v. All
projections used to score the final implied updates run in strict ('highest')
FP32, as in the preceding experiment. Training uses the existing 'high' setting.

For every alternative form a path from the all-high refresh to the all-strict
refresh. First change only m-transport arithmetic to strict while holding every
candidate basis and its v fixed. Sequential transport retains the high first
and second bases, not only the final basis. Then change the basis path to strict.
Warm v is identical throughout. Compare the full-update discrepancy, isolated
fixed-basis transport discrepancy and remaining discrepancy after strict m.
Use the same1% reference agreement threshold; retain every failure.

For QR split the latter part further: strict covariance-times-basis/QR with the
high ordering held fixed; strict ordering while keeping high v; finally strict
v reindexing while keeping the strict basis. Record per-axis ordering equality,
mismatched index fraction and v equality. These intermediate states are diagnostic
counterfactuals, not proposed v policies. Adjacent vector differences telescope;
their norms do not add and are not percentages of causal importance. Use the
strict final-update norm as a common denominator for stage differences and
verify vector closure <1e-5. Also retain fixed-basis coordinate-m and world-m
differences so denominator amplification is distinguishable from transport size.

No timing or learning claims for precision interventions. This gate diagnoses
the prior full-refresh failure; it does not relax the budget or establish that
strict FP32 improves training. Basis arithmetic/QR ordering remain paired to
each sampled state, and selected-age statistics are descriptive only.

Budget10,500 updates plus compile smoke and offline probes;900s internal/1200s
external, one lifetime GPU claim under private supervisor, no queue/restart.
Require5GiB free; retain compact configs, scalar probes, logs, frozen source,
plots and hashes within20MiB. No weights/checkpoints/tensor snapshots. Incoming
references and counterfactual tensors are discarded after each probe. Preserve
all prior experiments and caches; run CPU checks before freeze/launch.
