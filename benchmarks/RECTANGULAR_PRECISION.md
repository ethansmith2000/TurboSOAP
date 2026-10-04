# Rectangular natural-cycle retraction and precision gate

Replay the exact model/data recipe of the prior failed natural diagnostic: width
768 square + 4x expansion/contraction MLP, 64 rows, four repeated synthetic
batches, seed20261002, initialization plus 200 actual updates. Both/all and
smaller-side factors are separate arms. Factors reach3072. Two identical-seed
timing repeats reverse arm order. This is synthetic numerical evidence, not
learning validation or a comparison at equal quality.

Nine settings: QR10, QR20, original warm NS2, warm NS6/high, NS6/strict FP32,
NS6/BF16 covariance products, NS6/BF16 gauge, NS6/compiled gauge, and original
NS2 with the existing spectral-bound rotation cap. All warm schedules refresh
every10 with QR reset200. Adam(.9,.999), covariance beta.999, LR.0005, decay.05,
eps1e-8, basis LR.5 and cap.1 stay fixed. Model matmuls use BF16; persistent state
is FP32. No production optimizer/default changes. The conservative cap changes
the tracker, so record basis movement and covariance residual as well as geometry.

Keep per-step CUDA model/optimizer/whole intervals and wall time, cold init,
compile preflight and all refresh checks. Timings include the complete optimizer,
six retraction iterations where selected, moment handling and covariance products.
Diagnostics run after the timed interval. The subclass retains references only;
no diagnostic operations or tensor clones are inserted in the timed refresh.
Oracle memory is additional and identified in reported peak memory. Samples
synchronize; first-event costs are retained. One reset per cycle is sparse timing
evidence. Compiler preflight is separate; shadow evaluations never mutate state.

At every refresh, before a later reset can hide errors:

1. Check finite parameters/moments and normalized Gram error below.05, using
   strict FP32. Stop a failing case, retain its diagnostics, omit its cost summary.
2. Compare world momentum before/after actual transport. Separately compare
   actual coordinate m with its strict-FP32 reexpression using the actual bases.
3. Maintain a world-space .9 EMA of each actual gradient (skip initialization,
   as the optimizer does). Compare reconstructed stored m against that EMA.
   This detects accumulated basis/projection errors, without conflating differing
   parameter trajectories across arms.
4. For warm events, recompute the SAME algorithm in strict FP32 on the SAME
   covariance, incoming basis and m, retaining the SAME carried diagonal v.
   Compare implied post-refresh m/(sqrt(v)+eps) world updates: relative norm and
   cosine. This is a local precision shadow, not the next actual optimizer step,
   an exact second-moment oracle or a counterfactual training trajectory. It does
   not measure BF16 covariance error accumulated before the current state.
5. Record before/after normalized off-diagonal covariance residual and RMS basis
   movement for each factor. A stable but barely moving tracker is not validated
   as effective by the geometry check alone.

Predeclared provisional diagnostic budgets: maximum accumulated world-momentum
and per-transport world-momentum error<=1%, shadow update relative error<=1%,
cosine>=.999, and strict-shadow Gram error<.05. These screen large numerical
changes; they are not learning tolerances. Budget failures retain the full cycle
unless the existing geometry/finite guard stops it. Keep their conditional costs
labelled; do not promote them as passing candidates. No relaxed thresholds after
seeing results. Same-state FP32 shadows can themselves fail NS2 convergence;
record that rather than treating the shadow as an exact polar solution.

Resource budget:36 cases, at most7236 initialization/update calls,900seconds
internal and1050seconds external. One lifetime GPU claim, private supervisor,
no wait/queue/restart. Require5GiB free. No weights, checkpoints or tensor
snapshots. Keep compact evidence below20MiB and disposable compiler cache below
2GiB expected. In-memory oracle/refresh references are discarded after each case.

Prepare a fresh bundle with `PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python
benchmarks/rectangular_precision_gate.py --prepare --bundle <new-directory>`;
execute its frozen source with `--bundle` under the declared claim and budget.
