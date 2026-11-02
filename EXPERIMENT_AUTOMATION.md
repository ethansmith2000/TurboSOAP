> **Completed and closed, 2026-10-27:** All four ledger stages are completed. The scheduled monitor is disabled (app receipt verified; no next run). [Final report](reports/bf16_completion_20261027/REPORT.md) and `/workspace/soap-bf16-completed-backup-20261027.tar.gz` are saved. No remaining GPU work in this plan.

# Bounded SOAP experiment completion automation

The user authorized an automation to catch finished runs and proceed through
the necessary remaining experiments. T3 checks this thread every ten minutes.
The current frozen GPU job remains unchanged; this separately authorized
coordinator handles review and subsequent launches.

## Scope and limits

State: `/workspace/TurboSOAP/automation/remaining_experiments_20261027.json`.
The state pins the current replication, production-source hashes, stages,
budgets, scheduler ID and an eight-hour expiry. User corrections, pause or cancel
instructions take precedence. No subagents, new top-level threads or external
messages are needed. Report meaningful completions and failures in this thread;
keep unchanged-running checks brief.

Only one TurboSOAP GPU job may be queued or active at once. Every GPU operation
uses `/workspace/bin/gpu-claim`, and bounded long work uses supervisor with no
autostart/restart/retries. Respect other projects and FIFO admission. At most
45 additional GPU-minutes are authorized after the currently running replication:
reserve at most30 minutes for both refresh controls together and10 minutes for
the QR profile. Include compilation/startup while holding the GPU claim. Enforce
these reservations with real execution watchdogs, separately from queue time.
No sweeps, new datasets, larger models or further full-length training runs.

Momentum/Gram measurements are descriptive, not method-rejection thresholds.
Preserve historical flags. Judge loss, efficiency and actual stability. Nonfinite
or invalid state, source/input/reference mismatch, unexpected capture changes,
GPU failures and exceeded budgets require stopping the automation and reporting
the evidence. Do not automatically retry a failed GPU experiment or retune the
method. Small loss gaps of the sort already observed do not themselves trigger
an invented acceptance rule. A new method choice or material unexpected finding
requiring redesign should come back to the user.

## Reservation and duplicate prevention

Use `benchmarks/soap_automation_state.py` for atomic stage ownership. The CLI
never launches jobs; it records reservations under a file lock and atomic writes.

Before any stage work, check supervisor, shared GPU claims, bundle progress,
existing result files and the ledger. Reserve only the first pending stage whose
predecessors are completed. Choose a unique owner token and retain it. Record
the planned program and bundle before launching, transition to running, then
start the managed program once. An existing queued/running stage is monitored,
never submitted again. Reconcile result artifacts before treating an EXITED
program as success. Never erase claims or restart an ambiguous stage.

Before freezing follow-up code, verify the production-source hashes pinned in
the ledger. A changed optimizer requires a user-directed plan update; do not
silently overwrite it or treat a different method as the fixed experiment.
New harnesses and tests may be added for the already authorized controls.

Example, with the actual state path and a fresh owner token:

```bash
python benchmarks/soap_automation_state.py --state /workspace/TurboSOAP/automation/remaining_experiments_20261027.json status
python benchmarks/soap_automation_state.py --state /workspace/TurboSOAP/automation/remaining_experiments_20261027.json reserve --stage refresh_controls --owner OWNER --gpu-seconds 1800
python benchmarks/soap_automation_state.py --state /workspace/TurboSOAP/automation/remaining_experiments_20261027.json transition --stage refresh_controls --owner OWNER --status running --details-file /path/to/launch-details.json
```

The same owner completes the stage after auditing it. A failed stage transitions
to `needs_review`. If a preparing reservation has no known live work and cannot
be reconciled, stop and notify rather than reclaiming it and risking a duplicate.
Preparing for more than30 minutes is an unexpected condition requiring review.
On expiry, failures or finished work, disable the schedule with T3's
`update_scheduled_task(enabled=false)` using the stored scheduled task ID. Record
the reason locally. Disabling the monitor does not stop an already running GPU
job; stop a GPU job only for its own failure/watchdog or explicit user instruction.

## Stage 1 Review the current replication

Program: `soap-bf16-repl204`. Bundle:
`/workspace/optimizer_replay_results/soap_bf16_replicate_20261027_seed204_v1`.
If queued/running, do not launch follow-ups. Once completed, require supervisor
exit0, completed progress/results, source/input/support hashes, replay matching
and all event/observation/capture counts. Use the frozen runner's audit helpers.
Preserve raw results and merge the full BF16 seed204/205 paired quality evidence.
Seed205 reference/candidate is in
`reports/bf16_full_horizon_20261026_seed205/comparison.json`. Do not mix the
FP32-basis variant into replications of full BF16. Read current RESEARCH.md,
the replication protocol and DEMONSTRATION_EXPERIMENTS.md before interpretation.
Loss/perplexity deltas are primary; two model seeds do not establish statistical
equivalence. If proceeding would require a new method choice, stop for the user.

## Stage 2 Two short refresh controls

Prepare a new frozen protocol/runner and meaningful CPU checks, then run only
QR20 and QR50,2,000 calls each, sequentially on one GPU. Use model seed204 and
the same cached OWT order, full10k LR horizon, stock betas(.95,.95), covariance
.95, LR/decay/clipping/auxiliary optimizer/model/coverage as the replication.
Both controls use full BF16 M/V/C/Q and the existing CompiledSOAP arithmetic.
Only the refresh cadence changes: QR20 uses frequency20/repair20; QR50 uses
frequency50/repair50. These are pure QR controls with no warm events. Do not
change `bf16_soap.py` or the frozen production math to implement the controls.

Reuse the current full BF16 replication's first2,000 calls as the warm10/QR50
comparison. Keep the full1,024-row validation panel and matched initialization/
gradients. Count and test the actual cadence, including the skipped matrix
initialization call. Select momentum diagnostics at actual refresh ages and
adjust capture expectations: pure QR normally needs ordinary+QR shape graphs,
not the warm policy's12-graph assertion. Report complete shared timing cycles.

Allowed CPU work includes extending an opt-in harness/factory, labeling/counting
events, validating configuration and freezing source/data receipts. It does not
include an optimizer redesign, LR search or precision-policy change. Bound the
combined GPU execution to1,800 seconds including compilation/evaluation.
Audit the outputs and compare curves with the cached warm prefix. Short controls
cannot establish final-horizon superiority; narrow claims to their scope.

## Stage 3 Same state QR attribution profile

Prepare a small fixed-input comparison of original QR sorting/refinement and
the existing C@Q-reuse primitive on dimensions768,1536,2304,3072. Match dtype,
precision, shape/batching and inputs. Record numerical ordering/sign differences,
orthogonality and a meaningful update/state comparison outside timing. Near ties
can change finite-precision ordering; do not label algebraic equivalence as
bitwise agreement. Profile solver/GEMM costs with warmup and synchronization.

Hold one GPU for at most600 seconds. This is a profile, not a new training run.
Apply no new production change from the result. Separate batching, precision
and redundant-product attribution; current stock speed claims name the original
implementation until a stronger matched comparator is actually measured.

## Stage 4 Final synthesis and stop

Save a compact report with two-seed loss curves/deltas, memory, correctly scoped
timing, refresh-control findings and the QR attribution result. Keep the known
loss gap and diagnostic observations visible. Update the master research notes
and demonstration plan, preserve failure evidence and save a compact backup.
Do not launch a paper-scale benchmark suite or more experiments.

Mark all completed stages in the ledger and disable this scheduled task. Report
completion in this thread. If work stopped early, report the exact blocker and
completed stages without claiming the experiment plan finished.
