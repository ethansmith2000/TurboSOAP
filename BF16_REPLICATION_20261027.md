# Full BF16 replication at model seed 204

The user authorized the proposed second full model-seed comparison. Replicate
the fixed full BF16 method for 10,000 calls at seed 204, using the completed
stock SOAP QR10 control for quality and a short stock replay for timing and
reproduction checks. The seed 205 result remains unchanged.

## Fixed method and reference

Bundle: `/workspace/optimizer_replay_results/soap_bf16_replicate_20261027_seed204_v1`.
Runner: `benchmarks/bf16_seed204_replicate.py`, using the existing comparison
harness. Optimizer source and numerical policy are unchanged: BF16 M/V/C/Q,
compiled same-shape batches, FP32 master parameters and accumulation/solver
islands, warm refresh every 10 updates with QR repair every 50. There is no
FP32-basis substitution or hyperparameter retuning.

Stock matrix betas (0.95, 0.95), covariance beta 0.95, peak matrix LR 0.0015,
epsilon 1e-8, decay 0.05 and clip 1. Original auxiliary AdamW and grouping,
peak LR 0.0006. Original Transformer with 109,396,992 parameters; physical batch
8, accumulation 4, global batch 32. BF16 model autocast, eager model and compiled
FP32 CE. Same OpenWebText cohort and order, with 1,024-token rows and 1,023
targets per row. Keep the full 10,000-call schedule, 200-call linear warmup and
minimum LR ratio 0.1. Model seed 204 is the only recipe change from full BF16
seed 205; this is a model-initialization replication with a fixed data order.

The quality reference is the original completed `primary_stock_qr` seed 204
record from the OWT restart, final CE 3.411151532. Copy the raw record, its
source/input provenance and a normalized view into the new bundle. Normalization
adds the gradient-hash field used by the current harness without changing any
measurement. Verify source, inputs and support-file hashes before every worker.

## Two sequential phases

1. Replay original stock QR10 for 201 calls. Its initial weight/gradient hashes,
   loss/norm and full-panel CE must match the saved seed 204 reference. Require
   exact validation reproduction at calls 0, 50, 100 and 200 before starting
   the full candidate. Validation at 201 completes the replay record.
2. Run full BF16 from scratch for 10,000 calls at the same initialization. Require
   the same pinned startup values and exact repeated initial backward.

No weights or optimizer checkpoints are reused. Quality curves compare with
the completed full control, saving a full stock rerun. New training totals
10,201 calls and 333,939,936 targets, including one matrix-initialization call
per phase. Auxiliary AdamW updates on initialization calls as in prior runs.

## Observations and interpretation

Keep the same full validation, 21 refresh-diagnostic windows, eight alignment
observations, four evaluation tensor-hash checks and capture checks from the
[full-horizon protocol](BF16_FULL_HORIZON_20261026.md), with seed 204. All 96
factors and 48 physical moments are observed at the prescribed windows, including
the final refreshes. Save scalar observations only.

Per the user's clarification, finite momentum/Gram errors are **diagnostics**,
not method-rejection or loss-equivalence criteria. Preserve the historical 1%
and Gram threshold flags for comparability, but do not abort, adapt or reject
a method solely for crossing them. Compare loss curves, final CE/perplexity,
time and memory. No arbitrary loss-matching margin or statistical equivalence
claim is imposed after observing results. Nonfinite/invalid state, reference
reproduction failures, unexpected graph changes and resource failures remain
operational failures.

Measure paired timing at calls 101..200 of both new phases on the same GPU.
Also report the candidate's full steady mean over calls 101..10000. Cached full
stock timing is contextual, not a freshly paired wall-time measurement. Include
packing, casts, parameter writes and auxiliary AdamW in timed optimizer work;
exclude diagnostics/evaluation from timed scopes and include them in elapsed
time. The compiler cache is new for this bundle and shared sequentially.

## Management and completion

Managed program `soap-bf16-repl204`, one shared gpu-claim allocation, sequential
phases, no autostart/restart/retries/successor. Admission waits up to two hours;
execution is independently bounded to two hours, 90 minutes per phase and 24 GiB
peak allocated GPU memory. Expected execution is roughly 55–65 minutes after
admission. Other claims are respected.

Live state is recorded in `progress.json` and `runs/<arm>/events.jsonl`.
Completion audits reference reproduction, event/diagnostic counts, evaluation
preservation and compiled capture stability; it writes a comparison and copies
compact evidence to `reports/bf16_replicate_20261027_seed204/`. The comparison
labels the historical numerical flags as diagnostic and distinguishes the paired
timing window from the cached quality reference. No model/optimizer tensors
are saved. QR20/QR50 controls remain later steps in the demonstration plan.
