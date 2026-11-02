# BF16 state with FP32 bases confirmation

The completed seed205 full BF16 run ended at CE3.419354 versus stock QR10
3.409272, a+.010082 gap (1.013% higher perplexity), with18.66% lower measured step
time and half matrix-state memory. It remained finite, but actual-M error at
the first warm refresh exceeded the declared1% limit for three matrices, peaking
at1.0356%. Measurement audits passed; numerical qualification failed. The limit
remains unchanged.

The next candidate uses the existing BF16 M/V/covariance option with FP32 bases.
It passed the seed204 pilot with maximum actual-M error.5326% and both geometry
guards. Its matrix state is1.608GiB versus2.320GiB for stock, a30.68% reduction.
This confirmation tests full-horizon learning and numerical margin at seed205.
No optimizer mathematics or new numerical controller is introduced.

## Frozen experiment

Bundle: `/workspace/optimizer_replay_results/soap_bf16_hybrid_20261026_seed205_v1`.
Runner: `benchmarks/bf16_hybrid_confirm.py`. One GPU, sequential phases:

1. Original stock QR10 replay for201 calls at seed205. Its initial weights,
   gradients, loss/norm and full-panel CE must match the completed seed205
   control. Validation at calls0/50/100/200 must reproduce the stored values
   exactly before the full candidate starts.
2. BF16 M/V/C with FP32 Q, compiled warm10/QR50,10,000 calls from scratch at
   the same seed205. It must match the same pinned initial state and gradients.

The full10k stock QR10 result is copied and hash-pinned as the quality reference.
It is reused for the CE curve and memory comparison, saving about65 GPU-minutes.
The short replay provides a timing control on the newly claimed device. Report
paired prefix timing for exactly calls101..200 of both new phases, including
all refresh/repair events. The candidate's full9,900-call steady timing is also
reported, but cached full-control wall time is contextual rather than a newly
paired measurement. Diagnostic work is outside timed step scopes and included
in per-phase elapsed time.

All remaining recipe choices match the completed seed205 pair: original109M
Transformer and OpenWebText cohort/order, stock betas(.95,.95), covariance.95,
matrix peak LR.0015, epsilon1e-8, decay.05, clip1; original auxiliary AdamW;
physical8/accumulation4/global32;1024-token rows with1023 targets; full10k LR
schedule with200-call warmup/minimum ratio.1. Model seed205. Full1,024-row
validation panel. Source, input and copied reference hashes are verified before
each worker; no checkpoints or resume. New calls total10,201, with333,939,936
training targets. The quality reference's earlier10k calls are not counted as
new training.

## Checks and outputs

The full candidate keeps all21 refresh checks and eight alignment observations
from the [full-horizon protocol](BF16_FULL_HORIZON_20261026.md), including final
ages9950/9990. It repeats the initial backward exactly and checks evaluation
tensor/RNG/gradient preservation at0/1000/5000/10000. It must retain12 captured
graphs with zero graph breaks. Same finite-state/nonnegativeV gates and unchanged
original geometry guard (absolute<=.05, normalized<=.005) and physical-M<=1%
criterion. Accuracy failures mark the candidate unqualified; finite learning
continues. Startup/reproduction/nonfinite/resource/capture failures stop the
job without retry or adaptive changes.

Only the prefix and candidate are launched now. Further demonstrations remain
a draft in [DEMONSTRATION_EXPERIMENTS.md](DEMONSTRATION_EXPERIMENTS.md).
Compact results, audits and logs copy automatically to
`reports/bf16_hybrid_20261026_seed205/`. Every comparison labels the cached
reference and the100-call paired timing window. No model/optimizer tensors saved.

Managed program `soap-bf16-hybrid205`, no autostart/restart/retries/successor.
Shared gpu-claim admission waits up to2hours in FIFO order; no occupied GPU is
reclaimed. Expected execution roughly55–65minutes after admission.90-minute
phase limit,2-hour execution watchdog,24GiB allocation limit. Admission and
execution limits are separate. Read supervisor/claim status and `progress.json`
for live state; a queued program has not begun training.
