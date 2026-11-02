# Full-horizon BF16 confirmation, model seed 205

The user requested review of the completed five-arm pilot and launch of the next
experiment. The reviewed pilot is saved under `reports/bf16_learning_20261026/`.
The next experiment is one paired full-horizon confirmation, with a fresh model
initialization and a fixed BF16 policy. No search or automatic successor follows.

## Selection and scope

At2,000 calls on model seed204, full BF16 ended at CE3.993939 versus stock QR10
3.985905 and existing warm3.995753. Whole-step time was18.19% below the original
QR10 implementation, with half matrix-state memory (2.320->1.160GiB). Maximum
sampled late actual-M error was.5323%; the overall maximum remained.9924% at the
first warm refresh. The separate BF16 normalized Gram guard passed; the original
absolute Gram guard failed as expected. The small CE differences among compiled
variants do not establish a quality ranking. Full BF16 advances for its memory
benefit and successful sampled late checks. Original warm and compiled FP32
pilot evidence remain available for attribution.

This run addresses learning at the full schedule horizon and a new model seed.
It does not close efficient-QR attribution, QR20/QR50 controls, additional paired
seeds or literature claims. Stock QR10 remains the primary reference, with the
same original implementation used in the completed studies. No broad claim
against optimized or adaptive SOAP is supported by this pair.

## Frozen experiment

Bundle: `/workspace/optimizer_replay_results/soap_bf16_full_20261026_seed205_v1`.
Runner: `benchmarks/bf16_full_horizon.py` reuses the qualified storage harness.
Optimizer mathematics and precision policy are unchanged. The harness adds a
configurable model seed and a full-horizon schedule of observations.

Two fresh subprocesses run sequentially under one lifetime GPU claim:

1. `qr10`: original ResearchSOAP QR10 with FP32 state, stock control.
2. `bf16`: compiled batched warm10/QR50, BF16 M/V/C/Q, FP32 master weights and
   accumulation/solver islands, the same candidate as the completed pilot.

Model seed205 for both; data cohort/order remains identical to seed204. This is
a model-initialization replication, not an independently resampled dataset.
Each arm executes10,000 calls:327,360,000 targets and9,999 matrix updates after
one initialization call; auxiliary AdamW updates on all calls. Pair total:
654,720,000 training targets. Every arm starts from scratch; no resume or stored
model weights. Seed204's training results are not pooled into these timings.

All other recipe choices are fixed: the original109,396,992-parameter Transformer,
physical8/accumulation4/global32,1024-token blocks with1023 targets, BF16 model
autocast, eager model and compiled FP32 CE; matrix betas(.95,.95), covariance.95,
LR.0015, epsilon1e-8, decay.05 and clip1; auxiliary AdamW(.9,.999), peakLR.0006,
original decay grouping.10,000-call cosine schedule with200-call linear warmup
and minimum ratio.1. Same train.npy320000x1024 and full validation.npy1024x1024,
linked to the completed OWT cohort and hash-verified before each worker.

## Startup, observation and gates

Both arms must repeat their initial backward exactly. After the QR10 arm
completes, the BF16 worker derives its expected initial weight/gradient hashes,
loss/norm and full-panel CE from that paired control and checks them before
training. No seed204 gradient or CE is imposed on seed205. The candidate's
startup receipt identifies the paired reference explicitly.

Full1,024-row validation at calls0,50,100,200,500,1000,1500,2000,2500,3000,4000,
5000,6000,7000,7500,8000,9000,10000. Hash named parameters, present gradients and
listed optimizer tensors before/after evaluation at0/1000/5000/10000; also check
CPU/CUDA RNG and gradient identities. These hashes omit unlisted Python scalars
and model buffers and are not described as every possible state value.

All96 factors are checked at initialization and refresh ages
10,20,50,100,150,200,500,510,1000,1010,1500,1510,1990,2500,2510,5000,5010,7500,
7510,9950,9990. Actual-M transport is checked for all48 matrices at these21
refreshes. Coordinate offdiagonal covariance defects are observed at ages500,
1000,1500,1990,2500,5000,7500,9990. These are observations, with no adaptive
refresh or controller. All diagnostic work stays outside timed step scopes.

Carry the existing gates unchanged: finite state/nonnegativeV; original Gram
absolute<=.05 and normalized<=.005; separately BF16 normalized<=.005 and actual-M
error<=1%. Accuracy failures mark the result unqualified and do not change the
policy or thresholds; finite runs continue to retain the learning curve.
Nonfinite state, startup mismatch, unexpected capture changes or resource
failure stop the pair with no retry. BF16's expected failure of the original
absolute guard remains visible. No statistical equivalence or automatic winner
selection follows a single pair. The validation set is the previously used
research panel, not an untouched test set.

Steady timing uses calls101..10000:9,900 calls/198 complete50-call cycles. Include
parameter packing/writes and auxiliary AdamW. Per-arm elapsed includes startup,
compilation, checks, evaluations and diagnostics. A fresh bundle compiler cache
is shared sequentially; this is not an isolated cold-compilation comparison.
Capture counters at51/101/201/501/1001/2501/5001/7501/10000 must retain12 graphs
for BF16 and zero graph breaks. Peak allocation includes diagnostic work.

## Management and outputs

Managed program `soap-bf16-full205`; one shared gpu-claim allocation, sequential
arms; admission wait<=300s. No autostart/restart/retries/successor. Per-arm limit
90minutes; paired campaign limit3hours plus bounded admission;24GiB peak GPU
allocation. Expected duration about2–2.5hours on an RTX5090. Other jobs and GPU
claims remain untouched.

Freeze source, configuration, pilot selection evidence, CPU check XML and input
hashes. Every worker verifies source, input and support-file hashes. The live
`progress.json` and `runs/<arm>/events.jsonl` expose status. On completion the
parent audits all counts, initial matching, evaluation preservation and graph
counts, writes comparison/completion JSON, and copies compact scalar evidence
to `reports/bf16_full_horizon_20261026_seed205/`. No model/optimizer tensor
checkpoints are saved. A completed measurement and numerical qualification are
reported as separate statuses.
