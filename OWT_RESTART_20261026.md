> **Completed — 2026-10-26:** The declared calibration and four primary arms finished successfully. Our service exited and released its GPU claim. See `reports/owt_restart_20261026/completion_audit.json` and `comparison.json` for audited results. Stock QR/warm final CE3.411152/3.418128, previous QR/warm3.442197/3.451100; warm complete-step saving10.81%/11.15%. All numerical, state, source/input integrity and release checks passed. No successor is queued. The launch-status text below is retained as the preparation record.

# OpenWebText restart: stock SOAP betas and warm tracking

Prepared 2026-10-26 after the user authorized continuing with the existing shared
OpenWebText cache. This is a new experiment, not a replay of missing FineWeb
Trial213. The old machine's external run bundles and final Trial213 outcome
were not recovered. Historical successes and failures remain unchanged.

## Question and declared comparison

Compare held-out loss at matched updates and actual elapsed training progress
for QR10 versus warm10 with QR50 repair, at two matrix beta profiles:

- Stock beta profile: beta1 .95, adaptive beta2 .95, covariance decay .95.
- Previous beta profile: beta1 .9, adaptive beta2 .999, covariance decay .99.

The stock values match the official SOAP constructor (covariance beta .95 is
the effective value of upstream `shampoo_beta=-1`). This is a stock-beta profile
on the declared trainer/grouping, not a reproduction of an entire upstream
training recipe. The two profiles change beta1 as well as beta2/covariance;
between-profile differences cannot isolate any one decay's effect.

The QR path is the existing `ResearchSOAP`, verified against the pinned
upstream fixture including the new literal stock-beta case. The warm path is
the existing `FastSOAP` candidate, with the same ordinary parameter-update
ordering, epsilon/bias correction, and two-sided coverage. Its warm kernel
uses a skew-Frobenius cap .5, Jacobi damping .01, and two residual corrections;
BF16 operand products emit FP32 results. QR permutations reorder diagonal V;
warm refreshes leave diagonal V unchanged. Both paths transport first momentum
in the declared caller precision. No optimizer implementation is redesigned.

## Model, data, and execution

- Existing Transformer: width768, depth12, 12 heads, SwiGLU hidden1536,
  fused QKV/SwiGLU, RoPE, QK norm, tied embedding/head, no bias/dropout.
  Exactly 109,396,992 parameters. All48 hidden projection matrices receive
  full two-sided SOAP, giving96 factors with dimensions768/2304/3072.
- Embedding/tied head and normalization parameters use auxiliary AdamW,
  betas(.9,.999), peak LR .0006. Matrix LR is calibrated below. Weight decay
  .05 on linear matrix weights; no decay on embedding/norms; global clip1.
- Shared source: `/workspace/data/tokenized/openwebtext_gpt2_u16_1024_v1`.
  Dataset revision `79d93d786212f7344586290adb811d4ae6a1762c`, GPT-2 tokenizer
  revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Source split is at document
  level before packing. All41 declared shared-cache file hashes were verified.
- New fixed cohort: 320,000 unique training row IDs, selected with NumPy seed2205,
  and1,024 validation rows from the separate validation split, seed4205.
  Both beta profiles and both policies consume identical row order and validation
  examples. No within-arm training-row reuse for the full run. This does not
  establish semantic deduplication or restore historical input snapshots.
- Source blocks contain1,024 unpadded uint16 GPT-2 tokens. Inputs are the first
  1,023 and targets the last1,023; token0 is a valid target. No cross-block target
  is fabricated. Global batch32/physical8/accumulation4 yields32,736 targets/call.
  Selected token inputs occupy about628MiB plus row IDs, once, shared read-only
  between qualification versions. No raw dataset download or retokenization.
- Model seed204. FP32 parameters/optimizer state and BF16 model autocast;
  original eager transformer, compiled FP32 token cross-entropy, compiled warm
  refresh only. Flash attention. Caller float32 matmul precision `high` for both
  arms, reduced-precision BF16 reduction disabled. PyTorch2.11.0+cu128.
- Deterministic algorithms enabled and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
  Exact initial weights, repeated clipped gradients and initial validation CE
  must match across the four qualification arms on the selected device.
- No model weights, optimizer snapshots, resumable checkpoints or final weights
  are saved. Source/input hashes, row IDs, scalar logs, guards and results persist.

## Execution stages

1. CPU checks against pinned upstream and the warm policy, plus profile/coverage,
   pre-warm equality, refresh counts, and schedule tests. The frozen v3 source
   passes49 tests. Existing sources are copied, not edited for this campaign.
2. Four qualification arms at matrix LR .0005, each61 optimizer calls. The first
   call initializes SOAP state without a matrix update; the auxiliary branch
   still steps. Warm refreshes begin on call11 and QR repair is call51.
3. Baseline-only calibration: each profile's QR10 at peak matrix LRs
   .0005/.0015/.003 for500 calls, following the SAME full10,000-call cosine
   schedule and200-call linear warmup. Select the lowest call500 validation CE,
   then share that selected LR between QR and warm at that profile. This is a
   bounded early-trajectory calibration, not proof of a globally optimal LR.
4. Four fresh primary arms,10,000 calls/327.36M targets each. Full cosine horizon
   10,000, warmup200, minimum rate multiplier .1. Stock QR first/warm second;
   previous warm first/QR second. Reversing order does not remove temporal bias.
   All arms start fresh; calibration weights are not resumed.

Initial gradients are computed twice for repeatability without stepping. Those
extra backward passes are charged to elapsed startup, not counted as unique
training targets. Primary calls include one matrix-initialization call, so the
matrix age at call10,000 is9,999. Cadence is defined in matrix age.

## Guards, cost, and limitations

Hard stops: nonfinite loss/gradient/state; actual physical-momentum relative
error above .01 at first refresh and first repair; basis absolute Gram error
above .05 or Gram/sqrt(dimension) above .005; evaluation changing state on full
hash-checked calls; peak allocated GPU memory above24GiB; any arm above6hours.
Qualification also requires median complete step at calls22..61 below3seconds,
excluding actual-momentum diagnostic calls. The full campaign has a separately
declared supervisor/timeout wrapper and does not restart automatically.

Geometry observations occur at1/11/21/51/61, every250 calls and the final call.
Validation occurs at0/50/51/100/200/500, then every500 calls from1,000, and final.
All evaluation uses the same1,024 rows. Full model/gradient/optimizer hashes
before and after evaluation are checked at0/51/final; modes and gradient
identities and CUDA RNG are also checked. Final resident allocation must fall
below1MiB after deleting the model/optimizer and clearing the cache.

Report full scalar validation curves against actual elapsed time, including
startup/compilation/initial backward checks/evaluation/guards. Also report mean
optimizer and complete-step cost over full50-call cadence cycles101..10,000;
diagnostic synchronization outside the step remains charged to elapsed time.
Do not infer precise target crossing from sparse samples or combine unrelated
synthetic times with this learner. No universal .01CE quality threshold or
automatic default promotion is introduced.

This is one model seed and one fixed data order. Repeated validation panels are
not independent training replications. Calibration is short, main runs have a
new schedule, and other projects share the host. Full-factor costs do not imply
blocked-SOAP or width-independent speedups. Practical quality/time results
determine whether a later seed replication or hold-only control is worthwhile.

## Recovery and current locations

Active qualification bundle:
`/workspace/optimizer_replay_results/soap_owt_restart_20261026_v4`.
Code is frozen under `source/`, with a hash manifest in `provenance.json`.
All571 Python sources plus the upstream provenance fixture are retained.

The first startup (`soap_owt_restart_20261026`) failed exact gradient repeatability
before training. The deterministic second startup (`..._v2`) passed that check
but hit an event-logger keyword collision on its first optimizer call. Those
failures, logs, inputs and frozen sources remain preserved. v3 corrects the
logger without relaxing numerical gates. v3 then completed stock QR's61 calls
with all numerical/speed checks passed but stopped at its strict cleanup check.
A separate claimed minimal GEMM diagnostic showed a32MiB cuBLAS workspace
remaining after all GEMM tensors were deleted, and zero after clearing the
library workspace. v4 adds weak tensor/storage ownership observations and
clears only cuBLAS workspaces after training. Its first completed QR smoke has
zero live registered model/optimizer/gradient storages and zero final allocation;
two library workspaces accounted for64MiB before that clear. The original1MiB
final-allocation limit is retained. A separate initial supervisor wrapper
error occurred before launching any GPU child and was corrected by sourcing
the instance logging helper before enabling shell nounset.

All GPU work uses `/workspace/bin/gpu-claim` under supervisor. The CPU data
preparer uses `/venv/owt/bin/python`; GPU work uses `/venv/main/bin/python`.
Inspect `supervisorctl status soap-owt-qualification` and
`/var/log/portal/soap-owt-qualification.log`. Do not restart completed/failed
scientific attempts or overwrite their output directories. A corrected attempt
requires a new version with an explicit change record, as above.

The prepared learning service is `soap-owt-learning`, with `autostart=false`,
`autorestart=false`, no supervisor start retries, a12-hour external bound
including at most300seconds of queue waiting, and a fresh lifetime GPU claim.
It first requires a passed `qualification.json`, then performs the six declared
calibration arms and four primary arms. Any child failure stops the campaign;
there is no automatic retry or experiment beyond this declared comparison.
Its log is `/var/log/portal/soap-owt-learning.log`. Watch
`learn_status.json`, `calibration.json`, and eventually `comparison.json` in the
v4 bundle. The study's main model/data/horizon and learning-rate choices are
fully specified above; no implicit historical queue is restarted.

## Qualification result and learning launch

All four v4 qualification arms passed61 calls, with identical initial parameter
hashes, repeated initial gradient hashes and initial validation CE. Maximum
observed absolute Gram error was .017224 (<.05), worst physical-momentum
relative error .001083 (<.01), peak allocated GPU memory9.398GiB. Each arm
released all registered storages and ended at zero allocated bytes after
clearing library workspace. These short-run checks do not establish a learning
winner. The49 CPU checks are retained as `cpu_checks.xml`.

An additional same-weight/real-first-batch eager-versus-compiled CE screen
passed: global gradient relative difference .00138536 (0.1385%), cosine
.9999990404, absolute CE difference2.384e-7, unchanged model hash. The engineering
screens were relative difference<=.02, cosine>=.999 and CE difference<=.002,
inherited from the earlier execution-qualification framing; this is not a
training-equivalence guarantee. Receipt: `loss_execution_audit.json`; bounded
diagnostic source: `qualify_loss_execution.py` in the bundle.

The learning service was explicitly started after both qualification receipts
passed. It requires those receipts before acquiring the same physical GPU1
used for qualification. Stage order is six baseline LR calibrations, then the
four primary arms, with the failure/timeout limits above. Calibration/main
results are pending at launch; no completed primary quality or time-to-loss
claim is made. Inspect the live status before drawing conclusions or restarting.
