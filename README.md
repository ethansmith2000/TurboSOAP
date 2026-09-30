# TurboSOAP

TurboSOAP is SOAP with a warm, matmul-oriented eigenbasis tracker. It retains
the standard SOAP update:

```text
gradient -> project into covariance bases -> elementwise Adam -> project back
```

The implementation is intentionally narrow. Each covariance basis is
initialized with `eigh`, follows a damped Jacobi/double-bracket flow between
refreshes, and is periodically reset with QR or `eigh`.

See the [long-form method write-up](BLOG.md) for the motivation, equations,
state-transport details, limitations, and experiment plan.

## Installation and use

`soap.py` contains the optimizer and only requires PyTorch.

```python
from soap import SOAP

optimizer = SOAP(
    model.parameters(),
    lr=3e-3,
    weight_decay=0.01,
)

for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
```

`soap_block.py` provides the experimental `BlockSOAP` variant. It stores each
covariance factor in its current basis and updates the basis with disjoint exact
2x2 rotations or small Cayley blocks, avoiding dense tangent retraction on
ordinary refreshes:

```python
from soap_block import BlockSOAP

optimizer = BlockSOAP(
    model.parameters(),
    lr=3e-3,
    betas=(0.95, 0.99),
    shampoo_beta=0.999,
    precondition_frequency=10,
    basis_pairs_per_refresh=0,  # one complete parallel matching
    basis_pair_selection="round_robin",
    basis_block_size=16,        # damped near-identity Cayley blocks
    basis_block_rotation_cap=0.25,
    basis_block_stall_threshold=1e-6,
    basis_block_seed=0,
)
```

`round_robin` pair selection avoids a device synchronization and eventually
visits every coordinate pair. `top_correlation` greedily targets the strongest
current correlations but synchronizes once per active factor refresh.
With `basis_block_size > 2`, the tracker uses reproducibly shuffled blocks and
batched Cayley updates to correct several within-block correlations while
preserving a near-identity coordinate change. The refresh count and
`basis_block_seed` determine the schedule, so unrelated random-number use does
not change it. The default uses the local block step without probability
compensation. Set `basis_block_unbiased_scale=True` to multiply by the inverse
pair-inclusion probability before applying the block spectral cap; this is an
experimental estimator and can become too aggressive for small blocks in large
factors.
If the damped generator stalls on an equal-diagonal correlated block, the
tracker applies one capped Jacobi pair rotation within that block. This fallback
is selected on-device.

Run `python benchmarks/benchmark_basis_refresh.py` to compare one-factor dense
and block refresh time and diagonalization progress across factor sizes.

`benchmarks/replay_basis_tracking.py` replays rotating, shock, and equal-diagonal
covariance trajectories through full eigendecomposition, a held basis, exact
Jacobi pairs, and Cayley blocks. It reports off-diagonal error, eigen residual,
orthogonality, reset/refresh counts, update-device time, and wall time.

```bash
python benchmarks/replay_basis_tracking.py --device cuda:0 --trajectory all \
  --size 64 --steps 64 --reset-every 16 --block-size 16 \
  --output soap_replay.json
```

Use `--no-unbiased-scale` to test raw local block steps. The probability
compensation can be too aggressive when only a small fraction of pairs is sampled
and the spectral cap is active, especially as factor size grows.

`benchmarks/benchmark_optimizer_step.py` compares complete dense/block and
one-/two-sided optimizer steps on a rectangular tensor, including steady-state
device time, wall time, optimizer-state memory, active factors, and peak temporary
allocation:

```bash
python benchmarks/benchmark_optimizer_step.py --device cuda:0 \
  --rows 1024 --columns 256 --precondition-frequency 4 --block-size 64 \
  --output soap_optimizer_step.json
```

Add `--normalize-grads` to compare variants at matched per-step update RMS. Without
normalization, one-sided and two-sided preconditioning can have substantially
different effective step sizes and require separate learning-rate tuning.

The first `step()` observes a gradient, initializes covariance statistics and
their eigenbases, and does not update parameters. `TurboSlimSOAP` is an alias
for `SOAP`.

## Basis tracker

For one covariance factor, define

```text
B = Q.T @ C @ Q
```

TurboSOAP minimizes the off-diagonal covariance energy. Its skew generator is

```text
Omega_ij = B_ij (B_jj - B_ii)
```

and the implemented damped Jacobi generator is

```text
Omega_ij = B_ij (B_jj - B_ii) / ((B_ii - B_jj)^2 + tau^2)
```

where `tau` is controlled by `basis_jacobi_damping`. A tracker refresh applies

```text
Q <- retract(Q + Q @ (basis_lr * Omega))
```

after clipping excessive rotation. Small directions are never normalized
upward, so the update naturally vanishes in a diagonal basis.

## Main options

- `precondition_frequency`: optimizer updates between warm basis refreshes.
- `precondition_frequency_after_warmup`: optional deterministic refresh period
  after `precondition_frequency_warmup_steps`. Both default to `0`, which keeps
  one fixed cadence.
- `basis_residual_threshold`: optional normalized off-diagonal residual gate.
  At a nominal refresh, it measures
  `||offdiag(Q.T @ C @ Q)||_F / ||Q.T @ C @ Q||_F` and refreshes only when the
  threshold is crossed. The checked scalar transfer synchronizes CUDA, so this
  is an experimental diagnostic rather than the recommended throughput path.
- `basis_residual_max_age`: force a basis refresh after this many optimizer
  updates even if the residual remains below threshold.
- `basis_residual_warmup_steps`: retain the fixed nominal cadence for an initial
  bootstrap before residual gating begins.
- `basis_lr`: damped Jacobi step size.
- `basis_lr_age_compensation`: opt-in compounding of the tracker step over the
  actual age since the previous refresh:
  `1 - (1 - basis_lr) ** (age / reference_age)`. The result is still bounded
  by `basis_rotation_cap`.
- `basis_lr_reference_age`: reference age for that compounding; `0` uses
  `precondition_frequency`.
- `basis_rotation_cap`: maximum average column rotation in one refresh; `0`
  disables clipping.
- `basis_rotation_cap_mode="spectral_bound"`: use a conservative row-sum bound
  on the rotation spectral norm instead of the original average-column cap.
- `basis_jacobi_damping`: relative pairwise eigengap damping.
- `basis_stall_pair_threshold`: enable an exact largest-pair fallback when the
  dense flow stalls near equal Rayleigh quotients. Default: `0` (disabled,
  because checking the condition synchronizes the device).
- `basis_block_unbiased_scale`: inverse-probability scale for sampled pair
  generators. Default: `False`; the uncompensated local step was more stable
  across the included factor-size replay.
- `basis_block_stall_threshold`: relative on-device stall threshold for Cayley
  blocks larger than two. Default: `1e-6`; `0` disables the fallback.
- `basis_reset_frequency`: hard-reset interval in basis refreshes; `0` disables
  periodic resets.
- `basis_reset_max_age`: optional optimizer-step bound between later hard
  resets. This keeps reset age bounded when basis refreshes become sparse.
- `basis_reset_stagger`: distribute parameter resets across a centered
  one-period window before continuing at the configured cadence. This preserves
  long-run reset work while avoiding a single step that QR-resets every active
  tensor. Default: `False`.
- `basis_reset_method`: one-step orthogonal iteration (`"qr"`) or exact
  eigendecomposition (`"eigh"`). A reset reconstructs covariance in world
  coordinates and produces a fresh orthonormal basis, repairing accumulated
  singular-value drift.
- `basis_ns_iterations`: Newton--Schulz iterations in the warm retraction.
- `max_precond_dim`: skip covariance factors above this axis size.
- `precondition_mode="smaller_side"`: for 2D tensors, keep only the smaller
  covariance factor and basis to reduce optimizer state and update cost.
- `precondition_mode="aspect_ratio"`: retain both factors for square and mildly
  rectangular matrices, then keep only the smaller factor when
  `max(shape) / min(shape) >= precondition_aspect_ratio`. The threshold defaults
  to `2.0`.
- `merge_dims`: merge adjacent dimensions before constructing factors.
- `basis_track_stats_frequency`: collect tracker diagnostics every N refreshes
  to reduce synchronization overhead.
- `transport_second_moment_on_warm`: apply squared-overlap diagonal variance
  transport on warm rotations. Default: `False`; repeated diagonal projection
  discards cross-coordinate second moments, so this is experimental.

The default basis state, covariance EMA, and Adam moments are fp32. Gradient
outer products may be computed in bf16 on CUDA before fp32 accumulation.

## State transport

The first moment is transported only when the basis changes: it is projected
back through the old basis and into the new one. Warm gauge steps keep columns
continuous, so the elementwise second moment remains attached to those column
labels. Periodic hard resets transport it through squared basis overlaps,

```text
R = Q_old.T @ Q_new
variance_new = variance_old transformed along each mode by R**2
```

which handles reset permutations, signs, and nontrivial rotations under the
diagonal-covariance approximation.

## Diagnostics

On basis refreshes, `latest_basis_stats` reports averages over active factors:

- `diag_err`: off-diagonal energy relative to `||C||_F`.
- `corr_err`: RMS off-diagonal correlation.
- `orth_err`: normalized `||Q.T @ Q - I||_F`.
- `basis_drift`: normalized basis change.
- `hard_reset`: fraction of factors hard-reset in that refresh.

`just_gathered_basis_stats` indicates whether the current optimizer step
collected diagnostics. This can be less frequent than basis refreshes when
`basis_track_stats_frequency > 1`.

The LLM harness also reports actual active-factor basis updates separately from
legacy per-state cadence counters, residual checks/skips and refresh causes,
checked-residual percentiles, active hard-reset tensor/factor counts, the first
reset range in refreshes and optimizer steps, optimizer-time percentiles, and
timing split into ordinary, warm-refresh, and hard-reset steps.

## Benchmarks

`benchmarks/benchmark_optimizer_step.py` measures complete dense/block and
one-/two-sided optimizer steps. `benchmarks/train_teacher_mlp.py` runs the
deterministic short-training funnel, holding covariance beta fixed while
sweeping the Adam denominator beta for one-sided tracking.

## LLM training harness

`transformer.py` is a compact modern decoder: token embeddings feed the residual
stream directly, followed by pre-norm RMSNorm blocks, RoPE attention with optional
QK normalization, bias-free SwiGLU, a final RMSNorm, and a tied LM head. The
nonstandard input projection from the imported trainer is disabled by default;
`--input-projection` retains it as an explicit ablation. RMSNorm uses an explicit
`1e-5` epsilon under BF16. QK normalization is per head dimension and is applied
before RoPE. QKV and the two SwiGLU input projections remain fused by default for
the forward pass; `--no-fused-qkv` and `--no-fused-swiglu` expose physical-split
architecture controls.

SOAP treats each physical parameter tensor as one covariance problem. Under the
default smaller-side policy, fused QKV and SwiGLU input matrices share one
input-side basis and avoid the much larger output-side factor. Physical splitting
is therefore an explicit parameterization/memory ablation for SOAP rather than a
correctness requirement.

CUDA AdamW already uses PyTorch's fused implementation. `--compile` enables
`torch.compile`, and `--compile-fullgraph` requests the full-graph behavior used
by the larger target profile.

`train_llm.py` is self-contained and defaults to the promoted dense one-sided
configuration plus a structured synthetic dataset for offline checks:

```bash
/venv/main/bin/python train_llm.py \
  --optimizer soap --steps 1000 --output soap_llm.json
```

For a packed Hugging Face text run:

```bash
/venv/main/bin/python train_llm.py --dataset hf \
  --dataset-name Salesforce/wikitext --dataset-config wikitext-2-raw-v1 \
  --tokenizer-name gpt2 --optimizer soap --output soap_wikitext.json
```

For OpenWebText, use streaming plus explicit token limits. The supplied 1,000-step
configs reserve a deterministic validation prefix, materialize 2.5 million train
tokens and 262,144 validation tokens, and share a roughly 11 MB int32 token cache:

```bash
/venv/main/bin/python train_llm.py \
  --config configs/llm_openwebtext_one_sided_1000.json \
  --output soap_openwebtext_1000.json
```

### Width-768 target screen

The first target-regime screen uses a 95.2M-parameter decoder at width 768,
depth 8, 12 heads, batch 32, and sequence length 1024. Runs use BF16,
full-graph compilation, direct token embeddings, QK normalization, and fused
QKV/SwiGLU tensors. Smaller-side SOAP retains 32 factors: fused rectangular
projections share their input-side basis and omit the large output-side basis.

A seed-123 matrix-rate sweep covered `6.25e-5`, `1.25e-4`, `2.5e-4`, and
`5e-4`. The two lowest rates were close at step 50; `1.25e-4` was retained
because it was better at the step-25 validation and avoids selecting a rate
from a `0.011` single-seed endpoint difference. The matched three-seed result is:

| Profile | Mean validation loss | Mean step ms | Mean optimizer ms | State MiB |
|---|---:|---:|---:|---:|
| Fused AdamW, LR `4e-4` | 7.04498 | **134.67** | **1.76** | 726.58 |
| Smaller-side SOAP, LR `1.25e-4` | 6.89174 | 146.15 | 14.55 | 870.58 |
| Transport skew 0.52, Muon LR `0.06` | **6.59310** | 173.21 | 43.87 | **618.67** |

SOAP improves mean loss by `0.15323` over tuned AdamW while increasing total
step time by 8.5%. Its optimizer state is 144.0 MiB larger than AdamW and
251.9 MiB larger than Transport Muon. The result establishes a useful quality
and throughput middle point, but 50 steps is too short to choose a long-run
default or to resolve the two lowest SOAP rates.

The complete rate sweep, seed rows, refresh timing, memory, and source artifact
names are in
`../optimizer_replay_results/llm_openwebtext_modern_768x8_50step_summary.json`.
The next gate freezes `1.25e-4` and uses a larger int32-only token cache rather
than recycling the existing 2.5M training tokens.

The cache has now been expanded to 40M training tokens plus the validation
prefix, stored as 153.6 MiB of int32 IDs with no retained source text. At 200
steps and three matched seeds, smaller-side SOAP reaches validation loss
`5.90796 ± 0.01387`, versus `6.09585 ± 0.03048` for tuned AdamW and
`5.66059 ± 0.00539` for skew-controlled Transport Muon.

SOAP averages 146.81 ms per step and 13.54 ms in the optimizer, compared with
135.75/1.76 ms for AdamW and 174.44/43.73 ms for Transport. Its gain over AdamW
is `0.18789` loss for 8.1% more step time. The 870.58 MiB optimizer state remains
144.0 MiB above AdamW and 252.0 MiB above Transport.

All three runs complete 1,254 basis refreshes with no hard reset. The configured
staggered first-reset range is 21--59 refreshes per factor, while 200 training
steps provide only about 20 refresh opportunities per factor. A 1,000-step gate
is therefore needed both for convergence and to exercise reset behavior in this
larger transformer. Full curves and source artifacts are in
`../optimizer_replay_results/llm_openwebtext_modern_768x8_200step_three_seed_summary.json`.

The shared summary also includes the Transport warm-kernel efficiency control.
A one-power-iteration skew profile preserves the two-iteration profile's
three-seed mean loss (`5.66084` versus `5.66059`) while reducing its total step
time by 2.2%. This does not change the SOAP candidate: smaller-side SOAP remains
the lower-cost quality/throughput middle point, and its next distinctive test is
the 1,000-step run that crosses the staggered hard-reset window.

Two aspect-ratio profiles are included. Threshold 4 keeps both factors for the
square and 3:1 matrices in this decoder and drops the large factor only for the
6:1 feedforward input. It is the quality-oriented profile and uses staggered
hard resets every 40 basis refreshes. Threshold 2 also slims the 3:1 matrices
and is the memory-oriented profile; it now uses the same reset-40 staggered
schedule. The retained
`llm_openwebtext_aspect_ratio4_reset20_1000.json` configuration reproduces the
earlier threshold-4 reset-20 control, while
`llm_openwebtext_aspect_ratio2_reset20_1000.json` preserves the threshold-2
control. Explicit staggered profiles are also available as
`llm_openwebtext_aspect_ratio{2,4}_reset40_staggered_1000.json`:

```bash
/venv/main/bin/python train_llm.py \
  --config configs/llm_openwebtext_aspect_ratio4_1000.json \
  --output soap_openwebtext_ratio4.json

/venv/main/bin/python train_llm.py \
  --config configs/llm_openwebtext_aspect_ratio_1000.json \
  --output soap_openwebtext_ratio2.json

/venv/main/bin/python train_llm.py \
  --config configs/llm_openwebtext_aspect_ratio4_reset40_staggered_1000.json \
  --output soap_openwebtext_ratio4_staggered.json
```

Across three 1,000-step seeds, threshold 4 with reset 40 changed validation loss
by `+0.007%` relative to reset 20 while reducing measured optimizer time by
`13.5%` and total step time by `6.2%`. At 2,000 steps in seed 123, reset 40 gave
validation loss `6.26259` versus `6.26146` for reset 20 and reduced optimizer
time from `15.76` ms to `13.81` ms. Threshold 4 itself retained 28 of 32 factors
and averaged only `0.059%` higher validation loss than two-sided SOAP over three
2,000-step seeds while using `21.2%` less optimizer state. These compact-decoder
measurements motivate the supplied profile; `precondition_mode="all"` remains
the optimizer API default for untested model shapes.

At seed 123, staggering leaves the compact validation loss effectively
unchanged (`6.59621` versus `6.59629`) and reduces median hard-reset-step time
from `112.44` to `77.42` ms. On a wider 11.42M-parameter 192/576/1024 shape mix,
it replaces one `180.93` ms synchronized reset step with 16 distributed reset
steps having a `78.83` ms median; validation loss changes by `+0.00022` and mean
optimizer time is unchanged within `0.1` ms. Across compact seeds 11, 29, and
123, mean validation loss changes by less than `0.00001`. The supplied
threshold-4 profile therefore enables staggering; the optimizer API default
remains off, and
`llm_openwebtext_aspect_ratio4_reset40_synchronized_1000.json` preserves the
synchronized control.

For the threshold-2 memory profile, reset 40 plus staggering changes the
three-seed mean validation loss from `6.61610` to `6.61635` (`+0.0037%`). A
current-code seed-123 comparison halves observed reset work from 64 tensor/80
factor resets to 32/40, lowers median hard-reset-step time from `64.98` to
`54.81` ms, and changes mean optimizer time from `13.97` to `13.81` ms. The
supplied threshold-2 profile therefore uses reset 40 plus staggering while its
explicit reset-20 configuration remains available for reproduction.

The modern width-768, depth-8 gate now covers 1,000 steps at seeds 123, 456,
and 789. Smaller-side SOAP at matrix LR `1.25e-4` reaches validation loss
`4.61910 ± 0.01213`, versus `4.75432 ± 0.00774` for AdamW and
`4.23854 ± 0.00144` for one-power skew Transport. SOAP averages 148.34 ms per
step, 13.52 ms in the optimizer, 220,899 tokens/s, and 870.58 MiB of optimizer
state. It remains the middle quality/throughput point.

Every SOAP seed completes exactly 6,534 basis refreshes and 64 staggered hard
resets with no instability. Ordinary optimizer steps average 9.62 ms, warm
refresh steps 46.68 ms, and hard-reset steps 49.65 ms. A hard reset therefore
adds only 2.98 ms over a normal warm refresh on average; reducing unnecessary
basis refreshes is more promising than optimizing the reset itself. The next
method gate compares the fixed ten-step schedule with a device-side covariance
residual trigger. Full curves and reset timing are in
`../optimizer_replay_results/llm_openwebtext_modern_768x8_1000step_three_seed_summary.json`.

That refresh-policy gate adds two opt-in controls. Residual gating checks the
normalized off-diagonal covariance energy at each nominal refresh and reuses the
checked `Q.T @ C @ Q` product when it accepts the refresh. A deterministic
two-phase policy changes the fixed cadence after a chosen optimizer step. A
step-age bound preserves occasional hard resets under either sparse policy. All
controls default to disabled and preserve the preceding fixed-cadence behavior.

At 200 steps, residual threshold `0.9` with maximum age 40 reaches validation
loss `5.91150` with 161 active-factor refreshes, versus `5.93042` and 608 for the
observed fixed-10 control. Fixed periods 30 and 40 reach `5.95933` and `5.97329`
with 192 and 128 refreshes. This shows that the residual carries useful selection
information. Its checked host transfer, however, removes most end-to-end timing
benefit. At 1,000 steps, using the gate from initialization finishes `0.01587`
above the fixed-10 seed-123 loss. A fixed-10 bootstrap through step 200 removes
that gap in the same seed,
but only one later threshold crossing occurs; maximum age makes nearly every
remaining decision.

The synchronization-free equivalent is fixed period 10 through step 200 and
period 40 thereafter, with a 400-step hard-reset age bound. Its three-seed gate
is:

| Refresh policy | Validation loss | Optimizer ms | Step ms | Active refreshes |
|---|---:|---:|---:|---:|
| Fixed period 10 | **4.62583** | 13.64 | 148.51 | 3,168 |
| Period 10 to 40 after step 200 | 4.63066 | **11.37** | **148.14** | **1,248** |

The two-phase schedule removes 60.6% of active basis updates and 16.7% of
optimizer time. Mean validation rises by `0.00483`, equivalent to about 0.48%
relative perplexity, and two of three paired endpoint differences are positive.
Mean step time improves only 0.25%. Fixed period 10 remains the conservative
quality reference, while the two-phase and residual policies remain explicit
research controls for users willing to exchange a small measured loss gap for
less optimizer work.
The durable calibration, source paths, curves, counters, and paired statistics
are in
`../optimizer_replay_results/soap_refresh_policy_768_summary.json`.

Two cadence refinements complete the current short-horizon screen:

| Refresh policy | Validation loss | Optimizer ms | Step ms | Active refreshes |
|---|---:|---:|---:|---:|
| Fixed period 10 | **4.62583** | 13.64 | 148.51 | 3,168 |
| Period 10 to 20 after step 200 | 4.62719 | 12.12 | 147.02 | 1,888 |
| Period 10 to 40 after step 400 | 4.63244 | 11.92 | **145.22** | **1,728** |

Both refinements look better than fixed period 10 in seed 123 and have positive
paired differences in seeds 456 and 789. The gentler period-20 profile is the
balanced efficiency candidate: it raises mean validation by only `0.00136`
(about 0.14% relative perplexity), saves 11.1% optimizer time, and saves 1.0%
total step time. Its paired loss differences have standard deviation `0.01908`,
far larger than the mean gap, so three 1,000-step runs do not resolve a reliable
quality difference. The delayed period-40 profile trades 12.6% optimizer time
and 2.2% measured step time for a `0.00661` mean loss gap (about 0.66% relative
perplexity).

Each 1,000-step run sees 32.8M tokens, only 0.34 tokens per model parameter, and
ends near loss 4.6. These are screening runs rather than asymptotic training
evidence. Fixed period 10 remains the conservative default, but the sparse
profiles are viable opt-in efficiency tradeoffs. A longer no-reuse gate should
decide whether their small gaps persist, grow, or wash out.

That gate is staged as a matched 4,000-step seed-123 comparison. Its shared
OpenWebText cache contains 140M training tokens and 262,144 validation tokens as
int32 IDs (535 MiB), enough for 131.1M consumed training tokens without wrapping
the sampler. The fixed-10 and 10-to-20 configs are
`llm_openwebtext_modern_768x8_soap_fixed10_current_lr000125_batch32_seq1024_4000.json`
and
`llm_openwebtext_modern_768x8_soap_fixed10to20_warmup200_resetage400_lr000125_batch32_seq1024_4000.json`.
They remain pending until one healthy GPU is unclaimed.

Launch both arms serially on the same first-available GPU with one non-waiting
claim. The launcher skips completed outputs and resumes an interrupted arm from
its rolling checkpoint:

```bash
benchmarks/run_soap_refresh_4000_gate.sh
```

Skipping refreshes also reduces the tracker's cumulative corrective motion. A
sparse branch now implements this as the opt-in age-compensated basis step above.
The seed-123 1,000-step screen config is
`llm_openwebtext_modern_768x8_soap_fixed10to20_agecomp_warmup200_resetage400_lr000125_batch32_seq1024_1000.json`.
At the base ten-step age it preserves `eta=0.5`; after a twenty-step gap it uses
`eta_eff=0.75`, before rotation clipping. This tests an integrator change rather
than searching more uncorrected cadence schedules.

A 1,000-step, dimension-64 deterministic replay supports the mechanism. Against
uncorrected 10-to-20 tracking, age compensation reduces mean post-warmup
off-diagonal error by 43.9% under smooth rotation, 60.8% after an abrupt shock,
and 15.6% on the equal-diagonal stress trajectory. It stays orthogonal to about
`2e-7`. Compared with fixed period 10, compensated tracking is 12.3% worse on
smooth rotation, 21.6% better on mean post-shock error, and 51.9% worse in the
equal-diagonal case. The replay therefore advances the option to the LLM screen
without making it a default. Reproduce it with
`benchmarks/replay_age_compensation.py`; the durable result is
`../optimizer_replay_results/soap_age_compensation_replay.json`.

Long runs support atomic rolling checkpoints with exact shuffled-data position,
model, optimizer, scheduler, and RNG state:

```bash
/venv/main/bin/python train_llm.py \
  --config configs/llm_openwebtext_modern_768x8_soap_smaller_side_lr000125_batch32_seq1024_1000.json \
  --checkpoint /workspace/optimizer_checkpoints/soap.pt \
  --checkpoint-every 100

/venv/main/bin/python train_llm.py \
  --config configs/llm_openwebtext_modern_768x8_soap_smaller_side_lr000125_batch32_seq1024_1000.json \
  --resume /workspace/optimizer_checkpoints/soap.pt
```

Keep the original target `--steps` and schedule when resuming. CPU controls are
bitwise exact. Compiled CUDA controls reproduce every logged loss and reset
event; final FP32 optimizer state can differ by about `5e-10` because reduction
order after recompilation is not bitwise fixed.

`--hf-streaming` reads only the bounded sample. `--max-train-tokens` and
`--max-validation-tokens` cap RAM and token-cache usage, while `--token-cache`
stores token IDs as int32 and avoids repeated network reads and tokenization
across seeds. Batches are promoted to int64 only when loaded for embedding
lookup. Eager loading of
`Skylion007/openwebtext` is rejected unless `--allow-large-hf-download` is passed,
because the complete download plus generated dataset can occupy about 64 GB.

The trainer supports JSON configuration defaults, gradient accumulation,
checkpoint output, cosine/constant schedules, mixed precision, optional model
compilation, validation, CUDA timing, peak memory, optimizer-state size, and SOAP
factor diagnostics. Embeddings, the tied LM head, and vector parameters use the
unpreconditioned Adam-like path; internal matrix projections use SOAP. The
fallback path has its own `--soap-fallback-lr` (default `3e-4`) because SOAP's
matrix learning rate is generally too large for a tied token embedding.

## References

- SOAP: <https://arxiv.org/abs/2409.11321>
- Shampoo: <https://arxiv.org/abs/1802.09568>
