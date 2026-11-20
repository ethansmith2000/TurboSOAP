# Refresh study (2026-11-20): refresh cost and open-loop refresh quality

Scripts: `benchmarks/refresh_study_microbench.py`, `benchmarks/refresh_study_replay.py`,
`benchmarks/refresh_study_summarize.py`, shared code in `benchmarks/refresh_study_common.py`.
Nothing in `bf16_soap.py` or other existing files was changed; only production
functions are imported.

## Part 1: refresh timing microbenchmark

**Setup.** One RTX 5090 claimed exclusively (`gpu-claim`, GPU 5), with no other
job of this study running. Host load average was about 50 at start and 9.5 at
end, because another project's jobs were winding down on other GPUs. PyTorch
2.11.0+cu128. Settings match the 109M runs:
- deterministic algorithms with `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- FP32 matmul precision `high`;
- BF16 reduced-precision reductions disabled;
- 4 CPU threads;
- `torch.compile(fullgraph=True, dynamic=False, emulate_precision_casts=True)`.

Inputs per factor:
- **C** is an SPD exponential moving average (β = 0.95, 40 steps) of G Gᵀ, with
  G = U diag(λ^½) X, λ_i ∝ 1/i, and X Gaussian of size n × n/4. C is stored in BF16.
- **Q** is a sign-fixed QR of U(I + A), with A skew-symmetric and entries
  ~0.2/√n. Q is stored in BF16. The input r_C is about 0.20, in the range of the
  trained factors.

Timing uses CUDA events, 2 warmups, then two interleaved passes (forward and
reverse method order), with 8–40 repetitions sized to about 4 s per cell. The
tables report the median, and the JSON also has p25/p75.

Methods (per factor, batched):
- `warm_bf16`: production `warm_basis`.
- `warm_fp32`: FP32 C/Q storage and FP32 basis, BF16 compute; the paper's `compiled_fp32` configuration.
- `qr`: production `qr_basis`, including the sort.
- `eigh`: FP32 eigh, as at initialization.
- `cholqr`: Cholesky-QR in FP32 with the same sort as QR, and no shift.

`transport_moment` and the QR V permutation are timed per matrix shape.

### Reconciliation gate (passes)

| | measured | 109M reference | ratio |
|---|---|---|---|
| Production `CompiledSOAP.step`, ordinary event (4×12 matrices of the 109M shapes + fused AdamW on a 50257×768 stand-in) | 10.13 ms | 10.14 ms | 1.00 |
| … warm event | 61.09 ms | 61.61 ms | 0.99 |
| … QR event | 609.24 ms | 601.18 ms | 1.01 |
| Σ isolated warm refreshes (5×768, 1536, 2304, 3072; batch 12) + 4 transports, **compiled** | 49.84 + 5.06 = 54.90 ms | 61.61 − 10.14 = 51.47 ms | 1.07 |
| same, **eager** | 81.40 + 4.98 = 86.38 ms | 51.47 ms | 1.68 |
| Σ isolated QR refreshes + 4 transports + 4 V permutations, compiled | 593.74 + 5.06 + 0.38 = 599.19 ms | 591.04 ms | 1.01 |
| same, eager | 598.36 + 4.98 + 0.43 = 603.77 ms | 591.04 ms | 1.02 |
| QR primitive vs §5.4 (batch 12; 768 / 1536 / 2304 / 3072), eager | 30.84 / 81.36 / 143.46 / 219.36 ms | 30.91 / 80.54 / 141.57 / 215.22 ms | 1.00–1.02 |

The event-region timings reproduce the 109M optimizer means to within about 1%.
The isolated warm primitives are compiled per function, not inside the event
graph, and they overshoot the warm increment by about 3.4 ms (7%). Eager warm is
1.7× slower: elementwise work that the compiled graph fuses runs as separate
kernels. The paper's warm numbers are therefore compiled-execution numbers.

### Cost per factor (compiled; ms per call ÷ batch; ratios to production QR)

| n | warm (BF16) | warm/QR b12 | warm/QR b1 | FP32-storage warm/QR b12 | eigh/QR b12 | Cholesky-QR/QR b12 | Cholesky-QR failures (b12) |
|---|---|---|---|---|---|---|---|
| 256 | 0.067 | 0.108 | 0.887 | 0.113 | 7.26 | 0.110 | 0/12 |
| 512 | 0.068 | 0.044 | 0.439 | 0.047 | 7.77 | 0.068 | 0/12 |
| 768 | 0.069 | 0.027 | 0.287 | 0.030 | 2.53 | 0.067 | 0/12 |
| 1024 | 0.10 | 0.027 | 0.207 | 0.035 | 2.29 | 0.074 | 0/12 |
| 1536 | 0.33 | 0.049 | 0.116 | 0.063 | 2.19 | 0.092 | 0/12 |
| 2048 | 0.73 | 0.075 | 0.093 | 0.097 | 2.19 | 0.118 | 0/12 |
| 2304 | 1.02 | 0.086 | 0.086 | 0.112 | 2.30 | 0.127 | 0/12 |
| 3072 | 2.46 | 0.136 | 0.126 | 0.168 | 2.46 | 0.168 | 0/12 |
| 4096 | 5.54 | 0.200 | 0.207 | 0.246 | 2.78 | 0.225 | 0/12 |
| 6144 | 18.7 | 0.300 | 0.244 | 0.364 | 3.64 | 0.304 | **12/12 (NaN)** |
| 8192 | 45.5 | 0.406 | 0.331 | 0.482 | 5.13 | 0.377 | **12/12 (NaN)** |

Notes:
- **Small n.** Below n ≈ 1024 the batch-12 warm call takes about 0.8 ms
  regardless of n. It is latency-bound: the cost is fixed per call, not per flop.
- **Large n.** QR (cuSOLVER `geqrf`/`orgqr`, looped over the batch) grows more
  slowly than the warm step's ~7 full n×n BF16 products plus elementwise passes.
  So warm/QR rises from 0.03 at n=768–1024 to 0.41 at n=8192.
- **The 109M factor dimensions** (768–3072) sit at the most favourable part of
  the curve, with warm/QR at 0.03–0.14.
- **FP32 storage** costs warm 4–30% more than BF16 storage.
- **eigh** costs 2.2–7.8× QR.
- **Cholesky-QR** costs 1.6–2.7× warm for 768 ≤ n ≤ 2048 and within ±25% of
  warm at n ≥ 3072. It is fragile:
  - on these power-law inputs, every factor fails (NaN) at n ≥ 6144 (κ(ZᵀZ) ≈ κ(C)² ≈ n²);
  - its BF16-stored orthogonality error grows from 0.003 (n=256) to 0.034 (n=4096),
    against 0.0024 for QR.
  - **TF32 caveat (added later).** These Cholesky timings and failures used FP32
    matmul precision "high", so the Gram ZᵀZ, which is not BF16-exact, could
    run in TF32. With IEEE FP32 Grams, unshifted Cholesky-QR stays finite to
    n = 8192 at batch 1, with a worst column angle of 0.06 rad, and costs more.
    See Part 1b, "IEEE-FP32 Grams".
- **Failures.** The only rows that did not run are **eager** `warm_bf16` and
  `warm_fp32` at n=8192, batch 12: out of memory, because eager materializes
  several 12×8192² FP32 temporaries. The compiled versions ran.
- **Bit equality.** The parametrized ablation copy of `warm_basis` is
  bit-identical to production at default settings (eager and compiled; n =
  768, 1536, 2304, 3072; batch 12). Compiled production is not bit-identical to
  eager production, which is expected from fusion.

Figure: `refresh_cost_vs_dim.pdf`.

## Part 1b: larger n and a shifted CholeskyQR3 baseline (extension)

This extension was requested by the paper session (user-approved).
- **Run.** `benchmarks/refresh_study_microbench_ext.py`, exclusive GPU 0,
  03:57–04:38 UTC, with no other job of this study running. Host load was about
  68 at start (other projects' CPU work) and 5–20 during the large-n cells.
  Same inputs, settings and timing protocol as Part 1.
- **Shifted CholeskyQR3 (FP32).** The Gram G = XᵀX of the sorted X = Z[:,π]
  gets a shift s = 11(mn + n(n+1))·u·‖X‖₂², with m = n, u = 2⁻²⁴ and ‖X‖₂²
  from 20 power iterations on G. Then R₁ = chol(G + sI) and X₁ = X R₁⁻¹,
  followed by two unshifted CholeskyQR passes. There is no other safeguard.
- **Flop model.** Stated in the JSON. Warm is counted as 14n³, QR as 14/3 n³
  (2 for CQ + 4/3 for geqrf + 4/3 for orgqr), CholQR3 as 12n³, Cholesky-QR as
  16/3 n³, and eigh as 9n³ (a convention).
- **Continuity at n = 8192.** Times agree with Part 1 to within 2%: batch 1
  warm 36.5 vs 37.2 ms and QR 110.4 vs 112.4 ms; batch 12 warm 537 vs 545 ms
  and QR 1321 vs 1344 ms.

| n | batch | warm (ms/call) | QR | shifted CholQR3 | FP32 eigh | **warm/QR** | warm/CholQR3 | CholQR3/QR | achieved TFLOP/s: warm / QR / CholQR3 / eigh |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | 1 | 36.5 | 110.4 | 92.4 | 570 | **0.33** | 0.40 | 0.84 | 211 / 23 / 71 / 8.7 |
| 12288 | 1 | 120.1 | 267.7 | 265.4 | 1945 | **0.45** | 0.45 | 0.99 | 216 / 32 / 84 / 8.6 |
| 16384 | 1 | 283.9 | 533.2 | 581.6 | 3348 | **0.53** | 0.49 | 1.09 | 217 / 39 / 91 / 11.8 |
| 8192 | 4 | 177.5 | 439.9 | 485.3 | 2282 | **0.40** | 0.37 | 1.10 | 173 / 23 / 54 / 8.7 |
| 12288 | 4 | 590.9 | 1070.5 | 1428.0 | 7812 | **0.55** | 0.41 | 1.33 | 176 / 32 / 62 / 8.6 |
| 16384 | 4 | 1393.9 | 2149.4 | 3118.8 | 13445 | **0.65** | 0.45 | 1.45 | 177 / 38 / 68 / 11.8 |
| 8192 | 12 | 537.2 | 1321.3 | 1258.9 | 6843 | **0.41** | 0.43 | 0.95 | 172 / 23 / 63 / 8.7 |
| 12288 | 12 | 1794.9 | 3225.7 | 3959.9 | 23456 | **0.56** | 0.45 | 1.23 | 174 / 32 / 68 / 8.5 |
| 16384 | 12 | OOM | OOM | OOM | OOM | — | — | — | — |

All times are compiled.
- **Crossover: none measured up to n = 16384** at any batch. The warm step runs
  at the BF16 tensor-core rate (≈170–217 TFLOP/s under the 14n³ model) from
  n ≈ 2048 upward. QR's achieved rate is still climbing (23 → 39 TFLOP/s from
  n = 8192 to 16384), so warm/QR rises by about 1.6× per doubling of n.
- **Extrapolation is unreliable.** Naively it would put warm = QR near
  n ≈ 30k (batch 4) to 45k (batch 1). That assumes QR's rate keeps growing
  until it nears the FP32/TF32 limits, which was not measured. On this 32 GB
  GPU, n = 16384 at batch 12 is out of memory for every method. At batch 4 it
  is out of memory in eager mode for warm, Cholesky-QR and CholQR3 (compiled
  ran), and FP32-storage warm runs out of memory in eager mode there.
- **Shifted CholQR3 cost.** Relative to QR it is 3.4× at n = 256, 1.06× at
  1024, 0.62–0.84× for 1536–8192, and 0.99–1.09× at 12288–16384 (batch 1). At
  batch 12 it is 0.22–0.95× QR up to 8192.
- **Shifted CholQR3 failures (under "high", i.e. TF32 Grams).** It fails at
  exactly the sizes where unshifted Cholesky-QR fails, n ≥ 6144 (every element,
  at every batch). The shifted first pass succeeds; passes 2–3 fail.
  - With m = n the prescribed shift is 11(2n²+n)u·‖X‖₂²: 0.09·‖X‖₂² at n = 256,
    88·‖X‖₂² at 8192 and 352·‖X‖₂² at 16384.
  - At large n the first pass therefore mostly rescales X.
  - **These breakdowns are largely a TF32 artifact.** With IEEE-FP32 Grams the
    unshifted passes survive κ(X)² ≈ n² up to n ≈ 8192 on most elements (next
    subsection).
- **Orthogonality where it succeeds.** Normalized Gram error is
  0.00236–0.00238, the same BF16 floor as QR. Unshifted Cholesky-QR gives
  0.0027 at n = 256, rising to 0.034 at 4096.
- **Agreement with Householder QR.** See the accuracy subsection below. The
  in-run agreement fields were superseded because their reference recomputed
  the column sort in a different precision, so near-tie orders differed.
- **Rows that did not run.** Out-of-memory rows are recorded in the JSON.
  The final cell (n = 16384, batch 12) crashed the process on an uncaught
  FP32-copy OOM, after every method had already hit OOM there. Its rows were
  reconstructed from the log and marked; the script has since been patched.

Files: `microbench_ext.json` and `refresh_cost_vs_dim_ext.pdf`.

### Accuracy of the Cholesky-QR variants against Householder QR

Source: `benchmarks/refresh_study_cholqr_accuracy.py` → `microbench_ext_accuracy.json`.
Batch 1, same synthetic inputs, run on a shared GPU slot (quality only). The
methods ran under FP32 matmul "high" (TF32 Grams); IEEE-FP32 results are below.
- **Reference.** Householder QR of the same sorted Z = CQ, using each method's
  own column order.
- **Metric.** Column-wise angle after sign fixing, at "highest" precision.
- **BF16 floor.** The angle between Householder Q and its BF16 rounding:
  1.8e-3 max.

| n | production QR: max angle | unshifted CholQR: max / median / p99 angle, e_Q | shifted CholQR3: max / median / p99 angle, e_Q |
|---|---|---|---|
| 256 | 1.9e-3 | 3.3e-3 / 1.8e-3 / 2.9e-3, 2.7e-3 | 2.0e-3 / 1.7e-3 / 1.9e-3, 2.4e-3 |
| 1024 | 1.8e-3 | 6.6e-3 / 2.0e-3 / 4.4e-3, 3.3e-3 | 2.2e-3 / 1.7e-3 / 2.0e-3, 2.4e-3 |
| 2048 | 1.8e-3 | 1.2e-2 / 2.3e-3 / 8.1e-3, 4.8e-3 | 3.4e-3 / 1.8e-3 / 2.4e-3, 2.4e-3 |
| 3072 | 1.8e-3 | 3.1e-2 / 3.0e-3 / 2.2e-2, 1.0e-2 | 3.5e-3 / 1.8e-3 / 2.8e-3, 2.4e-3 |
| 4096 | 1.8e-3 | 0.11 / 4.5e-3 / 6.1e-2, 3.1e-2 | 3.8e-3 / 1.9e-3 / 3.1e-3, 2.4e-3 |
| 6144–16384 | 1.8e-3 | NaN (Cholesky breaks down at pivot ≈ 4700–5200) | NaN (passes 2–3 break down at pivot ≈ 4800–5000) |

- **Production QR** sits exactly at the BF16 floor. The earlier in-run "π/2"
  angles were the sort-order artifact.
- **Shifted CholQR3**, where it succeeds (n ≤ 4096 on these inputs), matches
  Householder QR to within 2× the BF16 floor, with medians at the floor. Its
  orthogonality also sits at the floor.
- **Unshifted Cholesky-QR** degrades steadily with n, to a 0.11 rad worst column
  at n = 4096.

### IEEE-FP32 Grams (no TF32): accuracy and cost

**Why rerun.** The panel asks for true IEEE-FP32 Grams, and the shift formula
assumes u = 2⁻²⁴. The accuracy and timing were therefore rerun with the
Cholesky methods under FP32 matmul "highest".
- **Accuracy** (`microbench_ext_accuracy_highest.json`): same inputs as above;
  only the method calls switch precision.
- **Timing** (`microbench_ext_ieee.json`): exclusive GPU, host load about 8.
  Inputs were generated under "highest", which differs slightly from the "high"
  runs. QR and warm times are taken from `microbench_ext.json`.

**Accuracy (batch 1).**
- **Shifted CholQR3** matches Householder QR at the BF16 floor (max column
  angle 1.76–1.89e-3, Gram error 2.34–2.36e-3) for every n up to 8192.
  It fails (NaN) at 12288 and 16384.
- **Unshifted Cholesky-QR** stays finite to 8192. Its worst column angle grows
  1.9e-3 → 0.013 (4096) → 0.061–0.064 (6144–8192), and it fails at 12288 and 16384.

**Failures in the timing run** (different input draws; "x/y" = failed elements
out of batch).

| n | CholQR3, b1 | CholQR3, b4 | CholQR3, b12 | Cholesky-QR, b1 | Cholesky-QR, b4 | Cholesky-QR, b12 |
|---|---|---|---|---|---|---|
| ≤ 4096 | 0 | — | 0 | 0 | — | 0 |
| 6144 | 1/1 | — | 0/12 | 0/1 | — | 1/12 |
| 8192 | 1/1 | 1/4 | 2/12 | 0/1 | 3/4 | 9/12 |
| 12288 | 0/1 | 4/4 | out of memory | 1/1 | 4/4 | out of memory |
| 16384 | 1/1 | out of memory | out of memory | 1/1 | out of memory | out of memory |

So at IEEE FP32 these inputs sit at the breakdown edge from n ≈ 6144: whether
an element fails depends on the draw.

**Cost (compiled, ms per call).**

| n | batch | CholQR3, TF32 | CholQR3, IEEE | QR (production) | IEEE CholQR3 / QR | warm / IEEE CholQR3 | Cholesky-QR, IEEE |
|---|---|---|---|---|---|---|---|
| 768 | 12 | 6.8 | 7.8 | 30.4 | 0.26 | 0.10 | 2.3 |
| 1536 | 12 | 20.0 | 28.9 | 79.7 | 0.36 | 0.14 | 10.0 |
| 2304 | 12 | 48.1 | 78.8 | 140.2 | 0.56 | 0.15 | 27.5 |
| 3072 | 12 | 95.7 | 173.8 | 213.7 | 0.81 | 0.17 | 60.4 |
| 4096 | 12 | 194.3 | 351.2 | 327.3 | 1.07 | 0.19 | 132.5 |
| 8192 | 12 | 1258.9 | 2554.7 | 1321.3 | 1.93 | 0.21 | 965.9 |
| 4096 | 1 | 17.3 | 21.2 | 27.3 | 0.78 | 0.27 | 8.2 |
| 8192 | 1 | 92.4 | 128.9 | 110.4 | 1.17 | 0.28 | 51.0 |
| 12288 | 1 | 265.4 | 412.4 | 267.7 | 1.54 | 0.29 | 160.5 |
| 16384 | 1 | 581.6 | 880.6 | 533.2 | 1.65 | 0.32 | 359.1 |

- **At the 109M factor sizes** (768–3072, batch 12), IEEE CholQR3 takes
  0.26–0.81× QR's time. Summed over the 109M refresh (5×768 + 1536 + 2304 +
  3072) that is 7.8·5 + 28.9 + 78.8 + 173.8 = 320 ms, against 594 ms for QR
  and 50 ms for warm.
- **From n ≈ 4096 (batch 12) or 8192 (batch 1) upward it is slower than QR.**
  At n ≥ 2048 IEEE FP32 reaches only 22–60 TFLOP/s, against TF32's 48–91 for
  CholQR3 at n ≥ 4096, on this GPU.

**Bottom line for the paper.**
- **Use IEEE-FP32 Grams.** With them, shifted CholQR3 is a faithful QR
  substitute on these synthetic spectra up to about n = 4096 (no failures, BF16
  floor accuracy), at 0.26–0.8× QR's time at 109M sizes. It is still slower
  than the warm step: 5–10× per dimension, and 320 vs 50 ms in the 109M sum.
- **Its cost advantage over QR is gone by n ≈ 4096–8192,** where it also
  starts to fail sporadically.
- **The TF32 numbers above overstate both its speed and its fragility.**
- **On trained factors it does not work as specified.** At IEEE FP32 on CPU
  (`replay_pb_checks.json`, age-10 snapshots, d = 256/512), passes 2–3 break
  down on 11 of 16 factors. 14 of 16 BF16-stored factors are slightly
  indefinite, so a shift or ridge larger than |λ_min| is needed, as the panel's
  §9.1 says.

## Part 2: open-loop replay and warm-step ablation

**Setup (approved config).**
- **Models.** 1-layer Transformers from the repo's `transformer.py`: fused QKV,
  fused SwiGLU with hidden size 2d, tied embeddings, RoPE, QK-norm, no bias,
  heads = d/64. Widths d ∈ {256, 512, 1024, 2048}. Each width has 8 factors of
  dimensions d×5, 2d, 3d and 4d, the 109M pattern scaled.
- **Driver.** `soap_reference.ResearchSOAP` QR10 with the stock betas
  (.95/.95/.95), built via `owt_restart.build_optimizer`. Matrix LR 1.5e-3 (not
  re-tuned per width), AdamW auxiliary LR 6e-4, weight decay .05, gradient clip
  1.0. Warmup 100, then cosine to 0.1× over 1,001 calls.
- **Data.** 32 rows × 1024 tokens per step from the cached OpenWebText at
  `/workspace/data/tokenized/openwebtext_gpt2_u16_1024_v1`. This is the same
  dataset revision and tokenizer as the paper but a different set of rows. Seed 204.
- **Training outcome.** Validation CE at age 1000: 5.40 / 5.08 / 4.94 / 4.87 for
  d = 256 / 512 / 1024 / 2048. Each run took 1.7–7.3 min on its own GPU.
- **Saved data.** C_L and C_R of every matrix at age 0 and every 10th age
  (101 snapshots), in BF16, under
  `/workspace/data/turbosoap_refresh_study_20261120/` (≈40 GB).

**Age alignment.** For d=256, FP32 C and the driver's Q were saved as well. An
offline chain of ResearchSOAP's QR refresh over the saved FP32 trajectory
reproduces the driver's Q **bit-exactly at all 100 refresh ages for all 8
factors**, so the snapshots are exactly the covariances each refresh saw.

**Replay.** Each schedule starts from production's eigh initialization at age 0
and is chained over the saved BF16 trajectory using the production `warm_basis`,
`qr_basis` and eigh initialization path, run eagerly. Schedules: QR10,
warm10/QR50, QR20, QR50, eigh10, Cholesky-QR10, init-only, plus the ablation
chains.

The parametrized default warm chain stays bit-identical to production at every
age, for every width and factor group.

**Metrics.** All are evaluated at FP32 "highest" precision and averaged
unweighted over a width's 8 factors.
- **r_C:** the paper's definition. *After* means immediately after the refresh
  at that age; *before* means the previous basis on the covariance at that age.
  *Interval* is mean(after_t, before_{t+10}).
- **r_C^w:** r_C of D^-½ B D^-½ with D = diag(B), restricted to coordinates with
  B_jj ≥ 1e-4·max B_jj, so every direction counts equally regardless of its variance.
- **Top-k overlap:** ‖U_kᵀ Q_S‖²_F / k. Here U_k holds the top-k eigenvectors of
  C and Q_S the k stored columns with the largest Rayleigh quotients, with
  k ∈ {1, 8, n/8}.
- **e_Q:** the normalized Gram error ‖QᵀQ − I‖_F / √n.

All results are **open-loop**: the covariances come from a QR10-driven run and
do not respond to the replayed bases.

### Replay quality (late window, ages 500–1000)

| d | schedule | r_C after | r_C before | r_C interval | r_C^w interval | top-1 after | top-n/8 after | e_Q after |
|---|---|---|---|---|---|---|---|---|
| 256 | QR10 | 0.139 | 0.274 | 0.206 | 0.461 | 0.923 | 0.922 | 2.36e-3 |
| 256 | warm10/QR50 | 0.262 | 0.318 | 0.290 | 0.622 | 0.845 | 0.841 | 1.91e-3 |
| 256 | QR20 | 0.211 | 0.299 | 0.255 | 0.573 | 0.889 | 0.882 | 2.36e-3 |
| 256 | QR50 | 0.292 | 0.336 | 0.315 | 0.660 | 0.834 | 0.823 | 2.36e-3 |
| 256 | eigh10 | 0.002 | 0.263 | 0.133 | 0.332 | 1.000 | 1.000 | 2.36e-3 |
| 512 | QR10 | 0.176 | 0.316 | 0.246 | 0.513 | 0.868 | 0.913 | 2.35e-3 |
| 512 | warm10/QR50 | 0.308 | 0.364 | 0.336 | 0.701 | 0.778 | 0.815 | 1.88e-3 |
| 512 | QR20 | 0.253 | 0.342 | 0.298 | 0.638 | 0.812 | 0.869 | 2.35e-3 |
| 512 | QR50 | 0.330 | 0.378 | 0.355 | 0.734 | 0.778 | 0.804 | 2.35e-3 |
| 512 | eigh10 | 0.002 | 0.298 | 0.150 | 0.371 | 1.000 | 1.000 | 2.36e-3 |
| 1024 | QR10 | 0.204 | 0.346 | 0.275 | 0.536 | 0.830 | 0.902 | 2.35e-3 |
| 1024 | warm10/QR50 | 0.341 | 0.397 | 0.370 | 0.746 | 0.736 | 0.794 | 1.87e-3 |
| 1024 | QR20 | 0.281 | 0.373 | 0.327 | 0.676 | 0.778 | 0.855 | 2.35e-3 |
| 1024 | QR50 | 0.364 | 0.412 | 0.389 | 0.779 | 0.730 | 0.787 | 2.35e-3 |
| 1024 | eigh10 | 0.002 | 0.324 | 0.163 | 0.392 | 1.000 | 1.000 | 2.35e-3 |
| 2048 | QR10 | 0.240 | 0.389 | 0.314 | 0.588 | 0.777 | 0.899 | 2.35e-3 |
| 2048 | warm10/QR50 | 0.389 | 0.448 | 0.419 | 0.823 | 0.668 | 0.781 | 1.85e-3 |
| 2048 | QR20 | 0.324 | 0.418 | 0.371 | 0.741 | 0.718 | 0.849 | 2.35e-3 |
| 2048 | QR50 | 0.412 | 0.462 | 0.437 | 0.856 | 0.663 | 0.777 | 2.35e-3 |
| 2048 | eigh10 | 0.002 | 0.359 | 0.180 | 0.433 | 1.000 | 1.000 | 2.35e-3 |

Two schedules are left out of this table:
- **Init-only:** r_C ≈ 0.92–0.95 at every width.
- **Cholesky-QR10:** **non-finite from the first refresh (age 10)**. That is
  800/800 factor-refresh events at d ≥ 512 and 791/800 at d = 256. Unshifted
  Cholesky-QR in FP32 cannot factor Z = CQ for these factors. Many directions
  are near-null, and some BF16-stored factors are slightly indefinite (4–15%
  negative eigenvalues in the right-side SwiGLU factor at d ≥ 512).

**Build-up across the four warm refreshes (warm10/QR50, late window; before / after each refresh):**

| d | 0 after QR | 10 | 20 | 30 | 40 | 50 before QR | QR10, phase-40 after |
|---|---|---|---|---|---|---|---|
| 256 | 0.153 | 0.291 / 0.269 | 0.308 / 0.287 | 0.322 / 0.301 | 0.330 / **0.309** | 0.337 | **0.141** |
| 512 | 0.187 | 0.321 / 0.303 | 0.357 / 0.341 | 0.372 / 0.356 | 0.380 / **0.364** | 0.386 | **0.176** |
| 1024 | 0.214 | 0.342 / 0.326 | 0.400 / 0.384 | 0.410 / 0.395 | 0.415 / **0.400** | 0.417 | **0.204** |
| 2048 | 0.259 | 0.398 / 0.381 | 0.455 / 0.439 | 0.459 / 0.444 | 0.452 / **0.436** | 0.472 | **0.220** |

Key observations (open-loop, single seed):

1. **Consistent with the paper's §5.5 cell.** Phase 40, just after the fourth
   warm refresh, gives warm10/QR50 r_C = 0.31 / 0.36 / 0.40 / 0.44 against
   QR10 0.14 / 0.18 / 0.20 / 0.22 for d = 256 → 2048. The 109M run gave
   0.38–0.39 against 0.20–0.21, which falls between our d = 512 and d = 2048.
   This is consistency, not reproduction: 1-layer models, open loop, ages ≤ 1000
   against a closed-loop 12-layer run at ages 1990/9990. r_C grows with width
   for every schedule.
2. **A warm refresh removes only 3–8% of the current r_C** (e.g. 0.291 → 0.269).
   A QR refresh removes 38–49% (QR10: 0.274 → 0.139 at d = 256). The
   10-age drift is larger than either. Even an exact refresh every 10 ages
   (eigh10) rises from 0.002 back to 0.26–0.36 before the next refresh.
3. **Warm refreshes close only 15–23% of the QR50 → QR10 gap** in the interval
   mean of r_C: 23% / 17% / 17% / 15% for d = 256 / 512 / 1024 / 2048. They
   close 12–19% of it in r_C^w. Top-1 alignment after refresh is 0.67–0.85,
   against 0.66–0.83 for QR50 and 0.78–0.92 for QR10.
4. **The weak directions barely move.** r_C^w for warm10/QR50 (0.62–0.82) stays
   near QR50 (0.66–0.86), against 0.46–0.59 for QR10. The likely mechanism is
   the gap normalization h_ij = (d_i − d_j)/max_k d_k. At age 750, 40–99% of
   each factor's eigen-directions have λ < 1% of λ_max, and those directions
   carry 4–41% of the variance. Every pair of such directions has |h| < 0.01,
   so the damping suppresses it.
5. **The large left factors dominate the factor mean.** The 3d and 4d left
   factors (QKV and SwiGLU-in) reach r_C 0.64–0.80 under warm10/QR50, against
   0.35–0.54 for QR10. The d and 2d factors sit at 0.13–0.29 against 0.06–0.17.
6. **QR20 beats warm10/QR50 at every width** (interval r_C 0.255–0.371 against
   0.290–0.419), at higher refresh cost. Is the warm refresh's r_C gain over
   QR50 cheaper per millisecond than QR20's?
   - **Corrected.** An earlier version said "1.3–1.9×" by dividing every width's
     gain by the fixed 109M cost.
   - **How it is computed now.** Each width uses its own factor dimensions
     (d×5, 2d, 3d, 4d), with compiled batch-12 per-factor costs from Part 1.
     Transports are excluded.
   - **Result.** The warm refreshes buy **2.9× / 2.3× / 0.92× / 0.35×** the
     per-millisecond r_C reduction of QR20's extra QRs, for d = 256 / 512 /
     1024 / 2048. The advantage reverses near d = 1024, as warm/QR rises with n
     (Part 1).
   - **At the 109M factor sizes** the ratio is about 1.4–1.5×, using the
     d = 512–1024 gains. There, refresh work per 50 ages is 594 ms (QR50),
     793 ms (warm10/QR50), 1,484 ms (QR20) and 2,969 ms (QR10).
   - **Scope.** This is an r_C statement, not a loss statement.
   - **Panel needs correcting.** `research/reviews/panel_20261120/P1_SYNTHESIS.md`
     §9.4 quotes the superseded 1.3–1.9×.
7. **Orthogonality is fine.** The warm basis's e_Q (1.85–1.91e-3) is slightly
   *lower* than the BF16-rounded QR baseline (2.35e-3).

### Warm-step ablation (one knob at a time inside warm10/QR50; late window)

Timing is per batch-12 call on 768-dim factors, compiled, from Part 1 on the
exclusive GPU. Default: 0.82 ms. The full table is in `tables.md` and `ablation.json`.

| change from default (0.01 / 0.5 / 2 / 1) | Δ r_C interval (d=256…2048) | Δ r_C^w interval | e_Q after | time (ms) |
|---|---|---|---|---|
| damping 0.001 | +0.016 / +0.012 / +0.014 / +0.014 | +0.006 / −0.002 / 0.000 / +0.001 | ≈ same | 0.83 |
| damping 0.003 | +0.009 / +0.007 / +0.008 / +0.010 | 0.000 / −0.006 / −0.004 / −0.002 | ≈ same | 0.83 |
| damping 0.03 | −0.010 / −0.008 / −0.008 / −0.009 | +0.007 / +0.012 / +0.012 / +0.013 | ≈ same | 0.83 |
| damping 0.1 | −0.016 / −0.012 / −0.012 / −0.014 | +0.020 / +0.022 / +0.023 / +0.026 | ≈ same | 0.82 |
| cap 0.25 | +0.009 / +0.006 / +0.007 / +0.008 | +0.013 / +0.010 / +0.013 / +0.016 | ≈ same | 0.82 |
| cap 1.0 | −0.011 / −0.009 / −0.009 / −0.010 | −0.015 / −0.011 / −0.013 / −0.018 | 1.9–2.0e-3 (+2–10%) | 0.83 |
| **no cap** | **diverges**: non-finite in 59–82% of factor-ages, first at age 30–40 | — | — | 0.78 |
| 0 Newton–Schulz corrections | +0.002 / +0.001 / +0.003 / +0.006 | ≈ +0.002 | **6.5e-3–1.0e-2 (3.5–5×)** | 0.46 |
| 1 correction | 0.000 | 0.000 | 2.1–2.2e-3 | 0.64 |
| 3 corrections | 0.000 | 0.000 | 1.76–1.81e-3 | 1.01 |
| 2 warm steps per refresh | −0.014 / −0.011 / −0.011 / −0.012 | −0.019 / −0.016 / −0.018 / −0.021 | ≈ same | 1.50 |

**Are 0.01 / 0.5 / 2 reasonable?** Yes, as points in this family. None of the
changes alters the picture: the best single change moves the interval r_C by
at most 0.016, while the QR50–QR10 gap is 0.11–0.12.
- **Corrections: 2 is right.** Zero corrections leave 3.5–5× the Gram error.
  One correction is already near the BF16 QR baseline. A third buys about 5%
  lower e_Q for 0.18 ms per call.
- **The cap is load-bearing.** Without it the step diverges. A cap of 1.0 was
  slightly better than 0.5 on this replay, at the cost of a little more Gram
  error.
- **Damping trades the two metrics against each other.** Larger δ lowers the
  Frobenius r_C, probably because the cap binds less often, but worsens the
  weak-direction metric r_C^w. Smaller δ is worse on r_C. So 0.01 is a sensible
  middle.
- **Two warm steps** give the largest gain on both metrics, for 1.8× the warm
  cost.

The limit is structural, not a tuning problem: the 10-age drift and the
max-normalized gap leave weak directions unrotated.

### Caveats

- **Open loop.** The covariances come from a QR10-driven trajectory, so closed
  loop TurboSOAP would see different C. The results describe tracking quality
  on a fixed trajectory, not loss.
- **Scale.** Single seed, 1-layer models, 1,000 steps (11 of 101 snapshots
  fall in LR warmup), and LR not re-tuned per width.
- **Data.** OWT rows differ from the paper's frozen cohort.
- **Covariance precision.** Saved C is the reference's FP32 running average
  rounded once to BF16. Production accumulates the average in BF16.
- **Execution mode.** The replay runs the production functions eagerly; the
  compiled production function differs at the rounding level (Part 1).
- **Metric definitions.** r_C^w depends on the 1e-4 coordinate floor. Factor
  means are unweighted; see the per-dimension breakdown in `replay.json`.
- **Cholesky-QR.** The replay chain ran with FP32 matmul precision "high", so
  the Gram ZᵀZ could use TF32. A follow-up CPU check at IEEE FP32
  (`replay_pb_checks.json`) still finds breakdown at the first refresh. That
  check used the age-10 snapshot with the production eigh initialization.
  - Unshifted Cholesky breaks down (info > 0) on 14/16 factors at d = 256 and 512.
  - BF16-stored C has 0–20% negative eigenvalues there (14 of 16 factors are
    indefinite), with λ_min/λ_max down to −3.8e-4.

  So this failure is not a TF32 artifact. Shifted CholQR3 at IEEE FP32 also
  breaks down in passes 2–3 on 11/16 of those factors.

Files:
- `replay.json` and `ablation.json`
- `replay_parts/` (full per-age, per-factor series, plus the alignment check)
- `tables.md`
- `replay_quality.pdf` and `ablation.pdf`
- training/replay logs in `logs/`

## Part 2b: alternative warm-step generators on the trained factors (extension)

This extension was requested by the paper session (user-approved), following
the review panel's P1 synthesis rows PB1–PB3 and dispute D1. It is an open-loop
replay on the same saved trajectories as Part 2, quality only.

**Code and runs.**
- `benchmarks/refresh_study_replay_pb.py` and `refresh_study_summarize_pb.py`.
- Replays ran on shared GPU slots. Timing ran on an exclusive GPU with no other
  job of this study running (host load 4–9).

**Generators.** Each is followed by Q + QA in BF16 and the production
Newton–Schulz (NS) corrections.
- **Production:** A = −½(B/s)·h/(h² + 0.01²), with h_ij = (d_i − d_j)/s and
  s = max|d|. Frobenius cap 0.5.
- **PB1, the QR tangent** (as in the panel's `checks/generator_compare.py`):
  A_ij = B_ij·sgn(d_j − d_i)/max(d_i, d_j, ρ·s), ties broken by column index,
  entries clamped to ±1. Variants:
  - ρ ∈ {1e-4, 1e-3} with Frobenius cap 0.5 or 1.0;
  - a spectral cap ‖A‖₂ ≤ 0.5, estimated by 8 power iterations (PB2);
  - one NS correction instead of two (ρ = 1e-4, Frobenius cap 0.5).
- **Relative-gap damping:** the production formula with m_ij = max(|d_i|, |d_j|, 1e-4·s)
  in place of s. Frobenius cap 0.5.
- **Jacobi factor 1 (D1/PB3):** production with −1 in place of −½. Frobenius cap 0.5.

**Checks.**
- The generic code path at production settings is bit-identical to `warm_basis`
  at every replayed age, for all widths and factor groups.
- Its chain equals Part 2's warm10/QR50 chain exactly.
- The timing inputs are also bit-identical (768–3072 at batch 12, and 8192 at batch 1).

**Schedules.** warm10/QR50, plus warm10 with no QR repair after the eigh
initialization. Part 2's QR10, QR20, QR50 and eigh10 chains are the references.

**Metrics.** Part 2's metrics, plus the panel's whitened ρ_rms: the RMS
off-diagonal correlation over coordinates with B_jj ≥ 1e-4·max. Also reported:
cap-binding rate and mean applied multiplier, non-finite rate, the 90th
percentile over factor-ages, and the mean over the 3d/4d left factors. All are
in `replay_pb.json` and `tables_pb.md`.

### Headline: % of the QR50 → QR10 gap closed (late window, interval means; d = 256 / 512 / 1024 / 2048)

| generator | schedule | r_C | r_C^w | ρ_rms | mean applied cap multiplier | e_Q ×1e3 |
|---|---|---|---|---|---|---|
| production | warm10/QR50 | 22 / 17 / 16 / 15 | 19 / 15 / 14 / 13 | 30 / 26 / 30 / 38 | 0.47 / 0.46 / 0.41 / 0.32 | 1.85–1.91 |
| PB1 ρ=1e-4, Frobenius 0.5 | warm10/QR50 | 10 / 5 / 3 / 2 | 20 / 14 / 11 / 8 | 31 / 26 / 26 / 30 | 0.08 / 0.04 / 0.02 / 0.01 | 1.87–1.94 |
| PB1 ρ=1e-4, Frobenius 1.0 | warm10/QR50 | 19 / 10 / 7 / 4 | 34 / 25 / 20 / 15 | 49 / 43 / 43 / 46 | 0.16 / 0.08 / 0.04 / 0.02 | 1.94–2.05 |
| PB1 ρ=1e-3, Frobenius 0.5 | warm10/QR50 | 11 / 6 / 4 / 3 | 21 / 17 / 15 / 12 | 31 / 28 / 29 / 36 | 0.09 / 0.06 / 0.03 / 0.02 | 1.87–1.94 |
| PB1 ρ=1e-3, Frobenius 1.0 | warm10/QR50 | 20 / 12 / 9 / 6 | 35 / 29 / 25 / 21 | 50 / 45 / 46 / 53 | 0.19 / 0.12 / 0.07 / 0.04 | 1.97–2.08 |
| **PB1 ρ=1e-4, spectral 0.5** | warm10/QR50 | 25 / 13 / 9 / 5 | **40 / 29 / 23 / 15** | **55 / 50 / 48 / 46** | 0.23 / 0.11 / 0.06 / 0.02 | 1.91–2.10 |
| relative-gap damping | warm10/QR50 | 2 / 0 / 0 / 0 | 4 / 3 / 2 / 1 | 6 / 5 / 5 / 6 | 0.01 / 0.01 / 0.00 / 0.00 | 1.90–1.96 |
| Jacobi factor 1 | warm10/QR50 | 24 / 18 / 18 / 15 | 20 / 16 / 15 / 13 | 30 / 27 / 31 / 39 | 0.35 / 0.37 / 0.29 / 0.17 | 1.85–1.90 |
| PB1 ρ=1e-4, Frobenius 0.5, 1 NS | warm10/QR50 | 11 / 5 / 3 / 2 | 20 / 15 / 11 / 8 | 31 / 26 / 26 / 30 | 0.08 / 0.04 / 0.02 / 0.01 | 2.15–2.28 |
| production | warm10, no QR | −54 / −51 / −50 / −50 | −60 / −52 / −45 / −30 | −57 / −48 / −36 / −14 | 0.46–0.32 | 1.73–1.81 |
| PB1 ρ=1e-4, Frobenius 1.0 | warm10, no QR | −30 / −61 / −110 / −193 | 14 / −1 / −12 / −15 | 32 / 24 / 21 / 28 | 0.15–0.02 | 1.79–1.94 |
| PB1 ρ=1e-4, spectral 0.5 | warm10, no QR | −11 / −38 / −70 / −163 | 26 / 10 / −2 / −12 | 46 / 39 / 35 / 33 | 0.23–0.02 | 1.79–2.03 |
| relative-gap damping | warm10, no QR | −493 / −485 / −449 / −385 | −100 / −75 / −63 / −40 | −159 / −121 / −107 / −78 | ≈0 | 1.79–1.91 |

Reference interval means for d = 256 / 512 / 1024 / 2048:

| reference | r_C | r_C^w | ρ_rms |
|---|---|---|---|
| QR10 | 0.206 / 0.246 / 0.275 / 0.314 | 0.461 / 0.513 / 0.536 / 0.588 | 0.78 / 1.14 / 1.34 / 1.88 |
| QR50 | 0.315 / 0.355 / 0.389 / 0.437 | 0.660 / 0.734 / 0.779 / 0.856 | 1.24 / 1.80 / 2.21 / 3.25 |
| eigh10 | 0.133 / 0.150 / 0.163 / 0.180 | 0.332 / 0.371 / 0.392 / 0.433 | 0.65 / 1.00 / 1.21 / 1.75 |

No variant produced non-finite values.

### Findings

1. **PB1 does not hold up as specified on trained factors.**
   - **The cap binds on effectively every refresh (99–100%),** and the applied
     multiplier collapses with width: 0.08 → 0.01 under Frobenius cap 0.5, and
     0.23 → 0.02 under the spectral cap.
   - **The ±1 clamp never fires:** 0 entries over 17,280 PB1 refresh events.
   - **Why ‖A‖ is large.** For d_i, d_j above the floor,
     |A_ij| = |B_ij|/max(d_i, d_j) ≤ |B_ij|/√(d_i d_j) = |ρ_ij|. So
     ‖A‖_F ≲ ρ_rms·√n_eff, and ρ_rms·√n_eff is 5–400 on these factors.
     Measured ‖A‖_F (`replay_pb_checks.json`; age-740 eigenbasis applied to C
     at age 750) respects the bound. It implies Frobenius-0.5 multipliers of
     0.003–0.2, matching the replay.
   - **Where the mass sits.** Pairs with both diagonals below 1% of the maximum
     carry 59–99.7% of ‖A‖_F² on 14 of the 16 factors checked. The exception is
     the attention-output left factor, whose head-head pairs carry 42–70%.
   - **Consequence.** The single global cap therefore starves the head
     rotations that r_C measures.
   - **Positive control.** On the panel's chained toy (n = 128,
     `generator_compare.py` setting), this PB1 implementation with the spectral
     cap 0.5 gives r_C 0.0769 / ρ_rms 0.2433. That reproduces the panel's 0.0767 / 0.2427
     (QR10 0.0759 / 0.2401), and the cap barely binds (mean multiplier 0.95).
     The same PB1 with Frobenius cap 0.5 gives 0.104 / 0.295, worse than
     production's 0.098 / 0.294. With no cap it diverges under the BF16
     two-correction retraction; the panel used an exact polar retraction.
     **So the PB1 direction works when a spectral cap leaves it alone. The
     failure on trained factors is specific to their large weak-direction mass.**
   - **Under warm10/QR50, PB1 closes less of the r_C gap than production at
     every width except d = 256 with the spectral cap** (25% vs 22%). At
     d = 2048 it closes 2–6% against 15%.
   - **It does better on the whitened metrics.** The spectral or Frobenius-1.0
     variants close 15–40% of the r_C^w gap (production 13–19%) and 43–55% of
     the ρ_rms gap (production 26–38%). The advantage shrinks with width.
   - **The floor ρ matters little.** ρ = 1e-3 is marginally better than 1e-4,
     because its multiplier is slightly larger.
2. **No generator can replace QR repair under the current caps.**
   - Without QR, every variant is worse than QR50 on r_C: production by
     −50 to −54%, PB1 by −11 to −291%, worsening with width.
   - The PB1 spectral and Frobenius-1.0 variants do beat QR50 on ρ_rms
     (21–46% of the gap closed), but not on r_C.
3. **Relative-gap damping is inert under a global cap.** Its multiplier is
   about 0.00–0.01, so it applies almost no motion. Without QR it collapses
   (r_C 0.85–0.91).
4. **Jacobi at factor 1 is the same as production.** With the cap binding,
   doubling the generator only halves the multiplier (0.17–0.37 against
   0.32–0.47). On trained factors, dispute D1 is decided by the cap, not by the
   gain factor.
5. **One NS correction gives identical quality.** PB1 with 1 NS differs from
   2 NS by at most 0.002 on every metric, with e_Q 2.1–2.3e-3, and is 23–28% faster.
6. **Whitened-metric caveat.** Even an exact refresh decorrelates within 10
   ages: eigh10's ρ_rms goes from 0.007 just after the refresh to 1.3–2.4 just
   before the next. QR50 sits at the same level. So ρ_rms "gap closed" mostly
   measures the value right after each refresh, averaged against a tail that
   re-saturates within one interval. ρ_rms sums ρ_ij² over each row, so values
   above 1 do not by themselves indicate noise: typical |ρ_ij| here are about
   0.03–0.1.
7. **What would be needed.** A bound that does not let tail pairs consume the
   step budget, for example a per-row or per-pair bound (PB2's row-ℓ₂ variant),
   or masking pairs whose diagonals are both below a noise floor. Neither was
   tested here.

### Cost (ms per call, compiled; exclusive GPU; batch 12 at n = 768 / 1536 / 2304 / 3072, batch 1 at 8192)

| generator | 768 | 1536 | 2304 | 3072 | 8192 |
|---|---|---|---|---|---|
| production | 0.84 | 3.88 | 12.40 | 29.13 | 36.49 |
| PB1, Frobenius cap (any ρ, either cap) | 0.83–0.84 | 3.87–3.88 | 12.40–12.43 | 29.15–29.19 | 36.55–36.64 |
| PB1, spectral cap 0.5 | 2.23 | 4.94 | 15.24 | 34.48 | 39.01 |
| relative-gap damping | 0.84 | 3.88 | 12.45 | 29.26 | 36.67 |
| Jacobi factor 1 | 0.84 | 3.88 | 12.44 | 29.27 | 36.71 |
| PB1, Frobenius 0.5, 1 NS | 0.65 | 2.84 | 8.88 | 21.07 | 26.38 |

- The PB1 generator itself is free.
- The 8-iteration spectral power estimate costs +1.1–5.3 ms per call. It is
  latency-bound at small n: +1.4 ms (2.7×) at n = 768.
- Eager timings are in `replay_pb_timing.json`.

Files:
- `replay_pb.json` and `replay_pb.pdf`
- `tables_pb.md`
- `replay_pb_timing.json`
- `replay_pb_parts/`
- logs in `logs/`
