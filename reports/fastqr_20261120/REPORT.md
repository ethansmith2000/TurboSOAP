# Fast-QR refresh variants for TurboSOAP (2026-11-20): implementation, checks, timing, 109M training

Brief: implement the refresh panel's top candidates (`research/reviews/panel_20261120/CANDIDATES.md`) as an
optimizer variant without touching production. Then time them, and get matched 109M evidence, including the
long-missing compiled-BF16 Householder QR10 control.

**Files (all new).**
- `bf16_soap_fastqr.py`: the variant optimizer.
- `benchmarks/fastqr_checks.py`: unit checks, trained-factor branch/chain comparison, compiled-IEEE check.
- `benchmarks/fastqr_timing.py`: 109M event regions and per-factor primitives.
- `benchmarks/fastqr_training.py`: frozen-bundle training runner.
- `benchmarks/fastqr_report.py`: collects everything into this directory (`summary.json`, figure, run evidence).
- `benchmarks/fastqr_stage2.py`: stage-2 gates, deltas, decision rule and figure (`stage2.json`).
- Raw training bundles are in `/workspace/data/fastqr_20261120/`.
- `bf16_soap.py`, `paper/` and all other existing files are unchanged. Nothing is committed.

**Status: stage 1 complete** (5 arms × 2,000 calls, seed 204, all audits passed). Answers to the brief are in
§5. **Stage 2 complete** (Householder QR10 and C2+C1, 10k calls, seeds 204/205, all gates passed). Its decision
rule was fixed before any result (§6), and the results are in §7:
- **Endpoint attribution:** "BF16 removes the QR10 advantage" on both seeds. Matched BF16 QR10 ends +0.0018 and
  +0.0003 against BF16 warm10/QR50, where FP32 QR10 ends −0.0112 and −0.0101.
- **C2+C1:** fails its pre-registered endpoint tolerance on seed 205 (+0.0037 against ≤ 0.003); passes on seed
  204 (+0.0018).
- **Stage 3** (BF16 QR50 to 10k on seeds 204/205, plus an FP32-storage QR10 screen at 2k) has been approved and
  is running. Its gates and decision rules were fixed in §8 before launch.
- **Stage 3 results (§10).**
  - **BF16 QR50:** ends within 0.003 of warm10/QR50 on both seeds (+0.0007, +0.0028). So warm refreshes add no
    endpoint value resolvable at two seeds.
  - **FP32-storage QR10 screen:** a gate failed (the arm is run-to-run nondeterministic on GPU), so its result is
    indicative only. It points to storage, with f = 0.91 and g = 0.62 (partial).
- **Stage 4 results (§11).**
  - **C6, S = 300:** passes at the margin (g(500) = 0.54, Δ@10k = +0.0026).
  - **C6, S = 1000:** fails on endpoint harm (+0.0081).
  - **Both:** fall behind production right after the switch, by 0.012–0.024 at call 1500.
  - **PB8:** dropped.
  - **Seed 205 for S = 300** awaits the user's decision.

## 1. Implementation (`bf16_soap_fastqr.py`)

`FastRefreshSOAP(CompiledSOAP)` changes only which function runs at a refresh event:

- `qr_kind='householder'` (default) is production `qr_basis`.
- `qr_kind='cholqr'` is C1, Gram-shifted CholeskyQR3 with no operator ridge:
  - Z = C Q, the same descending Rayleigh-quotient sort, and the same V permutation as `qr_basis`;
  - pass 1 factors ZᵀZ + s·I with s = `shift_rel`·λ_max(ZᵀZ) (8 power iterations, `shift_rel` = 1e-6);
  - then 2 unshifted CholeskyQR passes.
- **IEEE FP32.** Every FP32 product in the CholQR event region runs in IEEE FP32. `allow_tf32` is switched off
  around the compiled call only, because cuBLAS reads the flag at call time. The process stays at "high", so
  production's TF32 Z product and the Householder arms are untouched.
- **Fallback.** A factor whose Cholesky reports `info ≠ 0` in any pass, or whose output is non-finite, is recomputed
  on the host after the step:
  - Householder QR of the same sorted Z, so the V permutation stays valid;
  - its moment is re-transported from the pre-transport M;
  - the compiled region returns that M, the orders and the ok flags.
  - It costs one host synchronization per QR event. Every fallback is logged (call, matrix shape, side, index).
- `warm_ns_steps=1` is C2: production `warm_basis` with one Newton–Schulz correction instead of two.
  `warm_basis_ns(..., 2)` is a verbatim copy, checked bit-equal.
- **Schedules.** `precondition_frequency` K and `hard_reset_interval` k, as in production. Refreshes come every K
  ages; the refresh is QR (of `qr_kind`) when age % k == 0 and warm otherwise. k = K gives QR-K with no warm step.
- **Default dispatch.** With the defaults every event dispatches to production's own compiled functions
  (`ordinary_group`/`warm_group`/`qr_group`), so the default optimizer is production by construction.

## 2. Verification

### 2.1 Unit checks (`unit_cpu.json`, `unit_cuda_compiled.json`): all pass

| check | CPU eager | GPU compiled (109M shapes, 3 per shape) |
|---|---|---|
| default FastRefreshSOAP == production CompiledSOAP, 121 calls, every param/state bit-equal (warm10/QR50 and QR10) | pass | pass |
| `warm_basis_ns(·, 2)` == `warm_basis` (bit) | pass | pass |
| fast 'cholqr' event == production 'qr' event apart from the basis: params, C_L, C_R and V permutation bit-equal | pass | pass |
| real Cholesky failure (first-row-only gradient, so C_L is exactly rank 1): fallback Q == production `qr_basis`, M re-transported, other elements untouched, states finite | pass (bit-equal to `qr_basis`) | pass |

### 2.2 Compiled IEEE CholQR on GPU (`compiled_ieee_check.json`)

Five trained factors (n = 1024–4096). The compiled CholQR under IEEE is **bit-equal to eager IEEE** on every
factor, so the precision switch reaches the compiled region's cuBLAS calls. With TF32 Grams (relative Gram error
1e-2 to 4e-5):
- the same function **fails on 2 of 5** factors (e_Q 0.77–0.97);
- on a third (n = 4096) it **silently returns a wrong basis**: Cholesky succeeds and the output is finite and
  orthonormal, but it is 1.55 rad off Householder;
- IEEE is 1.9e-3 rad off, at the BF16 floor.

The ok flag cannot catch this TF32 failure mode, so IEEE is required, not optional.

### 2.3 Trained factors: CholQR3 against Householder on the same input, every refresh age

**Data.** The refresh-study snapshots (`/workspace/data/turbosoap_refresh_study_20261120/`): 1-layer models of
width d, 8 factors per width (5 of dim d, then 2d, 3d, 4d), C every 10 ages, ages 10–1000.

**Method.**
- Along a production Householder QR10 chain (BF16, from production's eigh initialization), each age branches:
  - production `qr_basis` at training precision;
  - C1 under IEEE;
  - FP32 Householder of the identical sorted Z.
- A CholQR10 chain with fallback is also run.
- d = 256 ran on CPU, d = 512–2048 on GPU, with IEEE for the Cholesky path.
- Files: `trained_d*.json`, `logs/trained_*.log`.

| width (factor n) | factor-ages | **C1 failures, shift 1e-6** | shift 1e-7 | worst factor (1e-6) | latest failing age | chain fallbacks (of 800) |
|---|---|---|---|---|---|---|
| d=256 (256–1024) | 800 | **1 (0.1%)** | 0 | 768-L 1/100 | 90 | 0 |
| d=512 (512–2048) | 800 | **3 (0.4%)** | 2 | 1536-L 2/100 | 230 | 6 |
| d=1024 (1024–4096) | 800 | **33 (4.1%)** | 17 | 4096-L 13/100 | 610 | 27 |
| d=2048 (2048–8192) | 800 | **88 (11%)** | 66 | 8192-L 47/100 | 810 | 89 |

- **Failures are not confined to early ages.** This contradicts the panel correction's "every failure at age
  ≤ 20", which came from ages ≤ 50 at d ≤ 1024.
  - They are densest in the first 100 ages.
  - They continue to age 290 (4096-L), 610 (1024-R) and 810 (8192-L).
- **The rate grows steeply with n.**
  - The largest left factors dominate (QKV 3d, SwiGLU-in 4d), plus the 2d/4d right factors at d ≥ 1024.
  - Almost all failures break in the first unshifted pass (pass index 1).
  - The rest are marginal: an eager rerun of the same input passes.
- **The 109M factor sizes (768–3072) sit between d = 512 and d = 1024.** At d=1024: 3072-L 3/100, 2048-R 8/100.
  - From these factors one would expect 1–4% fallbacks. **Measured in closed-loop 109M training it is 7–10%**
    (§4.2), so the 1-layer snapshots understate it.
- **Shift 1e-7 fails less often than 1e-6 at every width** (0 vs 1, 2 vs 3, 17 vs 33, 66 vs 88), with the same
  accuracy (first-half column angle ≤ 7.8e-4). This favours 1e-7, but stage 1 uses the specified 1e-6.

**Accuracy where C1 succeeds** (same sorted Z, FP32 before BF16 rounding):

| width | worst column angle, leading n/2 columns | worst, all columns | BF16 floor (Householder vs its BF16 rounding) | top-n/8 prefix-span distance (FP32) | e_Q of BF16 output: C1 / Householder (max) | max abs r_C difference |
|---|---|---|---|---|---|---|
| d=256 | 7.0e-5 | 7.8e-4 | 1.8–2.6e-3 | ≤ 6e-8 | 2.392e-3 / 2.392e-3 | 5e-5 |
| d=512 | 9.6e-5 | 1.4e-3 | 1.8–2.8e-3 | 0 | 2.37e-3 / 2.37e-3 | 5e-6 |
| d=1024 | 2.5e-4 | 8.1e-3 | 1.8–2.7e-3 | 0 | 2.37e-3 / 2.37e-3 | 1e-5 |
| d=2048 | 4.6e-4 | 9.3e-3 | 1.7–2.8e-3 | 0 | 2.36e-3 / 2.36e-3 | 3e-5 |

- **Where the differences sit.** Outside the trailing near-null columns, C1 matches Householder an order of
  magnitude below the BF16 floor. The all-column maxima above the floor are in trailing near-null columns, whose
  orientation is ill-determined for Householder too; they do not move r_C.
- **Sort order.** C1's sort keys come from an IEEE Z, production's from TF32. The orders agree at 99.8–99.97% of
  positions on average (minimum 97.1%) on GPU, and exactly on CPU.

**Chained.** CholQR10 with fallback and the Householder QR10 chain give the same late-window (ages 500–1000) r_C
interval at every width, inside the panel's 0.003 gate:

| width | Householder chain | CholQR chain |
|---|---|---|
| d=256 | 0.2066 | 0.2067 |
| d=512 | 0.2457 | 0.2458 |
| d=1024 | 0.2747 | 0.2747 |
| d=2048 | 0.3141 | 0.3140 |

## 3. Timing (`timing.json`)

**Setup.** One exclusive RTX 5090 (gpu-claim GPU 3), host load ≈ 14, with the 109M settings: deterministic,
FP32 "high", no BF16 reduced reductions, 4 threads.
- Event regions are `opt.step()` over 4×12 matrices of the 109M shapes, plus fused AdamW on a 50257×768 stand-in.
- Gradients are redrawn every call after a 40-call warm-in, so covariances are full rank.
- 30 interleaved repetitions; medians, with the IQR in the JSON.
- **No CholQR factor fell back during timing** (0 of 5,760 factor refreshes on these synthetic inputs), so the
  C1 times exclude fallback cost.

**Event regions, synchronized host wall** (CUDA-event times agree within 0.05 ms):

| event | measured (ms) | 109M training reference (ms) | ratio |
|---|---|---|---|
| ordinary (production) | 9.98 | 10.14 | 0.98 |
| warm, 2 NS (production) | 60.59 | 61.61 | 0.98 |
| **Householder QR (production)** | **613.34** | 601.18 | 1.02 |
| ordinary (FastRefreshSOAP; production function) | 10.04 | — | — |
| **warm, 1 NS (C2)** | **47.69** (−12.9) | — | — |
| **CholQR3 QR event (C1, IEEE)** | **360.94** (−252.4; 0.59×) | — | — |

**Amortized optimizer cost per call:** (Σ events)/cycle from the medians above. "Refresh/call" is that minus the
ordinary event.

| schedule | optimizer ms/call | refresh ms/call | vs production |
|---|---|---|---|
| warm10/QR50 (production) | 26.09 | 16.12 | — |
| Householder QR10 | 70.32 | 60.34 | +44.2 |
| Householder QR20 | 40.15 | 30.17 | +14.1 |
| **CholQR10 (C3)** | 45.07 | 35.10 | **+19.0** |
| **CholQR20 (C1-20)** | 27.53 | 17.55 | **+1.4** |
| warm10/CholQR50 (C1) | 21.05 | 11.07 | −5.0 |
| warm10 ns=1/QR50 (C2) | 25.06 | 15.08 | −1.0 |
| **warm10 ns=1/CholQR50 (C2+C1)** | 20.01 | 10.04 | **−6.1** |

**Per factor, batch 12, compiled** (ms per call; `primitives` in the JSON):

| n | Householder `qr_basis` | C1 (IEEE) | C1 with TF32 Grams (not usable) | warm 2 NS | warm 1 NS | IEEE Gram | `cholesky_ex` | IEEE triangular solve | one fallback (eager QR + transport, batch 1) |
|---|---|---|---|---|---|---|---|---|---|
| 768 | 31.2 | 7.6 | 6.6 | 0.80 | 0.63 | 0.34 | 1.2 | 0.67 | 3.0 |
| 1536 | 81.6 | 33.0 | 23.6 | 3.97 | 2.92 | 1.93 | 4.2 | 2.63 | 7.0 |
| 2304 | 143.8 | 88.7 | 57.6 | 12.2 | 8.8 | 6.32 | 9.8 | 6.58 | 12.3 |
| 3072 | 218.0 | 189.5 | 111.4 | 29.0 | 20.8 | 15.35 | 18.3 | 13.59 | 18.7 |

**Reconciliation.** Σ per-factor C1 over the 109M set (5×768 + 1536 + 2304 + 3072) = 349.1 ms, plus about 15 ms
of ordinary step and transports, gives about 364 ms, against the 360.9 ms event. Σ Householder = 599.5 ms, giving
about 614, against 613.3.

**Against the panel's gates.**
- C2 passes its timing gate: the warm event is 12.9 ms faster, against a required ≥ 8 ms.
- C1 misses the original ≤ 200 ms event gate (360.9 ms). That gate was set from TF32 Gram times. The panel's
  corrected IEEE estimate was ≈ 325 ms; the measurement is 11% above it.
- C1 stays cheaper than Householder at every 109M size, but only by 13% at n = 3072. There the three IEEE Grams,
  three Choleskys and three triangular solves take 24%, 29% and 22% of its time; the rest is the Z product, power
  iterations and casts.
- The C1 times exclude fallbacks. At 3% of 96 factors per event that would add about 30 ms.

## 4. 109M training, stage 1 (seed 204, 2,000 calls)

**Setup.**
- Bundle `/workspace/data/fastqr_20261120/stage1_seed204_c2000`, with frozen sources and sha256s. Compact evidence
  is in `runs/stage1_seed204_c2000/` and `stage_stage1_seed204_c2000/`.
- The training loop is `bf16_storage_screen.run` itself, with the paper's conventions: the 109M tied-embedding
  model, the frozen OWT rows (sha-matched), the 10k LR horizon, LRs, betas, clip and batch, and validation on all
  1,024 rows.
- The 5 arms ran sequentially on one exclusive RTX 5090 (gpu-claim GPU 3), 06:49–07:53 UTC; wall 64 min, about
  1.07 GPU-hours.

**Reproducibility gates (all passed).**
- **Startup.** Every arm reproduced the paper's seed-204 startup exactly: initial weights, first gradient, loss,
  norm and initial validation CE.
  - Attempt 1 failed the gradient check because the runner did not set the paper's recorded launch environment
    (`CUBLAS_WORKSPACE_CONFIG=:4096:8` and thread settings, `paper/REPRODUCIBILITY.md`). It is kept in
    `/workspace/data/fastqr_20261120/failed_attempts/`.
  - The runner now sets that environment and asserts it in every worker.
- **Bit-exact reproduction.** The fresh production warm10/QR50 arm **reproduced the paper's cached seed-204 run bit
  for bit** at all 8 validation points through call 2000 (max |ΔCE| = 0). The patched runner is therefore
  transparent, and this launch is matched to the paper's runs.
- **Audits.** Event counts are exactly as scheduled. There are no graph breaks. Unique compiled graphs stay flat
  from call 101 to 2000: 12 for warm schedules, 8 for QR-only schedules, so switching TF32 off around CholQR
  caused no recompiles. Final states are finite and peak memory is 8.22 GiB in every arm.

### 4.1 Validation CE and time

Step and optimizer times are means over calls 101–2000, synchronized host wall, as in the paper. The "QR event"
column is the mean optimizer time of a QR-type event, with the median in parentheses.

| arm | CE@500 | CE@1000 | CE@1500 | CE@2000 | step ms | optimizer ms | QR event ms | warm event ms |
|---|---|---|---|---|---|---|---|---|
| warm10/QR50, production (fresh) | 5.0015 | 4.3616 | 4.1224 | **3.9939** | 320.5 | 26.6 | 615.7 | 61.8 |
| **Householder QR10, BF16 compiled** (matched control) | 4.8857 | 4.3251 | 4.1231 | **3.9981** | 364.8 | 71.0 | 616.0 | — |
| **CholQR10 (C3)** | 4.8788 | 4.3235 | 4.1242 | **4.0000** | 343.8 | 52.3 | 429.4 (414.5) | — |
| **CholQR20 (C1-20)** | 4.9391 | 4.3481 | 4.1283 | **3.9964** | 325.6 | 31.7 | 438.3 (417.3) | — |
| **warm10 ns=1 / CholQR50 (C2+C1)** | 5.0035 | 4.3565 | 4.1273 | **3.9954** | 315.7 | 22.3 | 460.4 (429.3) | 48.6 |
| *paper, Householder QR20 (cached, other launch)* | 4.9369 | 4.3418 | 4.1234 | 3.9904 | 329.6 | 40.5 | — | — |
| *paper, Householder QR50 (cached, other launch)* | 5.0180 | 4.3647 | 4.1323 | 3.9987 | 313.6 | 22.9 | — | — |
| *paper, FP32 eager QR10 (stock; pilot prefix)* | 4.8776 | 4.3088 | 4.1084 | 3.9859 | 380.1 | — | — | — |

**Noise scale.** Production seed 205 − seed 204 is −0.0117, −0.0095, −0.0018 and −0.0039 at calls 500, 1000,
1500 and 2000. Differences of a few 1e-3 at call 2000 are single-seed noise. The fresh arms' forward/backward
time was 291–294 ms, against about 288 ms in the paper's launches. Compare step times within this launch.

Figure: `fastqr_ce_vs_step.pdf`/`.png`, CE at calls 500/1000/2000 against mean step time. Hollow markers are
cached runs from other launches.

### 4.2 CholQR fallbacks in closed-loop 109M training

| arm | factor refreshes | fallbacks | rate | calls 1–500 | 501–1000 | 1001–1500 | 1501–2000 | QR events with ≥ 1 fallback |
|---|---|---|---|---|---|---|---|---|
| CholQR10 | 19,104 | 1,412 | **7.4%** | 15.0% | 4.5% | 4.8% | 5.5% | 199/199 (max 45) |
| CholQR20 | 9,504 | 786 | **8.3%** | 18.0% | 4.5% | 4.9% | 6.1% | 99/99 (max 42) |
| warm10 ns=1 / CholQR50 | 3,744 | 368 | **9.8%** | 21.8% | 4.5% | 6.2% | 8.1% | 39/39 (max 40) |

- **The rate is higher than the trained 1-layer factors suggested** (0.4–4% at d = 512–1024).
  - It never decays to zero: 4.5–8% persists through call 2000 (at 10k: a 9–10% plateau, §7.3).
  - Every QR event has at least one fallback.
  - The QR events after four warm refreshes fail more often than those in a pure QR schedule.
- **By factor (CholQR10)**, the share of each factor's refreshes that fall back: QKV-left (n = 2304) 20%,
  SwiGLU-in-left (3072) 17%, attention-out-right 6%, QKV-right 6%, SwiGLU-out-right (1536) 5%, the rest ≤ 3%.
- **Cost.** Each fallback costs about 12.7 ms (event time against fallback count: r = 0.97; intercept 357 ms
  against the 361 ms no-fallback timing).
  - A CholQR10 event in training therefore averages 429 ms, against 616 ms for Householder: −30% rather than the
    −41% of §3.
  - The fallback is batched per factor group but runs eagerly, after a host synchronization.
- **Correctness.** All fallbacks were handled: finite states, and Gram errors at the BF16 floor (§4.3).

### 4.3 Basis orthogonality in training

- **Normalized Gram error** ‖QᵀQ − I‖_F/√n, max over all 96 factors at 11 diagnostic calls (6 right after QR
  events, and at ages 540/1040/1540/1990, i.e. after 4 warm refreshes): 0.00236–0.00237 in every arm except
  C2+C1, which is 0.00245.
- **Spectral ‖QᵀQ − I‖₂**, the C2 "remaining check":
  - production warm10/QR50: max 0.0084;
  - C2+C1: max **0.026**, at call 1041, i.e. after the 4th consecutive one-NS warm refresh;
  - QR-only arms: 0.005–0.006.
  - C2 passes the panel's < 0.07 gate with about 2.7× margin.

**Basis quality right after refresh.** Gram error cannot see a wrong-but-orthonormal basis (§2.2's TF32 case), so
also compare the per-factor coordinate residual r_C immediately after the QR events at calls 501/1001/1501/1951
(`coordinate_residuals` in `geometry.json`).

| call | CholQR10 mean r_C | Householder QR10 mean r_C | QKV-left, CholQR10 / Householder | SwiGLU-in-left, CholQR10 / Householder |
|---|---|---|---|---|
| 501 | 0.168 | 0.172 | 0.311 / 0.316 | 0.378 / 0.405 |
| 1001 | 0.230 | 0.213 | 0.393 / 0.437 | 0.545 / 0.485 |
| 1501 | 0.238 | 0.270 | 0.376 / 0.387 | 0.525 / 0.596 |
| 1951 | 0.198 | 0.193 | 0.359 / 0.357 | 0.522 / 0.500 |

- **No systematic difference.** CholQR is higher at some calls and lower at others, including on the
  fallback-prone large left factors.
- **C2+C1 against production** at the same calls: 0.293/0.291, 0.241/0.226, 0.235/0.233, 0.208/0.212.
- **Limits.** The arms follow different trajectories, so individual factors scatter (ratios 0.3–2.2). This is
  consistency evidence, not a same-input test. The same-input test is §2.3: under IEEE, no factor that passed
  `ok` was off Householder beyond 4.6e-4 rad (leading half of the columns) in 3,200 trained factor-ages.
- **No same-input check at the 109M closed loop exists**, because its covariances were not saved.

## 5. Answers

**Does CholQR10 dominate warm10/QR50?** **No, not at 2,000 calls.**
- CholQR10 is far ahead early: −0.123 CE at call 500 and −0.038 at 1000. That is ten times the seed spread.
- By call 1500 the lead is gone (+0.002), and at call 2000 it is +0.006, within noise.
- It costs +23.2 ms per step (+7.2%).
- At equal wall time (CholQR10's 2,000 calls = 688 s), production would have run about 2,145 calls.
- **At 10k (§7), by inference.** CholQR10 was not run at 10k. Householder QR10, which CholQR10 tracks within
  noise at 2k, does not beat warm10/QR50 at 10k on either seed. CholQR10 would also need to be better than
  Householder QR10 at 10k to dominate, and nothing suggests it is. So it would not dominate either; this is an
  inference, not a measurement.

**CholQR10 against the matched Householder QR10.**
- It is the same trajectory within noise: ΔCE −0.007, −0.002, +0.001 and +0.002 at calls 500–2000.
- It is 21 ms per step cheaper (343.8 vs 364.8 ms, −5.8%), including its 7.4% fallbacks.
- **Through 2k calls, C1 behaves as a drop-in for Householder QR events at 109M**, and CholQR10 dominates
  Householder QR10. The evidence: the same CE curve, same post-refresh r_C statistics, and same-input agreement
  on trained snapshots (§2.3, §4.3).
- **C1 alone is untested at 10k.** The only 10k runs containing C1 are the C2+C1 runs, and those cannot separate
  C1 from C2 (§7.3).
- **Silent failure is not excluded.** The `ok` flag does not detect a wrong-but-orthonormal basis, and the
  large left factors that pass sit near the failure edge (17–26% of their refreshes fall back).
- **Before production use, add an a-posteriori check.** For example, compare a few columns against one
  Householder solve, or bound ‖R_new − R_shifted‖.

**C2+C1 (warm10 ns=1 / CholQR50) against production.**
- CE agrees within noise: +0.002, −0.005, +0.005 and +0.0015.
- It is cheaper: optimizer 22.3 against 26.6 ms/call (−16%), step −4.8 ms (−1.5%).
- The saving is smaller than §3's −6.1 ms/call because of its 9.8% fallbacks.
- At 2k this is the only arm faster than production, and C2 passes its orthogonality gate.
- **At 10k (§7.3) it fails the pre-registered endpoint tolerance on seed 205** (+0.0037), and is +0.0018 on seed
  204. The 2k agreement does not establish equal CE at the endpoint.

**CholQR20 (C1-20).**
- Its CE follows the paper's Householder QR20 within noise (+0.002 to +0.006, different launch).
- It costs +5.1 ms per step against production (+1.6%). Householder QR20 costs +14.1 ms/call by the §3 timing
  (+15.5 ms per step across the paper's launches).

**The paper's pending items** (`\pending{}` in `05_results.tex` and `06_discussion.tex`):
- **Fresh warm10/QR50 in the same launch as a matched BF16 QR10:** done. It reproduces the cached run bit for bit.
- **Measured BF16 QR10 step time:**
  - 364.8 ms per step, 71.0 ms optimizer. The projections were 360 ms (seed 204) and 352.7 ms / 69.2 ms (seed 205).
  - warm10/QR50 is **12.1% faster per step** than the matched BF16 QR10; projected 12–12.8%.
- **Recovery fractions against BF16 QR10:**
  - warm recovers 12.5% of the QR50 − QR10 gap at call 500 (FP32-QR10 version: about 12%) and 7.9% at call 1000.
  - At calls 1500 and 2000, BF16 QR10 is no better than BF16 QR50 (gaps 0.009 and 0.0006). Warm (3.9939) is
    lower than both at call 2000, so the fraction is undefined there.
  - The paper's "about 37% at call 2000" against FP32 QR10 does not carry over to BF16 QR10.

**The paper's endpoint attribution** (resolved by stage 2 under the pre-registered rule; details in §7.2).

The refresh gap, warm − QR10, within each implementation:

| implementation | call 500 | 1000 | 1500 | 2000 | 10k |
|---|---|---|---|---|---|
| FP32-state eager ResearchSOAP (paper's stock arms), seed 204 | +0.118 | +0.061 | +0.020 | +0.010 | **+0.0070** |
| BF16 compiled, seed 204 | +0.116 | +0.037 | −0.001 | −0.004 | **−0.0018** |
| BF16 compiled, seed 205 | +0.098 | +0.038 | +0.005 | −0.003 | **−0.0003** |

- **Before stage 2:** at 2k, two readings were open: (a) the BF16 compiled implementation costs QR10 but not warm;
  (b) single-seed noise in one BF16 QR10 run.
- **Stage 2 favours (a).** Both seeds agree under the pre-registered rule: matched BF16 QR10 ends level with BF16
  warm10/QR50 at 10k (§7.2). Seed 204's run is bit-identical to stage 1 through call 2000, so seed 205 is the
  one new draw.
- **What remains open** is why. Either refresh cadence does not matter at the BF16 endpoint, or the BF16
  compiled QR10 path is itself handicapped. Stage 2 alone does not separate the two (§7.2). The stage-3
  FP32-storage screen (§10.2) points, indicatively, to BF16 storage.
- **Confound.** "Implementation" bundles BF16 storage, BF16 compute and the compiled code path.

**For the paper's thesis** ("approximate refresh" or "cheap exact-ish refresh"):
- C1 is exact-ish: it matches Householder at the BF16 floor on trained snapshots and gives the same training
  trajectory through 2k calls. C1 alone is untested at 10k.
- C1 is cheaper than Householder at every 109M size, but needs a Householder fallback and an accuracy guard
  (above):
  - 7–10% of all refreshes at 2k, 9.6–9.8% through 10k;
  - on the large left factors, 17–26% at 2k; through 10k, QKV-left 32% and SwiGLU-in-left 18%.
- Exact-ish refresh at QR10 cadence buys an early-training advantage, not an endpoint advantage. That holds at
  2k, and at 10k on both seeds for Householder QR10 (§7.2). It costs +7% (CholQR10) to +14% (Householder QR10)
  step time.
- C2+C1 saves 1.1–1.5% of step time, but fails the pre-registered 10k tolerance on seed 205 (+0.0037). On seed
  204 it is indistinguishable from production (§7.3). **No tested fast-QR variant both saves time and meets the
  10k tolerance on both seeds.**

## 6. Stage 2: 10,000 calls, seeds 204 and 205 (approved 2026-11-20; protocol fixed before results)

The user approved Householder QR10 and C2+C1. CholQR10 is not run at 10k: at 2k it follows Householder QR10 within
noise (§5).

**Runs.** One frozen bundle per (seed, arm), `/workspace/data/fastqr_20261120/stage2_seed{204,205}_{arm}_c10000`,
each on its own exclusive `gpu-claim` (launcher `logs/stage2_launch.sh`, queued 09:14 UTC).
- **Arms:** `bf16_hh_qr10` (production `CompiledSOAP`, Householder every 10) and `bf16_w10_cq50_ns1` (C2+C1).
- **Shared with stage 1:** the same frozen sources except the runner (the runner only adds late diagnostic calls
  5001/5041/9951/9991, so 2k protocols are unchanged), the same inputs and the same launch environment.

**Comparators (cached; no fresh production arm in this stage).**

| seed | BF16 warm10/QR50, 10k | status of comparator | FP32-state stock QR10, 10k | FP32-state stock warm, 10k |
|---|---|---|---|---|
| 204 | `bf16_replicate_20261027_seed204/runs/bf16`: 3.42239 | **prefix-proven**: its 8 CEs through call 2000 equal this study's fresh production arm to full precision | `owt_restart_20261026` `primary_stock_qr`: 3.41115 | `primary_stock_warm`: 3.41813 |
| 205 | `bf16_full_horizon_20261026_seed205/runs/bf16`: 3.41935 | **source- and startup-matched, not prefix-proven** (below) | same bundle, `runs/qr10`: 3.40927 | not run |

- **Seed-205 comparator.**
  - Same as here: `bf16_soap.py`, `transformer.py` and input shas.
  - Its `bf16_storage_screen.py` (7c7aee…) is the version used by the seed-204 replicate, which is prefix-exact
    against this study's runs.
  - Its outer runner `bf16_full_horizon.py` is an earlier revision (6e0b5d… against ef3f19…).
  - Stage-2 startup gates check weights, first gradient, loss, norm and initial CE against it.
- **Paper endpoint gap** (BF16 warm − FP32 QR10): +0.0112 (seed 204), +0.0101 (seed 205).
- **Production seed spread at 10k:** 0.0030.

**Gates (all must pass before any 10k delta is interpreted).**
1. Per-run startup gate and the stage-1 audits: schedules, graph breaks, recompiles, finiteness.
2. **Seed-204 prefix gate.** Both seed-204 runs must reproduce their stage-1 runs bit for bit through call 2000:
   all 8 validation CEs, plus C2+C1's 368 fallbacks in 3,744 factor refreshes.

**Decision rule for the endpoint attribution** (fixed before results).
- **Statistic:** per seed, Δ_s = CE(BF16 Householder QR10) − CE(cached BF16 warm10/QR50) at every validation
  call. Primary: call 10,000. Secondary: the mean over calls 5000–10000. Seeds are reported separately, never
  pooled.
- **Outcomes:**
  - **The refresh effect survives BF16:** Δ ≤ −0.007 on both seeds, i.e. BF16 QR10 recovers most of FP32 QR10's
    0.010–0.011 advantage. Then the paper's endpoint gap is a refresh-policy effect.
  - **BF16 removes the QR10 advantage:** Δ ≥ −0.003 on both seeds (within the seed spread). Then the paper's
    endpoint gap is mainly an implementation (BF16/compiled) effect on QR10.
  - **Otherwise unresolved:** the seeds split, or either seed falls between −0.007 and −0.003. Then the effect is
    partial, or noise, at two seeds.

**C2+C1 criterion.**
- Keeps production's endpoint if |CE(C2+C1) − CE(cached production)| ≤ 0.003 at call 10,000 on each seed.
- Its spectral ‖QᵀQ − I‖₂ must stay < 0.07 at every diagnostic call, including 5041 and 9991.
- Reported: the fallback rate per 1k-call window and per factor through 10k.

**Time.** These runs share the machine with other jobs and have no same-launch production arm, so step-time
claims stay with stage 1 and §3. Stage-2 step times are reported with fb_ms as the drift indicator.

**Cost:** about 4.1 GPU-hours; about 65 min wall once 4 GPUs are free.

## 7. Stage 2 results (10,000 calls, seeds 204 and 205)

Analysis: `benchmarks/fastqr_stage2.py`, writing `stage2.json` and the figure `fastqr_stage2_delta.{pdf,png}`.
Evidence is copied to `runs/stage2_*` and `stage_stage2_*`.

**All gates passed.**
- **Startup.** All four runs reproduce their seed's paper startup: weights, first gradient, loss, norm and
  initial CE. This is the first use of the seed-205 reference.
- **Seed-204 prefix.** Both runs equal their stage-1 runs bit for bit through call 2000:
  - every validation CE and every per-call loss;
  - for C2+C1, also the full refresh log and fallback log (368 of 3,744).
- **Seed-205 comparator, strengthened after the protocol was written.** On both seeds, each fresh run's per-call
  losses equal the cached production run bit for bit through call 11, where the schedules first diverge.
  - On seed 205 this confirms the cached run shares data order, model, forward/backward and ordinary optimizer
    steps.
  - Its warm/QR50 events run the same `bf16_soap.py` (sha-matched), but this check does not exercise them.
- **Audits.**
  - Event counts are exactly as scheduled: QR10 has 9,000 ordinary + 999 QR; C2+C1 has 9,000 + 800 warm + 199 QR.
  - There are no graph breaks, and unique compiled graphs stay flat (8 / 12) at every capture through call 10000.
  - Final states are finite and peak memory is 8.22 GiB in every run.
- **Wall time.** 56–63 min per run, on 4 GPUs between 09:47 and 11:08 UTC, 3.97 GPU-hours in total. The claims
  first waited 33–56 min behind other projects' sweeps.

### 7.1 Validation CE

Δ is the difference against the cached BF16 production run of the same seed. "Late" is the mean over calls
5000–10000 (7 points).

| seed | run | CE@2000 | 5000 | 7500 | 10000 | Δ@10k | late Δ |
|---|---|---|---|---|---|---|---|
| 204 | cached BF16 warm10/QR50 (production) | 3.9939 | 3.6417 | 3.4922 | 3.4224 | 0 | 0 |
| 204 | **BF16 Householder QR10** | 3.9981 | 3.6447 | 3.4953 | 3.4242 | **+0.0018** | +0.0027 |
| 204 | **C2+C1** | 3.9954 | 3.6428 | 3.4936 | 3.4242 | **+0.0018** | +0.0016 |
| 204 | *FP32-state stock QR10 (cached)* | 3.9859 | 3.6301 | 3.4816 | 3.4112 | −0.0112 | −0.0106 |
| 204 | *FP32-state stock warm (cached)* | 3.9958 | 3.6384 | 3.4879 | 3.4181 | −0.0043 | −0.0039 |
| 205 | cached BF16 warm10/QR50 (production) | 3.9900 | 3.6387 | 3.4889 | 3.4194 | 0 | 0 |
| 205 | **BF16 Householder QR10** | 3.9932 | 3.6396 | 3.4902 | 3.4197 | **+0.0003** | +0.0010 |
| 205 | **C2+C1** | 3.9910 | 3.6416 | 3.4922 | 3.4231 | **+0.0037** | +0.0037 |
| 205 | *FP32-state stock QR10 (cached)* | 3.9826 | 3.6302 | 3.4801 | 3.4093 | −0.0101 | −0.0091 |

Figure: `fastqr_stage2_delta.png`, the deltas from call 1500 to 10000. The grey band is ±0.003, the production
seed spread at 10k.

### 7.2 Endpoint attribution: BF16 removes the QR10 advantage (pre-registered outcome, both seeds)

- **Rule outcome.** Δ@10k is +0.0018 (seed 204) and +0.0003 (seed 205), both ≥ −0.003. The late-window means
  (+0.0027, +0.0010) agree.
- **No lead after call 2000.** BF16 QR10 is never better than BF16 warm10/QR50 on either seed (Δ between
  +0.0003 and +0.0053). FP32 QR10 is 0.007–0.012 better at every one of those points.
- **The paper's endpoint gap, decomposed (seed 204).** BF16 warm − FP32 QR10 = 0.0112, made of:
  - 0.0070 = FP32 warm − FP32 QR10, the refresh effect within FP32 eager;
  - 0.0043 = BF16 warm − FP32 warm, the implementation effect on warm.
- **Inside BF16 compiled, the refresh effect is gone:** warm − QR10 = −0.0018 (seed 204), −0.0003 (seed 205).
- **The implementation hurts QR10 more than warm.** BF16 QR10 − FP32 QR10 = +0.0130 (seed 204) and +0.0104
  (seed 205), against 0.0043 for warm (seed 204; there is no seed-205 FP32 warm run). This interaction holds
  under either reading below.
- **Two readings this design cannot separate:**
  - **(i)** At the BF16 endpoint, refresh cadence does not matter: warm10/QR50 already gets what QR10 would.
  - **(ii)** The BF16 compiled QR10 path is itself handicapped, so a better BF16 QR would recover part of
    FP32's 0.010. Candidate causes:
    - BF16 basis and moment storage rounded at every QR transport, 5× more often than QR50;
    - production `qr_basis` differs from ResearchSOAP's refresh (Z = CQ reuse, near-tie sort-order changes;
      `bf16_qr_attribution_20261027`).
  - "Implementation" also bundles storage precision, compute precision and compilation.
- **For the paper.**
  - **Supported:**
    - The +0.011 / +0.010 endpoint gap between BF16 warm10/QR50 and FP32 QR10 cannot be attributed to the
      refresh policy.
    - Within the matched BF16 implementation, warm10/QR50 reaches QR10's 10k CE (within 0.002 on both seeds)
      at 12.1% lower step time (stage 1, same launch).
  - **Not supported** while (ii) stands: "warm loses nothing against exact refresh".
- **The warm cells already exist; one QR10 cell is missing.** `bf16_learning_20261026` (seed 204, 2k, same
  startup) ran the warm10/QR50 schedule in three implementations:

  | implementation | warm10/QR50, CE@2000 | QR10, CE@2000 |
  |---|---|---|
  | FP32 eager (stock) | 3.9958 | 3.9859 (`qr10`) |
  | compiled, FP32 storage | 3.9916 (`compiled_fp32`) | **not run** |
  | compiled, BF16 storage | 3.9939 (`bf16`) | 3.9981 (this study) |

- **Separating experiment: run in stage 3** (§10.2; indicative only, a gate failed). Originally specified as one
  compiled FP32-storage QR10 arm, seed 204, 2k calls,
  about 15–20 GPU-minutes.
  - About 3.986 would point to BF16 storage precision.
  - About 3.998 would point to the compiled QR path (`qr_basis`, batching, compilation).

### 7.3 C2+C1 through 10k: fails its pre-registered tolerance on seed 205

- **Criterion.** Δ@10k is +0.0018 on seed 204 (pass) and **+0.0037 on seed 205 (fail; tolerance 0.003)**.
- **Facts that need no comparator** (fresh runs in the same launch):
  - C2+C1 − BF16 QR10 = 0.0000 at 10k on seed 204, +0.0034 on seed 205;
  - on seed 205, C2+C1 is above cached production at every validation call from 3000 (+0.0029 to +0.0045).
- **Reading.**
  - On seed 205 there is a consistent deficit of about 0.003–0.004 from call 3000 to the end. On seed 204,
    C2+C1 is indistinguishable from production: +0.0018 at 10k, the same as Householder QR10.
  - Two seeds cannot say whether this is a real cost or seed-205 noise.
  - The design also does not say which component would cause it, C2's single NS step or C1's CholQR3. The
    runner has C2-only (`bf16_w10_qr50_ns1`) and C1-only (`bf16_w10_cq50`) arms; they were not run.
- **Benefit.** Optimizer time is 22.5 / 22.8 ms against production's 26.4 / 26.1 ms (−3.6 ms per call, about
  −1.1% of a step; −1.5% in stage 1's same-launch comparison).
- **Fallbacks through 10k.** Totals are 1,840 and 1,872 of 19,104 factor refreshes (9.6%, 9.8%).
  - **By 1k-call window:** 12.7% / 13.8% in calls 1–1000, 6.9–7.1% in 1001–2000, then a 8.7–10.7% plateau to
    call 10000. The rate does not decay.
  - **Every QR event** (199 of 199) falls back on at least one factor; the maximum is 40 / 48 per event.
  - **By factor:** QKV-left 32%, SwiGLU-in-left 18%, QKV-right 10%, attn-out-right 7–8%, SwiGLU-out-right 5%,
    SwiGLU-in-right 4–5%, the rest ≤ 1%.
  - **Cost:** a mean CholQR event takes 465 / 470 ms, against Householder's 609 / 612 ms in the QR10 runs.
- **Orthogonality gate: passes.**
  - Maximum spectral ‖QᵀQ − I‖₂ is 0.026 (seed 204, call 1041) and 0.0105 (seed 205, call 1541).
  - At calls 5041 and 9991 it is 0.0069 / 0.0071 (seed 204) and 0.0071 / 0.0071 (seed 205).
  - Householder QR10 stays ≤ 0.006.

### 7.4 Time (shared machine; step-time claims stay with stage 1 and §3)

| seed | run | step ms | optimizer ms | fb ms | QR event ms | warm event ms |
|---|---|---|---|---|---|---|
| 204 | BF16 Householder QR10 | 361.8 | 70.3 | 291.6 | 608.5 | — |
| 204 | C2+C1 | 316.0 | 22.5 | 293.5 | 464.7 | 48.7 |
| 204 | cached production (other launch) | 314.1 | 26.4 | 287.6 | — | — |
| 205 | BF16 Householder QR10 | 363.1 | 70.5 | 292.6 | 611.8 | — |
| 205 | C2+C1 | 316.4 | 22.8 | 293.6 | 469.9 | 49.7 |
| 205 | cached production (other launch) | 309.5 | 26.1 | 283.5 | — | — |

- **Drift.** Forward/backward ran 4–10 ms slower in this launch than in the cached runs. Compare optimizer time
  across launches, and step time only within a launch.
- **Stability.** Within each run, fb varied by ≤ 3 ms across the ten 1k-call windows.
- **QR10 cost.** BF16 QR10's optimizer time (70.3 / 70.5 ms) matches stage 1's 71.0 ms, about +44 ms per call
  over production.

## 8. Stage 3: BF16 QR50 to 10k, and an FP32-storage QR10 screen (protocol fixed before results)

**Status: approved by the user and launched** (queued 11:48 UTC, 2026-11-20, `logs/stage3_launch.sh`). This
protocol was written before launch and is unchanged.

**Why.**
- **QR50 to 10k.** Stage 2 found matched BF16 QR10 level with BF16 warm10/QR50 at 10k on both seeds (§7.2).
  - If refresh cadence stops mattering at the BF16 endpoint, plain QR50 (no warm events) may match too. Then the
    warm refreshes would add nothing at the endpoint.
  - QR50 has only been run to call 2000: the paper's control, CE 3.99868, +0.0047 against warm on seed 204.
  - The paper (§6, conclusion) names this as untested.
- **FP32-storage QR10 at 2k.** This is the missing cell of §7.2. It separates BF16 storage from the compiled path
  and BF16 compute as the cause of BF16 QR10's deficit against FP32 QR10.

**Runs.** One frozen bundle each in `/workspace/data/fastqr_20261120/`:

| bundle | arm | configuration | calls | seed |
|---|---|---|---|---|
| `stage3_seed204_bf16_hh_qr50_c10000` | `bf16_hh_qr50` | production `CompiledSOAP`, frequency = repair = 50, BF16 state and basis (the paper's `bf16_qr50` control) | 10,000 | 204 |
| `stage3_seed205_bf16_hh_qr50_c10000` | `bf16_hh_qr50` | same | 10,000 | 205 |
| `stage3_seed204_fp32s_hh_qr10_c2000` | `fp32s_hh_qr10` | production `CompiledSOAP`, frequency = repair = 10, **FP32 state and basis**, BF16 compute (the cached `compiled_fp32` configuration of `bf16_learning_20261026`, at QR10 cadence) | 2,000 | 204 |

- **Shared with stage 2:** the same frozen sources, sha-identical except the runner, plus the same inputs, launch
  environment and validation/diagnostic schedule. The runner only adds the two arm specs and reads storage dtypes
  from the spec; existing arms are unchanged.
- **CPU check before freezing** (`logs/stage3_cpu_check.py` and `.log`):
  - `bf16_hh_qr50` builds an optimizer bit-identical to the paper's `bf16_qr50` over 120 calls.
  - `fp32s_hh_qr10` has FP32 state and basis, and is bit-identical to `compiled_fp32` through its first refresh.
- **Launch (after approval):** `logs/stage3_launch.sh`, three exclusive `gpu-claim --wait` claims, yielding to
  other projects' sweeps as before.
- **Cost:** about 2.1 GPU-hours. Each QR50 run takes about 55 min and the FP32-storage QR10 screen about 15 min,
  so about 1 h wall once 3 GPUs are free.

**Gates (all must pass before any delta is read).**
1. **Startup:** per-run gate as in stage 2.
2. **Seed-204 QR50 prefix:** per-call losses for calls 1–2000 and all 8 validation CEs must equal the cached
   control `bf16_refresh_controls_20261027/runs/bf16_qr50` bit for bit.
   - That control ran the same `bf16_storage_screen.py` (9df9ee…), and its warm sibling already reproduced bit
     for bit.
   - If it cannot be matched, I will say so and fall back to gates 1 and 3.
3. **First-refresh losses:** per-call losses must equal the cached run of the same storage configuration, bit for
   bit, through call 11, the first schedule divergence:
   - QR50, both seeds: the cached BF16 warm10/QR50 production run;
   - FP32-storage QR10: the cached `compiled_fp32` run. Its BF16 sibling in that bundle equals this study's
     production run for all 2,000 losses.
4. **Audits:**
   - event counts: QR50 has 1 cold + 9,800 ordinary + 199 QR; QR10 at 2k has 1 cold + 1,800 ordinary + 199 QR;
   - no graph breaks, flat compiled-graph counts and a finite final state.

**Decision rule A: endpoint value of warm refreshes (QR50, both seeds).**
- **Statistic:** per seed, Δ_s = CE(BF16 QR50) − CE(cached BF16 warm10/QR50) at call 10,000. Secondary: the mean
  over calls 5000–10000. Seeds are reported separately, never pooled.
- **Outcomes:**
  - **Warm refreshes add no endpoint value resolvable at two seeds:** Δ ≤ +0.003 on both seeds (this includes
    QR50 being better).
  - **Warm refreshes improve the endpoint:** Δ ≥ +0.006 on both seeds.
  - **Otherwise unresolved.**
- **Thresholds.**
  - 0.003 is the production seed spread at 10k, the same null band as stage 2.
  - 0.006 is twice that. There is no 10k reference effect to scale from; stage 2's 0.007 was 2/3 of FP32's
    0.0106, which is comparable.
- **Context only (not part of the rule):**
  - at 2k, QR50 trails warm by +0.0047 on seed 204;
  - QR50 − BF16 QR10 at 10k, comparing exact deterministic CEs across launches.

**Decision rule B: storage or code path (FP32-storage QR10, seed 204, 2k screen).**
- **Inputs.** Let X be the FP32-storage QR10 CE at call 2000. All recipes share betas 0.95/0.95/0.95, LR 0.0015,
  eps 1e-8 and weight decay 0.05 (`stock_betas` and the `CompiledSOAP` defaults).

  | implementation | QR10, CE@2000 | warm10/QR50, CE@2000 | refresh effect D = QR10 − warm |
  |---|---|---|---|
  | FP32 eager (stock, `bf16_learning_20261026`) | 3.98591 | 3.99575 | −0.00984 |
  | compiled, FP32 storage | **X** | 3.99156 (`compiled_fp32`) | X − 3.99156 |
  | compiled, BF16 storage (this study) | 3.99809 | 3.99394 | +0.00415 |

- **Two recovery fractions** (two questions).
  - **f, the level:** how much of BF16 QR10's deficit against FP32 eager QR10 does FP32 storage recover?
    f = (3.99809 − X) / 0.01219.
  - **g, the cadence effect (§7.2's question):** how much of the lost refresh effect does FP32 storage restore?
    g = (0.00415 − (X − 3.99156)) / 0.01399.
- **Outcomes (both fractions must agree):**
  - **BF16 storage accounts for most of it:** f ≥ 2/3 and g ≥ 2/3. g binds, so this needs X ≤ 3.98638.
  - **The compiled path or BF16 compute accounts for most of it, not storage:** f ≤ 1/3 and g ≤ 1/3. f binds,
    so this needs X ≥ 3.99403.
  - **Otherwise partial or unresolved.** Both f and g are reported.
- **Secondary:** the same at call 1500. There QR10 is 4.10837 / 4.12314 (FP32 eager / BF16) and warm is
  4.12819 / 4.12808 (FP32 eager / compiled FP32 storage).
- **Caveats.**
  - This is a single-seed 2k screen. The production seed spread at call 2000 (0.0039) is about 1/3 of either
    gap, so only a clear outcome is informative.
  - Compute stays BF16 (`CompiledSOAP`'s compute dtype), so "storage" means storage only.
  - The second outcome would not separate the compiled path from BF16 compute.

**Time.** Shared machine: no step-time claims. Optimizer ms are reported for context only.

## 9. Stage 4: C6 front-loaded schedule, S = 300 and S = 1000 (protocol fixed before results)

**Status: approved by the user (12:03 UTC, 2026-11-20) to run after stage 3.** It is chained to start when the
stage-3 launcher exits. This protocol was written before approval and is unchanged since, apart from the cost and
S = 1000 clarifications added before the ask.

**Why.** In BF16 the warm/QR gap is an early-training effect: matched BF16 QR10 leads warm10/QR50 by 0.116 (seed
204) and 0.098 (seed 205) at call 500, and by 0.037 at 1000, then is level from about call 2000 to 10k (§7). C6
(panel PB7, `research/reviews/panel_20261120/CANDIDATES.md`) targets that window: exact refreshes early, then
production.

**Arm** (`FrontLoadedSOAP` in `bf16_soap_fastqr.py`; production `CompiledSOAP` with only the event choice changed).
- **Schedule:** Householder QR at every 10th age up to and including age S. After S, production warm10/QR50 on the
  global age grid: warm at ages ≡ 0 mod 10, QR at ages ≡ 0 mod 50, so the first post-switch refresh is a warm at
  age S + 10.
- **QR events:** the QR-phase events run production's own `qr` event function. So a run reproduces `bf16_hh_qr10`
  bit for bit until the first post-switch refresh, which gives a free gate.
- **Event counts over 10k calls:**

  | S | QR events | warm events | extra QR events vs production |
  |---|---|---|---|
  | 300 | 223 | 776 | +24 |
  | 1000 | 279 | 720 | +80 |

- **CPU check before freezing** (`logs/stage4_cpu_check.py` and `.log`):
  - with S = 30, the arm is bit-identical to `CompiledSOAP(10, 10)` until the post-switch warm, then follows
    warm10/QR50's events exactly;
  - the runner's per-call event assertion agrees with the optimizer on every call;
  - the PB8 hook leaves the trajectory bit-identical.
- **Unit checks** were rerun on CPU after `bf16_soap_fastqr.py` gained the new class
  (`logs/unit_cpu_rerun_after_frontloaded.*`).

**Runs.** Seed 204 only, 10,000 calls, one exclusive `gpu-claim` each (`logs/stage4_launch.sh`):
- `stage4_seed204_bf16_fl300_w10_qr50_c10000`
- `stage4_seed204_bf16_fl1000_w10_qr50_c10000`
- **Sources:** the same frozen sources as stages 2 and 3, except `bf16_soap_fastqr.py` (adds the new class only)
  and the runner (adds the arms and the PB8 log).
- **Cost:** about 2 GPU-hours, about 1 h wall on 2 GPUs.
- **Seed 205:** run only for the chosen S, and only after the user has seen these results.

**Comparators** (all exact and deterministic; no new baselines):
- the cached seed-204 BF16 warm10/QR50 production run, which is bit-equal to this study's fresh production arm
  through call 2000;
- the stage-2 seed-204 BF16 Householder QR10 run.

Reference values (CE):

| call | production | BF16 QR10 | gap (production − QR10) |
|---|---|---|---|
| 500 | 5.00151 | 4.88567 | 0.11584 |
| 1000 | 4.36160 | 4.32509 | 0.03651 |
| 2000 | 3.99394 | 3.99809 | −0.00415 |
| 10000 | 3.42239 | 3.42416 | −0.00177 |

**Gates (all must pass before any result is read).**
1. **Startup:** as in stages 2 and 3.
2. **QR-phase prefix:** per-call losses equal the stage-2 seed-204 `bf16_hh_qr10` run bit for bit through call
   S + 11, and so do all validation CEs at calls ≤ S + 11.
3. **Audits:**
   - event counts as in the table above;
   - no graph breaks; the compiled-graph count rises from 8 to 12 only when the warm function first compiles,
     at call S + 11;
   - a finite final state;
   - no PB8 error.

**Decision rule (per S, seed 204).**
- **Definitions:**
  - gap closure g(c) = (CE_prod(c) − CE_C6(c)) / (CE_prod(c) − CE_QR10(c));
  - Δ(c) = CE_C6(c) − CE_prod(c).
- **By construction, g = 1 at every validation call ≤ S + 11.** For S = 1000 that includes calls 500 and 1000, so
  its early test is automatic and its real tests come after the switch.
- **S = 300 is the informative arm.** S = 1000 passes unless it harms the endpoint. A pass means QR10's transient
  early lead at lower cost, not an endpoint gain: QR10's own lead on seed 204 is gone by call 1500.
- **Primary (the panel's threshold):** g(500) ≥ 0.5, with g(1000) reported.
- **Secondary, no endpoint harm:** Δ(10000) ≤ +0.003, the production seed spread at 10k. Δ(10000) ≤ −0.003 would be
  an endpoint gain, reported as single-seed.
- **Outcomes:**
  - **C6 passes at S:** primary and secondary both hold.
  - **Fails, no early gain:** g(500) < 0.5.
  - **Fails, endpoint harm:** g(500) ≥ 0.5 but Δ(10000) > +0.003.
- **The panel's call-2000 clause is reported, not gating.**
  - Its text: does not widen the call-2000 gap.
  - BF16 QR10 itself sits at Δ(2000) = +0.0042, against a production seed spread of 0.0039 at call 2000. A run that
    follows QR10 through call S + 11 can inherit that.
  - So Δ(2000) − Δ_QR10(2000) is reported, with Δ at 1500–5000.
- **Choosing S for seed 205** (a proposal; the user decides):
  - if both pass, the cheaper S = 300, together with the size of S = 1000's extra early gain (mean Δ over calls
    500–1500);
  - if one passes, that one;
  - if neither passes, no seed-205 run is proposed.
- **Cost** (measured optimizer ms per call over calls 101–10000):
  - **Call S + 11 is excluded from every mean and reported separately.** There the warm function compiles for the
    first time, inside the timed step. In stage 1 that first warm call took 9–25 s, which would add about 1–2.6 ms
    per call to a 9,900-call mean. Production compiles all of its events before call 101.
  - **Within-run estimate:** extra QR events × (this run's mean QR-event ms − mean warm-event ms) / 10,000. This
    is immune to the 26.1–26.6 ms drift of production's optimizer time across launches.
  - **Cross-launch comparison:** production's 26.4 ms (cached) and 26.6 ms (stage 1, fresh).
  - **Projections:** about +1.3–1.5 ms (S = 300) and +4.4–4.9 ms (S = 1000), from the panel and §3.
  - **PB8 overhead check:** the run's ordinary-event mean against about 10.4 ms in earlier runs, to show the PB8
    hook (outside the timed region) adds nothing.
  - No cross-launch step-time claims.

**PB8 settling diagnostic** (the panel's test, observation only).
- **What is measured:** the p99.9 of |M/(√V + ε)| over all matrix-state coordinates, at the start of a call
  (outside the timed region):
  - `pre`: just before each refresh, after 9 ordinary steps;
  - `post`: right after the refresh step;
  - `mid`: after the refresh and 4 ordinary steps.
- **Simplifications:** bias correction is omitted, and the new gradient is not yet mixed in.
- **Effect on training:** read-only, verified on CPU to leave the trajectory bit-identical. It costs about 3,000
  samples per run.
- **Covering the QR10 arm:** the S = 1000 arm is the QR10 arm through call 1011, so its QR phase gives QR10's
  settling directly. The late QR50 repairs give QR samples after the switch.
- **Statistic:** R = p99.9(post) / p99.9(pre) for the same refresh, with the median and 95th percentile per
  (refresh kind, phase).
- **Rule (the panel's):** if the median R ≤ 2 in every (kind, phase) cell, PB8 is dropped. Otherwise it is kept
  for follow-up.

## 10. Stage 3 results

Analysis: `benchmarks/fastqr_stage3.py`, writing `stage3.json` and `fastqr_stage3.{pdf,png}`. Evidence is copied to
`runs/stage3_*`.

### 10.1 BF16 QR50 to 10k (rule A): warm refreshes add no endpoint value resolvable at two seeds

**All gates passed.**
- **Startup:** both seeds.
- **Seed-204 control:** per-call losses for calls 1–2000 and all 8 validation CEs are bit-equal to the paper's
  cached `bf16_qr50` control, so that control is reproduced and extended to 10k.
- **First refresh:** on both seeds, per-call losses equal the cached production run through call 11, first
  differing at call 12.
- **Audits:**
  - events are 1 cold + 9,800 ordinary + 199 QR;
  - no graph breaks, and 8 compiled graphs at every capture;
  - finite final state and 8.22 GiB peak memory.
- **Wall time:** 55 min per run.

| seed | CE@2000 | 5000 | 7500 | 10000 | Δ@10k vs cached warm10/QR50 | late Δ (5000–10000) |
|---|---|---|---|---|---|---|
| 204 | 3.9987 | 3.6433 | 3.4930 | 3.4231 | **+0.0007** | +0.0009 |
| 205 | 3.9938 | 3.6421 | 3.4921 | 3.4221 | **+0.0028** | +0.0033 |

- **Rule A outcome.** Δ ≤ +0.003 on both seeds, so warm refreshes add no endpoint value resolvable at two seeds.
- **Seed 205 is at the edge.**
  - Its late-window mean (+0.0033) falls in "between".
  - QR50 is +0.0017 to +0.0035 above production at every validation point from call 2500.
  - So if warm refreshes help the 10k endpoint, the effect is about 0.003 or less, the size of the seed spread.
    Two seeds cannot resolve it.
- **All three BF16 cadences end together.** At 10k, QR50 − BF16 QR10 is −0.0011 (seed 204) and +0.0025 (seed
  205). On each seed, QR10, warm10/QR50 and QR50 all end within 0.003 of one another.
- **Cadence matters early.**
  - At call 500, QR50 trails warm10/QR50 by 0.017 (seed 204) and 0.009 (seed 205); BF16 QR10 leads it by 0.116
    and 0.098.
  - By call 2000 all three are within 0.005, and they stay there.
- **Cost (context only).**
  - QR50's optimizer time is 22.3 / 22.0 ms per call, against production's 26.4–26.6. So the warm refreshes cost
    about 4 ms per call (about 1.3% of a step), and Householder QR10 costs about +44 ms.
- **For the paper.**
  - **Supported:**
    - in the BF16 compiled implementation, the 10k endpoint is insensitive to refresh cadence across QR10,
      warm10/QR50 and QR50, to within about 0.003 on two seeds;
    - warm10/QR50's measurable value is early in training;
    - plain QR50 is cheaper and ends level.
  - **Not resolved at two seeds:** an endpoint effect of the warm refreshes of about 0.003 (seed 205).
  - **Scope:** this holds for BF16 storage. With FP32 storage, QR10 led warm10/QR50 by 0.0045 at call 2000
    (§10.2, indicative).

### 10.2 FP32-storage QR10 screen (rule B): a gate failed, so the result is indicative only

**The first-refresh gate failed.** The protocol required per-call losses to equal the cached `compiled_fp32` run
bit for bit through call 11; they first differ at call 4, by about 3e-5 in loss. The startup gate and the audits
passed.

**Diagnosis.** Several 12-call GPU replays (`logs/diag_fp32_gate*.log`, `logs/diag2.py`,
`logs/run_compiled_fp32.py`; bundle `/workspace/data/fastqr_20261120/diag_fp32_gate/`):

| replay | first differing call against … |
|---|---|
| runner's own `compiled_fp32` case, today | the cached 10-26 run: **12**, i.e. bit-equal through call 11. The reference is reproducible |
| my `fp32s_hh_qr10` arm, three launches (stage 3, diag 1, diag 2) | each other: **4**. The arm is **not run-to-run deterministic** on GPU |
| my arm with the runner's own parity routine, and the `compiled_fp32` schedule through my runner | each other: 12 (deterministic); pure `compiled_fp32`: 4 |

- **Measured.**
  - With my parity routine, the arm varies from launch to launch.
  - With the runner's own parity routine, it is deterministic, but it still differs from the pure `compiled_fp32`
    path at call 4, by about 1e-5 from the third update on.
- **Hypotheses (untested).**
  - *Run-to-run variation:* my parity routine does not free cached GPU memory (`gc.collect(); empty_cache()`), as
    the runner's own does, so compile-time allocations leave a launch-dependent memory layout.
  - *Systematic offset:* the install path, e.g. the extra eigensolver workspace of my spectral diagnostic at call
    1, gives a different fixed layout.
  - *Mechanism:* the FP32-storage numerics depend on memory layout, for example through alignment-dependent cuBLAS
    algorithm choice.
- **The configuration itself is not in question.** It is bit-identical on CPU (§8) and agrees to 1e-5 on GPU.
- **BF16-storage runs are unaffected.** Every BF16 arm in stages 1–4 has reproduced bit for bit wherever it was
  checked.

**Indicative result (not gate-qualified; one draw of a nondeterministic run).**

| call | X (FP32-storage QR10) | f (level) | g (cadence effect) | rule-B cell |
|---|---|---|---|---|
| 2000 (primary) | 3.98701 | 0.91 | 0.62 | partial (g < 2/3; "storage" needs X ≤ 3.98638) |
| 1500 (secondary) | 4.11090 | 0.83 | 0.87 | storage |

- **What this suggests.**
  - FP32 storage recovers most of BF16 QR10's level deficit against FP32 eager QR10. f is 1.08, 0.73, 0.83 and
    0.91 at calls 500, 1000, 1500 and 2000; the early calls have the least chaotic drift.
  - Inside compiled FP32 storage, QR10 − warm is −0.0045 at call 2000 and −0.0172 at 1500, against +0.0042 and
    +0.0008 in BF16 storage. So FP32 storage restores part of the QR10 advantage over warm; the primary cell is
    partial.
- **Why it is only indicative.**
  - The primary cell lands 0.0006 from the "storage" boundary.
  - The run's own nondeterminism grows chaotically to about the seed spread (0.004) by call 2000.
  - The gate failed.
  - So rule B is reported as **partial / not gate-qualified**.
- **Making it decisive.**
  - A gate-qualified run first needs a 12-call replay that matches `compiled_fp32` through call 11. Using the
    runner's own parity routine alone did not achieve that; the runner's own diagnostics would also have to be
    used, which is untested.
  - Alternatively, repeat the arm 2–3 times to measure the draw-to-draw spread, about 15 GPU-min per draw.
- **GPU use.** The two 12-call diagnostics used a few GPU-minutes under the stage-3 approval; no separate approval
  was sought.

## 11. Stage 4 results: C6 front-loaded schedule (seed 204, 10k)

Analysis: `benchmarks/fastqr_stage4.py`, writing `stage4.json`, `stage4_pb8_pairs.json` and
`fastqr_stage4.{pdf,png}`. Evidence is copied to `runs/stage4_*`.

**All gates passed (both S).**
- **Startup:** passed.
- **QR-phase prefix:** per-call losses equal the stage-2 QR10 run bit for bit through call S + 11 (first
  difference at calls 312 and 1012), and so do all validation CEs up to then.
- **Event counts:** 223 QR + 776 warm (S = 300) and 279 + 720 (S = 1000), exactly as scheduled.
- **Graphs:** no graph breaks; the compiled-graph count goes 8 → 12 at the switch, as expected.
- **Final state and PB8:** finite, and no PB8 error.

| call | production | BF16 QR10 | C6, S = 300 | C6, S = 1000 |
|---|---|---|---|---|
| 500 | 5.0015 | 4.8857 | 4.9386 (−0.063) | 4.8857 (−0.116) |
| 1000 | 4.3616 | 4.3251 | 4.3534 (−0.008) | 4.3251 (−0.037) |
| 1500 | 4.1224 | 4.1231 | 4.1349 (**+0.0125**) | 4.1462 (**+0.024**) |
| 2000 | 3.9939 | 3.9981 | 4.0047 (+0.011) | 4.0181 (+0.024) |
| 5000 | 3.6417 | 3.6447 | 3.6472 (+0.0055) | 3.6550 (+0.013) |
| 10000 | 3.4224 | 3.4242 | **3.4250 (+0.0026)** | **3.4305 (+0.0081)** |

(Values in parentheses are Δ against production.)

**Pre-registered outcomes.**
- **S = 300: passes, but only just, on both criteria.**
  - g(500) = 0.543 against the 0.5 threshold (g(1000) = 0.225).
  - Δ@10k = +0.0026 against +0.003.
- **S = 1000: fails on endpoint harm** (Δ@10k = +0.0081). Its early test was automatic by construction.
- **Proposal under the rule:** S = 300 for seed 205. S = 1000's extra early gain is a mean Δ 0.023 lower over
  calls 500–1500.

**What the data say beyond the rule: switching from QR10 to warm10/QR50 has a cost.**
- **Both arms fall behind production immediately after the switch.**
  - The training loss starts diverging from QR10's at the first post-switch warm refresh.
  - By call 1500, S = 1000 is 0.024 behind production and S = 300 is 0.0125 behind.
  - The penalty grows with the length of the QR phase.
- **It decays slowly.** At 10k it is still +0.008 (S = 1000) and +0.0026 (S = 300). BF16 QR10 and plain
  warm10/QR50 both end within 0.002 of each other.
- **So C6 trades early-training gain for a mid-training deficit.** S = 300 is ahead of production only up to
  about call 1000 and behind it from call 1500 to 10k. Its 10k endpoint sits at the edge of the tolerance.
  - The rule's pass therefore means "the early lead at about +1 ms per call, with no endpoint harm resolvable at
    one seed". It is not "a free early gain".
- **Not explained by basis quality or by post-refresh overshoot.**
  - The coordinate residuals r_C after the switch match QR10's and production's: at call 1501 the mean is 0.265
    for S = 1000, 0.270 for QR10 and 0.233 for production.
  - The PB8 settling statistic does not move at the switch (below). It tests only the overshoot side, where large
    |M/(√V+ε)| follows a refresh.
  - The panel's slow side, V staying too large in newly cleaned directions, shrinks the ratio and leaves the upper
    tail unchanged. It is untested here.
  - The mechanism is open. One candidate is the warm step's interaction with moment and V statistics that have
    adapted to QR's column re-sorting; it is untested here.

**Cost** (optimizer ms per call over calls 101–10000; call S + 11, the 25.5 s first compile of the warm function,
is excluded):

| | mean optimizer ms | within-run extra | vs production (cached 26.4 / stage-1 26.6) | projection |
|---|---|---|---|---|
| S = 300 | 27.09 | +1.30 | +0.7 / +0.5 | +1.3–1.5 |
| S = 1000 | 30.43 | +4.40 | +4.0 / +3.8 | +4.4–4.9 |

Ordinary events take 10.27 / 10.29 ms against about 10.4 in earlier runs, so the PB8 hook adds no measurable cost.

**PB8 settling diagnostic: dropped by the panel's rule.**
- **Statistic:** the median R = p99.9(post)/p99.9(pre) per cell (n, median, p95):

  | cell | S = 300 | S = 1000 |
  |---|---|---|
  | QR in the QR phase | 30, 0.998, 1.11 | 100, 1.005, 1.05 |
  | warm, late | 776, 0.995, 1.00 | 720, 0.995, 1.00 |
  | QR, late | 193, 1.09, 1.12 | 179, 1.08, 1.11 |

- **Outcome:** every median is ≤ 2, so PB8 is dropped under the panel's rule, for the overshoot side only (see
  above). The largest single ratio is 1.55, after the first QR at age 10.
- **The S = 1000 arm doubles as the QR10 arm** through call 1011. So QR10's own post-refresh settling is also
  within about 5%.
- **No step at the switch.** p99.9 at the switch is 0.453 → 0.464 (last QR) → 0.451 (first warm).

**Next step (needs the user's decision).** Seed 205 for S = 300, about 1 GPU-hour, would show:
- whether the post-switch penalty replicates (so far it is single-seed);
- whether the borderline pass holds.
