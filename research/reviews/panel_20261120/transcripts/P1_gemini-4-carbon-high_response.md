## Where the Time Actually Goes (§2 Diagnosis)

Before proposing changes, we decompose the 109M timings and synthetic scaling curves against the hardware ceilings of an RTX 5090 (Blackwell GB202: ~209.5 TFLOPS dense BF16 with FP32 accumulate due to the 2× GeForce FP32-accumulator throttle, ~104.8 TFLOPS FP32/TF32, 96 MB L2, ~1.79 TB/s GDDR7 **[verify][recalled]**).

1. **Factor cubic skew ($\sum n^3$).** Across the 96 factors ($60\times 768,\; 12\times 1536,\; 12\times 2304,\; 12\times 3072$):
   $$\sum_{f=1}^{96} n_f^3 = 60(768^3) + 12(1536^3) + 12(2304^3) + 12(3072^3) = (27.18 + 43.49 + 146.77 + 347.89)\times 10^9 = 565.33\times 10^9.$$
   The 24 left factors of fused QKV ($2304$) and fused SwiGLU-in ($3072$) carry **87.5% of all $O(n^3)$ FLOPs**, while the 60 factors at $n=768$ carry **4.8% of FLOPs** across 5 separate batch-12 calls.

2. **Amortized mean optimizer budget ($26.08\text{ ms/call}$):**
   $$26.08\text{ ms} = \underbrace{10.14\text{ ms}}_{\text{ordinary body (100\%): } 38.9\%} + \underbrace{0.08 \times 51.47\text{ ms}}_{\text{warm increment ($4.12\text{ ms}$): } 15.8\%} + \underbrace{0.02 \times 591.04\text{ ms}}_{\text{QR increment ($11.82\text{ ms}$): } 45.3\%}.$$

   - **QR repair ($593.74\text{ ms}$ for 96 factors) runs at 4.44 TFLOPS (~2% of GPU peak).** `qr_basis` performs $Z = CQ$ ($2n^3$) plus Householder `geqrf` + `orgqr` ($\frac{4}{3}n^3 + \frac{4}{3}n^3 = \frac{8}{3}n^3$ **[verify][certain]**), totaling $\frac{14}{3}\sum n^3 = 2.638\text{ TFLOPs}$. Dividing $2.638\text{ TFLOPs} / 0.59374\text{ s} = 4.44\text{ TFLOPS}$ ($0.83\text{ TFLOPS}$ at $n=768$, $7.47\text{ TFLOPS}$ at $n=3072$). Because PyTorch's batched `torch.linalg.qr` loops sequentially over the batch on one CUDA stream **[verify][certain]**, 96 low-occupancy panel factorizations serialize back-to-back. Notice that the five $n=768$ groups alone take $5 \times 30.8 = 154.0\text{ ms}$ (26% of QR time for 4.8% of FLOPs).
   - **Ordinary step ($10.14\text{ ms}$) is ~70% Tensor Core GEMMs and ~30% pack/state bandwidth.** Each matrix group ($b=12$) executes 6 GEMMs ($\bar{G}$: $2m^2n + 2mn^2$; $U$: $2m^2n + 2mn^2$; $C_L, C_R$: $2m^2n + 2mn^2$), totaling $72mn(m+n)$ per group:
     $$(0.391_{2304\times 768} + 0.065_{768\times 768} + 0.652_{3072\times 768} + 0.196_{768\times 1536}) = 1.305\text{ TFLOPs}.$$
     At 209.5 TFLOPS peak, ideal GEMM time is $6.23\text{ ms}$ (~$7.2\text{ ms}$ achieved at $k=768$). One-third of this ($0.435\text{ TFLOPs}$ + $962\text{ MB}$ R/W) is `covariance_ema`, computing both triangles of symmetric matrices on every step. Another ~$1.5\text{–}2.2\text{ ms}$ goes to `torch.stack(params)`, `torch.stack(grads)`, and 48 `copy_` unpacks (~$1.4\text{ GB}$ redundant traffic) plus moment updates, and ~$0.7\text{ ms}$ to auxiliary AdamW.
   - **Warm refresh ($49.84\text{ ms}$) has two distinct regimes.** Summing §2.2's `warm, b12` across the 109M counts reproduces §2.1 identically: $60(0.069) + 12(0.331) + 12(1.02) + 12(2.46) = 4.14 + 3.97 + 12.24 + 29.52 = 49.87\text{ ms}$. `warm_basis` runs 7 GEMMs ($14n^3$ FLOPs):
     - *Latency-bound ($n \le 1024$):* Both `b1` and `b12` take a flat **$0.80\text{–}0.84\text{ ms/call}$** (~15 sequential cuBLAS + Triton launches at ~$53\text{ }\mu\text{s}$ each, amplified by deterministic workspace resets).
     - *Compute + FP32-buffer bandwidth-bound ($n \ge 2304$):* At $n=6144, 8192$, `b1` achieves $3247\text{ GFLOPs}/15.2\text{ ms} = 213.6\text{ TFLOPS}$ and $7697\text{ GFLOPs}/37.2\text{ ms} = 206.9\text{ TFLOPS}$—100% of the RTX 5090's BF16-FP32acc ceiling. Yet `b12` regresses by **22–23%** ($37.2 \to 45.5\text{ ms}$ at $n=8192$) because 5 unfused `mm32` calls materialize $12\times 8192\times 8192$ FP32 buffers ($3.22\text{ GB}$ each) to DRAM instead of fusing BF16 epilogues, blowing out the 96 MB L2 cache.

---

## Question A: Making Refreshes and the Method Faster

### Part (A-eng): Engineering at Strictly Fixed Mathematics

#### Candidate A1 — Fused `baddbmm` Epilogues, Cross-Group Factor Coalescing, and Contiguous Param/Grad Banks

1. **Exact update:**
   - **In `warm_basis` (BF16 production path):** Pre-cache `neg_eye = -torch.eye(n, dtype=torch.bfloat16, device=dev).expand(b, n, n)`. Because `0.0` and `-1.0` are exact in BF16 and cuBLAS `baddbmm` on BF16 uses `CUBLAS_COMPUTE_32F` internally before rounding once to BF16 **[verify][certain]**, replace 4 of the 5 `mm32` + Triton-cast chains with single fused `baddbmm` calls:
     ```python
     # Exact replacement for y0 and the 2 Newton-Schulz iterations (basis_dtype == compute_dtype == bf16):
     y = torch.baddbmm(q, q, a.to(torch.bfloat16), beta=1.0, alpha=1.0)
     for _ in range(2):
         res = torch.baddbmm(neg_eye, y.transpose(-1, -2), y, beta=1.0, alpha=1.0)
         y = torch.baddbmm(y, y, res, beta=1.0, alpha=-0.5)
     ```
   - **In `qr_basis`:** Replace `c, old = cov.float(), q.float(); z = torch.bmm(c, old)` (TF32 GEMM) with `z = mm32(cov, q)` (BF16 Tensor Core GEMM with FP32 output). Since BF16's 7 mantissa bits embed losslessly into TF32's 10 mantissa bits **[verify][certain]**, the multiplicands are bit-identical while running at 2× Tensor Core rate.
   - **Cross-group factor coalescing:** Compile one optimizer-wide step across all 4 groups and stack the five $n=768$ factor batches ($5\times 12$) into one $b=60$ tensor for `warm_basis`.
   - **Contiguous parameter/grad banks:** Allocate one `(12, m, n)` FP32 master bank `P_bank` per shape group and bind each module parameter as `nn.Parameter(P_bank[i])` (with a `post_accumulate_grad_hook` writing into a contiguous `(12, m, n)` BF16 `G_bank`), eliminating `torch.stack(params)`, `torch.stack(grads)`, and `unbind`/`copy_`.

2. **Cost:**
   - Same GEMM count ($7$ BF16 $n\times n$ in `warm_basis`, $6$ BF16 $m\times n$ in `matrix_step`), except `qr_basis`'s $Z=CQ$ shifts from 1 FP32/TF32 GEMM to 1 BF16-in/FP32-out GEMM.
   - Kernel count in `warm_basis` drops from ~15 to **9** (7 cuBLAS GEMMs + 2 Triton kernels for `sym(b)`/skew `a`/norm).
   - DRAM savings in `warm_basis`: each of the 4 replaced `mm32` chains previously wrote $4n^2\text{ B}$ (FP32) and read $4n^2\text{ B} + 2n^2\text{ B}$ to write $2n^2\text{ B}$ ($12n^2\text{ B}$), vs $4n^2\text{ B}$ for `baddbmm`—saving **$32n^2\text{ bytes/factor}$** ($2.15\text{ GB/factor}$ or $25.8\text{ GB}$ at $b=12, n=8192$).
   - Contiguous banks save $2\times 283\text{ MB}$ (`stack`/`copy_` of `P`) + $425\text{ MB}$ (`stack`+cast of `G`) $\approx \mathbf{0.99\text{ GB}}$ every call.

3. **Expected effect:** *High confidence.*
   - Ordinary step: saves **$0.8\text{–}1.1\text{ ms/call}$** ($10.14 \to \sim 9.1\text{ ms}$).
   - Warm refresh at 109M: coalescing $5\times(b=12, n=768) \to (b=60, n=768)$ cuts $4.14\text{ ms} \to \sim 1.7\text{ ms}$, and fused `baddbmm` saves ~$2.5\text{ ms}$ across $n=1536\text{–}3072$ (total warm event: **$-5\text{ ms}$**).
   - Warm refresh at $n=6144\text{–}8192$ (`b12`): recovers most of the L2/DRAM regression vs `b1`, saving **$2.5\text{–}7.0\text{ ms/factor}$** at $n=6144\text{–}8192$.
   - `mm32` in `qr_basis`: saves ~$4\text{ ms}$ per 109M QR event and ~$3.5\text{ ms/factor}$ at $n=8192$.

4. **Failure modes:** `neg_eye.expand(b, n, n)` has batch stride `0`; in some cuBLAS versions `cublasGemmStridedBatchedEx` requires distinct output and `C` buffers when `strideC == 0`—allocate a contiguous `(b, n, n)` `neg_eye` if `out` aliases `C`, or let PyTorch allocate a fresh output tensor (default in `torch.baddbmm`). Weight-tying on embeddings is unaffected since embeddings are in auxiliary AdamW.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** None—arithmetic is mathematically identical (`CUBLAS_COMPUTE_32F` accumulation rounded once to BF16).

6. **Cheap test:** §2.6 (ii) Microbenchmark: verify bitwise (or $\le 1\text{ ULP}$) equivalence of `warm_basis` outputs and check that `warm, b12` at $n=8192$ drops below $40\text{ ms/factor}$ and $b=60, n=768$ takes $<2.0\text{ ms}$ total ($<0.033\text{ ms/factor}$). Reject if `b12` at $n=8192$ saves $<5\%$.

---

#### Candidate A2 — Batched Triton Symmetric-Tile (`syrk`) Kernels for `covariance_ema` and Newton–Schulz Grams

1. **Exact update:** Both $G_b G_b^T$ / $G_b^T G_b$ (every call) and $Y_j^T Y_j - I$ (twice per warm refresh) are symmetric Gram matrices, yet `bmm`/`baddbmm` compute all $n^2$ tiles. Replace both with a batched Triton kernel grid `(cdiv(n, B) * (cdiv(n, B) + 1) // 2, batch)` that computes only upper-triangular blocks `pid_m <= pid_n` ($B=128$) in FP32 Tensor Core `tl.dot`, applies `beta * old + (1 - beta) * acc` (or `acc - (row == col)`), casts to BF16, and writes both `(pid_m, pid_n)` and transposed `(pid_n, pid_m)` (similar to symmetric Gram kernels used in Muon **[verify][recalled]**).

2. **Cost:**
   - Tile fraction computed is $\frac{N_t(N_t + 1)}{2 N_t^2} = \frac{1}{2}\left(1 + \frac{B}{n}\right)$ where $N_t = \lceil n / 128 \rceil$:
     - $n=768$ ($N_t=6$): $21/36 = 58.3\%$ of GEMM FLOPs;
     - $n=2304$ ($N_t=18$): $171/324 = 52.8\%$;
     - $n=3072$ ($N_t=24$): $300/576 = 52.1\%$;
     - $n=8192$ ($N_t=64$): $2080/4096 = 50.8\%$.
   - `covariance_ema` FLOPs across the 109M model drop from $0.435\text{ TFLOPs}$ to **$0.233\text{ TFLOPs}$** ($-46.4\%$), and `old` BF16 reads drop by $46.4\%$ ($241\text{ MB}$ saved/call).
   - `warm_basis` effective GEMM equivalents drop from $7$ to **$6.04\text{–}6.17$** ($-13.7\%$ FLOPs at $n=3072$, $-14.1\%$ at $n=8192$).

3. **Expected effect:** *Medium-high confidence* (assuming well-tuned Triton warp-specialized persistent tiles on Blackwell):
   - Ordinary step: saves **$0.9\text{–}1.2\text{ ms/call}$** ($10\text{–}12\%$ of every ordinary call).
   - Warm refresh: saves **$3.5\text{–}5.5\text{ ms}$** at 109M and **$5\text{–}6\text{ ms/factor}$** at $n=8192$.

4. **Failure modes:** Naive Triton GEMM configs on consumer Blackwell (`sm_120`) without tuned `num_stages` / TMA descriptors can achieve 75–85% of cuBLAS TFLOPS, eroding half of the 2× tile advantage; autotune over `BLOCK_M/N in {64, 128}`, `BLOCK_K in {64, 128}`.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** Strictly *improves* invariants—guarantees exact bitwise symmetry $C = C^T$ in BF16 (whereas `baddbmm` can produce 1-ULP asymmetry between $(i,j)$ and $(j,i)$ when split-K reduction orders differ).

6. **Cheap test:** §2.6 (ii) Microbenchmark on `covariance_ema` and `warm_basis` across $n \in [768, 8192]$. Reject if the Triton `syrk` kernel is $<1.35\times$ faster than `torch.baddbmm` at $n \ge 1536$.

---

#### Candidate A3 — Multi-Stream Factor-Parallel `QRRepair` + Asynchronous Post-`W`-Write Overlap

1. **Exact update:**
   - **Multi-stream QR (within `QRRepair`):** Instead of calling `torch.linalg.qr` on $(12, n, n)$ on one stream, pre-create $S = 8$ CUDA streams (each with pre-allocated static workspace so the caching allocator never crosses streams) and assign the 96 individual $(n, n)$ `qr` calls by Longest-Processing-Time-First (LPT): each stream gets 1–2 factors of $n=3072$, 1–2 of $n=2304$, 1–2 of $n=1536$, and 7–8 of $n=768$, bracketed by `stream.wait_event(start_ev)` and `end_ev[s].record(stream)`.
   - **Post-`W`-write pipeline overlap (across the training step):** In §1.2, `W` is completely updated at Step 3 using pre-refresh $Q_L, Q_R$, and Steps 5–6 (`WarmRefresh` or `QRRepair`, `V` permute, `transport_moment`) touch **only** optimizer state (`Q, V, M`), which is untouched during Call $k+1$'s $283.4\text{ ms}$ forward+backward pass. Split the compiled step into `step_and_cov_graph` (Steps 1–4, returning immediately to start Call $k+1$'s forward pass) and `refresh_and_transport_graph` (Steps 5–6, launched on low-priority background stream(s) and synchronized via CUDA event right before Call $k+1$'s optimizer call).

2. **Cost:** Identical arithmetic and total FLOPs ($2.638\text{ TFLOPs}$ for 96 QRs; $7.915\text{ TFLOPs}$ for 96 warms), zero extra memory beyond static workspaces (~$250\text{ MB}$ across 8 streams at 109M).

3. **Expected effect:** *High confidence.*
   - **Multi-stream QR alone:** Single-factor QR at $n=3072$ uses $7.5\text{ TFLOPS}$ (~7% of FP32 peak); running 8 streams raises aggregate GPU utilization from $4.4\text{ TFLOPS}$ to ~$15\text{–}22\text{ TFLOPS}$, shrinking 109M QR wall time from **$593.7\text{ ms} \to \sim 130\text{–}190\text{ ms}$** (critical path bounded by $2\times 18.1 + 2\times 11.8 \approx 60\text{ ms}$ single-factor latency plus SM contention).
   - **With Fwd+Bwd overlap:** Hides ~$40\text{ ms}$ of the $51.5\text{ ms}$ warm event and ~$150\text{–}220\text{ ms}$ of QR behind Call $k+1$'s $283.4\text{ ms}$ Fwd+Bwd, dropping **mean exposed optimizer wall time from $26.08\text{ ms/call} \to \sim 11.5\text{–}13.5\text{ ms/call}$**.

4. **Failure modes:** In PyTorch's ATen cuSOLVER wrapper, `DeviceThreadHandlePool` shares one `cusolverDnHandle_t` per CPU thread per GPU, calling `cusolverDnSetStream` before each op **[verify][recalled]**; if `geqrf` allocates temporary workspace via the default allocator across streams without static workspace pools, allocator syncs can serialize streams. At $n=8192$, 8 concurrent FP32 QRs need $8 \times 268\text{ MB} \approx 2.15\text{ GB}$ scratch. Also, overlapping Tensor-Core-heavy `WarmRefresh` with Fwd+Bwd stretches Fwd+Bwd by its Tensor Core time (~$35\text{–}40\text{ ms}$), saving mainly launch/memory gaps (~$10\text{–}15\text{ ms}$), whereas low-occupancy `QRRepair` overlaps heavily with Fwd+Bwd GEMMs.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** None—bitwise identical even under `torch.use_deterministic_algorithms(True)`, because Fwd+Bwd and background refresh operate on strictly disjoint tensors with deterministic per-stream kernels.

6. **Cheap test:**
   - Multi-stream QR: directly testable in §2.6 (ii) Microbenchmark over the 96 factors of 109M. Reject if 8-stream wall time exceeds $280\text{ ms}$ (<2× speedup).
   - Post-`W`-write overlap: cannot be tested in §2.6 (ii) alone (no Fwd+Bwd); test with a 200-call training timing run verifying bitwise-identical loss to the synchronous baseline and whole-step mean $<297\text{ ms}$.

---

### Part (A-eq): Equation-Level Speedups Preserving Quality

#### Candidate A4 — Regularized Column-Normalized Shifted Cholesky-QR2/3 (`sCholeskyQR`) Replacing Householder QR

1. **Exact update:** Why did §2.2's naive Cholesky-QR fail at $n=6144$? Because $\lambda_i \propto 1/i \implies \kappa(Z^T Z) \approx n^2$; right at $n = 4096 = 2^{12}$, $u_{\text{FP32}} \kappa(Z^T Z) = 2^{-24} \cdot (2^{12})^2 = 1.0$, and at $n = 6144$, $2^{-24} \cdot 6144^2 = 2.25 > 1$, making $Z^T Z$ numerically indefinite.
   We cure both rank-deficiency and conditioning using three exact-eigenvector-preserving steps based on CholeskyQR2 / Shifted CholeskyQR3 **[verify][recalled]**:
   - Compute $Z = \mathcal{F}(C, Q)$ via `mm32(cov, q)`; Rayleigh quotients $r_j = \sum_i \text{float}(Q_{ij}) Z_{ij}$; sort $\pi = \operatorname{argsort}_{\text{desc}}(r)$; scale $s = \max(\max_j |r_j|, 10^{-30})$.
   - **Isotropic shift before QR:** $Z_\delta = Z + (\delta s)\,\text{float}(Q)$ with $\delta = 2\times 10^{-4}$ (for 2-pass) or $\delta = 10^{-5}$ (for 3-pass). Note $Z_\delta = (C + \delta s I)Q$: since $C$ and $C + \delta s I$ have **identical eigenvectors in identical order**, the target eigenbasis is unchanged, while $\kappa(Z_\delta) \le \kappa(Q)(1 + \delta^{-1})$ is strictly bounded even if $\operatorname{rank}(C) = 1$.
   - **Column normalization:** $Z_0 = Z_{\delta,\pi} D_0^{-1}$ where $(D_0)_{jj} = \|z_{\delta,\pi(j)}\|_2$. Since $D_0 \succ 0$ is diagonal (hence upper-triangular with positive diagonal), by QR uniqueness $\operatorname{qr}(Z_0).Q \equiv \operatorname{qr}(Z_{\delta,\pi}).Q$ **[verify][certain]**, and now $\operatorname{diag}(Z_0^T Z_0) = I$, $\|Z_0\|_F^2 = n$.
   - **2-pass (or 3-pass) Cholesky-QR in IEEE FP32** (disable TF32 on $Z_k^T Z_k$ so precision is $2^{-24}$, not $2^{-10}$):
     ```python
     # Pass 1 (shifted so chol never fails):
     g1 = ieee_fp32_bmm(z0.transpose(-1, -2), z0) + (11 * n * 6e-8) * eye
     r1, _ = torch.linalg.cholesky_ex(g1, check_errors=False)
     z1 = torch.linalg.solve_triangular(r1, z0, upper=False, left=False).transpose(...) # Z0 @ R1^{-T} if lower L1: Z1 = Z0 @ L1^{-T}
     # More cleanly with upper=True:
     u1, _ = torch.linalg.cholesky_ex(g1, upper=True, check_errors=False)
     z1 = torch.linalg.solve_triangular(u1, z0, upper=True, left=False)
     # Pass 2 (unshifted polish to O(u_FP32) << 0.0024 BF16 floor):
     g2 = ieee_fp32_bmm(z1.transpose(-1, -2), z1) + 1e-7 * eye
     u2, _ = torch.linalg.cholesky_ex(g2, upper=True, check_errors=False)
     q_new = torch.linalg.solve_triangular(u2, z1, upper=True, left=False).to(basis_dtype)
     ```
     *(Note: `solve_triangular(u, z, upper=True, left=False)` solves $X U = Z \implies X = Z U^{-1}$ with zero host sync via `cholesky_ex` **[verify][certain]**).*

2. **Cost:** Per factor (2-pass):
   - 1 BF16-in/FP32-out $n\times n$ GEMM ($Z = CQ$: $2n^3$ FLOPs);
   - 2 IEEE-FP32 $n\times n$ Gram GEMMs ($4n^3$ FLOPs, or $2n^3$ with `syrk`);
   - 2 FP32 batched Cholesky `potrf` ($2 \times \frac{1}{3}n^3 = \frac{2}{3}n^3$ FLOPs **[verify][certain]**);
   - 2 FP32 `trsm` solves ($2 \times n^3 = 2n^3$ FLOPs **[verify][certain]**).
   Total: **$8.67n^3$ FLOPs** (6.67$n^3$ in fast IEEE FP32 BLAS-3/Cholesky + 2$n^3$ BF16 TC), zero Householder panel loops.

3. **Expected effect:** *High confidence.* Scaling from §2.2's 1-pass `Cholesky-QR` ($72.0\text{ ms}$ across 96 factors):
   - 109M QR event drops from **$593.7\text{ ms} \to \sim 110\text{–}130\text{ ms}$** (**4.6–5.4× faster**, cutting whole-optimizer mean by **~$9.4\text{ ms/call}$ (36%)**!).
   - Per factor (`b12`): $n=768$: $2.56 \to \sim 0.28\text{ ms}$ (9×); $n=3072$: $18.1 \to \sim 4.8\text{ ms}$ (3.8×); $n=8192$: $112 \to \sim 68\text{ ms}$ (1.65×, **0 NaNs**).
   - Orthogonality after BF16 storage matches Householder QR ($0.0024$), and $r_C$ matches within $<0.002$.

4. **Failure modes:**
   - If TF32 is accidentally left enabled on `g1`/`g2`, input truncation at $2^{-10}$ limits conditioning to $\kappa(Z_0) \lesssim 30$; must enforce IEEE FP32 on those two GEMMs (e.g., a Triton IEEE-FP32 `tl.dot(..., input_precision="ieee")` `syrk` kernel inside `torch.compile`).
   - Within tail eigenspaces where $\lambda_i, \lambda_j \ll \delta s = 2\times 10^{-4} s$, power-step separation slows to $\frac{\lambda_i + \delta s}{\lambda_j + \delta s} \approx 1$; adding Pass 3 (`sCholeskyQR3`) lets $\delta$ drop to $10^{-6}$ at +45% pass cost (~$165\text{ ms}$ total, still 3.6× faster than QR).

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** Pass 2 reaches FP32 orthogonality ~$10^{-6}$, well below the $0.0024$ BF16 storage floor. With $\beta_C = 0.95$, $C$ can still have decayed tail modes ($0.95^{50} \approx 0.077$ per 50 idle steps on inactive features), which $\delta s$ stabilizes cleanly.

6. **Cheap test:** Test in **both** §2.6 (i) Open-loop replay (all widths 256–2048) and §2.6 (ii) Microbenchmark ($n = 256\text{–}8192$, checking $n=6144, 8192$ specifically). Reject if any NaN occurs, if normalized Gram error exceeds $0.0026$, or if replay $\rho_{\text{rms}}$ / top-$k$ overlap deviates from `QR10` by $>1\%$.

---

#### Candidate A5 — Single Newton–Schulz Correction in `WarmRefresh` (5 GEMMs vs 7)

1. **Exact update:** Change `for _ in range(2):` in `warm_basis` to `for _ in range(1):`.

2. **Cost & algebraic error bound:**
   - Drops `warm_basis` from **7 to 5** $n\times n$ BF16 GEMMs ($14n^3 \to 10n^3$, **$-28.6\%$ FLOPs**; 1 BF16-out + 4 FP32-out or `baddbmm`).
   - **Exact derivation:** For $Q^T Q = I + E_{\text{in}}$ ($\|E_{\text{in}}\|_F/\sqrt{n} \approx 0.0024$) and skew $\tilde{A} = aA$ ($\tilde{A}^T = -\tilde{A}$), first-order skew terms cancel in $Y_0 = Q(I + \tilde{A})$:
     $$E_0 = Y_0^T Y_0 - I = (I - \tilde{A})(I + E_{\text{in}})(I + \tilde{A}) - I = -\tilde{A}^2 + (I - \tilde{A})E_{\text{in}}(I + \tilde{A}).$$
     One cubic Björck–Bowie step $Y_1 = Y_0(I - \frac{1}{2}E_0)$ maps each eigenvalue $e_0$ of $E_0$ to **[verify][certain]**:
     $$e_1 = (1 + e_0)\left(1 - \frac{1}{2}e_0\right)^2 - 1 = -\frac{3}{4}e_0^2 + \frac{1}{4}e_0^3.$$
     Under the cap $\|\tilde{A}\|_F/\sqrt{2} \le 0.5$, the **worst-case** concentration puts all norm in a single 2D plane ($\sigma_1 = \sigma_2 = 0.5 \implies e_0 = 0.25$), giving $|e_1| = \frac{3}{4}(0.25)^2 - \frac{1}{4}(0.25)^3 = 0.04297$ on 2 modes, contributing $\frac{\sqrt{2}\times 0.04297}{\sqrt{n}} = \mathbf{0.00219}$ at $n=768$ (already below the $0.0024$ BF16 floor!). When norm spreads across $\ge 10$ pairs ($\sigma \le 0.158$), $|e_1| \le 0.00047$. Furthermore, the single NS step at the *next* warm refresh contracts any prior $e_{\text{in}} = 0.043$ quadratically to $0.0014$.

3. **Expected effect:** *High confidence.*
   - 109M warm refresh drops from **$49.84\text{ ms} \to \sim 35.8\text{ ms}$** ($-28\%$, saving ~$1.1\text{ ms/call}$ mean).
   - Large-$n$ regime (where warm/QR reaches $0.406$): saves **$1.6\text{ ms/factor}$ at $n=4096$**, **$5.3\text{ ms/factor}$ at $n=6144$**, and **$13.0\text{ ms/factor}$ at $n=8192$ (`b12`)**.

4. **Failure modes:** If paired with a Question B change that raises the spectral cap above $\|\tilde{A}\|_2 = 0.55$ ($e_0 > 0.30 \implies |e_1| > 0.06$), a coherent multi-mode rotation can leave noticeable non-orthogonality affecting transpose `transport_moment`; keep $\|\tilde{A}\|_2 \le 0.45$ when using 1 NS step (or use Candidate B2's order-2 seed).

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** Second NS iteration in the current code mostly polishes values already drowned by the final `R_b(Y_1)` BF16 rounding ($0.0018$); independent of $\beta_C$.

6. **Cheap test:** §2.6 (i) Replay harness + §2.6 (ii) Microbenchmark. Reject if normalized Gram error over 4 chained warm refreshes ever exceeds $0.0040$ or if replay $r_C$ / $\rho_{\text{rms}}$ degrades by $>0.002$.

---

#### Candidate A6 — Subsampled Covariance EMA (Every $s=2$ Calls with $\beta_{C,2} = \beta_C^2$) + Dimension-Aware `transport_moment`

1. **Exact update:**
   - **Subsampled `covariance_ema`:** Since $C_L, C_R$ are never read on the 9 calls between refreshes, update $C_L, C_R$ only on even ages ($t \bmod 2 == 0$) with matched-half-life decay $\beta_{C,2} = \beta_C^2 = 0.9025$ and $1 - \beta_{C,2} = 0.0975$ (compiling an `ordinary_nocov` graph for odd ages).
   - **Dimension-aware `transport_moment`:** For $M \in \mathbb{R}^{m\times n}$, contract the *smaller* side's relative rotation $T$ first instead of passing both sides through world coordinates:
     - If $m > n$ ($2304\times 768$, $3072\times 768$): $T_R = \mathcal{B}(Q_R^T, Q_R^+)$, then $M^+ = \mathcal{B}(\mathcal{B}((Q_L^+)^T, \mathcal{B}(Q_L, M)), T_R)$. Cost: $4m^2n + 2mn^2 + 2n^3$ vs $4m^2n + 4mn^2$, saving $2n^2(m - n) > 0$ FLOPs ($6n^3$ saved at $m=4n$).
     - If $n > m$ ($768\times 1536$): $T_L = \mathcal{B}((Q_L^+)^T, Q_L)$, then $M^+ = \mathcal{B}(\mathcal{B}(T_L, M), Q_R^T, Q_R^+)$, saving $2m^2(n - m) > 0$ FLOPs.

2. **Cost:**
   - On odd calls (50% of all calls): ordinary-step GEMMs drop from **6 to 4** per matrix ($1.305 \to 0.870\text{ TFLOPs}$, **$-33.3\%$**), and $962\text{ MB}$ of $C_L, C_R$ BF16 R/W is eliminated completely.
   - `transport_moment`: saves $7.5\%\text{–}8.3\%$ FLOPs on rectangular shapes and reduces BF16 rounding depth on the smaller axis from 2 matmuls to 1.

3. **Expected effect:** *High confidence on speed, medium-high on loss.* Saves ~$2.5\text{ ms}$ on every odd step (**~$1.25\text{ ms/call}$** off whole-optimizer mean, or ~12% of ordinary cost) plus ~$0.4\text{ ms}$ on transport.

4. **Failure modes:** Effective sample count of the EMA drops from $\frac{1+\beta_C}{1-\beta_C} = 39$ to $\frac{1+\beta_C^2}{1-\beta_C^2} = 19.5$ steps **[verify][certain]**; however, each gradient already aggregates $32\times 1024 = 32,768$ tokens and rank $\ge 768$ inner contractions, so subsampling $s=2$ (or $s=5$ in Distributed Shampoo/SOAP variants **[verify][recalled]**) rarely hurts unless batch size is tiny.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** *Beneficial!* In BF16 (half-ULP $u_{\text{BF16}} = 2^{-8} = 0.00391$ **[verify][certain]**), `0.95 * C + 0.05 * G G^T` stalls completely whenever $|(GG^T)_{ij} - C_{ij}| < \frac{2^{-8}}{0.05}|C_{ij}| = 7.81\%|C_{ij}|$ (for mantissas near $1.0^+$). Doubling $1 - \beta_C$ to $0.0975$ **shrinks the BF16 EMA stagnation deadzone by $1.95\times$** (down to $4.0\%$).

6. **Cheap test:** Speed in §2.6 (ii) Microbenchmark (`ordinary_nocov` vs `ordinary`). Quality **cannot** be tested in §2.6 (i) (snapshots are fixed every 10 ages); test with a 2,000-call training control (seed 204), rejecting if CE at call 500 or 2000 degrades by $>0.002$.

---

## Question B: Tracking the QR10 Trajectory for Better Loss

### Foundational Analysis of the Three Core Questions

#### 1. Why `WarmRefresh` currently misses QR10 (and hits $r_C = 0.975$ in training)
Let $B = Q^T C Q = D + E$ with descending Rayleigh quotients $d_1 \ge \dots \ge d_n > 0$ and $E = \operatorname{offdiag}(B)$.
- **Exact first-order trajectory of `QRRepair`:** Since $CQ = Q B = Q(I + E D^{-1})D$ and $D \succ 0$ is diagonal, $\operatorname{qr}(CQ).Q = Q \cdot \operatorname{qr}(I + E D^{-1}).Q$. Writing $I + E D^{-1} = (I + K_{\text{QR}})(I + R_{\text{upper}})$ with skew $K_{\text{QR}}^T = -K_{\text{QR}}$ and upper-triangular $R_{\text{upper}}$ (zero for $i > j$), matching strictly lower entries ($i > j$) gives:
  $$(K_{\text{QR}})_{ij} = +\frac{B_{ij}}{d_j} \quad (i > j), \qquad (K_{\text{QR}})_{ij} = -\frac{B_{ij}}{d_i} \quad (i < j).$$
  Thus for $i > j$ ($d_j \ge d_i$):
  $$(B_{\text{QR}}^+)_{ij} \approx B_{ij} + (d_i - d_j)(K_{\text{QR}})_{ij} = B_{ij} + (d_i - d_j)\frac{B_{ij}}{d_j} = \mathbf{\frac{d_i}{d_j}\, B_{ij}} + O(\|E\|^2).$$
  Furthermore, on **any** $2\times 2$ SPD block $B = \begin{pmatrix} d_1 & b \\ b & d_2 \end{pmatrix}$—**including equal diagonals $d_1 = d_2$**—$I + K_{\text{QR}} = \begin{pmatrix} 1 & -b/d_1 \\ b/d_1 & 1 \end{pmatrix}$ has exact polar factor $\frac{1}{\sqrt{d_1^2 + b^2}}\begin{pmatrix} d_1 & -b \\ b & d_1 \end{pmatrix} \equiv \operatorname{qr}(B).Q$ to **all orders**.

- **Three compounding throttles in current `WarmRefresh`:** Substituting $A_{ij} = -\frac{1}{2}\frac{B_{ij}}{s}\frac{h_{ij}}{h_{ij}^2 + 0.01^2}$ ($s = d_1,\; h_{ij} = \frac{d_i - d_j}{d_1}$) into $(DA - AD)_{ij} = (d_i - d_j)A_{ij}$ gives (with cap $a$):
  $$(B_{\text{warm}}^+)_{ij} \approx \left(1 - \frac{a}{2}\frac{h_{ij}^2}{h_{ij}^2 + 0.01^2}\right) B_{ij}.$$
  1. **Factor-of-$\frac{1}{2}$ under-relaxation:** Exact $2\times 2$ Jacobi satisfies $\tan(2\theta) = \frac{2B_{ij}}{d_i - d_j} \implies \theta \approx \frac{B_{ij}}{d_i - d_j}$ **[verify][certain]** (the $2$'s cancel!). The $-\frac{1}{2}$ halves the step even at $a = 1, h_{ij} = 1$.
  2. **Global normalization by $s = d_1$ freezes power-law tails:** Under §2.2's $\lambda_i \propto 1/i$, every pair with $i, j > 100$ ($87\%$ of coordinates at $n=768$; $98.8\%$ at $n=8192$) has $d_i, d_j < 0.01 d_1 \implies |h_{ij}| < 0.01$, suppressing motion by $\frac{1}{2}\frac{h_{ij}^2}{0.01^2} < 3\%$ even when $d_i / d_j = 2$, while Adam's $M/(\sqrt{V}+\epsilon)$ gives those tail coordinates equal step weight!
  3. **$\sqrt{n}$-dependent Frobenius cap:** In §2.2, $A_0$ has entries $\sim 0.2/\sqrt{n}$ ($r_C \approx 0.20$). Over the ~$100n$ pairs with $h_{ij} \gg 0.01$, $\|A\|_F/\sqrt{2} \approx \sqrt{100n \cdot (0.1/\sqrt{n})^2} \approx 1.0$, triggering $a \approx 0.5$. Combined with $-\frac{1}{2}$, max contraction is $1 - 0.25 = 0.75$, predicting $r_C^+ \approx 0.20 \times 0.75 = \mathbf{0.150}$—matching §2.2's measured $0.150$! Worse, for a full-rank rotation of angle $\theta$ across all $n$ coordinates, $\|A\|_2 \sim \theta$ (governing Taylor/NS validity) while $\|A\|_F/\sqrt{2} \sim \theta\sqrt{n/2}$, throttling per-coordinate rotation to $0.5/\sqrt{1536} = \mathbf{0.73^\circ}$ per refresh at $n=3072$.
  4. **Why trained max $r_C$ hits $0.965\text{–}0.975$:** When a sharp gradient spike aligns across $k \approx 20$ old basis vectors ($B \approx \frac{\lambda_1}{k}\mathbf{1}_k\mathbf{1}_k^T$), all $k$ coordinates get **nearly equal diagonals** $d_1 \approx \dots \approx d_k \approx \lambda_1/k$, giving $r_C = \sqrt{1 - 1/k} = \sqrt{19/20} = \mathbf{0.9747}$ and $h_{ij} \approx 0$—completely stalling `WarmRefresh` until QR repair.

#### 2. Is $r_C$ the right proxy for loss? What to measure instead
**No.** Because $\|Q^T C Q\|_F^2 = \sum_i \lambda_i^2$ is dominated by the top few eigenvalues in a power-law spectrum, $r_C$ ignores misalignment across $>95\%$ of coordinates that Adam scales up to unit step size.
To derive the *exact* metric governing the SOAP step, let $g = C^{1/2}z$ ($z \sim \mathcal{N}(0, I_n)$) and compare the approximate whitened step $Q D_Q^{-1/2} Q^T g$ ($D_Q = \operatorname{diag}(Q^T C Q)$) against ideal $C^{-1/2}g = U z$. Expanding $Q = U(I + A)$ ($A^T = -A$ small, $D_Q = \Lambda + O(A^2)$, $B_{ij} = (\lambda_j - \lambda_i)A_{ij} + O(A^2)$):
$$\left(U^T(Q D_Q^{-1/2} Q^T C^{1/2} - I)U\right)_{ij} = A_{ij}\left(1 - \sqrt{\frac{\lambda_j}{\lambda_i}}\right) + O(A^2).$$
Summing squares over $\{i, j\}$ ($A_{ji} = -A_{ij}$):
$$\Delta_{\text{pre}}^2(Q; C) := \frac{1}{n}\mathbb{E}_z\|Q D_Q^{-1/2} Q^T g - C^{-1/2}g\|_2^2 = \frac{1}{n}\sum_{i < j} |A_{ij}|^2 (\sqrt{\lambda_i} - \sqrt{\lambda_j})^2\left(\frac{1}{\lambda_i} + \frac{1}{\lambda_j}\right) + O(A^3)$$
$$= \frac{1}{n}\sum_{i < j} \frac{B_{ij}^2}{\lambda_i \lambda_j}\cdot \underbrace{\frac{\lambda_i + \lambda_j}{(\sqrt{\lambda_i} + \sqrt{\lambda_j})^2}}_{\in [1/2,\, 1]\;\forall \lambda_i,\lambda_j > 0} + O(A^3).$$
Up to a universal factor in $[1/\sqrt{2}, 1]$, **the exact whitened update error is the regularized RMS off-diagonal correlation $\rho_{\text{rms}}$** (already in your §2.6 i harness!):
$$\rho_{\text{rms}}(Q; C) = \sqrt{\frac{1}{n(n-1)}\sum_{i \neq j} \frac{B_{ij}^2}{(d_i + \epsilon_C)(d_j + \epsilon_C)}}, \qquad \epsilon_C \approx 10^{-12}\text{–}10^{-10}.$$
Because your target is specifically **QR10** (not `eigh`), supplement $\rho_{\text{rms}}$ in §2.6 (i) with **relative preconditioner distance to the replayed QR10 trajectory**: $\|Q(D_Q+\epsilon_C I)^{-1/2}Q^T(C+\epsilon_C I)^{1/2} - Q_{\text{QR10}}(D_{\text{QR10}}+\epsilon_C I)^{-1/2}Q_{\text{QR10}}^T(C+\epsilon_C I)^{1/2}\|_F / \sqrt{n}$.

#### 3. Role of the Call-500 Gap, $V$, and Column Order
- **Call-500 gap (+0.124 CE):** At $k=1$, $Q$ comes from `eigh` of a *single* random-init batch gradient $G_1$ (rank-limited and noisy). Over calls 1–500 (spanning the 200-call warmup right across peak LR $1.5\times 10^{-3}$), the true eigenbases undergo $O(1)$-radian rotations across all $n$ dimensions. Throttled to $0.73^\circ$ RMS by the $\sqrt{n}$ Frobenius cap and blind below $0.01 d_1$, `warm10/QR50` tracks like `QR50` ($5.0015$ vs $5.0180$), wasting peak-LR steps; under a fixed 10k cosine decay, early lost progress becomes a persistent effective-step lag (~$0.005\text{–}0.007\text{ nats}$) stacked atop ongoing tracking/transport loss.
- **$V$ and column order:** Leaving $V$ unpermuted during *infinitesimal* non-crossing rotations is benign, **except**: (i) over 40 steps ($\approx 3$ EMA half-lives) eigenvalues cross, and B1's triangular QR generator *requires* descending $d$ so $d_i/d_j \le 1$ for $i > j$; (ii) much worse, **rotating $M$ linearly while only permuting (or freezing) $V$ breaks Adam's moment bound!**
  *2×2 proof:* Suppose Mode 1 has $M_1 = 1, V_1 = 1$ and Mode 2 has $M_2 = 10^{-3}, V_2 = 10^{-6}$ ($|M_i/\sqrt{V_i}| = 1$). A rotation by $\theta = 0.2\text{ rad}$ ($\pi = \text{id}$) transports $M_2^+ = -\sin\theta\, M_1 + \cos\theta\, M_2 \approx -0.199$, while leaving $V_2^+ = 10^{-6}$, producing a **$199\times$ step spike** $M_2^+/\sqrt{V_2^+} \approx -199$ on the next step (especially severe right after 50-step `QRRepair` corrects a $r_C = 0.975$ spike!). Candidate B4 fixes this exactly.

---

### Question B Candidates ($\le 2\times$ Warm Cost at $n = 768\text{–}3072$)

#### Candidate B1 — Sorted Triangular Power-QR Skew Generator (`K_QR`) + Block-Power Spectral Cap

1. **Exact update:** Replace Jacobi's symmetric gap generator and Frobenius cap in `warm_basis` (and permute $V$'s corresponding axis by `order`, just as in `qr`):
   ```python
   def warm_qr_skew_basis(cov, q, v_0_fixed, compute_dtype, basis_dtype):
       qb, cb = q.to(compute_dtype), cov.to(compute_dtype)
       b = mm32(torch.bmm(qb.transpose(-1, -2), cb), qb)
       b = 0.5 * (b + b.transpose(-1, -2))
       d = b.diagonal(dim1=-2, dim2=-1)
       order = d.argsort(dim=-1, descending=True)
       # Permute Q, d, and rows & cols of B into descending Rayleigh order:
       qb = qb.gather(-1, order.unsqueeze(-2).expand_as(qb))
       d = d.gather(-1, order)
       b = b.gather(-2, order.unsqueeze(-1).expand_as(b)).gather(-1, order.unsqueeze(-2).expand_as(b))
       s = d.abs().amax(-1, keepdim=True).clamp_min(1e-30)
       d_safe = torch.hypot(d.clamp_min(0.0), 1e-4 * s)          # local Rayleigh denominator, soft floor 1e-4*s
       lower = (b / d_safe.unsqueeze(-2)).tril(-1).clamp(-1.0, 1.0)
       k_qr = lower - lower.transpose(-1, -2)                    # exact 1st-order QR skew generator
       # 2-step block power iteration on deterministic fixed (n, 8) probe for spectral norm ||k_qr||_2:
       w1 = torch.bmm(k_qr, torch.bmm(k_qr, v_0_fixed.expand(b.shape[0], -1, -1)))
       w1_n = w1 / torch.linalg.vector_norm(w1, dim=-2, keepdim=True).clamp_min(1e-20)
       w2 = torch.bmm(k_qr, torch.bmm(k_qr, w1_n))
       spec_est = torch.linalg.vector_norm(w2, dim=-2).amax(-1, keepdim=True).sqrt().unsqueeze(-1)
       a = k_qr * (0.45 / spec_est.clamp_min(1e-12)).clamp(max=1.0)   # keeps ||a||_2 <= ~0.50 < sqrt(2)
       y = torch.baddbmm(qb, qb, a.to(compute_dtype), beta=1.0, alpha=1.0)
       for _ in range(2):   # (or 1 step with A5)
           gram = mm32(y.transpose(-1, -2), y)
           res = (gram - torch.eye(y.shape[-1], device=y.device, dtype=gram.dtype)).to(compute_dtype)
           y = torch.baddbmm(y, y, res, beta=1.0, alpha=-0.5)
       return y.to(basis_dtype).contiguous(), order
   ```
   *(Optional Jacobi-boost multiplier on `lower` for $i > j$: setting $r_{ij} = \frac{(d_j - d_i)_+}{d_{\text{safe},j}}$ and multiplying `lower` by $\frac{r_{ij} + 0.25}{r_{ij}^2 + 0.25} \in [1, 1.62]$ gives exact QR gain $1$ at both $d_i = d_j$ and $d_i \ll d_j$, while accelerating $d_i/d_j \in [0.2, 0.8]$ toward exact Jacobi.)*

2. **Cost:** **7 BF16 $n\times n$ GEMMs** (identical to current `warm_basis`; **5 GEMMs** with A5) + 4 FP32 $n\times 8$ block matvecs ($64n^2$ FLOPs = **0.15% of 1 GEMM** at $n=3072$) + `argsort`/gathers on $Q, B, V$ (~$1.2\text{ ms}$ across all 96 factors). Total 109M warm cost: **~$1.03\times$ current warm** (~$51\text{ ms}$, or ~$37\text{ ms}$ with A1+A5).

3. **Expected effect:** *High confidence.*
   - On §2.2 synthetic inputs ($r_C \approx 0.20$), single-step $r_C$ drops from $0.150 \to \mathbf{\sim 0.035\text{–}0.050}$ (approaching QR's $0.025$), and $\rho_{\text{rms}}$ drops by $>3\times$ across the power-law tail.
   - Eliminates equal-diagonal stalling ($r_C = 0.975$ spikes break immediately) and removes the $O(1/\sqrt{n})$ early-training rotation bottleneck, closing an estimated **60–80% of the loss gap** to QR10.

4. **Failure modes:**
   - If the fixed $(n, 8)$ probe $V_0$ were orthogonal to a rank-2 coherent skew spike (probability negligible if $V_0$ uses 8 columns of a normalized Walsh–Hadamard / DCT matrix), `spec_est` could underestimate $\|K_{\text{QR}}\|_2$, causing Newton–Schulz overshoot if $\sigma_0(Y_0) = \sqrt{1 + \alpha^2} > \sqrt{3}$ ($\alpha > \sqrt{2}$). Safeguard: also clamp `spec_est` below by $\|K_{\text{QR}}\|_F / \sqrt{2k_{\text{max}}}$ or verify $\max_j (Y_0^T Y_0 - I)_{jj} = \max_j \|a_{:,j}\|_2^2$.
   - Gathering both rows and columns of FP32 `b` adds $8n^2\text{ B}$ memory traffic ($0.85\text{ ms}$ across 96 factors at 109M)—can be fused into the Triton kernel reading `b[order[i], order[j]]` directly without materializing permuted `b`.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** Because $B$ uses an intermediate BF16 `bmm(qb.T, cb)`, tail blocks below $10^{-4} d_1$ contain BF16 rounding noise from dominant modes; `hypot(d, 1e-4 * s)` and `clamp(-1, 1)` prevent noise amplification while reaching **100× deeper** into the spectrum than $0.01 s$.

6. **Cheap test:** §2.6 (i) Open-loop replay (all widths 256–2048) + §2.6 (ii) Microbenchmark, plus §2.5 Fact 3's $2\times 2$ equal-diagonal test $C = \begin{pmatrix}1 & 0.5\\0.5 & 1\end{pmatrix}$. Reject if replay $\rho_{\text{rms}}$ is not at least halfway from current `warm10/QR50` to `QR10`, or if latency exceeds $1.15\times$ current warm.

---

#### Candidate B2 — Local-Scale Full-Gain Jacobi + Order-2 Exponential Seed (`I + A + ½A²`) with 1 NS Step

1. **Exact update:** Keeps symmetric unsorted `WarmRefresh` ($V$ untouched), fixing all three scaling bugs and replacing Euler + 2 NS with an order-2 Taylor/Padé exponential seed + 1 NS step:
   $$s_{ij} = \max(|d_i|, |d_j|, 10^{-4} s), \qquad g_{ij} = \frac{d_i - d_j}{s_{ij}}, \qquad A_{ij} = -\frac{B_{ij}}{s_{ij}}\frac{g_{ij}}{g_{ij}^2 + 0.02^2},$$
   scale $\tilde{A} = R_b(a A)$ using B1's block-power spectral cap ($0.50 / \hat{\sigma}_{\max}(A)$), then:
   $$S = R_b\!\left(\tilde{A} + \frac{1}{2}\mathcal{F}(\tilde{A}, \tilde{A})\right), \qquad Y_0 = R_b(Q + \mathcal{F}(Q, S)), \qquad Y_1 = R_b\!\left(Y_0 - \frac{1}{2}\mathcal{F}(Y_0, R_b(\mathcal{F}(Y_0^T, Y_0) - I))\right).$$

2. **Cost & algebraic identity:**
   - **6 BF16 $n\times n$ GEMMs** ($Q^T C$, $(Q^T C)Q$, $\tilde{A}^2$, $Q S$, $Y_0^T Y_0$, $Y_0 E_0$ = $12n^3$ FLOPs—**14.3% cheaper** than current 7-GEMM warm!).
   - **Exact orthogonality identity:** Since $\tilde{A}^T = -\tilde{A} \implies (\tilde{A}^2)^T = \tilde{A}^2$:
     $$\left(I - \tilde{A} + \frac{1}{2}\tilde{A}^2\right)\left(I + \tilde{A} + \frac{1}{2}\tilde{A}^2\right) = \left(I + \frac{1}{2}\tilde{A}^2\right)^2 - \tilde{A}^2 = I + \frac{1}{4}\tilde{A}^4.$$
     The $O(\tilde{A}^2)$ Gram error cancels identically! At $\|\tilde{A}\|_2 = 0.5$, $\|Y_0^T Y_0 - I\|_2 \approx \frac{1}{4}(0.5)^4 = 0.0156$, which **one** NS step crushes to $\frac{3}{4}(0.0156)^2 = \mathbf{0.00018}$ (13× below BF16 rounding), while matching $\exp(\tilde{A})$ to $O(\tilde{A}^3)$.

3. **Expected effect:** *Medium-high confidence.* Faster than current warm (**~$43\text{ ms}$ vs $49.8\text{ ms}$** at 109M; $-14\%$ at $n=8192$), with quadratic $O(\|E\|^2)$ off-diagonal contraction on all pairs with $|d_i - d_j| / \max(d_i, d_j) > 0.02$ across the entire spectrum.

4. **Failure modes:** Still vanishes at exact equal diagonals ($d_i = d_j$, so it will **not** clear $r_C = 0.975$ coherent equal-diagonal spikes until the 50-step QR repair), and simultaneous full-gain Jacobi can over-rotate tightly coupled 3-cycles $(i,j,k)$ when $\|E\|$ is large ($r_C > 0.5$) unless capped at $\|\tilde{A}\|_2 \le 0.45$.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** Computing $\tilde{A}^2$ in BF16 (`mm32`) loses precision only when $\|\tilde{A}\|_2 < 10^{-4}$, where $\frac{1}{2}\tilde{A}^2 < 10^{-8}$ is already zero in BF16 anyway.

6. **Cheap test:** §2.6 (i) Replay + §2.6 (ii) Microbenchmark. Comparing **B1 vs B2** in replay directly isolates whether equal-diagonal symmetry breaking + sorting (B1) matters beyond fixing local scale + spectral cap (B2). Reject B2 if replay max-$r_C$ spikes remain $>0.85$.

---

#### Candidate B3 — Regularized Column-Scaled Cholesky-QR1 + 1-Step BF16 Newton–Schulz ("Warm CholQR-NS")

1. **Exact update:** Instead of approximating QR in $B$-space, directly execute one regularized power-QR step and remove single-pass Cholesky-QR's orthogonality drift ($0.034$ at $n=4096$) with 1 BF16 Newton–Schulz polish:
   ```python
   def warm_cholqr_ns(cov, q, compute_dtype, basis_dtype):
       z = mm32(cov, q)                                              # 1 BF16 GEMM -> FP32
       qf = q.float()
       r = (qf * z).sum(-2)
       order = r.argsort(dim=-1, descending=True)
       s = r.abs().amax(-1, keepdim=True).clamp_min(1e-30).unsqueeze(-2)
       z_reg = z + (5e-4 * s) * qf                                   # kappa(z_reg) <= 2001 at ALL n
       z_ord = z_reg.gather(-1, order.unsqueeze(-2).expand_as(z_reg))
       z0 = z_ord / torch.linalg.vector_norm(z_ord, dim=-2, keepdim=True).clamp_min(1e-30)
       g = torch.bmm(z0.transpose(-1, -2), z0) + 1e-4 * torch.eye(q.shape[-1], device=q.device)
       u, _ = torch.linalg.cholesky_ex(g, upper=True, check_errors=False)
       y1 = torch.linalg.solve_triangular(u, z0, upper=True, left=False).to(compute_dtype)
       # 1 BF16 Newton-Schulz polish (valid here because R^{-1} ALREADY broke polar symmetry!):
       res = (mm32(y1.transpose(-1, -2), y1) - torch.eye(q.shape[-1], device=q.device)).to(compute_dtype)
       q_new = torch.baddbmm(y1, y1, res, beta=1.0, alpha=-0.5)
       return q_new.to(basis_dtype).contiguous(), order
   ```
   *(Also permute $V$ by `order`, as in `qr`.)*

2. **Cost:**
   - **3 BF16 $n\times n$ GEMMs** ($Z = CQ$, $Y_1^T Y_1$, $Y_1 E_1$: $6n^3$ FLOPs);
   - **1 FP32/TF32 Gram GEMM** ($2n^3$) + **1 FP32 `cholesky`** ($\frac{1}{3}n^3$) + **1 FP32 `solve_triangular`** ($n^3$).
   - Total time from §2.2 primitives: §2.2's 1-pass `Cholesky-QR` took $72.0\text{ ms}$ across the 96 factors (with FP32 $Z=CQ$); using BF16 `mm32` for $Z=CQ$ (~$-7\text{ ms}$) + 1 BF16 NS step ($\frac{2}{7}\times 49.8 \approx +14\text{ ms}$) gives **~$79\text{ ms}$** across all 96 factors (**$1.58\times$ current warm**, well under $2\times$ at $n=768\text{–}3072$, and **$7.5\times$ faster than Householder QR**; at $n=8192$, ~$52\text{ ms}$ vs $45.5\text{ ms}$ warm = **$1.14\times$ warm** and $2.15\times$ faster than QR!).

3. **Expected effect:** *Very high confidence.*
   - Why §2.5 Fact 2 does not block the NS step: Fact 2 applies to $CQ = QB$ ($B$ symmetric). After upper-triangular $U^{-1}$, $Y_1 = Q_{\text{QR}}(I + E_{\text{orth}})$ already has the QR basis with $\|E_{\text{orth}}\|_2 \lesssim 0.03$; 1 NS step reduces orthogonality error to $\frac{3}{4}(0.03)^2 \approx 0.0007$ (stored as $0.0018$ in BF16), perturbing $Q_{\text{QR}}$ by only $O(\|E_{\text{orth}}\|^2) < 10^{-3}$.
   - Tracks `QR10` almost identically down to $\lambda_i \sim 5\times 10^{-4}\lambda_1$ ($r_C: 0.20 \to \sim 0.026$ in 1 step), closing **80–95% of the warm-vs-QR10 basis gap**.

4. **Failure modes:** Because `g` uses default `high` precision (TF32, $u \approx 2^{-10}$), setting ridge $<10^{-4}$ without IEEE FP32 on `g` can let `cholesky_ex` hit non-positive pivots on rank-deficient $C$; with ridge $5\times 10^{-4} s$ + diagonal shift $10^{-4}$ on unit-column `z0`, `g` is diagonally dominant over TF32 noise at all $n \le 8192$, at the mild cost of freezing modes below $5\times 10^{-4}\lambda_1$ until the 50-step full QR repair.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** Eliminates the $0.034$ Gram drift of raw Cholesky-QR (returning $0.0018$ BF16 Gram error) and keeps $V$ continuously sorted every 10 calls.

6. **Cheap test:** §2.6 (i) Replay + §2.6 (ii) Microbenchmark across $n = 256\text{–}8192$. Reject if `b12` cost exceeds $2.0\times$ warm on the 109M factor mix, if any NaN appears at $n=8192$, or if replay $\rho_{\text{rms}}$ differs from `QR10` by $>5\%$ on modes $\lambda_i > 10^{-3}\lambda_1$.

---

#### Candidate B4 — Doubly-Stochastic $V$-Transport $V \leftarrow (T_L^{\circ 2})V(T_R^{\circ 2})^T$ + Pre-Update Refresh Ordering

1. **Exact update:** On refresh calls ($t \bmod 10 == 0$, both `warm` and `qr`):
   1. Update $C_L, C_R$ from $G_b$ **first**, and compute $Q_L^+, Q_R^+$ (no separate $V$ gather needed—subsumed below!).
   2. Form relative transitions $T_L = \mathcal{B}((Q_L^+)^T, Q_L) \in \mathbb{R}^{m\times m}$ and $T_R = \mathcal{B}((Q_R^+)^T, Q_R) \in \mathbb{R}^{n\times n}$.
   3. Transport old $M$ and $V$ into the new basis *before* incorporating $\bar{G}_t$:
      $$M \leftarrow \mathcal{B}(\mathcal{B}(T_L, M), T_R^T), \qquad P_L = \frac{T_L^{\circ 2}}{\operatorname{sum}_{\text{col}}(T_L^{\circ 2})}, \quad P_R = \frac{T_R^{\circ 2}}{\operatorname{sum}_{\text{col}}(T_R^{\circ 2})}, \qquad V \leftarrow R_b(\mathcal{F}(\mathcal{B}(P_L, V), P_R^T)).$$
   4. Project $\bar{G}_t = \mathcal{B}(\mathcal{B}((Q_L^+)^T, G_b), Q_R^+)$ and run moment EMA + parameter step $U_t$ in the **fresh** basis $Q^+$.

2. **Cost & variance derivation:**
   - Forming $T_L, T_R$: $2m^3 + 2n^3$ BF16 FLOPs. Applying $T_L, T_R$ to $M$ takes $2m^2n + 2mn^2$ (saving $2m^2n + 2mn^2$ vs 4-matmul world transport!). Applying $P_L, P_R$ to $V$ takes $2m^2n + 2mn^2$ BF16 FLOPs.
   - Net extra FLOPs per refresh over current `transport_moment`: just **$2m^3 + 2n^3$** ($2\sum n^3 = 1.13\text{ TFLOPs}$ across 96 factors $\approx \mathbf{+6.5\text{ ms}}$ per refresh event = **$+0.65\text{ ms/call}$ amortized**, $+12\%$ of a warm event!). *(If $m \gg n$ on warm steps, setting $T_L \approx I - \tilde{A}_L$ avoids the $2m^3$ GEMM entirely!)*
   - **Exact variance justification:** Under $\bar{G}^+ = T_L \bar{G} T_R^T$ with uncorrelated old-frame coordinates $\mathbb{E}[\bar{G}_{ab}\bar{G}_{cd}] = V_{ab}\delta_{ac}\delta_{bd}$, $\mathbb{E}[(\bar{G}^+)_{ij}^2] = \sum_{a,b} (T_L)_{ia}^2 V_{ab} (T_R)_{jb}^2 = ((T_L^{\circ 2})V(T_R^{\circ 2})^T)_{ij}$. When $Q^+ = Q\Pi$ is a pure permutation, $T^{\circ 2} = \Pi^T$ reproduces exact $V$-permutation; when a top mode ($V_1$) mixes by $\sin\theta$ into a tail mode ($V_2 \ll V_1$), $V_2^+ \ge \sin^2\theta\, V_1$, eliminating the $O(|\sin\theta|\sqrt{V_1/V_2})$ update explosion.

3. **Expected effect:** *Medium-high confidence.* Eliminates post-QR / post-warm step-size spikes (especially at `t % 50 == 0` when QR corrects drifted warm bases) and upgrades 10% of all parameter steps from the 10-step-stalest basis to a 0-step-fresh basis, targeting ~$0.002\text{–}0.005\text{ nats}$ steady-state CE improvement. *(Note: incompatible with A3's post-`W` async overlap unless refresh uses $C_{t-1}$, which shifts $C$ by 1 step at $\beta_C=0.95$.)*

4. **Failure modes:** Repeated multiplication by $P_L, P_R$ acts as a slight diffusive smoothing across $V$ when $T \neq I$ ($O(\theta^2)$ mixing every 10 steps); with $\beta_2 = 0.95$, $V$'s memory decays by $0.95^{10} = 0.60$ between refreshes anyway, preventing long-term variance equalization unless $\theta$ is large every refresh.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** Row-normalizing $T^{\circ 2}$ in FP32 before casting $P_L, P_R$ to BF16 removes the $0.0024$ Gram drift of $Q$ so $\sum_a P_{ia} = 1$ to BF16 precision (preserving $\sum_{ij} V_{ij}$), and $(1-\beta_1)\bar{G}_t$ avoids 4 BF16 roundings of world transport.

6. **Cheap test:** Timing in §2.6 (ii) Microbenchmark; quality **cannot** be tested in §2.6 (i) (no $M, V$, or loss). Smallest test: 2,000-call training run (seed 204), logging maxcoordinate update $\|M_f/(\sqrt{V_f}+\epsilon)\|_\infty$ on steps $t \equiv 1 \pmod{50}$ and CE at 500/2000. Reject if CE improves by $<0.001$ and update spikes do not drop.

---

#### Candidate B5 — Dense Early Burn-In (`QR10` for $t \le 300$, then `Warm10/QR50`)

1. **Exact update:** Change the schedule line to:
   ```python
   event = 'cold' if not age else 'ordinary' if age % 10 else ('qr' if (age <= 300 or age % 50 == 0) else 'warm')
   ```

2. **Cost:** Replaces 24 warm events ($t \in \{10,20,30,40,\dots,290\}\setminus 50\mathbb{Z}$) with 24 QR events over a 10,000-call run:
   $$\Delta t_{\text{mean}} = \frac{24 \times (601.18 - 61.61)\text{ ms}}{10{,}000} = \mathbf{+1.30\text{ ms/call}} \quad (\text{+0.42\% whole-step time; +0.18 ms/call with A4}).$$
   Zero steady-state cost for $t > 300$.

3. **Expected effect:** *High confidence.* Completely eliminates the $+0.124$ call-500 gap during the non-stationary $G_1$-init $\to$ post-warmup transition ($t \le 300$ spans the entire 200-call LR warmup plus 7.4 covariance half-lives past peak LR!), recovering the ~$0.004\text{–}0.007\text{ nats}$ of irreversible cosine-schedule lag at virtually zero amortized cost.

4. **Failure modes:** Does not fix late-horizon spike stalls ($r_C = 0.975$ at age 9990) if used *without* B1 or B3; at very short horizons ($\le 500$ steps) its overhead is $60\%$ of QR10 rather than $2\%$.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** By $t = 300$, single-batch init bias has decayed by $0.95^{300} = 2\times 10^{-7}$, handing `WarmRefresh` a well-conditioned, full-rank, stationary eigenbasis.

6. **Cheap test:** Partially testable in §2.6 (i) Replay ($t \in [0, 1000]$, checking whether post-$300$ $\rho_{\text{rms}}$ converges to a lower steady-state attractor); definitively tested in a **500-to-2,000 call training run**, rejecting if call-2000 CE drops by $<0.003$ when applied to current `warm10/QR50`.

---

## Pitfalls in the Current Design and Measurements

Beyond the `WarmRefresh` throttling bugs, $r_C$ blindness, and $M$-vs-$V$ mismatch analyzed above, watch out for four subtle pitfalls:

1. **BF16 EMA Stagnation ("Swamping")—Not Underflow—in `M`, `V`, and `C`:**
   BF16 has FP32's 8-bit exponent (min normal $2^{-126} \approx 1.18\times 10^{-38}$ **[verify][certain]**), so $V \sim 10^{-16}$ does *not* underflow. Instead, BF16's 7-bit mantissa (half-ULP $u_{\text{BF16}} = 2^{-8} = 0.00391$ near $1.0^+$ **[verify][certain]**) creates a hard stagnation deadzone in any EMA $X \leftarrow R_b(\beta X + (1-\beta)Y)$: whenever $|Y - X| < \frac{2^{-8}}{1-\beta}|X|$, the update rounds to **exact zero change**. At $\beta = 0.95$, all drifts smaller than **$3.9\%\text{–}7.8\%$** stall in `M`, `V`, and `C`. Worse, if you ever test standard SOAP/Adam $\beta_2$ or $\beta_C \in \{0.99, 0.999\}$ **[verify][recalled]** with BF16 state (without stochastic rounding or FP32 master moments), the deadzone balloons to **$39\%\text{–}391\%$** and optimization breaks silently.

2. **Confounded Speed Baseline in §2.1 & Pending `BF16 QR10` Control in §2.3:**
   - §2.1 compares **compiled BF16** `warm10/QR50` ($26.08\text{ ms}$) against **eager FP32** `QR10` ($95.23\text{ ms}$). From your own §2.1 primitives, a **matched compiled BF16 `QR10`** costs $0.90(10.14) + 0.10(601.18) = \mathbf{69.25\text{ ms/call}}$ ($352.7\text{ ms}$ step)—so warm refreshes save $43.2\text{ ms/step}$ (12.2%), while $27.8\text{ ms/step}$ came from compiling + BF16 ordinary steps. Meanwhile, `BF16 QR50` takes $22.86\text{ ms/call}$ ($3.56\text{ ms/call}$ faster than `warm10/QR50`), and `BF16 QR20` ($40.46\text{ ms/call}$) currently *beats* `warm10/QR50` on loss for $+14\text{ ms/call}$.
   - Without `BF16 QR10` at 10k calls, part of the $+0.011$ gap (likely $0.002\text{–}0.004\text{ nats}$ from BF16 EMA stagnation + 4-matmul BF16 `M` transport) cannot be attributed cleanly to `WarmRefresh`.

3. **TF32 Precision Leak Inside "FP32 Islands" (`float32_matmul_precision = 'high'`):**
   Setting `"high"` lets every `torch.bmm` on `float32` tensors silently truncate inputs to **TF32 (10 mantissa bits, $u \approx 10^{-3}$)** **[verify][certain]**. While benign when inputs were just cast from 7-bit BF16 (`qr_basis`'s `c @ old`), it silently degrades any true FP32 Gram product like $Z^T Z$ in Cholesky-QR or `gram` in your optional FP32-basis Newton–Schulz island (`else: gram = torch.bmm(y.T, y)` in `warm_basis`, which claims to *"measure the actual FP32 basis Gram"* but actually measures a 10-bit TF32 Gram!).

4. **Open-Loop Replay (§2.6 i) and Synthetic Spectrum (§2.2) Optimism:**
   - §2.2 uses $\lambda_i \propto 1/i$ ($\kappa(C) = n \le 8192$, full-rank), whereas trained transformer factors exhibit steeper power laws ($\kappa(C) > 10^6$) and transient low-rank spikes ($r_C = 0.975$).
   - Open-loop replay feeds clean `FP32 QR10` covariances; in closed loop, a badly preconditioned step perturbs weights into sharper curvature directions, creating positive feedback between basis error and gradient spikes, while hiding $M/V$ transport shocks completely.

---

## Summary: Rankings, Top Pitfalls, and Next Measurements

### Top 5 Candidates for Question A (Speed)
1. **A4 (`sCholeskyQR` replacing Householder QR, A-eq):** Attacks the single largest slice of optimizer time (45.3%), cutting 109M QR events ~4.8× ($594 \to \sim 120\text{ ms}$, **$-9.4\text{ ms/call}$ mean**) with bounded conditioning at all $n$.
2. **A3 (Multi-stream QR + post-`W` async overlap, A-eng):** Pure bit-exact engineering that parallelizes low-occupancy QR across SMs and hides refresh/transport behind Call $k+1$'s $283\text{ ms}$ Fwd+Bwd (**$26.1 \to \sim 12\text{ ms/call}$ exposed**).
3. **A1 (Fused `baddbmm` epilogues + $b=60$ factor coalescing + contiguous param banks, A-eng):** Eliminates $32n^2\text{ B/factor}$ of FP32 DRAM traffic (fixing the $n=8192$ `b12` slowdown), halves $n=768$ launch overhead, and removes `stack`/`copy_` every step.
4. **A2 (Triton symmetric `syrk` for `covariance_ema` and NS Grams, A-eng):** Cuts one-third of ordinary-step FLOPs nearly in half ($-1\text{ ms}$ every call) and trims 14% of warm FLOPs while enforcing exact symmetry.
5. **A5 (1-step Newton–Schulz in `WarmRefresh`, A-eq):** Provably stays at or below the $0.0024$ BF16 rounding floor while immediately cutting warm FLOPs by **28.6%** across all $n$ (especially valuable at $n=4096\text{–}8192$).

### Top 5 Candidates for Question B (Tracking QR10 & Loss)
1. **B1 (Sorted Triangular Power-QR Skew Generator `K_QR` + Block-Power Spectral Cap):** Matches `QRRepair` to first order across all scales (and to *all* orders on $2\times 2$ equal diagonals) at **1.0× warm cost**, simultaneously fixing the $-\frac{1}{2}$ damping, $0.01 d_1$ tail blindness, $O(1/\sqrt{n})$ Frobenius cap, and $r_C=0.975$ stalls.
2. **B3 ("Warm CholQR-NS": Regularized Column-Scaled Cholesky-QR1 + 1 BF16 NS step):** Directly executes the power+QR step with $0.0018$ BF16 orthogonality and zero $n \ge 6144$ NaNs at **~1.58× warm cost** (7.5× cheaper than QR), offering the tightest possible match to QR10.
3. **B4 (Doubly-stochastic $V \leftarrow (T_L^{\circ 2})V(T_R^{\circ 2})^T$ + pre-update refresh):** Cures the $O(|\sin\theta|\sqrt{V_1/V_2})$ post-refresh step explosion when $M$ rotates without $V$, and uses fresh bases on refresh steps for $+0.65\text{ ms/call}$.
4. **B5 (`QR10` burn-in for $t \le 300$, then `warm10/QR50`):** Erases the $+0.124$ call-500 warmup gap right across peak LR for just **$+1.30\text{ ms/call}$** (+0.4% step time) over 10k calls.
5. **B2 (Local-scale full-gain Jacobi + `I + A + ½A²` seed + 1 NS step):** A **0.86× warm cost** pure-Jacobi baseline with 4th-order pre-NS orthogonality that cleanly isolates whether equal-diagonal symmetry breaking (B1) is needed beyond fixing local scaling and the spectral cap.

### Top 3 Pitfalls
1. **The three compounding `WarmRefresh` throttles + equal-diagonal stall:** $-\frac{1}{2}$ gain, global $s = d_1$ gap normalization (freezing $>87\%$ of a power-law spectrum that Adam weights equally), and $\sqrt{n}$-scaling of $\|A\|_F$ ($0.73^\circ$ RMS cap at $n=3072$), plus $h_{ij} \approx 0$ paralysis on $k$-way split spikes ($r_C \approx \sqrt{1 - 1/k} = 0.975$).
2. **Rotating $M$ without mixing $V$:** Linear $M$ transport paired with unrotated/permuted-only $V$ violates $|M/\sqrt{V}| \le 1$ by up to $|\sin\theta|\sqrt{V_{\max}/V_{\min}}$ (~$10^2\times$) on the step immediately following large QR repairs.
3. **BF16 EMA mantissa stagnation (and TF32 leakage into "FP32" Gram islands):** A $3.9\%\text{–}7.8\%$ deadzone on all state updates at $\beta = 0.95$ (catastrophic if $\beta \to 0.99+$), alongside `high` matmul precision silently running FP32 Gram checks in 10-bit TF32.

### Up to 5 Measurements Wanted Next
1. **Complete the 10k-call (and 2k-call) compiled `BF16 QR10` control** (plus an FP32-moments / BF16-factors ablation) to split the $+0.011\text{ nat}$ gap into BF16 EMA stagnation vs warm-refresh tracking error, and give the true compiled BF16 speed baseline ($69.25\text{ ms/call}$ predicted).
2. **Inspect the 96-factor distribution of trained $r_C$ and $\rho_{\text{rms}}$ by layer & matrix role** (specifically which factors hit $r_C = 0.965\text{–}0.975$, their eigenvalue spectra $\lambda_1/\lambda_n$, and how often the Frobenius cap $a < 1$ fires at ages $50, 200, 2000, 9990$).
3. **Log per-step max coordinate update $\|M_f/(\sqrt{V_f}+\epsilon)\|_\infty$ and cosine similarity of $U_t$ against an on-the-fly shadow `QR10` step** right before ($t \equiv 49 \pmod{50}$) and right after ($t \equiv 51 \pmod{50}$) QR repair, testing the $M/\sqrt{V}$ transport-spike hypothesis directly.
4. **Run B1, B2, B3, and A4 in the §2.6 (i) open-loop replay harness** reporting $\rho_{\text{rms}}$ (with floors $\epsilon_C \in \{10^{-8}, 10^{-12}\}$) and preconditioner distance to `QR10`, checking correlation between replay $\rho_{\text{rms}}$ and §2.3's known ordering (`QR10 < QR20 < warm10/QR50 < QR50`).
5. **Run a 2,000-call training ablation of B5 (`QR10` for $t \le 300$, then unchanged `warm10/QR50`)** on seed 204 to measure directly how much of the steady-state $+0.008\text{–}0.011$ plateau is locked-in early peak-LR lag vs ongoing refresh error.
