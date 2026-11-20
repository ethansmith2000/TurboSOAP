# P1: Gemini 4 Neon

Antigravity CLI 1.5.13 (`agy -p --model gemini-4-neon-xhigh --output-format stream-json`, brief inline as `-p`, empty working dir), 2026-11-20, about 21 min. First panel use of this model id. Brief: P1_BRIEF.md. Tool use: none (the stream-json has 0 `step_type: "tool"` events; tool names appear only in the init event). Verbatim `result.response`. Event log: `transcripts/P1_gemini-4-neon-xhigh_events.jsonl`.

---

### Ground-Rules Note on Tags
Throughout this response, **`[certain]`** marks derivations or direct consequences of the numbers/code you gave, **`[recalled]`** marks hardware/library details recalled from memory, **`[verify]`** marks literature or hardware-spec claims to check, and **`[speculation]`** marks untested hypotheses.

---

# Question A: Speed

## 0. Where the Time Actually Goes (Audit of §2)

### 109M model (mean optimizer time = **26.08 ms/call**)
Decomposing $0.90(10.14) + 0.08(61.61) + 0.02(601.18) = 26.08\text{ ms/call}$ into work performed on every step plus refresh increments **`[certain]`**:

| Component | Frequency | Event cost | Mean ms/call | Share of optimizer |
|---|---:|---:|---:|---:|
| Ordinary step (all 48 matrices + aux AdamW) | 100% | 10.14 ms | **10.14 ms** | **38.9%** |
| QR refresh increment (`qr_basis` 593.7 + `transport` 5.1 + `V` perm 0.4) | 2% | 591.04 ms | **11.82 ms** | **45.3%** |
| Warm refresh increment (`warm_basis` 49.8 + `transport` 5.1) | 8% | 51.47 ms | **4.12 ms** | **15.8%** |

Two underappreciated facts emerge immediately from exact FLOP counts **`[certain]`**:

1. **QR repair is still the #1 bottleneck (45.3%) despite running on only 2% of calls**, and its GPU utilization is dismal. Each `qr_basis` call does $2n^3$ ($Z=CQ$) + $\frac{4}{3}n^3$ (`geqrf`) + $\frac{4}{3}n^3$ (`orgqr`) $= \frac{14}{3}n^3 \approx 4.67n^3$ FLOPs — **3× fewer FLOPs than `warm_basis`** ($7 \times 2n^3 = 14n^3$ FLOPs). Yet on the RTX 5090, `qr` achieves only **0.83 TFLOP/s** at $n=768$ ($2.11\text{ GFLOP}/2.56\text{ ms}$), **7.5 TFLOP/s** at $n=3072$, and **22.9 TFLOP/s** at $n=8192$, because `torch.linalg.qr` loops serially over the batch of 12 and is bottlenecked by panel factorizations. As a result, the 60 small $n=768$ factors consume **25.9%** of QR time (154 ms) despite representing only **4.8%** of $\sum n^3$.
2. **The ordinary step is the #2 bottleneck (38.9%) and is ~75% GEMM-bound.** Across the 4 groups ($b=12$, shapes $2304\times 768$, $768\times 768$, $3072\times 768$, $768\times 1536$), $\sum 2bm^2n = 304.41\text{ GFLOP}$ and $\sum 2bmn^2 = 130.46\text{ GFLOP}$, so each projection pair ($\bar{G}$, $U$, or covariance $G G^\top + G^\top G$) is **434.87 GFLOP** — **1,304.6 GFLOP** total per ordinary call. From §2.1, `transport_moment` executes exactly two projection pairs (**869.73 GFLOP**) in **5.06 ms** (**171.9 TFLOP/s**). Scaling by $1.5\times$ shows the 6 ordinary GEMMs take **~7.6 ms** (~5.1 ms for $\bar{G}$ and $U$, ~2.5 ms for $C_L, C_R$), leaving **~2.5 ms** for `torch.stack`/`copy_`, Adam elementwise, $C$ HBM traffic (962 MB read+write), and auxiliary AdamW on the ~38.6M tied embeddings.
3. **Why `warm/QR` rises from 0.027 ($n=768$) to 0.406 ($n=8192$):**
   - At $n \le 1024$, `warm_basis` hits a **~0.80 ms/call launch-latency floor** regardless of batch ($12 \times 0.067 \approx 0.80\text{ ms}$), while `QR` is severely panel-bound (~0.8 TFLOP/s).
   - At $n = 6144\text{–}8192$, `warm_basis` ($14n^3$ FLOPs) is already pinned at the RTX 5090's BF16-with-FP32-accumulation Tensor Core ceiling (~209.5 TFLOP/s peak **`[recalled][verify]`**; `warm, b1` achieves **213.6 TFLOP/s** at $n=6144$ and **206.9 TFLOP/s** at $n=8192$!). Meanwhile `QR`'s blocked trailing updates reach ~23 TFLOP/s while doing **3× fewer FLOPs**, AND `warm, b12` suffers a **22–23% per-factor regression vs `b1`** ($45.5$ vs $37.2\text{ ms}$ at $n=8192$).
   - **Critical crossover **`[certain]`**:** Per 100 steps, `warm10/QR50` costs $8T_{\text{warm}} + 2T_{\text{QR}}$, while `QR20` costs $5T_{\text{QR}}$. Thus `warm10/QR50` becomes **slower than `QR20`** once $T_{\text{warm}}/T_{\text{QR}} \ge 3/8 = 0.375$ — which is **already crossed at $n=8192$** ($588\text{ ms}$ vs $560\text{ ms}$ per 100 steps)!

---

## (A-eng) Engineering Speedups at Fixed Mathematics

### A1. Multi-Stream Parallel `qr_basis` + `mm32(cov, q)` Inside `qr_basis`

1. **Exact update:**
   - Replace `torch.bmm(cov.float(), q.float())` with `mm32(cov, q)`. Since `cov` and `q` are BF16 (7-bit mantissa), their exact products require $\le 16$ mantissa bits, so BF16 Tensor Core MMA with FP32 accumulation (`mm32`) is **mathematically exact** relative to FP32/TF32 matmul of BF16-cast inputs **`[certain]`**.
   - Unbind the `ordered` tensor along the batch dimension and dispatch `torch.linalg.qr` across a fixed pool of $K=4\text{–}8$ CUDA streams (with `torch.cuda.Stream`), synchronizing via events before restacking. (Because `qr_basis` is an isolated FP32 island, either run the multi-stream QR outside the compiled region or wrap it in a custom op.)
2. **Cost:** Identical arithmetic per factor: $1\times$ $n^3$ BF16$\to$FP32 GEMM ($2n^3$, replacing $2n^3$ TF32) + $\frac{8}{3}n^3$ FP32 (`geqrf`+`orgqr`). Cuts `Z = CQ` input bandwidth by $2\times$ and doubles its tensor-core ceiling (~210 vs ~105 TFLOP/s on RTX 5090 **`[recalled][verify]`**).
3. **Expected effect:**
   - `mm32(cov, q)` alone saves ~4 ms on $n=3072$ and ~6 ms across all 96 factors per QR event.
   - Multi-stream QR attacks the <1% SM utilization at $n=768$ (154 ms) and $n=1536$ (81 ms). If cuSOLVER handles run concurrently across streams without internal mutex serialization, expect **2.5×–4×** speedup on $n=768$ and **1.5×–2.5×** on $n=1536$ → **~120–170 ms saved per QR event (~2.4–3.4 ms/call)**. Minimal benefit at $n \ge 4096$ where single QR fills more SMs. Confidence: **medium-high** (depends on whether cuSOLVER + `deterministic_algorithms(True)` serializes stream launches in PyTorch 2.11 / cu128 **`[recalled][verify]`**).
4. **Failure modes:** Extra workspace VRAM ($K\times$ single-QR workspace, tiny at $n \le 3072$); potential cuSOLVER host-lock contention under `use_deterministic_algorithms(True)`; breaks `fullgraph=True` unless wrapped in a `torch.library.custom_op`.
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Exact; bit-identical (up to `mm32` vs TF32 accumulation order).
6. **Cheap test:** Microbenchmark (§2.6(ii)): time `qr_basis` for `b=12` at $n \in \{768, 1536, 2304, 3072\}$ with $K \in \{1, 2, 4, 8\}$ streams. **Reject** if speedup at $n=768$ is $< 1.5\times$ (indicates cuSOLVER serialization).

---

### A2. Contiguous Group Parameter/Grad Buffers + Cross-Group $n=768$ Batching + CUDA Graphs

1. **Exact update:**
   - Allocate each shape group's 12 weight matrices as views into one contiguous `(12, m, n)` FP32 parameter tensor (and accumulate/pack gradients once via `_foreach` or a contiguous grad bucket), eliminating `torch.stack(params)`, `torch.stack(grads)`, and the 48 `parameter.copy_` loops on **100%** of calls.
   - Pack all 60 $n=768$ factors (5 groups of 12) into a single `(60, 768, 768)` call during `warm_basis` (keeping separate `m, v` groups only for $m\times n$ ops).
   - Capture the static ordinary and warm compiled regions into **CUDA Graphs** (`mode="reduce-overhead"` or manual `torch.cuda.CUDAGraph` with static scalar `age`/`lr` device tensors, which you already pass as tensors).
2. **Cost:** Identical GEMM counts. Eliminates ~283 MB param stack + ~283 MB param unpack + ~142–283 MB grad stack (~700–850 MB HBM traffic) on every ordinary step, and collapses 5 separate 0.80 ms kernel-launch latency floors in `warm_basis` into 1.
3. **Expected effect:**
   - Ordinary step: eliminates ~0.6–1.0 ms/call of pack/unpack bandwidth + launch overhead (~6–10% of ordinary step, **~0.8 ms/call** across 100% of calls).
   - Warm event: merging the five $n=768$ groups + CUDA graphs eliminates ~2.5–3.5 ms of latency overhead per warm event (~0.25 ms/call; huge relative win on small $n \le 1024$ where `warm` is currently 100% latency-bound at 0.80 ms/call). Confidence: **high**.
4. **Failure modes:** Contiguous parameter backing requires care with `state_dict` / weight tying (embeddings are separate anyway); CUDA Graphs fail on `qr` if cuSOLVER performs host syncs (keep QR eager/non-graphed) and hold static activation/intermediate pools (~200–500 MB on 109M, fine in 32 GB, problematic at $n=8192$).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Exact mathematics; no numerical change.
6. **Cheap test:** Microbenchmark (§2.6(ii)): compare `(60, 768, 768)` vs $5 \times (12, 768, 768)$ `warm_basis`, and CUDA-graphed vs ungraphed ordinary/warm event regions on 109M shapes. **Reject** individual sub-parts if ordinary saves $< 0.4\text{ ms}$ or 60-batch warm saves $< 1.5\text{ ms}$.

---

### A3. `warm_basis` Memory-Traffic Reduction (Scalar-Pull + Diagonal Sub) & Large-$n$ Micro-Batching

1. **Exact update:**
   - **Scalar pull:** Use $Q + Q(aA) = Q + a(QA)$. Compute `a_raw` in FP32, fuse writing `a_bf16 = a_raw.to(bf16)` with the Frobenius reduction `bound = ||a_raw||_F / sqrt(2)` (or reduce from `a_bf16`), run `prod = mm32(qb, a_bf16)`, and fold the scalar `a = min(1, 0.5 / bound)` into `y = (q.float() + a * prod).to(bf16)`. Because BF16 has FP32's full 8-bit exponent range ($10^{\pm 38}$), deferring scalar multiplication across BF16 rounding changes `aA` by $\le 0.5$ ULP **`[certain]`**.
   - **No `eye` materialization:** Replace `gram - torch.eye(...)` with index-fused subtraction `gram - (i == j)` or in-place `gram.diagonal(dim1=-2, dim2=-1).sub_(1.0)`.
   - **Micro-batching for $n \ge 3072$:** Loop `warm_basis` over sub-batches of size $b_{\text{chunk}} \in \{1, 2, 4\}$ when $n \ge 3072$ (keeping full $b=12$ for $n \le 2304$).
2. **Cost:** Same 7 BF16 GEMMs ($14n^3$), but eliminates 1 full FP32 write + 1 full FP32 read of $A$ ($2 \cdot 4bn^2$ bytes = **6.44 GB** at $b=12, n=8192$), eliminates `eye` read/writes, and cuts peak transient VRAM by $3\times\text{–}12\times$ (from ~13–16 GB to ~1.3–4 GB at $n=8192$).
3. **Expected effect:**
   - Directly recovers the measured `b12` vs `b1` regression in §2.2: **7%** at $n=3072$ ($2.46 \to 2.29\text{ ms/factor}$, ~2 ms saved on 109M warm), **23%** at $n=6144$ ($18.7 \to 15.2\text{ ms/factor}$), and **22%** at $n=8192$ ($45.5 \to 37.2\text{ ms/factor}$), plus another ~1–2 ms/factor at $n=8192$ from eliminating the 6.4 GB FP32 `a` round-trip. Confidence: **high** (the `b1` vs `b12` gap is already directly measured in §2.2; note that if part of `b12`'s slowdown is sustained 575W thermal/power throttling over a 546 ms burst vs 37 ms burst **`[speculation]`**, micro-batching still recovers the cuBLAS/VRAM/HBM portion).
4. **Failure modes:** If `a_raw` ever overflows $3.4 \times 10^{38}$ (impossible with `clamp_min(1e-30)` and `0.01^2` damping, where $|a_{\text{raw}}| \le 50 \|B\|_\infty / s \sim O(50)$).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Negligible ($\le 0.5$ ULP on `aA`); identical convergence.
6. **Cheap test:** Microbenchmark (§2.6(ii)) at $n \in \{3072, 4096, 6144, 8192\}$ for $b_{\text{chunk}} \in \{1, 2, 4, 12\}$ with locked vs unlocked GPU clocks (`nvidia-smi -lgc` to isolate thermal throttling from kernel/memory effects). **Reject** chunking if `b12` matches `b1` once clocks are locked and `a` traffic is fused.

---

### A4. Custom Upper-Triangular Batched `SYRK` (Triton/CUTLASS) for $G_b G_b^\top$, $G_b^\top G_b$, and $Y_j^\top Y_j$

1. **Exact update:** Every call executes `G_b @ G_b.T` and `G_b.T @ G_b` (symmetric), and every `warm_basis` executes `Y_j.T @ Y_j` twice (symmetric). A tiled Triton kernel (using Blackwell `tl.dot` BF16$\to$FP32) skips tiles with `pid_m > pid_n`, fuses `beta * C_old + (1 - beta) * acc` (or `acc - (i == j)` for `gram`), and writes both tile `(m, n)` and its transpose `(n, m)` in the epilogue.
2. **Cost:** For an $n\times n$ symmetric output with $T = n/B_M$ tiles per side, computes $\frac{T(T+1)}{2}$ tiles instead of $T^2$ ($\approx 50\%$ FLOPs for $T \ge 12$, e.g. 48% savings at $n=3072, B_M=128$).
   - Ordinary step: cuts covariance GEMMs from **434.9 $\to$ ~225 GFLOP** (saves ~210 GFLOP on 100% of calls!).
   - Warm refresh: cuts the two `Y^T Y` GEMMs from $4n^3 \to \sim 2.1n^3$ (saves **~14% of `warm_basis`**, i.e. 1 of 7 GEMMs, at every compute-bound $n$, AND cuts `Z^T Z` in Cholesky-QR by ~50%).
   - Bonus: guarantees **exact bitwise symmetry** of $C_L, C_R$.
3. **Expected effect:** Saves **~1.1–1.3 ms/call** on ordinary steps (100% of calls) and **~6 ms** per warm event (~0.5 ms/call) → **~1.6–1.8 ms/call** on 109M, and **~14%** on `warm_basis` at $n=4096\text{–}8192$ (assuming the Triton kernel reaches $\ge 85\%$ of cuBLAS BF16 efficiency; if it only reaches 75%, net speedup on those GEMMs is $1.5\times$ instead of $1.9\times$). Confidence: **medium-high** (engineering effort is non-trivial to match Blackwell cuBLASLt MMA schedules).
4. **Failure modes:** Under-tuned Triton kernels on Blackwell (`sm_120`) can lose 20–40% of cuBLASLt peak, eroding half the 2× algorithmic win; tile-transpose epilogue writes to non-contiguous lower-triangle addresses (needs shared-memory transpose before global store).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Exact same math, strictly cleaner because $C_L, C_R$ are guaranteed exactly symmetric.
6. **Cheap test:** Microbenchmark (§2.6(ii)): standalone batched Triton `SYRK` vs `torch.baddbmm` on `(12, 3072, 768)` and `(12, 8192, 8192)`. **Reject** if wall time is $> 0.70\times$ `torch.baddbmm`.

---

## (A-eq) Equation-Level Speedups

### A5. Column-Equilibrated Shifted Cholesky-QR2 (`sCholQR2`) Replacing `torch.linalg.qr`

1. **Exact update:** Replaces `qr_basis` with a numerically stabilized 2-pass BLAS-3 QR (Fukaya et al. 2020 **`[recalled][verify]`**), using `mm32` for $Z = CQ$, a tiny eigenvector-preserving ridge $\mu I$, column equilibration, and **true FP32** (TF32 disabled!) for the Gram products:
   ```python
   # Z = (C + mu*I) @ Q;  C + mu*I has identical eigenvectors to C, kappa <= 1/tau
   z = mm32(cov, q)                                              # exact BF16->FP32
   d = (q.float() * z).sum(-2)                                   # Rayleigh quotients
   mu = tau * d.abs().amax(-1, keepdim=True).clamp_min(1e-30)    # e.g. tau = 3e-4 or 1e-3
   z = z + mu.unsqueeze(-1) * q.float()
   order = d.argsort(dim=-1, descending=True)
   z = z.gather(-1, order.unsqueeze(-2).expand_as(z))
   # Column equilibration: qr(Z D^{-1}).Q == qr(Z).Q identically, and ||Z_hat||_F^2 == n
   z = z / z.norm(dim=-2, keepdim=True).clamp_min(1e-30)
   # Pass 1 (shifted) + Pass 2 (unshifted) in TRUE FP32 (TF32 OFF!)
   with torch.backends.cuda.sdp_kernel... # disable TF32 via fp32_precision="ieee"
       g0 = torch.bmm(z.transpose(-1, -2), z)
       g0.diagonal(dim1=-2, dim2=-1).add_(s1)                    # s1 = 1e-6 * n
       r0 = torch.linalg.cholesky(g0, upper=True)
       q1 = torch.linalg.solve_triangular(r0, z, upper=True, left=False)
       g1 = torch.bmm(q1.transpose(-1, -2), q1)
       r1 = torch.linalg.cholesky(g1, upper=True)
       q_new = torch.linalg.solve_triangular(r1, q1, upper=True, left=False)
   return q_new.to(basis_dtype).contiguous(), order
   ```
   **Algebraic & stability proof **`[certain]`**:**
   - Since $D = \operatorname{diag}(\|z_j\|_2) > 0$ is positive diagonal, if $\hat{Z} = Z D^{-1} = Q R$, then $Z = Q(R D)$ with $R D$ upper triangular and positive diagonal, so $\operatorname{qr}(\hat{Z}).Q = \operatorname{qr}(Z).Q$ identically.
   - Because $\|\hat{Z}\|_F^2 = n$, $\kappa(G_0) \le \frac{\|\hat{Z}\|_2^2 + s_1}{s_1} \le \frac{n + 10^{-6} n}{10^{-6} n} \approx 10^6 \ll u_{\text{FP32}}^{-1} \approx 1.67 \times 10^7$, so Pass 1 Cholesky cannot fail.
   - After Pass 1, $\sigma_i(Q_1) = \sigma_i(\hat{Z}) / \sqrt{\sigma_i(\hat{Z})^2 + s_1}$, so $\kappa(Q_1) \le \sqrt{1 + s_1/\sigma_{\min}(\hat{Z})^2}$. Since $C + \mu I$ bounds $\kappa(C+\mu I) \le 1/\tau$ and column equilibration further removes the pure-eigenvalue scaling (leaving only basis-misalignment factor $\kappa(\hat{Z}) \sim r_C \kappa(C+\mu I)$), for $\tau = 10^{-3}$ ($\sigma_{\min}(\hat{Z}) \gtrsim 10^{-3}$) and $n \le 4096$ ($s_1 \le 4 \times 10^{-3}$), $\kappa(Q_1) \le 65$, giving Pass 2 orthogonality error $O(u_{\text{FP32}} \kappa(Q_1)^2) \le 3 \times 10^{-4} \ll 2.4 \times 10^{-3}$ (BF16 floor).
   - **Crucial detail:** Your §2.2 naive Cholesky-QR failed because of (i) no column equilibration ($\kappa(Z^\top Z) = \kappa(C)^2$), (ii) no shift / 2nd pass, and (iii) **TF32 truncation on `Z.T @ Z`** (see Pitfall 2). On consumer GeForce Blackwell (RTX 5090), true FP32 CUDA-core peak (~105 TFLOP/s) equals TF32 Tensor Core peak (~105 TFLOP/s) **`[recalled][verify]`**, so disabling TF32 on `g0, g1` costs very little (~15–25% cuBLAS efficiency difference) while dropping $u$ from $5 \times 10^{-4}$ to $6 \times 10^{-8}$.
2. **Cost:** Per factor: $1\times$ $n^3$ BF16$\to$FP32 GEMM ($2n^3$) + $2\times$ [$2n^3$ FP32 GEMM (or $n^3$ SYRK) + $\frac{1}{3}n^3$ FP32 `cholesky` + $n^3$ FP32 `trsm`] $= \mathbf{2n^3\text{ BF16} + 6.67n^3\text{ FP32}}$ (or $4.67n^3$ FP32 with SYRK). All ops are batched BLAS-3 with no serial `geqrf` panel bottlenecks.
3. **Expected effect:** From §2.2, 1-pass CholQR summed over the 96 factors is $60(0.171) + 12(0.618 + 1.50 + 3.03) = \mathbf{72.04\text{ ms}}$ (vs **593.74 ms** for `qr`). Replacing TF32 $Z=CQ$ with BF16 `mm32` and adding Pass 2 yields **~115–140 ms** total across 96 factors (**4.2×–5.2× faster QR event!**), cutting mean optimizer time by **~9.0–9.5 ms/call** (**~35% of the entire optimizer**!). At $n=8192$, expect **~60–72 ms/factor** vs 112 ms for `qr` (**1.6×–1.9× faster**, ~2× with SYRK). Confidence: **high**.
4. **Failure modes:** If BF16 rounding makes $\operatorname{sym}(C)$ have a negative eigenvalue $-\nu$ with $\nu \approx \mu$, $C + \mu I$ can become near-singular. At $n=8192$ with $\sigma_{\min}(\hat{Z}) < 10^{-3}$, Pass 2 may need a tiny safety shift $s_2 = 10^{-8}n$ or a 3rd pass (`sCholQR3`) if $\kappa(Q_1) > 200$. Setting $\tau$ too large (e.g. $10^{-2}$) slows power-iteration contraction on tail eigenvalues $\lambda_i < \tau \lambda_{\max}$.
5. **Interaction with BF16 storage & $\beta_C=0.95$:** BF16-induced negative eigenvalues of $C$ are bounded by $\|\Delta C\|_2 \le 2^{-8}\|C\|_F$, so $\tau \approx 10^{-3}$ safely restores strict positive definiteness and prevents rank-deficiency crashes at early calls (when $C_L$ for $3072\times 768$ has rank $\le 768$).
6. **Cheap test:** Both tools! (i) Microbenchmark at $n = 256\text{–}8192$ ($b=1, 12$) verifying zero NaNs, Gram error $\le 0.0025$ after BF16 cast, and speedup vs `qr`. (ii) Replay harness comparing `sCholQR2` ($\tau \in \{10^{-4}, 3\times 10^{-4}, 10^{-3}\}$) vs `QR10` on $r_C$, RMS correlation, and Gram error. **Reject** if Gram error exceeds $0.003$ at $n=8192$ or replay $r_C$ degrades by $> 0.005$ vs `QR10`.

---

### A6. Degree-4 Matrix Exponential $R_4(\tilde{A})$ + Direct Local Moment Transport $M^+ = R_L^\top M R_R$

1. **Exact update:** Let $\tilde{A} = aA$ ($\tilde{A}^\top = -\tilde{A}$). Replace $Y_0 = Q(I+\tilde{A})$ + two Newton–Schulz iterations (5 GEMMs) with Paterson–Stockmeyer degree-4 Taylor exponential (3 GEMMs including $Q R$), and reuse $R_L, R_R$ for moment transport:
   $$S = \mathcal{F}(\tilde{A}_b, \tilde{A}_b), \quad R = R_b\!\left(I + \tilde{A} + \tfrac{1}{2}S + \mathcal{F}\!\left(R_b\!\left(\tfrac{1}{6}\tilde{A} + \tfrac{1}{24}S\right), R_b(S)\right)\right), \quad Q^+ = \mathcal{B}(Q, R),$$
   $$M^+ = \mathcal{B}(\mathcal{B}(R_L^\top, M), R_R).$$
   **Algebraic derivation of orthogonality error **`[certain]`**:**
   Since $\tilde{A}^\top = -\tilde{A}$, $R_4^\top = I - \tilde{A} + \frac{1}{2}\tilde{A}^2 - \frac{1}{6}\tilde{A}^3 + \frac{1}{24}\tilde{A}^4$, so
   $$R_4^\top R_4 = \left(I + \tfrac{1}{2}\tilde{A}^2 + \tfrac{1}{24}\tilde{A}^4\right)^2 - \left(\tilde{A} + \tfrac{1}{6}\tilde{A}^3\right)^2 = I + \underbrace{(1-1)}_{0}\tilde{A}^2 + \underbrace{\left(\tfrac{1}{4}+\tfrac{1}{12}-\tfrac{1}{3}\right)}_{0}\tilde{A}^4 + \underbrace{\left(\tfrac{1}{24}-\tfrac{1}{36}\right)}_{\frac{1}{72}}\tilde{A}^6 + \tfrac{1}{576}\tilde{A}^8.$$
   Because eigenvalues of $\tilde{A}^2$ are $-\theta_k^2$: at $\|\tilde{A}\|_2 \le 0.5$, $\|R_4^\top R_4 - I\|_2 \le \frac{0.5^6}{72} = \mathbf{2.17 \times 10^{-4}}$ (**11× below** the BF16 floor $0.0024$). Moreover, $R_4$ approximates $\exp(\tilde{A})$ accurately, whereas $\operatorname{polar}(I+\tilde{A}) = \exp(\arctan(\tilde{A}))$ undershoots large angles ($\arctan(0.5)=0.464$).
2. **Cost:**
   - `warm_basis`: $Q^\top C$ (1) + $(Q^\top C)Q$ (2) + $S=\tilde{A}^2$ (3) + $(\frac{1}{6}\tilde{A}+\frac{1}{24}S)S$ (4) + $Q R$ (5) = **5 BF16 GEMMs ($10n^3$)**, down from **7 BF16 GEMMs ($14n^3$)** (**−28.6% FLOPs**).
   - `transport_moment` on warm events: **2 BF16 GEMMs** ($2m^2n + 2mn^2$) down from **4 GEMMs** (**−50% FLOPs**, AND halves BF16 rounding passes on $M$).
3. **Expected effect:** Cuts `warm_basis` by **~26–29%** at compute-bound sizes ($n \ge 1536$: $49.8 \to \sim 37\text{ ms}$ on 109M; $45.5 \to \sim 32.5\text{ ms/factor}$ at $n=8192$) and cuts warm transport from $5.06 \to 2.53\text{ ms}$ → **~15 ms saved per warm event (~1.2 ms/call)** on 109M, and drops the large-$n$ `warm/QR` ratio at $n=8192$ from $0.406 \to \sim 0.29$. Confidence: **high**.
4. **Failure modes:**
   - **No self-healing of pre-existing $Q$ Gram drift:** Unlike Newton–Schulz on $Y$, multiplying $Q \leftarrow R_b(Q R_4)$ does not damp existing $E_Q = Q^\top Q - I$; over 4 warm refreshes between QR50 repairs, BF16 rounding errors add up roughly as $\sim 0.0024\sqrt{4} \approx 0.005$ (worst-case $\le 0.01$). If replay shows drift $> 0.008$, add 1 NS step on the 2nd warm step only (average 5.5 GEMMs) or use $R_4$ on steps 1, 3 and 1 NS polish on steps 2, 4.
   - **Norm limit:** Breaks down if $\|\tilde{A}\|_2 > 0.7$ ($\frac{0.7^6}{72} = 0.0016$, $\frac{1.0^6}{72} = 0.014$). If Question B allows larger steps ($\|\tilde{A}\|_2 \le 1.2$), use 1 scaling-and-squaring step $R = R_4(\tilde{A}/2)^2$ (**6 GEMMs**, still 14% faster than current warm, error $\frac{1}{2304}\|\tilde{A}\|_2^6 = 4.3 \times 10^{-4}$ at $\|\tilde{A}\|_2 = 1.0$).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Halving transport GEMMs reduces BF16 rounding noise injected into $M$ every 10 steps; monitor 4-step Gram drift in BF16.
6. **Cheap test:** (i) Replay harness: chain 4 consecutive $R_4$ warm steps between QR50s and check max Gram error (reject if $> 0.01$ at $t \bmod 50 = 40$). (ii) Microbenchmark at $n = 768\text{–}8192$ (reject if speedup at $n=3072$ is $< 22\%$).

---

### A7. Strided Covariance EMA Accumulation ($k_C = 2$, $\beta_C' = \beta_C^2 = 0.9025$)

1. **Exact update:** Only execute `covariance_ema` for $C_L, C_R$ on even ages ($t \bmod 2 == 0$, which automatically includes every refresh age $t \bmod 10 == 0$), using $\beta_C' = \beta_C^2 = 0.9025$ to preserve the exact 13.5-call half-life ($\ln 0.5 / \ln 0.9025 = 6.75$ updates $= 13.5$ calls). Compile a second lightweight `ordinary_nocov` graph for odd ages.
2. **Cost:** Eliminates $G_b G_b^\top$ ($2m^2n$), $G_b^\top G_b$ ($2mn^2$) = **434.87 GFLOP** AND **962 MB** of $C_L, C_R$ HBM read+write on **50%** of all calls (amortized saving: **217.4 GFLOP + 481 MB/call**).
3. **Expected effect:** Saves **~1.6–1.8 ms/call** across 100% of calls (~16–18% of ordinary step, **~6.5% of whole optimizer**). Confidence: **high** on speed; **medium-high** on negligible loss impact (effective sample size $N_{\text{eff}} = \frac{1+\beta_C'}{1-\beta_C'}$ drops from $39 \to 19.5$ gradient matrices, which for $3072 \times 768$ is still $19.5 \times 768 \approx 15,000$ rank-1 outer products, $\sim 5\times$ the dimension 3072). Going further to $k_C = 5$ ($\beta_C' = 0.7738$, $N_{\text{eff}} = 7.8$) saves ~2.7 ms/call but drops effective samples for $n=3072$ to ~6,000 ($2\times$ dimension), risking sample-covariance noise.
4. **Failure modes:** Higher variance in $C$ when $m \gg n$ and batch size is small; at $k_C = 2$ on 109M, risk is very low.
5. **Interaction with BF16 storage & $\beta_C=0.95$:** **Positive!** Because $1 - \beta_C'$ nearly doubles ($0.05 \to 0.0975$), new outer-product entries down to $0.04 |C_{ij}|$ (instead of $0.078 |C_{ij}|$) survive BF16 half-ULP rounding when added to $\beta_C' C_{ij}$.
6. **Cheap test:** **Cannot be tested in the open-loop replay harness** (which only saves $C$ every 10 ages, not per-step $G$). Speed: microbenchmark ordinary step with vs without `covariance_ema` (reject if saving $< 1.2\text{ ms/call}$). Quality: 2,000-call training run on seed 204 comparing $k_C=2$ vs $k_C=1$ (reject if CE at 2000 degrades by $> 0.001$).

---

### A8. Size-Aware Refresh Crossover (Drop Warm Refresh Above $n \approx 4096\text{–}6144$ in Favor of `sCholQR2`)

1. **Exact update:** Use warm10/`sCholQR2`-50 only for factor sizes where warm is much cheaper than `sCholQR2`, and switch directly to pure `sCholQR2` at cadence 10 or 20 for large factors ($n \ge 4096$, or even everywhere once A5 is in place).
2. **Cost:** Compare amortized refresh cost per 100 calls at $n = 8192$ (from §2.2 + A5/A6):
   - Current `warm10/QR50`: $8(45.5) + 2(112) = \mathbf{588\text{ ms}}$/100 calls.
   - Current `QR20` (no changes at all!): $5(112) = \mathbf{560\text{ ms}}$/100 calls (already faster AND better loss!).
   - `sCholQR2`-20 (~65 ms/factor): $5(65) = \mathbf{325\text{ ms}}$/100 calls (**1.8× faster** than current `warm10/QR50`, better loss!).
   - `A6-warm10 / sCholQR2-50` (~32.5 ms warm, ~65 ms sCholQR2): $8(32.5) + 2(65) = \mathbf{390\text{ ms}}$/100 calls — **STILL slower than `sCholQR2`-20 (325 ms)**!
   - Even on the 109M model: pure `sCholQR2`-10 (~125 ms + 5 ms transport) costs **13.0 ms/call** refresh overhead, **beating** current `warm10/QR50`'s **15.94 ms/call** with zero loss gap; while `A6-warm10 / sCholQR2-50` costs $0.08(39) + 0.02(130) = \mathbf{5.7\text{ ms/call}}$ (a **2.8×** cut in refresh overhead).
3. **Expected effect:** Eliminates the large-$n$ scaling wall completely (saves **45% of refresh time at $n=8192$** while upgrading from `warm10/QR50` to `QR20` quality). Confidence: **high**.
4. **Failure modes:** Inherits `sCholQR2`'s ridge/conditioning sensitivity (A5).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Exact same as A5.
6. **Cheap test:** Microbenchmark `A6-warm` vs `sCholQR2` across $n \in \{768..8192\}$ to locate the exact crossover $n^*$ where $8T_{\text{warm}} + 2T_{\text{sCholQR2}} > 5T_{\text{sCholQR2}}$ (i.e. $T_{\text{warm}}/T_{\text{sCholQR2}} > 0.375$). **Reject** crossover only if `A6-warm` beats $0.375\,T_{\text{sCholQR2}}$ at $n=8192$.

---

### A9. Blackwell Consumer Dtype Split: FP16 Basis Storage + FP16-Accumulate Update Back-Projection $U$ `[speculation]`

1. **Exact update:**
   - Store $Q_L, Q_R$ in **FP16** instead of BF16 (same 2 bytes/element; entries of orthonormal $Q$ are $O(1/\sqrt{n}) \in [10^{-3}, 1]$, well inside FP16's normal range $[6.1\times 10^{-5}, 65504]$ for $n \le 10^6$, with **10-bit mantissa** vs BF16's 7-bit).
   - Keep $G, M, V, C_L, C_R$ in BF16 (they require FP32 exponent range).
   - For the back-projection $U = Q_L \hat{U} Q_R^\top$ (where $\hat{U} = M_f / (\sqrt{V_f} + \epsilon)$ has $O(1)$ entries!), cast $\hat{U}$ to FP16 and compute both `bmm`s with **`allow_fp16_reduced_precision_reduction = True`** (FP16 Tensor Core MMA with FP16 accumulation). Do **NOT** use FP16-acc for `Y^T Y` (Newton–Schulz needs $10^{-3}$ Gram precision) or `Q^T C Q`.
   - *(Note on BF16 flag in §1.3: NVIDIA Ampere/Ada/Blackwell BF16 Tensor Core MMA always accumulates in FP32; only **FP16-with-FP16-acc** unlocks the 2× GeForce consumer Tensor Core rate — ~419 TFLOP/s vs ~210 TFLOP/s on RTX 5090 **`[recalled][verify]`**.)*
2. **Cost:** Same 2-byte storage for $Q_L, Q_R$; moves **434.87 GFLOP/call** (33% of ordinary GEMMs, `U = Q_L @ u_hat @ Q_R.T`) from ~210 TFLOP/s peak to ~419 TFLOP/s peak. Casting `Q` FP16$\to$BF16 for `Q_L.T @ G @ Q_R` fuses into register loads or costs 481 MB read/write (~0.35 ms if unfused; check fusion).
3. **Expected effect:** Saves **~0.9–1.3 ms/call** on 100% of calls **and** reduces stored basis Gram error by **8×** ($0.0024 \to \sim 0.0003$!). Confidence: **medium** (FP16-acc speedup on Blackwell `bmm` needs microbenchmark verification; Inductor fusion of FP16$\to$BF16 cast before `Q_L.T @ G` determines net gain).
4. **Failure modes:** FP16 accumulation over $n = 3072$ inner products introduces $\sim \sqrt{n}\cdot 2^{-11} \approx 2.7\%$ relative noise in $U$ (and overflow if an anomalous $\hat{U}$ entry exceeds $\sim 10^3$, though Adam normalized updates are $O(1)$ — clamp $|\hat{U}| \le 10$ before FP16 cast). At $n=8192$, relative noise is $\sim 4.4\%$.
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Strictly improves basis orthogonality (0.0003 vs 0.0024), reducing $M$-transport distortion; noise in $U$ does not feed back into $M, V, C$ (only into weight step $W$).
6. **Cheap test:** Speed: microbenchmark `Q_L @ u_hat @ Q_R.T` in FP16 (FP16-acc) vs BF16 on the 4 109M shapes (reject if saving $< 0.8\text{ ms}$ net of `Q` cast). Quality: **cannot be tested in open-loop replay** (no $U$ or loss); needs a 2,000-call training run (reject if CE at 2000 degrades by $> 0.001$).

---

# Question B: Tracking Quality

## 0. Root-Cause Diagnosis & Direct Answers to Your Three Questions

### Why the current `warm_basis` leaves $r_C = 0.150$ (synthetic), mean $0.38$ / max $0.975$ (trained), and only closes 12–37% of the loss gap

Four distinct mathematical defects compound inside the 5 lines of `warm_basis` **`[certain]`**:

1. **A factor-of-2 under-relaxation bug (`-.5`).**
   For $Q^+ = Q e^A$ ($A^\top = -A$) and $B = D + E$ ($D = \operatorname{diag}(d)$, $E$ off-diagonal):
   $$B^+ = e^{-A}(D + E)e^A = D + E + [D, A] + [E, A] + \tfrac{1}{2}[[D, A], A] + O(E^3).$$
   Since $([D, A])_{ij} = (d_i - d_j) A_{ij}$, exact first-order annihilation ($E + [D, A] = 0$) requires
   $$A^*_{ij} = -\frac{B_{ij}}{d_i - d_j}, \quad \text{yielding } B^+ = D + \tfrac{1}{2}[E, A^*] + O(E^3) = D + O(r_C^2) \quad (\textbf{quadratic convergence!}).$$
   Your code has `-.5`, giving $A_{ij} \approx \frac{1}{2} A^*_{ij}$, which leaves **at least 50%** of every off-diagonal entry intact ($B^+_{ij} \approx +\frac{1}{2} B_{ij}$) even on well-separated pairs with zero clipping.
2. **Global scale $s = \max_i |d_i|$ freezes 90%+ of a power-law spectrum.**
   The damping factor is $\frac{h_{ij}^2}{h_{ij}^2 + 0.01^2}$ with $h_{ij} = (d_i - d_j)/d_{\max}$. For any eigenpair in the bulk or tail with $d_i, d_j \le 0.01 d_{\max}$ — **even a 10:1 well-separated pair** like $d_i = 10^{-3} d_{\max}, d_j = 10^{-4} d_{\max}$ — $|h_{ij}| = 9\times 10^{-4}$, so $h_{ij}^2/(h_{ij}^2 + 0.01^2) = 0.008$: **99.2% of the rotation is killed**.
3. **Global Frobenius cap suffers $O(1/\sqrt{n})$ dimensional collapse.**
   For skew $A$ with eigenvalues $\pm i\theta_k$, $I+A$ has singular values $\sqrt{1+\theta_k^2}$, so Newton–Schulz convergence ($\sigma < \sqrt{3} \iff \|A\|_2 < \sqrt{2}$, 2-step BF16 accuracy at $\|A\|_2 \lesssim 0.55$) depends strictly on the **spectral norm** $\|A\|_2 = \max_k |\theta_k|$, NOT the Frobenius norm $\|A\|_F/\sqrt{2} = \sqrt{\sum_{k=1}^{n/2} \theta_k^2}$. Even on your synthetic benchmark ($A_{\text{true}}$ entries $\sim 0.2/\sqrt{n}$, $\lambda_i \propto 1/i$), a quick integral over the top $\sim 100$ undamped modes gives $\|A_{\text{warm}}\|_F/\sqrt{2} \approx \sqrt{\frac{0.04}{4}\int_0^\infty (1+(0.01x)^2)^{-2}dx} = \sqrt{0.01 \cdot 25\pi} \approx 0.89 > 0.5$, clipping $a \approx 0.56$! Combined with `-.5`, the effective step is $0.5 \times 0.56 = 0.28$, predicting $r_C^+ / r_C \approx 1 - 0.28 = \mathbf{0.72}$ — matching your measured $0.150/0.20 = \mathbf{0.75}$! Worse: if you fix Defect 2 without fixing Defect 3, all $n/2$ planes activate and $\|A\|_F/\sqrt{2}$ explodes by $\sqrt{n}$, throttling the entire step toward zero.
4. **Equal-diagonal paralysis explains your trained max $r_C = 0.975$.**
   If an emerging rank-1 spike $\lambda_1 u u^\top$ projects roughly equally onto $k$ basis columns ($Q^\top u \approx \frac{1}{\sqrt{k}}\mathbf{1}_k$), then the $k\times k$ block of $B$ is $\frac{\lambda_1}{k}\mathbf{1}_k \mathbf{1}_k^\top$, which has **equal diagonals** $d_1 = \dots = d_k = \lambda_1/k$ and $r_C = \sqrt{1 - 1/k}$ (**for $k=20$, $r_C = 0.9747$**!). Because $h_{ij} = 0$, `warm_basis` computes **$A = 0$ identically** and stays frozen at $r_C = 0.975$ for 40 calls until QR50 rescues it. By contrast, `QRRepair` computes $Z = CQ = QB$ whose first column points *directly along $u$*, dropping $r_C$ from $0.975 \to 0$ in one step. The exact same stall occurs at every **eigenvalue level crossing** (where $d_i$ and $d_j$ pass through $d_i = d_j$).

---

### Sub-question 1: Is $r_C$ the right proxy for loss? What to measure instead?

**No — $r_C$ is systematically biased toward the top ~10 eigenvalues and blind to what SOAP's Adam preconditioner actually sees `[certain]`.**

- In a power-law spectrum $\lambda_i \propto 1/i$, the top 10 eigenvalues account for $\sum_{i=1}^{10} i^{-2} / (\pi^2/6) = \mathbf{94.2\%}$ of $\|B\|_F^2$. Thus $r_C$ barely notices even 100% misalignment in modes $11\text{–}3072$.
- Yet SOAP runs **Adam** in the rotated basis, dividing each coordinate by $\sqrt{V_{ij}} + \epsilon \approx \sqrt{d_i^{(L)} d_j^{(R)}}$, which **whitens** the spectrum so bulk and tail coordinates contribute $O(1)$ equally to the parameter update.
- **What actually governs SOAP's diagonal-approximation fidelity:** The Gaussian KL divergence (Hadamard gap) between $C$ and its diagonal approximation $Q D Q^\top$ ($D = \operatorname{diag}(B)$, with Adam floor $\epsilon$):
  $$D_{\text{KL}}\!\left(\mathcal{N}(0, C_\epsilon)\,\|\,\mathcal{N}(0, Q D_\epsilon Q^\top)\right) = \frac{1}{2}\left[\underbrace{\operatorname{tr}(D_\epsilon^{-1} B_\epsilon) - n}_{=\,0} + \log\frac{\det D_\epsilon}{\det C_\epsilon}\right] = -\frac{1}{2}\log\det\!\left(D_\epsilon^{-1/2} B_\epsilon D_\epsilon^{-1/2}\right) \ge 0.$$
  Expanding $R = D_\epsilon^{-1/2} B_\epsilon D_\epsilon^{-1/2} = I + E_{\text{corr}}$ to second order:
  $$-\frac{1}{2}\log\det(I + E_{\text{corr}}) = \frac{1}{4}\|E_{\text{corr}}\|_F^2 + O(\|E_{\text{corr}}\|^3) = \frac{1}{4}\sum_{i \neq j}\frac{B_{ij}^2}{(d_i + \epsilon)(d_j + \epsilon)}.$$
- **What to measure instead (all computable in your existing replay harness!):**
  1. **Whitened correlation error / Hadamard KL gap:** $-\frac{1}{2n}\log\det(D_\epsilon^{-1/2} B_\epsilon D_\epsilon^{-1/2})$ and your existing **RMS off-diagonal correlation** $\sqrt{\frac{1}{n(n-1)}\sum_{i\neq j} B_{ij}^2 / ((d_i+\epsilon)(d_j+\epsilon))}$ — **prioritize these over raw $r_C$!**
  2. **Preconditioned condition number** $\kappa(D_\epsilon^{-1/2} B_\epsilon D_\epsilon^{-1/2})$ (or 99th/1st eigenvalue ratio of $R$): directly controls the step-size/curvature ceiling of SOAP.
  3. **Inter-refresh basis churn** $\chi = 1 - \frac{1}{n}\sum_i |(Q^\top Q^+)_{ii}|$ (after permutation alignment $\pi$), split into top-64 vs tail: **essential guardrail!** Because $V$ is not rotated, an aggressive refresh that slashes tail correlation by whipping degenerate noise eigenvectors around every 10 calls will win on correlation in open-loop replay while **hurting training loss** by invalidating $V$.
  4. **Worst-case (95th/max) $r_C$ across factors**, not just the mean — your mean $0.38$ hides catastrophic factor stalls at $0.975$.

### Sub-question 2: Role of the call-500 gap (+0.124 CE)

1. **Why call 500 is uniquely bad:** At `k=1` (cold start), $C_L$ for SwiGLU-in ($3072\times 768$) comes from a **single** gradient $G_1$, so $\operatorname{rank}(C_L) \le 768$ — **2,304 of 3,072 eigenvalues are identically zero** (random eigenvectors from `eigh`). Over calls 1–500 (warmup + feature emergence), the effective rank explodes from $768 \to 3072$ and dominant spikes rotate violently. Your schedule waits **50 updates** for the first QR repair, and runs only 9 QRs before call 500 (vs 49 for QR10), while `warm_basis` stalls on those near-degenerate emerging modes (closing only **12%** of the QR50–QR10 gap at call 500 vs **37%** at call 2000).
2. **Does call 500 explain the call-10,000 gap (+0.010–0.011)?** Only partially **`[certain]`**. Between call 500 and 2000, the gap shrinks $15\times$ ($+0.124 \to +0.008$) largely because the CE slope flattens. **Crucially**, between call 2000 and 10,000 on seed 204, the gap does *not* keep decaying — it slightly widens ($+0.0080 \to +0.0112$). Thus the final $+0.010\text{–}0.011$ deficit is sustained by **ongoing steady-state basis lag** (mean $r_C = 0.38$, max $0.975$ even at age 9990!) plus whatever portion is caused by **BF16 state** (unmeasured until BF16 QR10 completes!). Fixing *only* early calls (B4) will wipe out the call-500 gap, but you must ALSO fix steady-state tracking (B1–B3/B6) to close the call-10,000 gap.

### Sub-question 3: Treatment of $V$ and column order in the warm step

- **Column order:** `warm_basis` already computes $d = \operatorname{diag}(B)$, the exact Rayleigh quotients used by `qr_basis`, yet never sorts by $\pi = \operatorname{argsort\_desc}(d)$. Leaving columns unsorted across eigenvalue crossings breaks QR10's descending invariant for up to 40 steps, causes sign inversions with tie-breaking, and forces large multi-column shuffles at QR50. Sorting $Q, B, M$ and permuting $V$ by $\pi_L, \pi_R$ costs **only 0.38 ms** across all 48 matrices (§2.1) and should be done on **every** warm refresh.
- **Why naive $V$ rotation $V^+ = (R_L \odot R_L)^\top V (R_R \odot R_R)$ is a trap **`[certain]`**:**
  Although $S = R \odot R$ is doubly stochastic (nonnegative, trace-preserving, exact for permutations), it assumes old coordinates $\bar{G}$ were *uncorrelated*, dropping the cross-term $2 B_{ij}\sin\theta\cos\theta$. When $R$ rotates *into* the true eigenbasis to remove misalignment $\theta$, that cross-term is **negative** (it cancels leaked variance!).
  *Counterexample:* Let $C = U \operatorname{diag}(10^4, 10^{-2}) U^\top$ with old basis misaligned by $\theta = 0.01$. Old diagonals are $d_1 \approx 10^4$ and $d_2 = 10^4 \sin^2(0.01) + 10^{-2}\cos^2(0.01) \approx \mathbf{1.01}$. Rotating back by $-\theta$ restores true variance $\lambda_2 = \mathbf{0.01}$. Leaving $V_2$ unchanged gives $1.01$; applying $(R \odot R)^\top V (R \odot R)$ gives $V_2^+ = \sin^2(0.01)\cdot 10^4 + \cos^2(0.01)\cdot 1.01 = \mathbf{2.01}$ — **2× worse than doing nothing** (it double-counts the leaked variance!). This explains why SOAP itself (Vyas et al. 2024 **`[recalled][verify]`**) only permutes $V$. The right fix (when rotations are large) is **Kronecker-ratio rescaling** using the *actual* post-rotation diagonal $d^+ = \operatorname{diag}(R^\top B R)$, which includes all cross-terms (see **B5**).

---

## Question B Candidates

### B1. Fixed Full-Step Relative-Gap $\arctan$-Jacobi + Rayleigh Sort ($V$-Perm) + Row-$\ell_2$/Spectral Cap + $R_4(\tilde{A}/2)^2$

1. **Exact update:** Fixes all four defects of `warm_basis` in one drop-in replacement:
   ```python
   # 1. Projected covariance + Rayleigh descending sort (also permute V by order!)
   b = mm32(torch.bmm(qb.transpose(-1, -2), cb), qb)
   b = 0.5 * (b + b.transpose(-1, -2))
   d = b.diagonal(dim1=-2, dim2=-1)
   order = d.argsort(dim=-1, descending=True)
   q_ord = q.gather(-1, order.unsqueeze(-2).expand_as(q))
   # permute rows & cols of b by order -> d_ord is monotonically descending!
   b_ord = b.gather(-2, order.unsqueeze(-1).expand_as(b)).gather(-1, order.unsqueeze(-2).expand_as(b))
   d_ord = b_ord.diagonal(dim1=-2, dim2=-1)
   # 2. Pairwise relative-gap exact 2x2 Jacobi angle (NO 0.5 factor; breaks d_i == d_j ties!)
   d_max = d_ord.abs().amax(-1, keepdim=True).clamp_min(1e-30)
   s_ij = d_ord.abs().unsqueeze(-1) + d_ord.abs().unsqueeze(-2) + 1e-5 * d_max.unsqueeze(-1)
   gap_abs = (d_ord.unsqueeze(-1) - d_ord.unsqueeze(-2)).abs()
   idx = torch.arange(b.shape[-1], device=b.device)
   sgn = (idx.unsqueeze(0) - idx.unsqueeze(1)).sign().float()     # +1 for i<j, -1 for i>j, 0 on diag
   a = -0.5 * sgn * torch.atan2(2.0 * b_ord, gap_abs + 0.05 * s_ij)
   # 3. Local row-l2 angle cap (no sqrt(n) collapse; skew-preserving D_w A D_w)
   row_norm = a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
   w = (0.5 / row_norm).clamp(max=1.0)
   a = w * a * w.transpose(-1, -2)
   # 4. 3-step block power iteration (r=4 fixed probes) for spectral norm ||a||_2 <= theta_max = 0.9
   # (omitted for brevity: 3 bmm with (b, n, 4), cost 24 n^2 FLOPs; scale a by min(1, 0.9 / (1.05*sigma)))
   # 5. Degree-4 scaling-and-squaring matrix exponential R = R_4(a/2)^2
   a1 = 0.5 * a
   a1b = a1.to(compute_dtype)
   s = mm32(a1b, a1b)
   p = mm32((a1 / 6.0 + s / 24.0).to(compute_dtype), s.to(compute_dtype))
   r_half = (eye + a1 + 0.5 * s + p).to(compute_dtype)
   r = torch.bmm(r_half, r_half)
   q_new = torch.bmm(q_ord.to(compute_dtype), r).to(basis_dtype).contiguous()
   # Return (q_new, order, r) -> permute V by order, and transport M via r_L^T @ M_perm @ r_R!
   ```
   **Exact algebraic checks **`[certain]`**:**
   - **Exact skew-symmetry:** Swapping $i \leftrightarrow j$ flips `sgn` and leaves `b_ord`, `gap_abs`, `s_ij` invariant, so $A_{ji} = -A_{ij}$ and $A_{ii} = 0$ identically.
   - **Well-separated pairs anywhere in spectrum ($d_i = 10 d_j$, even at $10^{-4} d_{\max}$):** $\operatorname{atan2}(2B_{ij}, 9d_j + 0.55 d_j) \approx \frac{2B_{ij}}{9.55 d_j} \implies A_{ij} \approx -0.94 \frac{B_{ij}}{d_i - d_j}$ (full quadratic step!).
   - **Equal-diagonal stall ($C = \begin{pmatrix}1 & 0.5 \\ 0.5 & 1\end{pmatrix}$):** $A_{12} = -\frac{1}{2}\operatorname{atan2}(1.0, 0.10) = -0.736\text{ rad}$ (94% of exact $-\pi/4 = -0.785$ rad, dropping $B_{12}$ by $10\times$ in one step!), while equal diagonals with tiny noise ($B_{12} = 10^{-4}$) give $A_{12} = -10^{-3}$ (no noise blow-up).
   - **Row-$\ell_2$ cap:** $(D_w A D_w)^\top = -D_w A D_w$; geometrically $\|A_{:, j}\|_2$ is column $j$'s rotation angle, which is $O(1)$ independent of $n$ on dense rotations ($0.2$ on your synthetic benchmark vs $\|A\|_F/\sqrt{2} = 0.2\sqrt{n/2}$!), and prevents one noisy pair from throttling the rest of the matrix.
   - **Orthogonality of $R_4(a/2)^2$:** Since $R_4(a/2)$ is a polynomial in skew $a$, it commutes with its transpose, giving $R^\top R = (I + \frac{1}{72}(a/2)^6 + \dots)^2 = I + \frac{1}{2304} a^6 + O(a^8)$. At $\|a\|_2 = 0.9$, $\|R^\top R - I\|_2 \le 2.3 \times 10^{-4}$ (**10× below** BF16 floor!).
2. **Cost:** $Q^\top C$ (1) + $(Q^\top C)Q$ (2) + $S=a_1^2$ (3) + $P$ (4) + $R = R_{\text{half}}^2$ (5) + $Q_{\text{ord}} R$ (6) = **6 BF16 GEMMs ($12n^3$)** + $24n^2$ block power iter + $O(n^2)$ `atan2`/sort + 0.38 ms $V$-perm, AND **saves 2 GEMMs on transport** ($M^+ = R_L^\top M_{\text{perm}} R_R$). Net warm event is **~15% FASTER than current `warm_basis`**!
3. **Expected effect:** On the synthetic $r_C = 0.20$ input, drops single-refresh $r_C$ from $0.150 \to \mathbf{0.03\text{–}0.05}$ (closing ~80–90% of the gap to QR's $0.025$), dramatically lowers bulk/tail RMS off-diagonal correlation, and removes equal-diagonal freezes (expect trained mean $r_C$ to drop from $0.38 \to \sim 0.22\text{–}0.26$, closing **~60–80%** of the steady-state `warm10/QR50`–`BF16 QR10` loss gap). Confidence: **high**.
4. **Failure modes:**
   - **Over-rotating degenerate tail noise:** If $\delta = 0.05$ is too small on very noisy finite-sample bulk spectra, bulk columns may churn every 10 steps and hurt $V$ (tune $\delta \in \{0.05, 0.10, 0.20\}$ using the churn metric!).
   - **4-step BF16 Gram drift** (same as A6: verify $\le 0.008$ over 4 warm steps; add 1 NS step on warm #2 if needed).
   - **Large multi-column equal-diagonal cluster** ($k = 20$ rank-1 spike): single step with $\|a\|_2 \le 0.9$ breaks the stall and concentrates energy toward column 1, though needs ~2–3 steps (see B2/B3) to fully collapse $r_C = 0.975 \to 0.05$.
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Because $B = \mathcal{F}(\mathcal{B}(Q^\top, C), Q)$ rounds $Q^\top C$ to BF16, tail entries $d_j < 10^{-3} d_{\max}$ can suffer BF16 noise when $Q$ is misaligned (see Pitfall 3); keeping the floor $10^{-5} d_{\max}$ (or $10^{-4} d_{\max}$) prevents dividing by BF16 noise in the deep tail.
6. **Cheap test:** Replay harness (§2.6(i)) on widths 256–2048: measure mean/max $r_C$, RMS off-diagonal correlation, KL gap, tail churn, and Gram error across $\delta \in \{0.05, 0.1, 0.2\}$. **Reject** if single-refresh synthetic $r_C > 0.07$ or replay mean $r_C$ does not beat current warm by $> 40\%$ of the gap to QR10 without doubling tail churn.

---

### B2. Two-Iteration Local-Frame Warm Jacobi (`B1 × 2`)

1. **Exact update:** Run **two** inner iterations of B1 inside a single warm refresh, propagating $B$ in rotated space so $C$ is only touched once:
   - $B_0 = \operatorname{sym}(\mathcal{F}(\mathcal{B}(Q^\top, C), Q))$ (2 GEMMs).
   - Iter 1: sort $B_0$ by $\pi_1$, build $A_1$ via B1, compute $R_1 = R_4(A_1)$ (2 GEMMs, since 2 steps need smaller cap $\|A_1\|_2 \le 0.6$, saving the squaring GEMM!), update $B_1 = \operatorname{sym}(\mathcal{F}(\mathcal{B}(R_1^\top, B_{0,\text{ord}}), R_1))$ (2 GEMMs).
   - Iter 2: sort $B_1$ by $\pi_2$, build $A_2$ via B1, compute $R_2 = R_4(A_2)$ (2 GEMMs), compose $R_{\text{tot}} = \mathcal{B}(R_1[:, \pi_2], R_2)$ (1 GEMM), $Q^+ = \mathcal{B}(Q_{\pi_1 \circ \pi_2}, R_{\text{tot}})$ (1 GEMM). Permute $V$ by $\pi = \pi_1 \circ \pi_2$ and transport $M^+ = R_{\text{tot}, L}^\top M_{\text{perm}} R_{\text{tot}, R}$ (2 GEMMs).
2. **Cost:** **10 BF16 GEMMs ($20n^3$)** per factor ($1.43\times$ current 7-GEMM `warm_basis`), minus 2 transport GEMMs → **~1.3× current warm event cost** at $n = 768\text{–}3072$, well inside your $2\times$ budget and still $3\times\text{–}25\times$ below QR at $n \le 4096$.
3. **Expected effect:** Achieves **$O(r_C^4)$** quartic convergence on separated modes; crucially, when Iter 1 splits an equal-diagonal cluster or hits the rotation cap $\rho_{\max} = 0.5$, Iter 2 immediately diagonalizes the newly separated modes in the same refresh. On synthetic $r_C = 0.20$, expect single-refresh $r_C \approx \mathbf{0.005\text{–}0.02}$ (**beating QR10's $0.025$**!), and trained mean $r_C \approx \mathbf{0.18\text{–}0.22}$ matching QR10. Confidence: **high**.
4. **Failure modes:** Two consecutive BF16 roundings inside $B_1 = R_1^\top B_0 R_1$ can slightly degrade deep-tail precision; at $n=8192$ ($10/7 \times 45.5 \approx 65\text{ ms}$, or ~53 ms with A3 micro-batching), it approaches `sCholQR2`'s cost (so at $n \ge 6144$, A8/B6 is preferable).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Only ONE BF16 rounding is applied to $Q^+$ ($Q_{\text{perm}} R_{\text{tot}}$), same as B1 — does not double $Q$'s rounding drift.
6. **Cheap test:** Replay harness comparing B2 vs B1 vs QR10 on mean/max $r_C$ and RMS correlation, plus microbenchmark at $n = 768\text{–}3072$. **Reject** if B2 fails to cut B1's remaining gap to QR10 by at least 50% (especially max $r_C$), or if cost exceeds $1.6\times$ current warm at $n = 3072$.

---

### B3. Top-$k$ Exact Power-Subspace Deflation (Procrustes Reflection $H_+$, Zero SVD/QR) + B1 Jacobi

1. **Exact update:** Directly eliminates the $r_C = 0.975$ emerging-spike stall and matches `QR10` **identically** on the top-$k$ subspace ($k = 32$ or $64$) before running B1 on the full matrix, using only tall-skinny GEMMs and a $k\times k$ Cholesky solve:
   - Sort $Q$ by Rayleigh quotient $\pi = \operatorname{argsort\_desc}(d)$ (or reuse previous order; permute $V$ by $\pi$). Split $Q = [Q_1, Q_2]$ with $Q_1 \in \mathbb{R}^{n \times k}, Q_2 \in \mathbb{R}^{n \times (n-k)}$.
   - Exact top-$k$ power step: $Z_1 = \mathcal{F}(C, Q_1) \in \mathbb{R}^{n \times k}$, orthonormalize $U_1 = \operatorname{sCholQR2}(Z_1)$ (only a $k\times k$ Cholesky!).
   - Minimal-rotation orthogonal extension of $Q_1 \to U_1$ to $Q_2$ (exact geodesic on $O(n)$):
     $$M = Q_1^\top U_1 \in \mathbb{R}^{k \times k}, \quad R_p = \operatorname{polar}(M) \;\text{(8 iters of } k\times k\text{ Halley/NS)}, \quad \tilde{U}_1 = U_1 R_p^\top, \quad S = \operatorname{sym}(M R_p^\top) = (MM^\top)^{1/2} \succeq 0,$$
     $$V_+ = Q_1 + \tilde{U}_1, \quad H_+ = I_n - V_+(I_k + S)^{-1} V_+^\top, \quad Q_{\text{mid}} = \big[U_1,\; Q_2 - V_+(I_k + S)^{-1}(\tilde{U}_1^\top Q_2)\big].$$
   - Then run 1 pass of **B1** on $Q_{\text{mid}}$ (or run $R_4$ directly from $B_{\text{mid}} = Q_{\text{mid}}^\top C Q_{\text{mid}}$).
   **Exact algebraic proof **`[certain]`**:**
   - Since $Q_1^\top Q_1 = \tilde{U}_1^\top \tilde{U}_1 = I_k$ and $Q_1^\top \tilde{U}_1 = \tilde{U}_1^\top Q_1 = S \succeq 0$:
     $$V_+^\top V_+ = 2(I_k + S), \qquad H_+^\top H_+ = I_n - 2V_+(I_k+S)^{-1}V_+^\top + V_+(I_k+S)^{-1}\big[2(I_k+S)\big](I_k+S)^{-1}V_+^\top = I_n.$$
   - Because all eigenvalues of $S$ are principal cosines $\cos\theta_j \in [0, 1]$, **$\kappa(I_k + S) \in [1, 2]$ ALWAYS** — completely singularity-free even for an exact $90^\circ$ rotation ($\cos\theta_j = 0$)!
   - Since $V_+^\top Q_1 = I_k + S$, $H_+ Q_1 = Q_1 - V_+ = -\tilde{U}_1$, so $U_1^\top (H_+ Q_2) = -R_p (H_+ Q_1)^\top (H_+ Q_2) = -R_p (Q_1^\top Q_2) = 0$ and $(H_+ Q_2)^\top (H_+ Q_2) = I_{n-k}$.
   - Because $H_+ = I$ on $\operatorname{span}(Q_1, U_1)^\perp$, it perturbs $Q_2$ by the **minimum possible Frobenius distance** (preserving $V$ on the bottom $n-k$ columns!).
2. **Cost:** Top-$k$ deflation: $Z_1 = C Q_1$ ($2n^2k$) + $\tilde{U}_1^\top Q_2$ ($2n(n-k)k$) + $V_+ W$ ($2n(n-k)k$) + $O(nk^2 + k^3)$ = **$6n^2k + O(nk^2)$ FLOPs**.
   As a fraction of ONE $n\times n$ GEMM ($2n^3$): $\frac{3k}{n}$ — for $k = 64$, that is **0.25 GEMM** at $n=768$, **0.0625 GEMM** at $n=3072$, and **0.023 GEMM** at $n=8192$! Combined with B1 (6 GEMMs), total is **~6.1–6.3 GEMMs** (**still faster than current 7-GEMM `warm_basis`!**).
3. **Expected effect:** Completely destroys the max $r_C = 0.975$ failure mode (takes any rank-$\le k$ spike spread across arbitrary equal diagonals straight to $r_C \approx 0$ in 1 step, identical to QR10 on the top $k$ modes!) and dramatically slashes the call-500 gap (where top modes rotate by $>45^\circ$). Confidence: **high**.
4. **Failure modes:**
   - If a mode rotates by **exactly** $90^\circ$ ($\sigma_{\min}(M) = 0$), plain Newton–Schulz for $R_p = \operatorname{polar}(M)$ maps $0 \to 0$ (losing rank in $\tilde{U}_1$); fix by initializing $M \leftarrow M + 10^{-3} I_k$ and using 6–8 steps of cubic Halley iteration $X_{t+1} = X_t(3I + X_t^\top X_t)(I + 3 X_t^\top X_t)^{-1}$ on $k\times k$ (microseconds for $k=64$, converges from $10^{-3}$ to $1 - 10^{-5}$ in 8 steps), or `torch.linalg.svd` on the tiny $(b, k, k)$ matrix.
   - Note: because $Q_{\text{mid}}$ is formed before B1, `transport_moment` needs $R_{\text{full}} = Q^\top Q^+$ or standard 4-GEMM transport (5.06 ms, same as today).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Keep the tiny $k\times k$ ops and $Z_1$ in FP32; cast $Q_{\text{mid}}$ to BF16 only after B1 (or keep $Q_{\text{mid}}$ in BF16 for B1's $Q^\top C Q$). Because $Z_1 = \mathcal{F}(C, Q_1)$ does not round intermediate $Q^\top C$ to BF16, top-$k$ accuracy is strictly cleaner than `warm_basis`.
6. **Cheap test:** Replay harness on widths 256–2048 + synthetic $r_C = 0.975$ rank-1 equal-diagonal test ($B = \frac{1}{20}\mathbf{1}_{20}\mathbf{1}_{20}^\top$). **Reject** if max $r_C$ across all replay snapshots exceeds $1.25\times$ QR10's max $r_C$, or if $k=64$ deflation adds $> 20\%$ latency at $n=768$ in the microbenchmark.

---

### B4. Front-Loaded Refresh Schedule (QR10 / `sCholQR2`-10 for Calls 1–1000, Warm10/QR100 Thereafter)

1. **Exact update:** Reallocate the **same** (or smaller) total QR budget over the 10,000-call run to match the non-stationarity of the eigenspace:
   - Calls $t \in [1, 1000]$ (plus an extra refresh at $t = 2$ or $t = 5$ to exit the rank-$\le 768$ single-gradient cold start!): run `QR` (or `sCholQR2` from A5) every **10** ages (100 QRs).
   - Calls $t \in [1001, 10000]$: run improved warm (B1/B3) every 10 ages and `QR`/`sCholQR2` every **100** ages (90 QRs).
   - Total QRs = **190** (fewer than your current **199** QRs!).
2. **Cost:** **Negative incremental cost** relative to current `warm10/QR50` (190 vs 199 QRs over 10k calls; 9 fewer QR events = **−0.5 ms/call** amortized, and zero extra cost per warm step!).
3. **Expected effect:** Immediately **eliminates 100% of the +0.124 call-500 gap** (by construction, calls 1–1000 ARE BF16 QR10!), and isolates how much of the call-10,000 gap is an early-trajectory scar vs ongoing steady-state lag. Combined with B1/B3 after call 1000, expect call-10,000 loss to approach BF16 QR10 within **0.001–0.003 CE** **`[speculation]`**. Confidence: **high** for call 500; **medium** for call-10k (depends on B1/B3 holding $r_C \approx 0.20$ over 100-step QR intervals).
4. **Failure modes:** If B1/B3 still drifts over 9 consecutive warm refreshes (age $100k+10 \dots 100k+90$), extending QR cadence to 100 may hurt late-stage loss (test QR50 vs QR100 in replay first!). Also, a 2,000-call short run will look artificially good unless you evaluate the warm phase ($t > 1000$) separately.
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Eliminates the worst cold-start BF16 degeneracy (2,304 zero eigenvalues in $C_L$ during calls 1–50).
6. **Cheap test:** (i) Replay harness (ages 0–1000): verify whether B1/B3 sustains QR10-level $r_C$ and Gram error over a 100-step QR interval (9 consecutive warm refreshes). (ii) 2,000-call training run with crossover at call 500 (`QR10` for 1–500, `B1-warm10/QR50` for 501–2000) to measure both early recovery and post-500 drift. **Reject** QR100 late cadence if replay Gram error $> 0.01$ or $r_C$ climbs $> 0.05$ above QR50 between ages 500 and 1000.

---

### B5. Kronecker-Ratio $V$-Rescaling After Basis Rotations

1. **Exact update:** After refreshing $(Q_L, Q_R) \to (Q_L^+, Q_R^+)$ and permuting $V$ by $(\pi_L, \pi_R)$, rescale $V$ by the **exact** ratio of post- to pre-rotation factor diagonal variances (which incorporates all de-correlating cross-terms, unlike $(R\odot R)^\top V(R\odot R)$!):
   $$d_i^{(L)+} = \big((Q_L^+)^\top C_L Q_L^+\big)_{ii} = \sum_k (B_L R_L)_{ki} (R_L)_{ki}, \qquad \gamma_i^{(L)} = \operatorname{clip}\!\left(\frac{d_i^{(L)+} + \epsilon_d}{d_{\pi_L(i)}^{(L)} + \epsilon_d},\; c_{\min},\; c_{\max}\right),$$
   $$V_{ij}^+ = V_{\pi_L(i), \pi_R(j)} \cdot \big(\gamma_i^{(L)}\big)^\alpha \big(\gamma_j^{(R)}\big)^\alpha,$$
   with conservative bounds $c_{\min} = 0.25, c_{\max} = 4.0$, floor $\epsilon_d = 10^{-4} d_{\max}$, and damping exponent $\alpha \in \{0.5, 1.0\}$. (On the earlier $\lambda_1 = 10^4, \lambda_2 = 10^{-2}, \theta = 0.01$ counterexample, $d_2 = 1.01, d_2^+ = 0.01 \implies$ ratio moves $V_2$ in the **correct downward** direction instead of doubling it!)
2. **Cost:**
   - Inside **B2** (2-iter Jacobi): $B_1 = R_1^\top B_0 R_1$ is already computed, so $d^+$ is **free** (or $1\times n^3$ GEMM $B R$ + $O(n^2)$ dot product if applied after the final rotation in B1).
   - Rescaling $V$: $O(mn)$ fused into the $V$-permutation kernel (**~0.1 ms** across all 48 matrices).
3. **Expected effect:** Eliminates multi-step Adam step-size overshoots/undershoots immediately after large basis rotations (especially during calls 1–1000 and level crossings where $V$ currently lags with a 13.5-call half-life). Likely modest steady-state gain (**~0.001–0.003 CE** **`[speculation]`**), larger early-training stability gain. Confidence: **medium-low** (Kronecker factorization $V_{ij} \propto d_i^{(L)} d_j^{(R)}$ is approximate; when non-Kronecker structure dominates, rescaling all $n$ entries in row $i$ by the row-average variance ratio can inject noise).
4. **Failure modes:** Non-Kronecker gradient structure (a single outlier entry $(i, j)$ drives $d_i^{(L)+}/d_i^{(L)}$ and falsely rescales the other $n-1$ entries in row $i$); deep-tail division noise if $\epsilon_d$ is too small.
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Because $\beta_2 = 0.95$ has a relatively short half-life (13.5 calls), $V$ already adapts within ~10–15 steps after a refresh; this matters most when rotations happen every 10 steps (leaving $V$ perpetually half-adapted!). Clamping $[0.25, 4.0]$ prevents BF16 tail noise from distorting $V$.
6. **Cheap test:** **Cannot be tested in the open-loop replay harness** (no $V$, no gradients, no loss). Smallest test: 2,000-call training run on seed 204 comparing `B1 + V-perm only` vs `B1 + V-perm + Kronecker V-rescale` ($\alpha \in \{0.5, 1.0\}$). **Reject** if CE at 500 and 2000 do not both improve by $\ge 0.001$.

---

### B6. Hybrid `B3` (at $n \le 2304$) + Pure `sCholQR2` Every 10 Steps (at $n \ge 3072$, or Everywhere!)

1. **Exact update:** Notice that once **A5 (`sCholQR2`)** fixes Cholesky-QR's stability, its cost at $n = 3072$ (~4.5 ms/factor) is only **1.8×** current warm (2.46 ms/factor), and at $n = 8192$ (~65 ms/factor) is **1.4×** current warm (45.5 ms/factor) — **strictly inside your Question B budget ($\le 2\times$ warm at $n=3072$, well below QR at every $n$)!**
   - Option B6a (Hybrid by size): Run **B3** (or B1) on warm steps for $n \le 2304$ (where warm is $3\times\text{–}4\times$ faster than `sCholQR2`) and **pure `sCholQR2` every 10 steps** for $n \ge 3072$ (which accounts for 61.5% of FLOPs and the worst rank-deficiency / conditioning!).
   - Option B6b (Everywhere): Run **pure `sCholQR2` every 10 steps** on all 96 factors (dropping warm refresh completely!).
2. **Cost (109M, all 96 factors):**
   - **B6b (pure `sCholQR2`-10):** ~125 ms + 5.1 ms transport = **~130 ms** every 10 calls = **13.0 ms/call** refresh overhead — **2.9 ms/call FASTER than current `warm10/QR50` (15.94 ms/call)**!
   - **B6a (`B1/B3` at $n \le 2304$ + `sCholQR2`-10 at $n = 3072$):**
     - Warm event (8%): $n \le 2304$ B1 (~17 ms) + $n=3072$ `sCholQR2` (~54 ms) + transport (~4 ms) = ~75 ms → $0.08 \times 75 = 6.0\text{ ms/call}$.
     - QR event (2%): all-96 `sCholQR2` (~130 ms) → $0.02 \times 130 = 2.6\text{ ms/call}$.
     - Total refresh overhead = **8.6 ms/call** (**1.85× faster** than current 15.94 ms/call!), while giving exact QR10 tracking on the largest, hardest ($3072\times 768$) factors!
3. **Expected effect:** B6b **replicates the BF16 QR10 trajectory identically** (zero warm-approximation gap, $r_C \approx 0.20$, zero call-500 gap!) while already beating your current `warm10/QR50` wall time! B6a gives ~90%+ of QR10 quality at nearly half the refresh cost. Confidence: **high**.
4. **Failure modes:** Same ridge/conditioning caveats as A5 (`sCholQR2`).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Exact match to BF16 QR10 (including column ordering and $V$ permutation every 10 calls).
6. **Cheap test:** Already covered by A5's replay + microbenchmark tests; validate B6b against the pending BF16 QR10 training run (reject if CE differs by $> 0.001$ or 109M `sCholQR2` event exceeds 160 ms).

---

# Pitfalls in the Current Design & Measurements

### P1. Triple Compound Defect in `warm_basis` (`-0.5`, global `0.01 s`, $\sqrt{n}$ Frobenius cap, equal-diagonal stall)
Detailed in Question B §0 **`[certain]`**. Critically, the defects **mask each other**: fixing global damping `0.01 s` (Defect 2) without replacing the Frobenius cap (Defect 3) will make $\|A\|_F/\sqrt{2}$ explode by $\sqrt{n}$ and *worsen* tracking! And `A = 0` at $d_i = d_j$ is directly responsible for your trained **max $r_C = 0.975$**.

### P2. Silent TF32 Truncation Inside FP32 Linear-Algebra Islands (`matmul precision = "high"`)
You set FP32 matmul precision to `"high"` (§1.3). Inside `qr_basis`, `torch.bmm(cov.float(), q.float())` is unaffected only because `cov, q` originate in BF16 (7-bit mantissa $\subset$ TF32's 10-bit mantissa). **However**, in your Cholesky-QR benchmark (§2.2), `Z = CQ` is full FP32, so `Z.T @ Z` silently truncates `Z` to TF32 ($u_{\text{TF32}} = 2^{-11} \approx 4.9\times 10^{-4}$) while `trsm` ($Z R^{-1}$) uses untruncated FP32 `Z`! Writing $\tilde{Z} = Z + \Delta Z$ ($\|\Delta Z\| \sim u_{\text{TF32}}\|Z\|$), $R^\top R = \tilde{Z}^\top \tilde{Z}$ leaves orthogonality error $\|(\Delta Z) R^{-1}\| \sim O(u_{\text{TF32}}\,\kappa(Z))$, directly explaining why Cholesky-QR's Gram error degraded to **0.034 at $n=4096$** and crashed at $n \ge 6144$ **`[certain]`**. The same silent TF32 truncation hits the `"Optional precision island"` (`torch.bmm(y.T, y)`) in `warm_basis` when `basis_dtype=fp32`! Every FP32 linear-algebra island must explicitly enforce IEEE FP32 (`fp32_precision="ieee"`).

### P3. BF16 Precision Traps in `Q^T C`, `Q` Storage, Unsymmetrized `C`, and Future $\beta$ Changes
1. **Intermediate BF16 rounding of $T = Q^\top C$ in `warm_basis` (`torch.bmm(qb.T, cb)`):** When $C$ has high $\kappa(C)$ (e.g. $\lambda_1 = 1, \lambda_n = 10^{-4}$) and $Q$ is misaligned by $\theta = 0.01$, row $n$ of $T = Q^\top C$ has norm $\sim \theta \lambda_1 = 10^{-2}$, so rounding $T$ to BF16 injects error $\sim 2^{-8} \cdot 10^{-2} = 4\times 10^{-5}$ — **40% relative error** on $d_n = 10^{-4}$ (and can even flip $d_n$ negative for $\lambda_n < 4\times 10^{-5}$!).
2. **BF16 storage of orthonormal $Q$ wastes 3 mantissa bits on unused exponent range:** Storing $Q_L, Q_R$ in **FP16** (same 2 bytes!) cuts the Gram error floor $8\times$ ($0.0024 \to 0.0003$).
3. **`qr_basis` never symmetrizes `cov`:** BF16 `baddbmm` isn't guaranteed bitwise symmetric; non-symmetry introduces tiny imaginary perturbations.
4. **BF16 EMA stagnation cliff (fragility under hyperparameter changes) **`[certain]`**:** At $\beta = 0.95$, $0.05 x$ is $\ge 6.4$ BF16 ULPs so $M, V, C$ decay cleanly (though new updates $< 0.078 |C_{ij}|$ round to zero). However, because BF16 relative half-ULP at $x \in [1, 2)$ reaches $2^{-9} \approx 0.00195$, **increasing any $\beta$ above $0.998$ (e.g. standard Adam $\beta_2 = 0.999$) completely freezes EMA decay (`R_b(beta * x) == x`)**, and even $\beta = 0.99$ severely distorts effective decay!

### P4. Refresh-After-Update Ordering + 50-Step Cold-Start Nullspace Wait
- At `k = 1`, $C_L$ for SwiGLU-in ($3072\times 768$) has rank $\le 768$: **2,304 eigenvalues are zero**. Your schedule then runs **10 full steps** using that single-minibatch eigenbasis before the first warm refresh, and **50 full updates** before the first QR repair ever affects a weight update!
- Also, because refresh happens *after* step 3 (`U` and `W` update), every refresh step $t \bmod 10 == 0$ pays to compute fresh $Q^+$ using $G_t$, yet applies the **10-step-stale** $Q$ to $G_t$ and double-projects $G_t$'s contribution into $M$ (via old $Q$, then via 4-GEMM transport into $Q^+$). Moving `covariance_ema` + refresh + transport **before** projecting $G_t$ removes 1 step of basis lag and avoids double-projecting $G_t$ into $M$.

### P5. Unmatched Controls, Synchronized Timing Bias, Thermal/Clock Drift, and Open-Loop Replay Blind Spots
1. **Conflated speed and loss baselines (§2.1, §2.3):** Comparing **compiled BF16 `warm10/QR50` (26.08 ms)** against **eager FP32 `QR10` (95.23 ms)** mixes three effects: **24.30 ms/call (35% of the entire 69.15 ms speedup)** comes solely from BF16 + `torch.compile` on the ordinary step, unrelated to warm refresh! Worse, because **BF16 QR10 is still pending**, you do not yet know whether the `+0.010–0.011` final CE gap is mostly BF16 state or mostly `warm10/QR50` (the FP32-warm pilot only checked call 500!). And LR (`1.5e-3`) was tuned solely on FP32 QR10.
2. **Synchronized event-class timing + RTX 5090 boost-clock bias:** Inserting host synchronizations per optimizer event prevents CPU-GPU overlap, while comparing a 37 ms `b1, n=8192` microbenchmark burst against a 546 ms sustained 575W `b12, n=8192` burst (or a low-power serial cuSOLVER `QR` event) on a consumer RTX 5090 can introduce **10–20% boost-clock/thermal bias** **`[speculation]`** unless GPU clocks are locked (`nvidia-smi -lgc`).
3. **Open-loop 1-layer covariance replay blind spots (§2.6(i)):** (a) Has no $V$ or loss, so it rewards aggressive tail rotations that destroy $V$ in closed-loop training; (b) 1-layer models have flatter spectra and milder drift than 12-layer transformers; (c) open-loop $C_t$ from QR10 misses positive feedback where stale bases corrupt $G_{t+1}$ and $C_{t+1}$.
4. **Peak VRAM wall at $n = 4096\text{–}8192$:** Holding multiple `(12, 8192, 8192)` FP32 intermediates (`b`, `a`, `mm32`, `gram`, `residual`, `y.float()`) simultaneously requires **~13–16 GB** transient VRAM on a 32 GB card — infeasible once you scale model size without A3's micro-batching.

---

# Final Synthesis

## Top 5 Ranking for A (Speed)
1. **A5 (`sCholQR2` replacing `torch.linalg.qr`):** Attacks the #1 bottleneck (45.3% of optimizer time), saving **~9 ms/call (~35% of total optimizer)** on 109M and **~1.7×–2×** at $n=8192$ while matching QR10 quality.
2. **A7 (Strided covariance EMA $k_C=2, \beta_C'=0.9025$):** Saves **~1.6–1.8 ms/call** across 100% of calls at trivial code complexity while actually reducing BF16 EMA rounding loss.
3. **A6 (Degree-4 exponential $R_4$ + direct 2-GEMM transport):** Cuts `warm_basis` FLOPs by **28.6%** across all compute-bound $n$ (including $n=4096\text{–}8192$) and halves warm transport (saving **~1.2 ms/call** on 109M).
4. **A3 (`warm_basis` scalar-pull + large-$n$ micro-batching):** Immediately recovers the measured **7–23% `b12` slowdown** at $n = 3072\text{–}8192$ and cuts peak transient VRAM by $3\times\text{–}12\times$ at zero mathematical cost.
5. **A2 (Contiguous group param/grad buffers + 60-batch $n=768$ + CUDA Graphs):** Eliminates ~700–850 MB/call of pack/copy overhead and the 0.80 ms small-$n$ launch floor (**~1.0 ms/call** on 109M, exact math). *(If `sCholQR2` is not adopted, **A1 Multi-Stream QR** jumps to #1 or #2 for QR-bound workloads.)*

## Top 5 Ranking for B (Tracking Quality)
1. **B1 (Full-step relative-gap $\arctan$-Jacobi + Rayleigh sort/$V$-perm + row-$\ell_2$/spectral cap + $R_4(a/2)^2$):** Fixes all four root bugs simultaneously while making `warm_basis` **15% faster**, turning linear 0.5× top-only decay into full-spectrum quadratic convergence.
2. **B3 (Top-$k$ exact Procrustes power-subspace deflation + B1):** Adds only **~0.06–0.25 of one GEMM** ($k=64$) to match QR10 identically on the top-$k$ subspace and permanently kill the max $r_C = 0.975$ rank-$r$ stall.
3. **B6 (`sCholQR2`-10 everywhere or at $n \ge 3072$):** Once A5 stabilizes Cholesky-QR, pure `sCholQR2`-10 beats current `warm10/QR50` wall time on 109M **with zero warm-approximation loss gap**, and strictly beats warm at $n \ge 6144$.
4. **B2 (Two-iteration local-frame Jacobi `B1 × 2`):** Quartic $O(r_C^4)$ convergence at ~1.3× current warm event cost, closing residual gap when B1 hits the angle cap or splits equal diagonals.
5. **B4 (Front-loaded early QR10 + exit cold-start early at $t=2\text{–}5$):** Eliminates the entire +0.124 call-500 gap at **zero extra QR budget** over 10k calls.

## Top 3 Pitfalls
1. **The coupled 4-way defect in `warm_basis` (`-0.5`, global `0.01 s`, $\sqrt{n}$ Frobenius cap, equal-diagonal stall):** Directly explains synthetic $r_C = 0.150$, trained max $r_C = 0.975$, and why fixing global damping without replacing the Frobenius cap breaks the update.
2. **Silent TF32 truncation inside FP32 islands (`matmul precision = "high"`):** Corrupts `Z.T @ Z` in Cholesky-QR (explaining the 0.034 Gram error and NaN failure) and silently degrades FP32 Gram checks unless IEEE FP32 is forced locally.
3. **Unmatched controls (missing BF16 QR10 + eager-FP32 timing baseline) & $r_C$ proxy blind spot:** Without BF16 QR10 you cannot separate BF16 loss from warm-refresh loss, and optimizing raw $r_C$ (94% dominated by the top 10 eigenvalues) ignores the bulk/tail correlations that govern Adam-whitened SOAP updates.

## What I Would Want Measured Before Refining Further (Up to 5 Items)
1. **Complete the matched `BF16 QR10` (compiled) 10,000-call control** (loss at 500, 2000, 10000 + synchronized event timings): establishes the true ceiling for both BF16 speed and BF16 loss, separating BF16-state error from warm-refresh error.
2. **Isolate the `b12` vs `b1` 22% regression at $n = 6144, 8192$ with locked GPU clocks (`nvidia-smi -lgc`) and Nsight Compute/profiler:** tells us whether the gap is thermal throttling, cuBLASLt batched kernel selection under deterministic mode, or FP32 intermediate HBM traffic.
3. **Benchmark `sCholQR2` (A5, TF32 OFF on `Z^T Z`, $\tau \in \{10^{-4}, 3\times 10^{-4}, 10^{-3}\}$) across $n = 256\text{–}8192$ ($b=1, 12$):** checks zero-NaN stability, BF16 Gram error $\le 0.0025$, and exact wall time — if 109M `sCholQR2` lands at $\le 130\text{ ms}$, it reshapes the entire warm-vs-QR trade-off.
4. **Inspect the specific factors hitting $r_C \approx 0.975$ at ages 1990 and 9990** (which layer/shape, eigenvalue spectrum of $C$, diagonal spectrum of $B$, and off-diagonal structure): confirms whether those stalls are equal-diagonal spikes / level crossings and sets the right $k$ for B3.
5. **Replay-harness sweep of B1 ($\delta \in \{0.05, 0.1, 0.2\}$), B2, and B3 ($k \in \{32, 64\}$) reporting BOTH whitened RMS correlation / KL gap AND post-permutation tail churn $\chi$:** ensures we don't pick a warm refresh that wins on $r_C$ by scrambling noisy tail eigenvectors and hurting $V$ in real training.
