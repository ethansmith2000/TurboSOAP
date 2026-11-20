# P2: Gemini 4 Neon

Antigravity CLI 1.5.13 (`agy -p --model gemini-4-neon-xhigh --sandbox --output-format stream-json`), 2026-11-20, about 16 min (936 s). Prompt: `prompts/P2_gemini-4-neon-xhigh.md`, given as `prompt.md` in an empty working dir (labels it saw: X = Fable, Y = GPT-6.2-astra, Z = Gemini 4 Carbon). Tool use: 9 `read_file` calls, all on `prompt.md`, and 49 `search_web` calls; no other tools. Verbatim `result.response`. Event log: `transcripts/P2_gemini-4-neon-xhigh_events.jsonl`.

---

### 1. What changed for you

Cross-checking Document 2 against the moderator checks (**R3**) and the Part 2 trained-covariance replay (**R5**) forces concrete revisions across my Stage-1 candidates and pitfalls.

#### Contradicted
- **Pitfall P2 & A5 (TF32 "directly explains" §2.2's `0.034` Cholesky-QR Gram error and $n \ge 6144$ NaNs):** **Contradicted by M4 and superseded by R5.** M4's 10-bit TF32 Gram emulation produced only `0.0027–0.0043` Gram error at $n = 512\text{–}4096$ ($\kappa = 800\text{–}7200$), nowhere near §2.2's `0.034`. More importantly, R5 revealed that on *trained* BF16 covariances, unshifted FP32 Cholesky-QR fails in **791/800 events at $d=256$** and **800/800 at $d \ge 512$ from age 10**, because trained factors are near-null and **4–15% indefinite** in the right-side SwiGLU factor—not merely because of TF32 or $\kappa(C)^2 u_{\text{FP32}}$.
- **B1 expected single-refresh $r_C$ (`0.150 → 0.03–0.05`):** **Contradicted by M3.** On $n=256$ (input $r_C = 0.217, \rho_{\text{rms}} = 0.489$), my Stage-1 B1 (`PB3a`, as specified) achieved only $r_C = 0.121$, $\rho_{\text{rms}} = 0.341$, and top-$k$ distance $0.012$—beaten by uncapped factor-1 Jacobi (`0.052 / 0.278 / 0.004`) and the QR tangent `PB1` (`0.081 / 0.260 / 0.002`). Auditing my own Stage-1 code (lines 719–723) exposes the self-inflicted bug: `w = (0.5 / row_norm).clamp(max=1.0); a = w * a * w.T` **squares** the row damping factor on every entry between two capped rows ($w_i w_j = 0.25$ when `row_norm = 1`), compounding with `0.05 * s_ij` damping and `atan2` compression to severely under-rotate.
- **P1 / B1 magnitude of the Frobenius-cap slack under the current generator:** **Contradicted in magnitude by M2.** M2 measured pre-cap $\|A\|_F/\sqrt{2} = 0.63\text{–}1.10$ across $n = 128\text{–}1024$ with no trend in $n$ and $\|A\|_F/(\sqrt{2}\|A\|_2) = 2.5\text{–}5$ (not $\sqrt{n/2}$). That matched my own synthetic integral (`0.89`) *with* `0.01 s` damping on (see §6 for why M2 mis-tested the *un-damped* $\sqrt{n}$ claim), so on the *current* damped generator the slack is $O(1)$ rather than $O(\sqrt{n})$.

#### Weakened
- **A5 & B6 (`sCholQR2` as a near-exact, 2-pass drop-in QR10 replacement at $\tau = 10^{-3}$):** **Substantially weakened by M4 and R5.** M4 showed an operator ridge $\tau = 10^{-3} d_{\max}$ already shifts columns from Householder QR by up to **$2.6 \times 10^{-2}$** at $n=4096$ (`3.1e-3` at $\tau = 10^{-4}$). Combined with R5's 4–15% negative eigenvalues and near-null modes, 2-pass `sCholQR2` cannot be assumed safe: clearing BF16 indefiniteness requires symmetrizing $C$, using an adaptive operator ridge $\tau > |\lambda_{\min}(C)|$, and running **3-pass shifted CholeskyQR3** plus an in-graph fallback—raising cost from ~125 ms toward ~170–200 ms on 109M and potentially dulling tail rotations.
- **A6 & B1 pure $R_4$ retraction (no Newton–Schulz):** **Weakened by R5.** R5 proved `ns=0` degrades `e_Q` by $3.5\times\text{–}5\times$ (`6.5e-3–1.0e-2`), whereas **`ns=1`** achieves `0.000` delta in $r_C$ and $r_C^w$ across all widths with `e_Q = 2.11–2.23e-3` (still below QR's `2.35e-3`) while cutting $n=768$ time from `0.82 → 0.64 ms`. Self-healing `ns=1` (5 GEMMs, or 6 GEMMs with order-2 seed `I + A + ½A²`) dominates non-self-healing $R_4$.
- **B5 (Kronecker-ratio $d^+/d$ rescaling of $V$):** **Weakened by M5.** While M5 confirmed my $2\times 2$ counterexample against $T \circ T$ (`2.01` vs exact `0.01`) and confirmed exactness when $M,V,C$ are consistent, M5 showed unclamped $d^+/d$ blows up ($M/\sqrt{V} = 3.9\times 10^4$) if $V$ lags $C$. Even clamped ($[0.25, 4.0]$), $d^+/d$ can amplify spikes by $2\times$ on non-Kronecker mismatch.
- **B1 "high confidence" loss prediction (`60–80%` steady-state CE gap closed) & P3 BF16 EMA stagnation:** **Weakened by R5 and M6.** R5 showed 10-age drift dominates even exact `eigh10` (`r_C: 0.002 → 0.26–0.36` before next refresh), and BF16 QR10 is still missing, making numerical CE predictions premature. M6 showed sample noise reduces the BF16 EMA drift gap from `4.2%` to `0.6–0.8%` at $\beta_C = 0.95$.

#### Confirmed
- **Roofline & large-$n$ scaling (R4, M1):** Confirmed `warm, b12` is pinned at the RTX 5090 BF16-FP32acc ceiling (`165–174 TFLOP/s`; `207–213` at `b1`), QR runs at `0.8–22.9 TFLOP/s`, and `warm10/QR50` (`588 ms`) already loses to `QR20` (`560 ms`) at $n=8192$.
- **Weak-direction freeze & coupled damping–cap pathology (R5):** R5 confirmed 40–99% of eigenvalues sit below $0.01\lambda_{\max}$, keeping `warm10/QR50` $r_C^w$ (`0.62–0.82`) stuck near `QR50` (`0.66–0.86`). Crucially, R5's damping ablation confirmed my P1 coupling diagnosis: lowering $\delta$ from `0.01 → 0.001` *worsened* $r_C$ by `+0.012 to +0.016` at every width because activating more pairs tightened the global Frobenius cap, while removing the cap diverged (`59–82%` NaN).
- **Exact algebra (M1):** Confirmed $R_4^\top R_4 = I + A^6/72 + A^8/576$, top-$k$ Procrustes orthogonality/mapping, `high` $\Rightarrow$ `allow_tf32=True`, and `3d/4d` left factors dominating error (R5: $r_C = 0.64\text{–}0.80$).

#### Single biggest change to my view
My Stage-1 mental model assumed trained covariances behaved like the §2.2 synthetic full-rank $1/i$ spectrum, which led me to treat 2-pass `sCholQR2` (my A5/B6) as a near-certain drop-in replacement for Householder QR that could even obsolete warm refreshes, while trying to patch symmetric inverse-gap Jacobi (my B1) instead of targeting QR's own generator. **R5 and M3 overturned that thesis:** trained BF16 factors are near-null and 4–15% indefinite—killing unshifted Cholesky-QR 99–100% of the time and forcing heavy ridge/3-pass stabilization—which makes fixing the warm refresh itself essential, and M1/M3 showed the **ordered QR-tangent generator (PB1)** (which I missed completely in Stage 1) is the right first-order target for QR10 rather than patched inverse-gap Jacobi.

---

### 2. Prior work (web)

All external sources below were checked via web search only (**in search results only**; full PDFs were not opened). Details not visible in search summaries are marked `[unverified]`.

#### 2.1 Refresh frequency and cost in SOAP / Shampoo
- **FOAM (adaptive Shampoo refresh & damping):** *FOAM: Frequency and Operator-Error Based Adaptive Damping Method for Reducing Staleness-Oriented Error for Shampoo*, Kyunghun Nam & Sumyeong Ahn (ICML 2026), [arXiv:2606.02365](https://arxiv.org/abs/2606.02365) / [OpenReview](https://openreview.net/forum?id=0T8i3uXq3O) *(in search results only)*. Analyzes preconditioner staleness via convergence/stability bounds, showing numerical damping suppresses staleness error, and dynamically adjusts both damping and eigendecomposition refresh frequency from an online operator-error estimate *(exact estimator formula `[unverified]`)*. **Pooled candidates:** Establishes the core mechanism of **PB12** (adaptive refresh trigger) for Shampoo, though not inside static `torch.compile(fullgraph=True)` graphs.
- **FOAM (cheaper one-sided SOAP):** *Factored One-sided Adam-Moment for Practical and Scalable SOAP*, Jaemyung Yu, Byeongho Heo, Sangdoo Yun, & Dongyoon Han (NeurIPS 2026 / Nov 2026), [arXiv:2611.16905](https://arxiv.org/abs/2611.16905) *(in search results only)*. Drops the output-side Kronecker factor (input-only preconditioning, exact when output curvature is diagonal) and replaces SOAP's dense rotated second moment $V$ with a row-column factored estimator, cutting optimizer memory by ~2× and reaching Muon-class speed while matching SOAP/Muon loss at 124M–760M scale. **Pooled candidates:** Outside Stage 1's "no one-sided preconditioning" rule, so none of PA/PB proposes it—yet notably, in our 109M model the output-side $3d/4d$ left factors represent 87.5% of cubic FLOPs and the worst $r_C = 0.64\text{–}0.80$ (R5).
- **Basis-coordinate reparametrization & subspace updates:** *Reparametrizing Shampoo and SOAP for Subspace Basis Updates and BFloat16 Storage*, Alan Milligan, Zikun Xu, Simon Lacoste-Julien, Felix Dangel, & Wu Lin (ICML 2026 / May 2026), [arXiv:2605.26327](https://arxiv.org/abs/2605.26327) *(in search results only)*. Introduces a mathematically equivalent basis-coordinate reparametrization of Shampoo, KL-Shampoo, and SOAP preconditioners that (i) enables **subspace basis updates** (updating only a subset of basis vectors via QR while keeping the rest fixed) and (ii) prevents the performance degradation of naive **BF16 storage** *(exact update equations and $V$ handling inside the updated subspace `[unverified]`)*. **Pooled candidates:** Directly anticipates **PB5** (top-$k$ panel QR) and **PB6** (grouped block QR)—making subspace QR basis updates **already known**—and addresses the root cause of R5's BF16 $C$ indefiniteness more cleanly than **PA14/PB11** (FP16 $Q$ only).
- **Other SOAP / Shampoo / KL-Shampoo / EShampoo & schedule work:**
  - *SOAP: Improving and Stabilizing Shampoo using Adam*, Nikhil Vyas, Depen Morwani, Rosie Zhao, et al. (2024 / ICLR 2025), [arXiv:2409.11321](https://arxiv.org/abs/2409.11321) *(in search results only)* — introduces SOAP and warm-started one-step power iteration + QR (our QR10 baseline).
  - *Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner*, Runa Eschenhagen et al. (NeurIPS 2025), [arXiv:2506.03595](https://arxiv.org/abs/2506.03595), and follow-up *Clarifying Shampoo: Adapting Spectral Descent to Stochasticity and the Parameter Trajectory*, Runa Eschenhagen et al. (2026), [arXiv:2602.09314](https://arxiv.org/abs/2602.09314) *(in search results only)* — introduces **EShampoo** (decoupling eigenvalues from eigenbases without grafting) and adaptive/warm-started QR refresh criteria based on operator error (again aligning with **PB12**).
  - *Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization*, Wu Lin, Scott C. Lowe, Felix Dangel, Runa Eschenhagen, Zikun Xu, & Roger B. Grosse (2025 / ICLR 2026), [arXiv:2509.03378](https://arxiv.org/abs/2509.03378) *(in search results only)* — introduces **KL-Shampoo / KL-SOAP**, decoupling frequent eigenvalue updates from lazy/adaptive eigenbasis refreshes under the exact KL objective from §2.4/Question B.
  - *A Refresh Interval Does Not Fully Specify a Refresh Schedule: Learning-Rate-Dependent Placement in Kronecker-Preconditioned Language Model Training*, Xinyu Wang, Zhiwei Zhai, Yu Gu, Zijun Yu, Sicheng Lyu, Jingrui Tian, Lei Ding, Ziyang Song, & Xiao-Wen Chang (Oct 2026), [arXiv:2610.18695](https://arxiv.org/abs/2610.18695) *(in search results only)* — proves that at **fixed refresh budget**, placement strongly affects loss and **reverses with LR**: higher LRs favor earlier placement (directly supporting **PB7** front-loading!), while lower LRs favor later placement (warning against single-LR schedule tuning).
  - *4-bit Shampoo for Memory-Efficient Network Training*, Sike Wang et al. (NeurIPS 2024), [arXiv:2405.18144](https://arxiv.org/abs/2405.18144) *(in search results only)* — establishes that quantizing eigenvectors $Q$ with orthogonality rectification vastly outperforms quantizing raw $C$ (consistent with R5's BF16 $C$ indefiniteness).
  - *Convergence Rate Analysis of SOAP with Arbitrary Orthogonal Projection Matrices* (Apr 2026), [arXiv:2604.14816](https://arxiv.org/abs/2604.14816), and *Convergence of Rotation-based Matrix Optimizers: A Unified Analysis of SOAP, Conda, and SPlus* (Sep 2026), [arXiv:2609.24522](https://arxiv.org/abs/2609.24522) *(in search results only)* — establish SOAP convergence for arbitrary approximate orthogonal bases **provided $Q_t$ is conditionally independent of $G_t$**, which theoretically favors approximate warm refreshes while **flagging PB10** (refresh-before-update breaks conditional independence).

#### 2.2 CholeskyQR2 / shifted CholeskyQR3
- **Core theory & shift effect on $Q$:** *Roundoff error analysis of the CholeskyQR2 algorithm*, Yusaku Yamamoto, Yuji Nakatsukasa, Yuka Yanagisawa, & Takeshi Fukaya (2015), *ETNA* 44:306–326, [pdf](https://etna.math.kent.edu/vol.44.2015/pp306-326.dir/pp306-326.pdf) *(in search results only)*, proves CholeskyQR2 achieves $O(u)$ orthogonality/residual when $\kappa_2(X) \lesssim u^{-1/2}$ ($\approx 4100$ in FP32). *Shifted Cholesky QR for Computing the QR Factorization of Ill-Conditioned Matrices*, Takeshi Fukaya, Ramaseshan Kannan, Yuji Nakatsukasa, Yusaku Yamamoto, & Yuka Yanagisawa (2020), *SIAM J. Sci. Comput.* 42(1):A477–A503, [doi:10.1137/18M1218212](https://doi.org/10.1137/18M1218212) *(in search results only)*, extends this to $\kappa_2(X) \lesssim u^{-1}$ by adding a Gram shift $s \approx 11(mn + n(n+1))u\|X\|_2^2$ in Pass 1 ($X^\top X + sI = R_1^\top R_1$, $X_1 = X R_1^{-1}$) followed by unshifted CholeskyQR2 on $X_1$. *An improved Shifted CholeskyQR based on columns*, Yuwei Fan, Haoran Guan, & Zhonghua Qiao (2024/2025), *J. Sci. Comput.* / [arXiv:2408.06311](https://arxiv.org/abs/2408.06311) *(in search results only)*, tightens $s$ using column norms $[X]_g$ to avoid over-shifting.
  - **Crucial distinction on $Q$:** For **full-rank** $X$, Fukaya's Gram shift $sI$ in Pass 1 **does not alter exact $Q$**, because $R_1$ is upper triangular so $X_1 = X R_1^{-1}$ preserves every nested column subspace $\operatorname{span}(x_1,\dots,x_k)$ and Passes 2–3 recover $X = Q(R_3 R_2 R_1)$ to $O(u)$. By contrast, adding an **operator ridge** $Z_\tau = (C + \tau I)Q$ alters the operator itself and changes $Q$ on modes $d_i \lesssim \tau$ (confirmed in M4).
- **Rank-deficient and indefinite inputs:** The literature establishes that **CholeskyQR2 and Shifted CholeskyQR3 require full column rank**—they are **already shown to fail on rank-deficient inputs**. While a Gram shift $+sI$ absorbs negative roundoff eigenvalues in $X^\top X$ on Pass 1, $X_1 = X R_1^{-1}$ remains rank-deficient so Pass 2's unshifted Cholesky still crashes (and shifting Pass 2 leaves non-orthogonal columns $Q^\top Q = I - s R^{-\top} R^{-1}$). Handling rank deficiency requires column pivoting, randomized sketching, or LU preconditioning: e.g., *CholeskyQR with Randomization and Pivoting for Tall Matrices (CQRRPT)*, Maksim Melnichenko, Oleg Balabanov, Riley Murray, James Demmel, Michael W. Mahoney, & Piotr Luszczek (2023), [arXiv:2311.08316](https://arxiv.org/abs/2311.08316), and *A new randomized CholeskyQR based on LU decomposition with partial pivoting (RCLUPP)*, Yuwei Fan, Haoran Guan, & Zhonghua Qiao (2024), [arXiv:2412.06551](https://arxiv.org/abs/2412.06551) *(both in search results only)*.
- **Mixed-precision variants:** *Mixed-precision Cholesky QR factorization and its case studies on multicore CPU with multiple GPUs*, Ichitaro Yamazaki, Stanimire Tomov, & Jack Dongarra (2015) *(in search results only; arXiv ID `[unverified]`)* shows accumulating $X^\top X$ in doubled precision cuts orthogonality error from $O(\kappa^2 u)$ to $O(\kappa u)$. *Numerically Stable Cholesky-QR via Mixed-Precision Randomized Preconditioning (MRCQR)*, James E. Garrison, Chao Chen, & Ilse C. F. Ipsen (June 2026), [arXiv:2606.18411](https://arxiv.org/abs/2606.18411) *(in search results only)*, proves low-precision randomized preconditioners work up to $\kappa \lesssim 10^8$ (FP32) or $10^4$ (FP16) before higher-precision CholeskyQR. *Reduced and Mixed-Precision Block Householder QR Factorization (`mpWYQR`)* (Oct 2026), [arXiv:2610.38537](https://arxiv.org/abs/2610.38537) *(in search results only)*, accelerates WY-blocked Householder QR via mixed-precision trailing GEMMs.
- **Pooled candidates:** **PA1/PB4**'s shifted/column-equilibrated CholeskyQR is **already known**, and the NLA literature confirms that **Gram-only shifting fails on R5's rank-deficient factors**—making the operator ridge $(C + \tau I)Q$ (which lifts both rank deficiency and BF16 indefiniteness, at the cost of damping small modes) structurally necessary. **PA4** aligns with `mpWYQR`.

#### 2.3 First-order (QR-tangent) and Jacobi-type tracking of eigenbases under drift
- **Classical foundations:** Orthogonal/subspace iteration with Rayleigh–Ritz projection ($Z = CQ$, QR, $B = Q^\top C Q$) is established by G. W. Stewart (*Accelerating the Orthogonal Iteration for the Eigenvalues of a Hermitian Matrix*, *Numer. Math.* 1969; *Simultaneous Iteration for Computing Invariant Subspaces of Non-Hermitian Matrices*, *Numer. Math.* 1976, *in search results only*), converging at $|\lambda_{k+1}/\lambda_k|$ per step. Tracking drifting subspaces via QR + lightweight Jacobi sweeps is established by Pierre Comon & Gene H. Golub (*Tracking a few extreme singular values and vectors in signal processing*, *Proc. IEEE* 1990) and Marc Moonen, Paul Van Dooren, & Joos Vandewalle (*SIAM J. Matrix Anal. Appl.* 1990–1993) *(both in search results only)*; parallel/block Jacobi orderings by Richard P. Brent & Franklin T. Luk (*SIAM J. Sci. Stat. Comput.* 1985, *in search results only*); skew-symmetric tangent/geodesic updates $Q\exp(A)$ on orthogonal manifolds by Alan Edelman, Tomás A. Arias, & Steven T. Smith (*The Geometry of Algorithms with Orthogonality Constraints*, *SIAM J. Matrix Anal. Appl.* 1998, *in search results only*); and first-order QR perturbation $K_{ij} = E_{ij}/d_j$ ($i>j$) by Rajendra Bhatia (*Matrix Analysis*, 1997, *in search results only*).
- **Pooled candidates:** While the perturbation identity is classical, using the **explicit first-order ordered QR-tangent $K^{\text{QR}}$ + BF16 Tensor-Core retraction (PB1)** as a decomposition-free surrogate for SOAP's power-QR refresh appears **new** in the optimizer literature `[unverified beyond search results]`. **PB3** (simultaneous damped Jacobi) and **PB5/PB6** (panel/block subspace updates) are adaptations of known NLA methods (with PB5/PB6 overlapping Milligan et al. 2026).

#### 2.4 Batched or concurrent small QR on GPUs (cuSOLVER, MAGMA) at $n = 768\text{–}3072$, batch 12
- **NVIDIA & PyTorch facts:** Per NVIDIA's [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) and [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html) docs and PyTorch's `BatchLinearAlgebra.cpp` / `CUDABlas.cpp` *(all in search results only)*, `cublas<t>geqrfBatched` exists in cuBLAS (dispatched by PyTorch only for small $n \le 256$), whereas **cuSOLVER lacks a native batched `orgqr`**, causing PyTorch's `torch.linalg.qr` at $n = 768\text{–}3072$ to execute a **serial C++ batch loop** over `cusolverDnSgeqrf` + `cusolverDnSorgqr`. Crucially, PyTorch's `torch.linalg.qr(..., mode='r')` **skips `orgqr` completely** *(confirmed in search results only)*.
- **MAGMA & achievable performance:** Azzam Haidar, Stanimire Tomov, Ahmad Abdelfattah, & Jack Dongarra (ICL/UTK, *in search results only*) show GPU-resident blocked `magma_sgeqrf_batched` excels up to $n = 512\text{–}1024$ for large batches (hundreds of GFLOP/s to $>1$ TFLOP/s on prior-gen GPUs), whereas at $n = 2048\text{–}3072$ and tiny batch $b = 12$, batched kernels hit register/shared-memory/L2 limits and **multi-stream non-batched `cusolverDnXgeqrf`** is recommended *(exact RTX 5090 `b=12` GFLOP/s numbers `[unverified]`)*. Per the NVIDIA RTX Blackwell Architecture Whitepaper *(in search results only)*, RTX 5090 (GB202, 170 SMs) peaks at **104.8 TFLOP/s** FP32/TF32, **209.5 TFLOP/s** BF16 (FP32 acc), and **419 TFLOP/s** FP16 (FP16 acc).
- **Pooled candidates:** **PA2** (multi-stream QR), **PA3** (`mode='r'` + `trsm` + NS), and **PA4** (blocked batched Householder) are **already known** GPU-DLA techniques; at $b=12$, PA2+PA3 captures most of what is achievable without writing custom CUDA/MAGMA kernels.

#### 2.5 Handling Adam's second moment when the basis changes (PB8)
- Vyas et al. ([arXiv:2409.11321](https://arxiv.org/abs/2409.11321)) only permutes $V$ (no dense rotation); Yu et al. ([arXiv:2611.16905](https://arxiv.org/abs/2611.16905)) replaces dense $V$ with a factored row/column estimator; Lin et al. ([arXiv:2509.03378](https://arxiv.org/abs/2509.03378)) updates eigenvalues directly via KL minimization; Milligan et al. ([arXiv:2605.26327](https://arxiv.org/abs/2605.26327)) leaves untouched subspace coordinates invariant; and 2026 convergence proofs ([arXiv:2604.14816](https://arxiv.org/abs/2604.14816), [arXiv:2609.24522](https://arxiv.org/abs/2609.24522)) require no dense $V$-mixing *(all in search results only)*.
- **Pooled candidates:** Neither **PB8a** ($T \circ T$ mixing) nor **PB8b** ($d^+/d$ rescaling) is established as beneficial in the published SOAP literature `[unverified beyond search results]`, and M5's counterexamples explain why SOAP and its 2026 successors avoid dense $V$-transport.

---

### 3. Ranking

Recall the 109M time split (`26.08 ms/call`: **45.3% QR increment**, **38.9% ordinary**, **15.8% warm increment**), the R4 roofline (`warm` pinned at 165–213 TFLOP/s for $n \ge 2048$; `warm/QR = 0.41` at $n=8192$), and the five R5 facts (near-null and 4–15% indefinite trained BF16 factors; 99–100% failure of unshifted Cholesky-QR; uncapped warm diverges yet surviving ages beat QR10; warm barely moves weak directions; `QR20` beats `warm10/QR50` per refresh but not per ms).

#### Speed candidates (PA1–PA14)
1. **PA1 (Stabilized Cholesky-QR for QR events):** Highest potential speedup (−8 to −9 ms/call on the 45.3% QR share, plus large-$n$ scaling), **but** R5's 4–15% indefiniteness and near-null modes require `sym(C)` + adaptive operator ridge + IEEE-FP32 3-pass `sCholQR3` + fallback; drops below PA2/PA3 if the ridge required for 0% fallback degrades $r_C^w$.
2. **PA2 (Concurrent QR over CUDA streams):** Bit-exact and immune to R5's indefiniteness/rank-deficiency, attacking the 0.8–7.5 TFLOP/s serial cuSOLVER loop across all 96 factors (especially 154 ms on $n=768$ and 81 ms on $n=1536$); stacks multiplicatively with PA3.
3. **PA3 (R-only Householder `mode='r'` + ridged `solve_triangular` + 1 NS):** Skips the entire serial `orgqr` loop (~40% of QR time) in pure PyTorch with $O(\kappa u)$ rather than $O(\kappa^2 u)$ conditioning, needing only a $10\times$ smaller ridge than PA1 to survive R5's rank-deficient factors.
4. **PA6 (Fewer warm products — specifically PA6a `ns=1` [5 GEMMs] or PA6b `I+A+½A²` + 1 NS [6 GEMMs]):** R5 already proved `ns=1` gives `0.000` quality delta with `e_Q = 2.11–2.23e-3` (< QR's `2.35e-3`) for −22% to −29% warm time, and R4 proved only fewer products help warm at $n \ge 2048$. *(R6 sensitivity: if R6 shows PB1/Jacobi-1 requires cap $\theta_{\max} \ge 0.6$, use 6-GEMM PA6b instead of 5-GEMM PA6a, halving its FLOP cut.)*
5. **PA12 (Overlap refresh/transport with next Fwd+Bwd):** Exact math; genuinely hides a large fraction of low-occupancy QR (0.8–7.5 TFLOP/s) behind the 283 ms Fwd+Bwd, though overlapping Tensor-Core-saturated warm (165–174 TFLOP/s) mostly shifts time into Fwd+Bwd stretch; incompatible with PB10.
6. **PA9 (Contiguous param/grad banks + batch-60 $n=768$ coalescing + CUDA graphs):** Exact math targeting the 38.9% ordinary base (~0.8–1.0 ms/call on 100% of steps) plus collapsing five 0.82 ms latency-bound $n=768$ warm calls into one.
7. **PA11 (Strided covariance EMA $k_C=2, \beta_C^2=0.9025$):** Trivial code saving ~1.2–1.8 ms/call on 100% of steps while halving the BF16 EMA dead-band, though untestable in open-loop replay and requires a 2k-call loss check.
8. **PA10 (Fused `baddbmm` epilogues + large-$n$ sub-batching):** Exact math that fixes the 22–23% `b12` vs `b1` regression at $n=6144\text{–}8192$ and cuts peak transient VRAM $3\times\text{–}12\times$, with smaller (~0.3–0.4 ms/call) 109M impact.
9. **PA8 (Symmetric `syrk` Triton kernels for $GG^\top, G^\top G, Y^\top Y$):** Cuts ~17% of ordinary and ~14% of warm FLOPs (~1.0–1.6 ms/call) and enforces exact $C$ symmetry, but carries high Blackwell kernel-tuning risk against cuBLASLt.
10. **PA13 (Size-aware crossover: drop warm at $n \ge 6144$ for PA1):** Essential at $n=8192$ where `warm10/QR50` (`588 ms`) already loses to `QR20` (`560 ms`), but zero effect on 109M ($n \le 3072$) and contingent on PA1 surviving trained indefiniteness. *(R6 sensitivity: if R6 + PA6a brings 5-GEMM warm close to QR10 quality, the crossover shifts above $n \approx 10\text{k}$, lowering PA13's urgency.)*
11. **PA4 (Custom batched, blocked Householder QR / `mpWYQR`):** High ceiling and immune to R5 indefiniteness, but heavy custom CUDA engineering only justified if both PA1 and PA2+PA3 fail to achieve $\ge 2\times$ QR speedup.
12. **PA5 (`mm32` for $Z=CQ$ inside QR):** Exact 1-line win, but worth only ~0.10 ms/call standalone on 109M; fold directly into PA1/PA2/PA3.
13. **PA7 (Cheaper $M$ transport: $R_L^\top M R_R$ or short-side reassociation):** Entire transport is only 0.51 ms/call, so 8–50% savings yield only 0.04–0.20 ms/call; take as a freebie if $R$ is already formed, not as a primary optimization.
14. **PA14 (FP16 basis storage + FP16-acc back-projection $U$):** R5 already shows BF16 warm `e_Q = 1.85e-3 < QR 2.35e-3`, while FP16 accumulation injects 2.7–4.4% noise into $U$ and cast traffic offsets part of the 419 TFLOP/s gain.

#### Tracking candidates (PB1–PB12)
1. **PB1 (Ordered QR-tangent generator $K_{ij} = B_{ij}\operatorname{sgn}(d_j - d_i)/\max(d_i, d_j, \rho s)$):** Matches QR to first order (M1), eliminates the $\frac{1}{2}$ factor and equal-diagonal stall, unfreezes weak directions ($r_C^w$), and won M3 on chained $\rho_{\text{rms}}$ (`0.243` vs QR10 `0.240`) at $\le 1\times$ warm cost—provided negative BF16 $d_j$ are clamped at 0 before flooring and PB2 caps large steps. *(R6 sensitivity: if R6 shows PB1 overshoots or stalls on the $r_C = 0.64\text{–}0.80$ left factors, PB5 or PB4 moves to #1; if relative-gap/factor-1 Jacobi beats PB1 on both $r_C$ and $r_C^w$ in R6, PB3 moves to #1.)*
2. **PB2 (Spectral / row-$\ell_2$ cap instead of Frobenius cap):** Mandatory partner to PB1/PB3: R5 proved uncapped steps diverge (59–82% NaN) while the Frobenius cap induces the damping–cap trade-off (lowering $\delta \to 0.001$ hurt $r_C$ by `+0.014`); a 4-step block power norm costs negligible $O(n^2)$ FLOPs and decouples head from tail. *(R6 sensitivity: if R6 tests PB1/Jacobi under the old Frobenius cap and sees weak gains, PB2 immediately explains and unlocks them.)*
3. **PB5 (Head-block / panel power step on top $k$ columns + orthogonal tail completion):** Directly solves R5's worst regime ($3d/4d$ left factors at $r_C = 0.64\text{–}0.80$, outside first-order linearity) while remaining immune to tail indefiniteness/rank-deficiency for only $+3k/n$ of a GEMM, and aligns with Milligan et al. (2026); best paired with PB1 on the remaining bulk. *(R6 sensitivity: jumps to #1 if R6 shows first-order generators leave $3d/4d$ left factors $> 0.10$ above QR10; drops to #5 if PB1 already closes them.)*
4. **PB4 (Exact QR at every refresh via PA1, no warm step):** Replicates QR10 by construction inside the budget at $n \ge 1536$, but downgraded from my Stage-1 view because R5's 99–100% failure of unshifted Cholesky-QR forces an operator ridge (perturbing $Q$ per M4) and 3-pass cost. *(R6 sensitivity: jumps to #1–#2 if all three R6 warm generators fail to close $\ge 50\%$ of the QR50→QR10 $r_C^w$ gap.)*
5. **PB3 (Full-gain relative-gap Jacobi variants — PB3c or fixed PB3a without `w^2` bug):** M3 showed factor-1 Jacobi beat PB1 on raw $r_C$ (`0.052` vs `0.081` one-shot; `0.0742` vs `0.0767` chained), though still vulnerable to equal-diagonal stalls and gap blowups. *(R6 sensitivity: directly under test in R6; swaps with PB1 if relative-gap or capped factor-1 Jacobi beats PB1 on trained $r_C^w$ with lower churn.)*
6. **PB7 (Front-loaded schedule: QR10 for ages $\le 300\text{–}500$, then warm10/QR50 or QR100):** Eliminates the $+0.124$ call-500 peak-LR gap for $\approx 0\text{–}1.3\text{ ms/call}$ amortized, directly supported by Wang et al. ([arXiv:2610.18695](https://arxiv.org/abs/2610.18695)); complementary to PB1/PB5 because it does not fix late-horizon drift.
7. **PB9 (Sort columns at warm refreshes and permute $V$):** Tiny cost (0.38 ms/event) that maintains QR10's descending invariant across warm intervals and is needed for PB5's top-$k$ panel and PB1's tie-breaking.
8. **PB6 (Grouped block QR rotations):** Only $1\times n^3$ GEMM (attractive at $n \ge 4096$) and immune to small gaps inside blocks, but drops cross-block head-to-tail couplings on dense $3d/4d$ factors unless interleaved over multiple sweeps.
9. **PB12 (Adaptive QR trigger):** Supported by Nam & Ahn (2026) and Eschenhagen et al. (2025), and R5 shows huge heterogeneity ($3d/4d$ at $r_C = 0.64\text{–}0.80$ vs $d/2d$ at $0.13\text{–}0.29$), **but** dynamic triggers break `fullgraph=True`; a static *per-shape* split (e.g., QR20 on $3d/4d$ left factors, warm10/QR50 on $d/2d$) captures most of the benefit graph-cleanly.
10. **PB8 ($V$ after refresh — $T\circ T$ mixing or $d^+/d$ rescaling):** M5 proved both have sharp failure modes (double-counting leaked variance vs stale-$C$ spike), while permutation-only causes no spike under $\beta_1=\beta_2=\beta_C=0.95$; defer until matched BF16 QR10 isolates any residual $V$-lag loss.
11. **PB11 (Basis storage precision: FP16 $Q$ or FP32 master):** R5 showed warm BF16 `e_Q = 1.85e-3` already beats QR's `2.35e-3` and M2 showed BF16 warm matches float64 to 3 digits; the real BF16 precision trap is $C$, not $Q$.
12. **PB10 (Refresh before update):** Alters the target algorithm, blocks PA12 overlap, and breaks the conditional independence of $Q_t$ from $G_t$ assumed in 2026 SOAP convergence theory ([arXiv:2604.14816](https://arxiv.org/abs/2604.14816)).

---

### 4. Your top 3 experiments

Because R6 is already testing basic warm generators in replay while **no** QR-speed replacement has yet been validated on R5's near-null/indefinite trained factors, my top 3 experiments test: (1) the complete indefiniteness-safe, spectrally-capped 6-GEMM QR-tangent warm refresh, (2) adaptive-ridge 3-pass `sCholQR3` as a QR replacement, and (3) multi-stream R-only Householder + ridged `trsm` + 1 NS as the Householder-stable QR speedup.

#### Experiment 1: Indefiniteness-Safe Spectrally-Capped QR-Tangent + Order-2 Seed & 1 NS (`PB1 + PB2 + PA6b`)

1. **Exact spec (34 lines, 6 BF16 GEMMs vs 7 in production):**
```python
def warm_qr_tangent_o2(C_bf16, Q_bf16, rho=1e-3, theta_cap=0.65):
    # C_bf16, Q_bf16: [b, n, n] BF16 -> returns (Q_new_bf16, order)
    # Sweep rho in {1e-4, 1e-3} and theta_cap in {0.50, 0.65}.
    b, n, _ = Q_bf16.shape
    dev = Q_bf16.device
    C_sym = (0.5 * (C_bf16.float() + C_bf16.float().transpose(-1, -2))).to(torch.bfloat16)
    B = torch.bmm(torch.bmm(Q_bf16.transpose(-1, -2), C_sym), Q_bf16, out_dtype=torch.float32)
    B = 0.5 * (B + B.transpose(-1, -2))
    d = B.diagonal(dim1=-2, dim2=-1)
    order = d.argsort(dim=-1, descending=True)
    Q_ord = Q_bf16.gather(-1, order.unsqueeze(-2).expand_as(Q_bf16))
    B_ord = B.gather(-2, order.unsqueeze(-1).expand_as(B)).gather(-1, order.unsqueeze(-2).expand_as(B))
    d_ord = B_ord.diagonal(dim1=-2, dim2=-1)
    s = d_ord.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)          # [b, 1]
    denom = d_ord.clamp_min(0.0).clamp_min(rho * s).unsqueeze(-2)        # [b, 1, n]; guards negative BF16 d_j
    L = (B_ord / denom).tril(-1).clamp(-1.0, 1.0)                        # +B_ij / d_j for i > j (exact QR sign)
    A = L - L.transpose(-1, -2)                                          # skew-symmetric QR tangent
    # 4-step block power iteration (r=4 deterministic DCT probe) for ||A||_2:
    i_idx = torch.arange(n, device=dev, dtype=torch.float32).unsqueeze(-1)
    k_idx = torch.arange(1, 5, device=dev, dtype=torch.float32).unsqueeze(0)
    V = torch.cos(3.141592653589793 * (i_idx + 0.5) * k_idx / n)         # [n, 4]
    V = (V / V.norm(dim=0, keepdim=True)).unsqueeze(0).expand(b, n, 4)
    for _ in range(4):
        W = torch.bmm(A, torch.bmm(A, V))                                # -(A^T A) V
        V = W / W.norm(dim=-2, keepdim=True).clamp_min(1e-20)
    sigma = torch.bmm(A, V).norm(dim=-2).amax(dim=-1, keepdim=True).unsqueeze(-1)
    scale = (theta_cap / (1.15 * sigma).clamp_min(1e-12)).clamp(max=1.0)
    At = (scale * A).to(torch.bfloat16)
    # Order-2 exponential seed (error 0.25*theta^4 <= 0.045) + 1 NS polish (error <= 0.0015):
    A2 = torch.bmm(At, At, out_dtype=torch.float32)                      # GEMM 3
    K = (At.float() + 0.5 * A2).to(torch.bfloat16)
    Y0 = (Q_ord.float() + torch.bmm(Q_ord, K, out_dtype=torch.float32)).to(torch.bfloat16)  # GEMM 4
    G0 = torch.bmm(Y0.transpose(-1, -2), Y0, out_dtype=torch.float32)    # GEMM 5
    G0.diagonal(dim1=-2, dim2=-1).sub_(1.0)
    Y1 = (Y0.float() - 0.5 * torch.bmm(Y0, G0.to(torch.bfloat16), out_dtype=torch.float32)).to(torch.bfloat16)  # GEMM 6
    return Y1.contiguous(), order
```
2. **Where it runs:** Both (i) **replay harness** (`warm10/QR50`, ages 0–1000, $d \in \{256, 512, 1024, 2048\}$; also report 3d/4d left factors separately) and (ii) **microbenchmark** (compiled `ms/factor`, $b \in \{1, 12\}$, $n = 256\text{–}8192$, plus 109M warm event).
3. **Decision metric and threshold:** Late-window $r_C$ interval and $r_C^w$ interval each close **$\ge 60\%$ of the `QR50 → QR10` gap** at all four widths (e.g., at $d=2048$, $r_C \le 0.363$ and $r_C^w \le 0.695$), top-$n/8$ overlap $\ge 0.85$ at all widths, `e_Q <= 2.5e-3`, zero NaNs, AND compiled time $\le 0.95\times$ current warm at $n=3072$ (`<= 2.34 ms/factor`).
4. **Probability of passing:** **65%** (high confidence on speed, stability, and $d/2d$ factors; primary risk is whether a single capped first-order tangent closes $\ge 60\%$ on the high-drift $3d/4d$ left factors where $r_C = 0.64\text{–}0.80$).

#### Experiment 2: Adaptive-Ridge Column-Equilibrated Shifted CholeskyQR3 (`PA1 + PB4`)

1. **Exact spec (34 lines, 1 BF16 GEMM + 3 IEEE-FP32 CholeskyQR passes + in-graph fallback):**
```python
def scholqr3_ridge(C_bf16, Q_bf16, rho=1e-3):
    # C_bf16, Q_bf16: [b, n, n] BF16 -> returns (Q_new_bf16, order)
    # Sweep rho in {3e-4, 1e-3, 3e-3}. Note: inside torch.compile, enforce IEEE FP32
    # on the 3 Gram bmm calls via a custom op or fp32_precision="ieee" outside the graph.
    b, n, _ = Q_bf16.shape
    C_sym = (0.5 * (C_bf16.float() + C_bf16.float().transpose(-1, -2))).to(torch.bfloat16)
    Z = torch.bmm(C_sym, Q_bf16, out_dtype=torch.float32)                # 1 BF16->FP32 GEMM
    Qf = Q_bf16.float()
    d = (Qf * Z).sum(dim=-2)                                             # [b, n] Rayleigh quotients
    order = d.argsort(dim=-1, descending=True)
    s = d.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)              # [b, 1]
    neg_floor = (-d.amin(dim=-1, keepdim=True)).clamp_min(0.0)           # clears observed BF16 indefiniteness
    tau = torch.maximum(rho * s, 2.0 * neg_floor).unsqueeze(-1)          # [b, 1, 1]
    Z = (Z + tau * Qf).gather(-1, order.unsqueeze(-2).expand_as(Z))      # (C_sym + tau*I) Q_ord
    Z = Z / Z.norm(dim=-2, keepdim=True).clamp_min(1e-30)                # column equilibration (||Z||_F^2 = n)
    eye = torch.eye(n, device=Z.device, dtype=torch.float32).expand(b, n, n)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ok = torch.ones(b, dtype=torch.bool, device=Z.device)
        Y = Z
        for sh in (2e-5 * n, 1e-7 * n, 0.0):                             # shifted Pass 1, safety Pass 2, exact Pass 3
            G = torch.bmm(Y.transpose(-1, -2), Y)                        # IEEE FP32 Gram
            if sh > 0.0:
                G.diagonal(dim1=-2, dim2=-1).add_(sh)
            U, info = torch.linalg.cholesky_ex(G, upper=True, check_errors=False)
            ok = ok & (info == 0)
            U_safe = torch.where(ok.view(b, 1, 1), U, eye)
            Y = torch.linalg.solve_triangular(U_safe, Y, upper=True, left=False)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    Q_ord = Q_bf16.gather(-1, order.unsqueeze(-2).expand_as(Q_bf16))
    Q_new = torch.where(ok.view(b, 1, 1), Y.to(torch.bfloat16), Q_ord)   # zero-sync fallback
    return Q_new.contiguous(), order
```
*(Diagnostic note: also log `fallback_rate = (~ok).float().mean()`. If `2.0 * neg_floor` misses unaligned negative eigenvalues on SwiGLU-right, add a 3-step power iteration on $sI - C_{\text{sym}}$ in $6n^2$ FLOPs to bound $-\lambda_{\min}(C_{\text{sym}})$ directly.)*
2. **Where it runs:** Both (i) **replay harness** chained every 10 ages (no Householder QR repair, directly testing PB4 vs `QR10`) AND as the QR50 repair inside `warm10/QR50`, and (ii) **microbenchmark** ($n = 256\text{–}8192$, $b \in \{1, 12\}$, plus 109M QR event).
3. **Decision metric and threshold:** In replay as a QR10 replacement: **0.0% NaN / `< 0.5%` fallback rate** across all 800 events per width, `e_Q <= 2.5e-3`, and late-window $r_C$, $r_C^w$ intervals within **`+0.015`** of Householder `QR10` (top-$n/8$ overlap within `-0.015`) at all widths; in microbenchmark: 109M 96-factor QR event **`<= 220 ms`** ($\ge 2.7\times$ faster than 593.7 ms) and $n=8192$ `<= 105 ms/factor` with zero NaNs.
4. **Probability of passing:** **55%** (high confidence on speed and zero NaNs, real risk that the $\tau$ required to suppress 4–15% negative eigenvalues in SwiGLU-right pushes $r_C^w$ $> 0.015$ above QR10).

#### Experiment 3: Multi-Stream R-Only Householder QR + Ridged `trsm` + 1 NS (`PA2 + PA3`)

1. **Exact spec (34 lines, skips `orgqr` completely, $O(\kappa u)$ conditioning, $K \in \{1, 4, 8\}$ CUDA streams):**
```python
def multistream_ronly_qr(C_bf16, Q_bf16, streams, rho=1e-4):
    # C_bf16, Q_bf16: [b, n, n] BF16; streams: list of K pre-allocated CUDA streams (K in {1, 4, 8})
    # Returns (Q_new_bf16, order). Also time K=1 vs full qr(..., mode='reduced') to measure geqrf/orgqr split.
    b, n, _ = Q_bf16.shape
    C_sym = (0.5 * (C_bf16.float() + C_bf16.float().transpose(-1, -2))).to(torch.bfloat16)
    Z = torch.bmm(C_sym, Q_bf16, out_dtype=torch.float32)                # 1 BF16->FP32 GEMM
    Qf = Q_bf16.float()
    d = (Qf * Z).sum(dim=-2)
    order = d.argsort(dim=-1, descending=True)
    s = d.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    neg_floor = (-d.amin(dim=-1, keepdim=True)).clamp_min(0.0)
    tau = torch.maximum(rho * s, 2.0 * neg_floor).unsqueeze(-1)          # 10x smaller ridge than CholQR (kappa*u)
    Z_ord = (Z + tau * Qf).gather(-1, order.unsqueeze(-2).expand_as(Z)).contiguous()
    cur = torch.cuda.current_stream(Z_ord.device)
    R_list = [None] * b
    K = len(streams)
    for st in streams:
        st.wait_stream(cur)
    for i in range(b):
        st = streams[i % K]
        with torch.cuda.stream(st):
            R_list[i] = torch.linalg.qr(Z_ord[i], mode='r').R            # geqrf only; bypasses orgqr
    for st in streams:
        cur.wait_stream(st)
    R = torch.stack(R_list, dim=0)
    r_diag = R.diagonal(dim1=-2, dim2=-1)
    r_floor = (1e-6 * s).expand_as(r_diag)
    r_safe = torch.where(r_diag.abs() < r_floor, torch.where(r_diag >= 0, r_floor, -r_floor), r_diag)
    R.diagonal(dim1=-2, dim2=-1).copy_(r_safe)                           # preserves Householder sign convention
    Y = torch.linalg.solve_triangular(R, Z_ord, upper=True, left=False).to(torch.bfloat16)
    G = torch.bmm(Y.transpose(-1, -2), Y, out_dtype=torch.float32)       # 1 BF16 NS polish (2 GEMMs)
    G.diagonal(dim1=-2, dim2=-1).sub_(1.0)
    Y = (Y.float() - 0.5 * torch.bmm(Y, G.to(torch.bfloat16), out_dtype=torch.float32)).to(torch.bfloat16)
    return Y.contiguous(), order
```
2. **Where it runs:** Both (i) **microbenchmark** ($K \in \{1, 4, 8\}$ at $n = 256\text{–}8192$ and the 109M 96-factor QR event, also recording `geqrf` vs `orgqr` breakdown at $K=1$) and (ii) **replay harness** (chained every 10 ages across $d=256\text{–}2048$ to verify exact QR10 fidelity under $\rho = 10^{-4}$).
3. **Decision metric and threshold:** 109M 96-factor QR time **`<= 260 ms`** ($\ge 2.3\times$ speedup over 593.7 ms, saving $\ge 6.7\text{ ms/call}$), AND replay $r_C, r_C^w$ intervals within **`+0.005`** of Householder `QR10` with `e_Q <= 2.4e-3` and zero NaNs across all widths.
4. **Probability of passing:** **75%** (very high confidence on replay accuracy and $\ge 1.6\times$ from `mode='r'` alone; main uncertainty is whether cuSOLVER under `use_deterministic_algorithms(True)` serializes stream launches).

---

### 5. Disputes (D1–D6) and your strongest disagreement

#### Updated positions on D1–D6
- **D1 (Target generator):** **PB1 (ordered QR-tangent) with PB2 spectral cap and non-negative $d_j$ floor**, not full-gain Jacobi—M1/M3 confirmed PB1 matches QR to first order without gap singularities or equal-diagonal stalls and won on chained $\rho_{\text{rms}}$ (`0.243` vs `0.253`/`0.254` vs QR10 `0.240`), though R6 must check whether PB1 stays controlled on trained $3d/4d$ factors ($r_C \approx 0.7$).
- **D2 ($V$ after a refresh):** **Leave $V$ unchanged (permute only, matching QR10) for now**—M5 showed both $T \circ T$ (doubles leaked variance during alignment) and unclamped $d^+/d$ rescaling ($3.9\times 10^4$ spike when $V$ lags $C$) fail on realistic regimes, whereas consistent $M,V,C$ ($\beta_1=\beta_2=\beta_C=0.95$) produce no spike under leave-unchanged.
- **D3 (Why Cholesky-QR failed):** **Neither my TF32-only claim nor $\kappa(C)^2 u_{\text{FP32}}$ alone explains everything:** M4 refuted TF32 as the sole cause of §2.2's `0.034` (`0.0043` emulated) and showed unshifted FP32 survives $\kappa(C)=7200$ via column-scale invariance; on trained factors (R5), the dominant failure is **near-null directions and 4–15% negative BF16 eigenvalues** killing unshifted FP32 Cholesky-QR from age 10.
- **D4 (Size of engineering win):** **Closer to Reviewer X on ordinary-step slack (~1.5–2.5 ms/call); Reviewer Z overstates async overlap on warm events (`26 → 12 ms/call`) while having a genuine win on QR overlap.** Overlapping Tensor-Core-saturated `warm_basis` (165–174 TFLOP/s, R4) with BF16 Fwd+Bwd competes for the same Tensor Cores and stretches Fwd+Bwd, whereas overlapping low-occupancy QR (0.8–7.5 TFLOP/s) can genuinely hide ~5–8 ms/call.
- **D5 (Loss predictions):** **Reviewer Y is right.** Until the matched compiled BF16 QR10 control separates BF16 state loss from refresh loss, and closed-loop training tests how open-loop $r_C^w$ translates through dominant 10-age drift (R5), any "60–80% CE closure" claim is indefensible.
- **D6 (Is leaving $V$ unchanged at a warm step acceptable?):** **Yes, provided columns are permuted whenever warm refreshes reorder them (PB9).** Under continuous small rotations $I+A$, $V$'s error is $O(A^2)$ while $M$'s is $O(A)$, and M5 showed consistent $\beta=0.95$ states produce no post-refresh spike.

#### Single strongest disagreement with another reviewer
- **Reviewer & claim:** **Reviewer Z, Candidate B3 ("Warm CholQR-NS": 1-pass TF32 column-equilibrated Cholesky-QR with operator ridge `5e-4 * s`, Gram shift `1e-4 * I`, and 1 BF16 Newton–Schulz step)**, claimed with *"Very high confidence"* to have *"zero $n \ge 6144$ NaNs"*, *"0.0018 BF16 orthogonality"*, and *"80–95% of the warm-vs-QR10 basis gap closed"* (Document 5, lines 1666–1703). *(Note: Reviewer X's B1/A1 shares the same 1-pass shifted-Gram + NS structural defect.)*
- **Why I disagree (two independent mathematical failures):**
  1. **Singular-value collapse from the `1e-4` Gram shift that 1 NS step cannot repair:** For $G = Z_0^\top Z_0 + \nu I$ ($\nu = 10^{-4}$) and $Y_1 = Z_0 U^{-1}$, the exact singular values of $Y_1$ are
     $$\sigma_i(Y_1) = \frac{\sigma_i(Z_0)}{\sqrt{\sigma_i(Z_0)^2 + \nu}} \qquad \Big(Y_1^\top Y_1 = I - \nu U^{-\top} U^{-1}\Big).$$
     When $Q$ is misaligned with a strong mode (as in R5's $3d/4d$ left factors with $r_C = 0.64\text{–}0.80$ or equal-diagonal clusters), a strong eigenvalue $\lambda_1$ leaks by $\sin\theta \sim 0.3$ into a weak column $\lambda_j \ll \lambda_1$. Even *after* column equilibration, dividing column $j$ by $\|z_j\| \approx 0.3\lambda_1$ leaves its orthogonal tail component with singular value $\sigma_{\min}(Z_0) \approx \frac{\lambda_j + 5\times 10^{-4}\lambda_1}{0.3\lambda_1} \approx 1.7 \times 10^{-3}$. Plugging $\sigma = 1.7\times 10^{-3}$ and $\nu = 10^{-4}$ into the formula above gives
     $$\sigma_{\min}(Y_1) = \frac{1.7\times 10^{-3}}{\sqrt{2.9\times 10^{-6} + 10^{-4}}} \approx \mathbf{0.168} \implies e_0 = \sigma_{\min}^2 - 1 = \mathbf{-0.972}.$$
     One Newton–Schulz step ($\sigma_1 = \frac{3}{2}\sigma - \frac{1}{2}\sigma^3$) only boosts small singular values by $1.5\times$: $\sigma_1 = 0.250$, leaving **$e_1 = -0.938$**—a catastrophic **94% orthogonality defect**, contradicting Reviewer Z's claim that $\|E_{\text{orth}}\|_2 \lesssim 0.03 \to 0.0007$!
  2. **TF32 Gram noise + R5 indefiniteness overwhelm `1e-4` and `5e-4 * s`:** Reviewer Z leaves `torch.bmm(z0.T, z0)` in default `"high"` (TF32, $u_{\text{TF32}} = 2^{-11} \approx 4.9\times 10^{-4}$). On a unit-column matrix at $n = 8192$, TF32 Gram perturbation scales as $O(\sqrt{n}\,u_{\text{TF32}}) \approx \mathbf{0.044}$—**440× larger** than his `1e-4` Gram shift—causing `cholesky_ex` to hit non-positive pivots (returning garbage/NaN `U` silently since `check_errors=False` is unchecked!). On R5's SwiGLU-right factors with 4–15% negative BF16 eigenvalues, a fixed `5e-4 * s` operator ridge can also fail to lift $\lambda_{\min}(C)$ above zero.
- **Measurement that settles it:** Run Reviewer Z's verbatim `warm_cholqr_ns` (lines 1670–1685) on (a) the saved R5 trained snapshots ($d = 256\text{–}2048$) and (b) the §2.2 synthetic microbenchmark at $n = 4096, 6144, 8192$ with `"high"` precision, logging `info != 0` rate, **spectral Gram error** $\|Q^\top Q - I\|_2$, and `e_Q`. My prediction: nonzero `info != 0` failures at $n \ge 4096$ and on SwiGLU-right, plus spectral Gram errors $> 0.3$ (and `e_Q > 0.02`) on the misaligned $3d/4d$ left factors even when `info == 0`.

---

### 6. Errors in the moderator checks (R3)

Auditing M1–M7 numerically reveals **four genuine specification/methodology errors** (M4, M2, M3, M5) and **two narrower framing gaps** (M1, M7):

1. **M4 (Cholesky-QR numerics) fails to reproduce §2.2's `0.034` Gram error and misdiagnoses unshifted Cholesky-QR conditioning:**
   - **Discrepancy:** §2.2 measured BF16-stored Gram error `0.034` at $n = 4096$ (and all NaN at $n \ge 6144$) for unshifted 1-pass Cholesky-QR. M4 tested the same algorithm at $n = 4096$ and obtained only `0.0026` (FP32) and `0.0043` (emulated TF32)—**8× to 13× smaller than §2.2**—and did not run $n \ge 6144$. A check that fails to reproduce the baseline phenomenon by an order of magnitude cannot cleanly isolate its cause.
   - **Root cause of the mismatch:** M4 used clean full-rank FP32 $C$ with $\kappa \le 7200$, omitting two features of §2.2: (i) the 40-step EMA of $G G^\top$ with $G \in \mathbb{R}^{n \times n/4}$ (Wishart noise on $\sim 10n$ effective samples inflates $\kappa$ by $\sim 3.7\times$) and (ii) **BF16 storage of $C$**, which injects Wigner rounding noise $\|\Delta C\|_2 \sim \frac{2 u_{\text{BF16}}}{\sqrt{3n}}\|C\|_F \approx 9\times 10^{-5}$ at $n=4096$, right against $\lambda_{\min}(C) \sim \frac{1}{3n} \approx 8\times 10^{-5}$—pushing $C_{\text{BF16}}$ to near-singularity at $n = 4096$ and indefiniteness at $n = 6144$.
   - **Hidden subtlety M4 overlooked:** M4's own unshifted FP32 pass succeeded with `0.0026` at $\kappa(C) = 7200$, where $\kappa(C)^2 u_{\text{FP32}} = 7200^2 \cdot 5.96\times 10^{-8} = \mathbf{3.09} > 1$! That worked because unshifted Cholesky is nearly invariant to positive diagonal column scaling $Z = \hat{Z} D$: its real breakdown parameter is the **column-equilibrated** condition number $\kappa(\hat{Z}) \ll \kappa(C)$ (around 15–50 when $Q$ is nearly aligned), not $\kappa(C)$. That also explains why M4's elementwise 10-bit TF32 rounding only gave `0.0043` ($\lambda_j$ cancels out in $(\Delta Z)R^{-1}$ on aligned columns!).

2. **M2 (The cap and the half step) mis-tests the $\sqrt{n}$ Frobenius-cap claim:**
   - M2 ran the **production** generator—with **global `0.01 s` damping still turned on**—measured pre-cap $\|A\|_F/\sqrt{2} = 0.63\text{–}1.10$ independent of $n$, and concluded this *"contradicts in magnitude"* my claim that $\|A\|_F/\sqrt{2}$ *"explodes by $\sqrt{n}$"*.
   - **Why this mis-specifies the claim:** With `0.01 s` damping on and $\lambda_i \propto 1/i$, only the top $k_{\text{eff}} \approx 100$ modes are active regardless of $n$, giving $\sim 200n$ active entries of size $\sim 0.1/\sqrt{n}$ and $\|A\|_F^2 \approx 200n \cdot (0.1/\sqrt{n})^2 = 2.0$ (**constant in $n$**, matching my own Stage-1 analytical prediction of `0.89` in Document 2, line 662!). My $\sqrt{n}$ claim was explicitly conditional on **removing global `0.01 s` damping while keeping the Frobenius cap**: once all $n^2$ entries of $A_{\text{true}} \sim 0.2/\sqrt{n}$ activate, $\|A\|_F/\sqrt{2} \approx \frac{0.2}{\sqrt{2}}\sqrt{n}$ grows strictly as $\sqrt{n}$ (`1.6` at $n=128$, `4.5` at $n=1024$, `12.8` at $n=8192$) while $\|A\|_2 \approx 0.4$ stays $O(1)$. M2 also replaced 2-step BF16 NS with exact polar retraction, masking retraction error on uncapped steps.

3. **M3 (Generators, one-shot and chained) confounds uncapped vs capped generators and misses PB9's purpose:**
   - In both tables, M3 compared **uncapped** factor-1 Jacobi and **uncapped** PB1 against **capped** PB3a (which also suffered from my `w * a * w.T` bug). Because R5 proved `no cap` diverges on 59–82% of trained factor-ages and M3 itself reported uncapped PB1 blowing up $\rho_{\text{rms}}$ to `0.676` at $r_C = 0.45$, M3's headline chained result (`QR tangent 0.0767 / 0.243`) does **not** test a deployable capped PB1+PB2 algorithm.
   - M3's remark *"Sorting columns made no difference to the QR tangent"* is an artifact of testing distinct synthetic eigenvalues in an open-loop toy: `sgn(d_j - d_i)` is mathematically permutation-equivariant whenever all $d_i$ are distinct, so sorting only affects ties ($d_i = d_j$) and downstream $V$/QR50 alignment, neither of which M3 tested.

4. **M5 ($V$ after a refresh) drops B5's floor and clamp in Row 2 and uses an inconsistent 2D state:**
   - Row 2 reports `M/sqrt(V) = 3.9e4` for `rescale to d^+/d` by applying **unclamped** $d_2^+/d_2 = 10^{-6}/0.03947 = 2.53\times 10^{-5}$ to $V_2 = 10^{-6}$. My actual Stage-1 B5 spec (Document 2, lines 810–812) included both floor $\epsilon_d = 10^{-4} d_{\max}$ and bounds $c_{\min}=0.25, c_{\max}=4.0$, under which $|M_2^+/\sqrt{V_2^+}| \le 198/\sqrt{0.25} = \mathbf{396}$ (or `280` at $\alpha = 0.5$)—**100× smaller** than `3.9e4`. (M5 tested Reviewer X's unclamped B7 under a generic label.)
   - Moreover, Row 2 assumes $C$ is 100% rotated ($d_2 = 0.0395$) while $V_2 = 10^{-6}$ is 0% rotated; in 2D with $\beta_2 = \beta_C = 0.95$, $V \equiv \operatorname{diag}(Q^\top C Q)$ identically, so Row 2's mismatch can only occur via **non-Kronecker** matrix structure (where a single entry $V_{ij}$ differs from the row/column mean $d_i^{(L)} d_j^{(R)}$), not stale $\beta$ lag.

5. **M1 (Jacobi vs QR relative error `4.2–140`) compares against the wrong target:**
   - Measuring the inverse-gap Jacobi generator's distance to one-step `qr(D+E).Q` tests how far `eigh` ($\infty$ power steps, denominator $d_j - d_i$) sits from 1-step QR (denominator $d_j$): the ratio $d_j/(d_j - d_i) \to \infty$ for close eigenvalues, which is an expected difference of *targets*, not an algebraic error in Jacobi (as M3 confirms by showing factor-1 Jacobi beats PB1 on raw $r_C$).

6. **M7 (Basis-frame covariance $B = Q^\top C Q$) overcounts subspace-update cost and omits multi-step non-orthogonality drift:**
   - M7 states rotating $B$ always costs 2 products ("so the warm step keeps 7 products"), which misses both directions: (i) under **rank-$k$ or block subspace updates** (Milligan et al. 2026; PB5/PB6), rotating $B$ costs only $O(n^2 k)$ or $O(n^2 b) \ll 2n^3$ while eliminating the BF16 $Q^\top C$ dynamic-range loss; (ii) conversely, after a Householder QR event where $R = Q^\top Q^+$ is not pre-formed, rotating $B$ requires **3** products ($R = Q^\top Q^+$ plus $R^\top B R$), not 2. Finally, M7 checked only a single-step identity (`1.8e-3`), leaving unmeasured the multi-step drift in $\bar{G}\bar{G}^\top = Q_L^\top G (Q_R Q_R^\top) G^\top Q_L$ when BF16 $Q_R Q_R^\top \neq I$.
