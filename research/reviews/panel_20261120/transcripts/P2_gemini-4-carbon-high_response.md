### 1. What changed for you

Reading [prompt.md](file:///tmp/claude-0/-workspace/1ee48bb6-0438-41d0-828f-790445f70266/scratchpad/p2_carbon/prompt.md)’s moderator checks (R3), roofline table (R4), and trained-covariance replay (R5) forces five concrete revisions to my Stage-1 claims and confirms five others.

#### Claims contradicted or weakened

1. **Candidate B1 (`warm_qr_skew_basis`) one-shot $r_C$ prediction ($0.20 \to 0.035\text{–}0.050$) — contradicted by M3.**
   On synthetic one-shot inputs ($n = 256$, input $r_C = 0.217$; $n = 512$), M3 measured $r_C = 0.081$ and $0.099$ for uncapped PB1, versus $0.052$ for factor-1 Jacobi and $0.038$ for exact QR. My Stage-1 estimate neglected two effects: (i) on moderately separated pairs ($d_i / d_j \in [0.3, 0.8]$), first-order QR contracts $B_{ij}$ only by $(1 - d_i/d_j)$, leaving $(d_i/d_j)B_{ij}$ whereas full-gain Jacobi zeroes $B_{ij}$ to first order; and (ii) at large misalignment ($r_C = 0.45$), second-order $[E, K_{\text{QR}}]$ cross-terms degrade uncapped PB1 ($\rho_{\text{rms}} = 0.676$ vs $0.485$ for uncapped half-step). Conversely, M3 **confirmed** B1 where SOAP’s Adam update actually weights coordinates—on whitened $\rho_{\text{rms}}$ ($0.260$ one-shot vs $0.278$ Jacobi; $0.243$ chained vs $0.240$ QR10 and $0.253$ Jacobi) and top-$k$ subspace distance ($0.002$ vs $0.004$).
2. **$O(\sqrt{n})$ Frobenius-to-spectral cap slack (Question B §1.3, Top Pitfall 1) — weakened in scaling law by M2, reframed by R5.**
   My Stage-1 worst-case dense-rotation scaling ($0.73^\circ$ RMS at $n = 3072$) overstated the $n$-dependence under production’s global damping: M2 measured $\|A\|_F/\sqrt{2} \in [0.63, 1.10]$ across $n = 128\text{–}1024$ with no $\sqrt{n}$ trend and $\|A\|_F/(\sqrt{2}\|A\|_2) \in [2.5, 5]$ (matching my separate 100-mode integral $\|A\|_F/\sqrt{2} \approx 1.0 \implies a \approx 0.5$). More importantly, R5’s `cap=None` ablation proved the cap is **load-bearing** on trained covariances ($59\text{–}82\%$ non-finite factor-ages, first at age 30–40, because pre-refresh $r_C$ on $3d/4d$ factors reaches $0.64\text{–}0.80$ and drives $\|A\|_2 > \sqrt{2}$ past Newton–Schulz’s convergence basin).
3. **Candidate B4 ($V \leftarrow (T_L^{\circ 2}) V (T_R^{\circ 2})^T$) and $2\times 2$ $M/\sqrt{V}$ spike realism — substantially weakened by M5.**
   M5 reproduced my $198\times$ spike *only* when $M$ and $V$ are inconsistent with $C$. When $M, V, C$ share $\beta_1 = \beta_2 = \beta_C = 0.95$, pre-refresh variance leakage already inflates $V_2 = 0.0395 \gg 10^{-6}$, preventing upward spikes under any rule—and on a de-correlating rotation ($\theta = 0.01$, $\lambda_1 = 10^4, \lambda_2 = 10^{-2}$), $T^{\circ 2}$ mixing **doubles** stale variance ($1.01 \to 2.01$ vs true $0.01$). I drop $T^{\circ 2}$ mixing (`PB8a`).
4. **Equal-diagonal stalls and warm sorting (Question B §1.4, B1 `order`) — refined by M3 and R5.**
   M3 showed sorting makes zero difference to PB1 on $r_C$ and $\rho_{\text{rms}}$ (because $A^{\text{QR}}_{ij} = B_{ij}\operatorname{sgn}(d_j - d_i)/\max(d_i, d_j)$ is permutation-equivariant). R5 (Observations 2 & 5) showed that high trained $r_C$ is driven primarily by **large 10-age covariance drift** (even exact `eigh10` rebounds from $0.002 \to 0.26\text{–}0.36$ every 10 ages) concentrated in the **$3d, 4d$ left factors** ($r_C = 0.64\text{–}0.80$), rather than isolated $d_i = d_j$ ties alone.
5. **Candidate A4 / B3 ridge $Z + \tau Q$ under BF16 indefiniteness, A1 epilogue upside, and Stage-1 CE confidence — revised by R4 & R5.**
   - **R5 indefiniteness hazard:** R5 revealed that trained BF16 covariances are not merely rank-deficient—$4\text{–}15\%$ of eigenvalues in right-side SwiGLU factors ($d \ge 512$) are **negative**. In my Stage-1 A4/B3, adding $+\tau Q$ blindly to $CQ$ can cancel a negative BF16 eigenvalue $\lambda_k \approx -\tau$, creating a near-zero singular value in $(C + \tau I)Q$. Stabilized Cholesky-QR (`PA1`/`PB4`) must explicitly suppress negative-Rayleigh directions and guard Pass-2 pivots.
   - **R4 roofline:** `warm, b12` already achieves $165\text{–}174\text{ TFLOP/s}$ at $n = 2304\text{–}8192$ ($\sim 80\%$ of RTX 5090 BF16 Tensor-Core peak), confirming A1’s diagnosis of the $22\%$ `b12` vs `b1` regression at $n \ge 6144$ ($170$ vs $210\text{ TFLOP/s}$) while capping A1’s epilogue gain at 109M sizes (`n ≤ 3072`) to $<8\%$.
   - **D5 loss claims:** Without the pending `BF16 QR10` run, claiming "high confidence" in 60–80% (B1) or 80–95% (B3) CE gap closure overstated calibration between open-loop $r_C^w$ and closed-loop cross-entropy.

#### Claims confirmed

- **Exact algebraic identities (M1):** Confirmed QR’s first-order generator $K_{ij} = E_{ij}/d_j$ ($i > j$) ($2\times 10^{-3}\text{–}2\times 10^{-2}$ error vs $4.2\text{–}140$ for Jacobi), single-NS residual $e_1 = -\frac{3}{4}e_0^2 + \frac{1}{4}e_0^3$ (A5), order-2 seed orthogonality $I + \frac{1}{4}A^4$ (B2), rectangular transport FLOP savings of $8.3\% / 0\% / 7.5\% / 8.3\%$ (A6b), and `high` precision enabling TF32 in PyTorch 2.11 (Pitfall 3).
- **Candidate A5 (`ns=1` in `warm_basis`) — confirmed directly on trained covariances in R5:** Across all widths $d \in \{256, 512, 1024, 2048\}$, `ns=1` changed late-window $r_C$ interval by **`0.000`** and $r_C^w$ interval by **`0.000`**, kept $e_Q = 2.11\text{–}2.23\times 10^{-3}$ (still below QR’s $2.35\times 10^{-3}$), and cut compiled $n = 768, b = 12$ time from $0.82 \to 0.64\text{ ms}$ ($-22\%$).
- **Global $0.01 d_{\max}$ gap normalization freezes weak directions (Question B §1.2) — confirmed by R5 Obs. 4:** At age 750, $40\text{–}99\%$ of eigen-directions sit below $0.01\lambda_{\max}$ (carrying $4\text{–}41\%$ of variance), leaving `warm10/QR50` $r_C^w$ ($0.62\text{–}0.82$) stuck near `QR50` ($0.66\text{–}0.86$) vs `QR10` ($0.46\text{–}0.59$).
- **Cholesky-QR breakdown and BF16 EMA deadband (A4, Pitfall 1) — confirmed by M4, M6, and R5:** M6 confirmed a $4.2\%$ noiseless BF16 EMA stagnation gap ($0.6\text{–}0.8\%$ with noise); M4 refuted pure TF32 as sole cause of $0.034$ at $n = 4096$; and R5 confirmed unshifted FP32 Cholesky-QR fails on $791\text{–}800/800$ trained refreshes from age 10.

#### Single biggest change to my view

**Promoting `PB4` (indefiniteness-hardened `sCholeskyQR2` every 10 ages, eliminating the warm/QR split altogether—or at least on $n \ge 1536$) ahead of `PB1` (`K_QR` warm generator) to #1 for Question B.** In Stage 1, I treated `sCholeskyQR2` primarily as a 50-age QR replacement (A4) and ranked polynomial $B$-space `K_QR` (B1) #1 for 10-age tracking. R5 shows that on trained covariances, 10-age drift is non-perturbative on the dominant $3d/4d$ factors ($r_C = 0.64\text{–}0.80$), trapping any first-order $B$-space step between NS divergence (uncapped) and incremental 3–8% progress (capped)—whereas running `PA1` every 10 ages (`PB4`) executes the exact global `QR10` power step at $\sim 11\text{–}13\text{ ms/call}$, beating current `warm10/QR50` ($15.94\text{ ms/call}$) on **both** speed and tracking.

---

### 2. Prior work (web)

*(Methodology note: all sources below were verified via web search summaries/snippets—**in search results only**, not opened as full PDFs/webpages—and any detail beyond those summaries is marked `[unverified]`.)*

#### 2.1 Refresh frequency and cost in SOAP and Shampoo

- **Nam & Ahn (2026) FOAM (adaptive Shampoo refresh + damping):**
  *“FOAM: Frequency and Operator Error-Based Adaptive Damping Method for Reducing Staleness-Oriented Error for Shampoo”*, Kyunghun Nam and Sumyeong Ahn, 2026, [https://arxiv.org/abs/2606.02365](https://arxiv.org/abs/2606.02365) (ICML 2026; **in search results only**). Decomposes stale-preconditioner degradation in Shampoo into *operator-gap staleness error* and *regret staleness error*, proving preconditioner damping $\epsilon_t$ acts as a numerical stabilizer against basis drift. Rather than refreshing on a fixed clock, FOAM tracks a real-time operator-error proxy $h_t$: when $h_t > \tau$, it increases $\epsilon_t$, triggering an EVD refresh only when $\epsilon_t > \epsilon_{\max}$ (cutting EVD calls by $>80\%$, down to $5\text{–}10\%$ of steps).
  - *Status of pooled candidates:* Establishes that **PB12 (adaptive refresh trigger)** is **already known** in principle for Shampoo (paired with dynamic $\epsilon_t$, which none of us coupled to PB12), though host-triggered branching still breaks static `torch.compile(fullgraph=True)` graphs.
- **Yu et al. (2026) FOAM (cheaper SOAP):**
  *“FOAM: Factored One-sided Adam-Moment for Practical and Scalable SOAP”*, Jaemyung Yu, Byeongho Heo, Sangdoo Yun, and Dongyoon Han, 2026, [https://arxiv.org/abs/2611.16905](https://arxiv.org/abs/2611.16905) (NeurIPS 2026; **in search results only**). Halves SOAP’s state memory and reaches Muon-class wall-clock speed on $124\text{M–}760\text{M}$ LLMs via two structural changes: (i) **input-only preconditioning** (dropping the output-side left Kronecker factor entirely, which in our 109M model eliminates the $2304$ and $3072$ factors responsible for $87.5\%$ of $\sum n^3$ FLOPs and R5’s $r_C = 0.64\text{–}0.80$ bottleneck), and (ii) a K-FAC-motivated row-by-column factored Adam second moment.
  - *Status:* Known structural simplification excluded by Document 1 §1.3 (*"no one-sided preconditioning, and no coverage change"*), but independently corroborates R5 Obs. 5 that output-side $3d/4d$ factors dominate SOAP’s cost and staleness.
- **Milligan et al. (2026) (basis-coordinate statistics & subspace QR in BF16):**
  *“Reparametrizing Shampoo and SOAP for Subspace Basis Updates and BFloat16 Storage”*, Alan Milligan, Zikun Xu, Simon Lacoste-Julien, Felix Dangel, and Wu Lin, 2026, [https://arxiv.org/abs/2605.26327](https://arxiv.org/abs/2605.26327) (ICML 2026; **in search results only**). Shows naive BF16 preconditioner storage degrades Shampoo/SOAP/KL-SOAP, and resolves both compute and BF16 precision by storing $(\lambda_i, Q_i, P_i)$ with $P_i := Q_i^T C_i Q_i$ directly **in basis coordinates** (eliminating $C_i$) and updating only a selected **subspace** of $Q_i$ per refresh via higher-precision subspace QR (evaluating top-$k$/greedy, cyclic, and random block selections) blended with unchanged complement columns.
  - *Status:* Directly establishes that moderator check **M7** (basis-frame covariance $B = Q^T C Q$), **PB5** (top-$k$/subspace QR basis update with orthogonal tail completion), and **PB6** (block/cyclic subspace QR) are **already known** in the 2026 BF16 SOAP literature! Reviewer X’s closed-form Procrustes reflection $H_+ = I - V_+(I_k + S)^{-1}V_+^T$ completion inside PB5 appears algebraically distinct from Householder/block-QR embedding `[unverified whether Milligan et al. test Procrustes completion internally]`.
- **SOAP, Distributed Shampoo, EShampoo, KL-Shampoo/Pro-KLShampoo, and SPlus:**
  - *“SOAP: Improving and Stabilizing Shampoo using Adam”*, Nikhil Vyas, Depen Morwani, Rosie Zhao, Mujin Kwun, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham Kakade, 2024 (ICLR 2025), [https://arxiv.org/abs/2409.11321](https://arxiv.org/abs/2409.11321) (**in search results only**): establishes warm-started 1-step power-QR (`QRRepair`) every 10 steps, linear $M$ transport, and permuted-only $V$.
  - *“A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale”*, Hao-Jun Michael Shi et al., 2023, [https://arxiv.org/abs/2309.06497](https://arxiv.org/abs/2309.06497) (**in search results only**): establishes strided factor accumulation (**PA11 already known**), multi-stream/distributed factor concurrency (**PA2 already known**), and FP32 promotion requirements for ill-conditioned BF16 factors.
  - *“Purifying Shampoo: Investigating Shampoo’s Heuristics by Decomposing its Preconditioner”* (EShampoo), Runa Eschenhagen, Aaron Defazio, Tsung-Hsien Lee, Richard E. Turner, and Hao-Jun Michael Shi, 2025 (NeurIPS 2025), [https://arxiv.org/abs/2506.03595](https://arxiv.org/abs/2506.03595) (**in search results only**): decouples per-step eigenvalue updates in the cached basis from infrequent eigenbasis refreshes (removing grafting) and triggers refreshes adaptively via a warm-started QR off-diagonal criterion (**PB12 already known**).
  - *“Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization”* (KL-Shampoo / KL-SOAP), Wu Lin, Scott C. Lowe, Felix Dangel, Runa Eschenhagen, Zikun Xu, and Roger B. Grosse, 2025 (ICLR 2026), [https://arxiv.org/abs/2509.03378](https://arxiv.org/abs/2509.03378) (**in search results only**), and *“A New Perspective on Shampoo’s Preconditioner”*, Depen Morwani et al., 2024, [https://arxiv.org/abs/2406.17748](https://arxiv.org/abs/2406.17748) (**in search results only**): establish that Frobenius error ($r_C$) overweights top eigenvalues, whereas Gaussian KL divergence $-\frac{1}{2}\log\det(D^{-1/2}B D^{-1/2}) \approx \frac{1}{4}\sum_{i\neq j} B_{ij}^2/(d_i d_j)$ (equivalent to $r_C^w$ / $\rho_{\text{rms}}$) is the right scale-invariant objective.
  - *“Pro-KLShampoo: Projected KL-Shampoo with Whitening Recovered by Orthogonalization”*, Ruotong Sun and Ermin Wei, 2026, [https://arxiv.org/abs/2605.06316](https://arxiv.org/abs/2605.06316) (**in search results only**), and *“SPlus: A Stable Whitening Optimizer”*, Kevin Frans, Sergey Levine, and Pieter Abbeel, 2025, [https://arxiv.org/abs/2506.07254](https://arxiv.org/abs/2506.07254) (**in search results only**): exploit spike-and-flat spectra via low-rank subspace tracking + Muon-like orthogonalization (related to **PB5**) or instantaneous normalization.
  - *Novelty of warm-refresh candidates:* Replacing intermediate SOAP QR steps with dense Tensor-Core skew steps (**PB1** QR-tangent or **PB3** relative-gap Jacobi) + polynomial/NS retractions (**PA6**) is **new** outside this project’s own *TurboSOAP* lineage.

#### 2.2 CholeskyQR2 and shifted CholeskyQR3

- **Core references (all seen in search results only):**
  - *“Roundoff error analysis of the CholeskyQR2 algorithm”*, Yusaku Yamamoto, Yuji Nakatsukasa, Yuka Yanagisawa, and Takeshi Fukaya, 2015, *ETNA* 44:306–326, [http://etna.ricam.oeaw.ac.at/vol.44.2015/pp306-326.dir/pp306-326.pdf](http://etna.ricam.oeaw.ac.at/vol.44.2015/pp306-326.dir/pp306-326.pdf).
  - *“Shifted CholeskyQR for Computing the QR Factorization of Ill-Conditioned Matrices”*, Takeshi Fukaya, Ramaseshan Kannan, Yuji Nakatsukasa, Yusaku Yamamoto, and Yuka Yanagisawa, 2020 (*SIAM J. Sci. Comput.* 42(1):A477–A503; preprint 2018), [https://arxiv.org/abs/1809.11085](https://arxiv.org/abs/1809.11085).
  - *“Condition numbers and equilibration of matrices”*, A. van der Sluis, 1969, *Numer. Math.* 14:14–23, [https://doi.org/10.1007/BF02165096](https://doi.org/10.1007/BF02165096).
  - *“Mixed-Precision Cholesky QR Factorization and its Case Studies on Multicore CPU with Multiple GPUs”*, Ichitaro Yamazaki, Stanimire Tomov, and Jack Dongarra, 2015, *SIAM J. Sci. Comput.* 37(3):C307–C330, [https://doi.org/10.1137/14M0973773](https://doi.org/10.1137/14M0973773).
  - *“Numerically Stable Cholesky-QR on GPU via Mixed-Precision Randomized Preconditioning”* (MRCQR), James E. Garrison, Chao Chen, and Ilse C. F. Ipsen, 2026, [https://arxiv.org/abs/2606.18411](https://arxiv.org/abs/2606.18411).
- **What is established:**
  1. *Shift choice and effect on $Q$:* Literature sharply distinguishes a **Gram shift** $R_1^T R_1 = X^T X + sI$ ($s \approx 11(mn + n(n+1))u\|X\|_2^2$ in Fukaya et al. 2020) from an **operator ridge** $Z_\tau = (C + \tau I)Q$. A 1-pass Gram shift leaves $Q_1 = X R_1^{-1}$ **non-orthogonal** ($Q_1^T Q_1 = I - s R_1^{-T}R_1^{-1}$, shrinking weak columns), whereas following Pass 1 with unshifted `CholeskyQR2` (`sCholeskyQR3`) multiplies only by upper-triangular factors $(R_1 R_2 R_3)^{-1}$, recovering $\operatorname{qr}(X).Q$ to $O(u)$ with **zero distortion of $Q$** whenever $\kappa_2(X) \lesssim u^{-1}$. By contrast, operator ridge $Z_\tau = (C + \tau I)Q$ (`PA1`/`PB4`) preserves $C$’s exact eigenvectors asymptotically while damping finite-step rotation on modes $d_i < \tau$ from $1 - d_i/d_j$ to $1 - (d_i+\tau)/(d_j+\tau)$ (explaining M4’s $3.1\times 10^{-3}\text{–}2.6\times 10^{-2}$ column shift).
  2. *Rank-deficient or slightly indefinite inputs:* Unshifted `CholeskyQR` and `CholeskyQR2` are **already shown to fail** for $\kappa_2(X) \gtrsim u^{-1/2}$ ($\approx 4096$ in FP32) or $\operatorname{rank}(X) < n$ (Yamamoto et al. 2015). Even `sCholeskyQR3` fails on exact rank deficiency ($\sigma_{\min}(X) = 0 \implies \sigma_{\min}(Q_1) = 0$, breaking unshifted Pass 2) unless Pass 2 carries a floor shift or column regularization restores full rank. On R5’s slightly indefinite BF16 $C$ ($\lambda_{\min}(C) < 0$), $Z_\tau = (C + \tau I)Q$ can hit $\lambda_k + \tau \approx 0$ unless negative-Rayleigh columns are damped before adding $\tau Q$.
  3. *Mixed precision:* Yamazaki et al. (2015) prove higher-precision Gram accumulation lifts stability to $\kappa(X) \lesssim u_{\text{high}}^{-1/2}$; conversely, TF32 Gram truncation ($u_{\text{TF32}} = 2^{-11}$) degrades 1-pass error to $O(u_{\text{TF32}}\kappa(\hat{Z}))$. Using BF16 Tensor Cores for $Z = CQ$ (`PA5`), column equilibration (van der Sluis 1969), and IEEE-FP32 `sCholeskyQR2` (`PA1`/`PB4`) is a **known NLA building block**, newly adapted here to SOAP’s indefinite BF16 power step.

#### 2.3 First-order (QR-tangent) and Jacobi tracking under drift; subspace iteration + Rayleigh–Ritz

- **Core references (all seen in search results only):**
  - W. W. Symes, 1982, *“The QR algorithm and scattering for the finite nonperiodic Toda lattice”*, *Physica D* 4(2):275–280, [https://doi.org/10.1016/0167-2789(82)90069-0](https://doi.org/10.1016/0167-2789(82)90069-0).
  - Moody T. Chu, 1984, *“The generalized Toda flow, the QR algorithm and the center manifold theory”*, *SIAM J. Algebraic Discrete Methods* 5(2):187–201, [https://doi.org/10.1137/0605020](https://doi.org/10.1137/0605020).
  - Roger W. Brockett, 1991, *“Dynamical systems that sort lists, diagonalize matrices and solve linear programming problems”*, *Linear Algebra Appl.* 146:79–91, [https://doi.org/10.1016/0024-3795(91)90021-N](https://doi.org/10.1016/0024-3795(91)90021-N).
  - G. W. Stewart, 1969, *“Accelerating the orthogonal iteration for the eigenvectors of a Hermitian matrix”*, *Numer. Math.* 13:362–376, [https://doi.org/10.1007/BF02165413](https://doi.org/10.1007/BF02165413).
  - Pierre Comon and Gene H. Golub, 1990, *“Tracking a few extreme singular values and vectors in signal processing”*, *Proc. IEEE* 78(8):1327–1343, [https://doi.org/10.1109/5.58320](https://doi.org/10.1109/5.58320).
  - Marc Moonen, Paul Van Dooren, and Joos Vandewalle, 1992, *“An SVD updating algorithm for subspace tracking”*, *SIAM J. Matrix Anal. Appl.* 13(4):1015–1038, [https://doi.org/10.1137/0613061](https://doi.org/10.1137/0613061).
- **What is established:** Symes (1982) and Chu (1984) show continuous QR iteration is the Toda isospectral flow $\dot{B} = [B, B_{>0} - B_{>0}^T]$; linearizing discrete 1-step QR on $B = D + E$ yields **PB1** ($K_{ij} = B_{ij}/d_j$ for $i > j$), which breaks $d_i = d_j$ symmetry via column order, unlike gap/double-bracket flows (Brockett 1991) when $N = D$. Moonen et al. (1992) and Comon & Golub (1990) establish Jacobi-type tracking under *slow* drift, whereas *simultaneous* uncapped first-order Jacobi ($A_{ij} = -B_{ij}/(d_i - d_j)$, **PB3** uncapped) is **known to fail/diverge** outside the local basin $\|E\|_2 \ll \min |d_i - d_j|$ because non-commuting rotations interfere—explaining R5’s `cap=None` blowup. Finally, Stewart (1969) proves plain orthogonal iteration (`QR10`) converges linearly at $\lambda_{j+1}/\lambda_j$, while a $k$-dimensional Rayleigh–Ritz / subspace projection accelerates separation to $\lambda_{k+1}/\lambda_j$ (supporting **PB5**).

#### 2.4 Batched or concurrent small QR on GPUs ($n = 768\text{–}3072$, batch 12)

- **Core references (all seen in search results only):**
  - Ahmad Abdelfattah, Azzam Haidar, Stanimire Tomov, and Jack Dongarra, 2018, *“Batched one-sided factorizations of tiny matrices using GPUs: Challenges and countermeasures”*, *J. Comput. Sci.* 26:226–236, [https://doi.org/10.1016/j.jocs.2018.01.005](https://doi.org/10.1016/j.jocs.2018.01.005).
  - Jack Dongarra, Azzam Haidar, Jakub Kurzak, Piotr Luszczek, Stanimire Tomov, and Asim YarKhan, 2015, *“Batched matrix computations on hardware accelerators based on GPUs”*, *IJHPCA*, [https://doi.org/10.1177/1094342014567546](https://doi.org/10.1177/1094342014567546).
  - PyTorch CUDA `linalg.qr` / `BatchLinearAlgebraLib.cpp` docs & issues, PyTorch Contributors, 2024–2026, [https://pytorch.org/docs/stable/generated/torch.linalg.qr.html](https://pytorch.org/docs/stable/generated/torch.linalg.qr.html).
- **What is achievable:** cuSOLVER lacks a natively batched `orgqr` API, and fused shared-memory batched `geqrf` kernels only fit $n \le 128\text{–}256$; hence PyTorch’s `BatchLinearAlgebraKernel` loops sequentially over `b = 12` calling single-matrix `cusolverDnSgeqrf` + `cusolverDnSorgqr` (explaining §2.2’s $1.0\times$ batch scaling). Multi-stream dispatch (**PA2**, known) typically yields **$2\text{–}3.5\times$** at $n = 768\text{–}1024$ and **$1.1\text{–}1.4\times$** at $n = 3072$; MAGMA-style blocked WY batched Householder QR (**PA4**, known) reaches **$2\text{–}4\times$** (~$10\text{–}35\text{ TFLOP/s}$) via `gemmStridedBatched` trailing updates, but its $n/n_b = 12\text{–}48$ serial panel barriers keep it **$2\text{–}4\times$ slower** than BLAS-3 `sCholeskyQR2` (**PA1**). Skipping `orgqr` via `mode='r'` + `trsm` (**PA3**, known) saves ~40–50%.

#### 2.5 Handling Adam's second moment when the basis changes (PB8)

- **Core references (all seen in search results only):**
  - Nikhil Vyas et al., 2024, *SOAP*, [https://arxiv.org/abs/2409.11321](https://arxiv.org/abs/2409.11321).
  - Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong Tian, 2024, *“GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection”*, [https://arxiv.org/abs/2403.03507](https://arxiv.org/abs/2403.03507) (ICML 2024).
  - Thomas Robert, Mher Safaryan, Ionut-Vlad Modoranu, and Dan Alistarh, 2024, *“LDAdam: Adaptive Optimization from Low-Dimensional Gradient Statistics”*, [https://arxiv.org/abs/2410.16103](https://arxiv.org/abs/2410.16103).
- **What is established:** SOAP and GaLore leave $V$ unrotated across basis updates (SOAP only permutes $V$ on eigenvalue sort). LDAdam explicitly analyzes second-moment transport across changing bases $U_{t-1} \to U_t$, showing $(U_t^T g)^{\circ 2} \neq U_t^T (g^{\circ 2})$: diagonal variance projection $V^+ = (T \circ T)V(T \circ T)^T$ (**PB8a**, **already known and shown biased without error feedback**) drops off-diagonal cross-terms $2 B_{ij}\sin\theta\cos\theta$, which M5 confirmed doubles leaked variance ($1.01 \to 2.01$). Clamped Kronecker-marginal ratio rescaling $V_{ij}^+ = V_{\pi_L(i),\pi_R(j)}(d_i^{(L)+}/d_{\pi_L(i)}^{(L)})^\alpha (d_j^{(R)+}/d_{\pi_R(j)}^{(R)})^\alpha$ (**PB8b**) appears **new** in SOAP `[unverified in workshop literature]`.

---

### 3. Ranking

#### Pooled A (Speed) — all 14 candidates ranked

1. **PA1 (Stabilized Cholesky-QR for QR events):** Cuts the #1 bottleneck ($45.3\%$ of optimizer time, $594 \to \sim 105\text{–}130\text{ ms}$, saving $\sim 9.3\text{–}9.8\text{ ms/call}$), provided negative-Rayleigh columns are damped and IEEE-FP32 Gram shifts + `cholesky_ex` fallback guard against R5’s $4\text{–}15\%$ indefinite, near-singular BF16 factors.
2. **PA12 (Overlap refresh with next forward/backward):** Because Step 3 updates $W$ before Steps 5–6 touch only $(Q, M, V)$, pipelining refresh/transport behind Call $k+1$’s $283.4\text{ ms}$ Fwd+Bwd hides up to $\sim 14\text{ ms/call}$ of refresh wall time with **zero** mathematical lag.
3. **PA13 (Size-aware schedule: drop warm at large $n$, run PA1 every 10–20 ages):** R4 shows `warm` hits the BF16 Tensor-Core limit for $n \ge 2048$ ($8T_{\text{warm}} + 2T_{\text{QR}} > 5T_{\text{QR}}$ at $n=8192$); running PA1 on $n \ge 2304$ (or $n \ge 4096$) is faster *and* fixes R5’s $3d/4d$ left-factor bottleneck ($r_C = 0.64\text{–}0.80$).
4. **PA9 (Contiguous param/grad banks + batch-60 $n=768$ coalescing + CUDA graphs):** Saves $\sim 0.8\text{–}1.1\text{ ms}$ on **100%** of ordinary calls by removing $\sim 1\text{ GB}$ of `stack`/`copy_` traffic and eliminates 4 of 5 latency-bound $0.82\text{ ms}$ $n=768$ warm launches at exact mathematics.
5. **PA6 (Fewer warm products — specifically PA6a `ns=1` or PA6b `I+A+½A²` + 1 NS):** R5 proved `ns=1` (PA6a, 5 products) costs `0.000` in $r_C$ and $r_C^w$ ($e_Q = 2.1\text{–}2.2\times 10^{-3} < 2.35\times 10^{-3}$ QR) while saving $22\%$ at $n=768$ and $28.6\%$ FLOPs at large $n$ (use PA6b’s 6-product seed if PB2 raises $\|A\|_2$ to $0.65$; reject NS-free PA6c/d because open-loop $Q R_4$ never heals prior BF16 Gram drift).
6. **PA8 (Symmetric `syrk` kernels for $GG^T, G^TG$ and NS Grams):** Halves tile FLOPs on one-third of every ordinary step ($\sim 1.0\text{ ms/call}$) and trims 1 of 7 warm GEMMs ($\sim 14\%$) while enforcing bitwise symmetry on BF16 $C$.
7. **PA11 (Strided covariance EMA every 2nd call with $\beta_C^2 = 0.9025$):** Saves $\sim 1.3\text{–}1.7\text{ ms/call}$ across all calls and halves M6’s BF16 EMA deadband ($4.2\% \to \sim 2.1\%$), ranked below PA9/PA8 only because halving effective sample count requires a closed-loop loss check.
8. **PA10 (Fused epilogues, no FP32 temporaries, and sub-batching at large $n$):** Recovers the $22\text{–}23\%$ `b12` vs `b1` slowdown and $13\text{–}16\text{ GB}$ FP32 VRAM spike at $n \ge 6144$, though R4 caps its 109M (`n ≤ 3072`) gain below $\sim 8\%$.
9. **PA3 (R-only Householder `mode='r'` + `solve_triangular` + 1 NS):** Best fallback if `PA1` encounters pathological indefiniteness, cutting QR time $\sim 40\%$ by skipping `orgqr`, but still limited by cuSOLVER’s serial `geqrf` batch loop and requiring a ridge on R5’s near-null factors.
10. **PA2 (Concurrent QR over CUDA streams):** Speeds up $n \le 1536$ Householder QR by $\sim 2\text{–}3\times$ at exact math, but largely superseded once `PA1` replaces `geqrf`/`orgqr` with natively batched BLAS-3/Cholesky kernels.
11. **PA5 (`mm32` BF16 Tensor Cores for $Z = CQ$ in QR):** Free bit-exact $2\times$ GEMM speedup already folded into `PA1`, worth only $\sim 4\text{–}6\text{ ms}$ per QR event ($\sim 0.1\text{ ms/call}$) standalone.
12. **PA7 (Cheaper $M$ transport):** M1 confirmed $7.5\text{–}8.3\%$ rectangular FLOP savings, but total transport is $0.51\text{ ms/call}$ ($5.06\text{ ms} \times 10\%$), capping wall-time savings at $\le 0.25\text{ ms/call}$.
13. **PA4 (Custom batched, blocked Householder QR):** High custom-kernel complexity while $n/n_b = 12\text{–}48$ serial panel steps keep it $2\text{–}4\times$ slower than BLAS-3 `PA1`.
14. **PA14 (FP16 basis storage + FP16-accumulate back-projection):** R5 Obs. 7 shows BF16 $e_Q = 1.88\times 10^{-3}$ is already benign, whereas FP16-acc back-projection adds $\sim 3\text{–}4\%$ update noise and cross-dtype cast overhead.

#### Pooled B (Tracking) — all 12 candidates ranked

1. **PB4 (Exact QR at every refresh via hardened `PA1`, no warm step):** Reproduces `QR10` by construction (eliminating R5’s $10$-age large drift, weak-direction freeze $r_C^w$, and $3d/4d$ $r_C = 0.64\text{–}0.80$ in one step—beating `QR20` per refresh *and* per ms) at $\sim 11\text{–}13\text{ ms/call}$ vs $15.9\text{ ms/call}$ today. *(R6 contingency: if R6 shows PB1 closes $>75\%$ of the `QR50 -> QR10` $r_C^w$ gap on trained replay, a size-split hybrid—`PB1+PB2` at $n \le 1536$, `PB4` at $n \ge 2304$—beats pure `PB4` on quality-per-ms, though `PB4` stays #1 for unconditional tracking.)*
2. **PB5 (Head-block power step on top-$k$ columns + Procrustes tail completion + warm step):** Non-perturbatively collapses R5’s $r_C = 0.64\text{–}0.80$ left-factor drift and low top-1 overlap ($0.67$) without hitting any Taylor cap, costing **less** than current 7-GEMM warm ($6 + \frac{2k}{n}$ GEMMs when reusing $Z = CQ$ + 1 NS).
3. **PB1 (QR-tangent generator $A_{ij} = B_{ij}\operatorname{sgn}(d_j - d_i)/\max(|d_i|, |d_j|, \rho s)$, paired with PB2):** Unfreezes R5’s $40\text{–}99\%$ weak directions at zero extra GEMMs and beat factor-1 Jacobi on whitened $\rho_{\text{rms}}$ in M3 ($0.243$ vs $0.253$ chained), provided PB2 caps large drifts. *(R6 contingency: swaps #3/#5 with `PB3` if R6 shows factor-1 / relative-gap Jacobi beats `PB1` on trained late-window $r_C^w$ with $0\%$ non-finite events.)*
4. **PB2 (Row-$\ell_2$ + spectral cap instead of global Frobenius cap):** Mandatory partner to PB1/PB3—R5 showed `cap=None` diverges ($59\text{–}82\%$) while unfreezing weak modes (`damping=0.001`) makes the global Frobenius cap over-throttle $r_C$ ($+0.014$); local row-$\ell_2$ scaling $D_w A D_w$ + block-power spectral cap $\|A\|_2 \le 0.65$ prevents cross-mode throttling while guaranteeing NS stability.
5. **PB3 (Full-gain relative-gap Jacobi variants, 1–2 steps with PB2):** R5’s surviving `cap=None` trajectories ($r_C = 0.036\text{–}0.052 < \text{QR10}$!) and `warm_steps=2` (#1 ablation gain in R5) show locally scaled full-gain Jacobi can beat 1-step QR on separated modes once stabilized by PB2. *(R6 contingency: rises to #3 ahead of `PB1` if R6’s factor-1 or relative-gap Jacobi wins on trained $r_C^w$ across all widths.)*
6. **PB7 (Front-loaded schedule: QR10 / PB4 for ages $\le 300\text{–}500$):** Wipes out the $+0.124$ call-500 cold-start loss gap for $\le 0.4\%$ amortized run time (keep late repairs at 50 rather than stretching to 100, given R5’s rapid 10-age drift).
7. **PB8 (Adjusting $V$ after refresh — specifically clamped PB8b $d^+/d$ rescaling; reject PB8a):** M5 refuted PB8a ($T\circ T$ doubles stale variance $1.01 \to 2.01$) while showing PB8b ($d^+/d$, clamped to $[0.25, 4.0]$) cures the 206-step variance-decay lag on de-correlated tail modes; requires a training run since QR10 itself only permutes $V$.
8. **PB9 (Sort columns at warm refreshes and permute $V$):** Costs $0.38\text{ ms}$ per event and aligns $V$ with `QRRepair` / `PB5`, though M3 proved sorting alone leaves PB1’s $r_C$ and $\rho_{\text{rms}}$ unchanged.
9. **PB12 (Adaptive QR trigger when $a < 0.5$):** Principled (Nam & Ahn 2026; Eschenhagen et al. 2025) for catching $3d/4d$ spikes, but breaks static compilation and becomes redundant under `PB4` or `PB5`.
10. **PB6 (Grouped block QR rotations):** Misses dense head-to-tail couplings on R5’s $3d/4d$ factors ($O(b/n)$ off-diagonal coverage per step) and risks small-QR launch overhead.
11. **PB10 (Refresh before the update):** Changes the reference SOAP map (M1) and blocks `PA12`’s post-$W$-write async overlap with Fwd+Bwd.
12. **PB11 (FP16 / FP32-master / stochastic-rounded basis storage):** R5 Obs. 7 ($e_Q = 1.85\text{–}1.91\times 10^{-3}$) and M2 (`BF16` matches `float64` to 3 digits) show basis rounding is not the tracking bottleneck.

---

### 4. Your top 3 experiments

All three experiments run directly inside the existing **Part 2 replay harness** and **Part 1 microbenchmark** with zero training runs required.

#### Experiment 1: Indefiniteness-Hardened `sCholeskyQR2` Every 10 Ages (`PA1` + `PB4`)

1. **Exact spec** (replaces *both* `warm_basis` and `qr_basis`; note caller/wrapper must disable TF32 on the two FP32 Gram `bmm`s via `torch.backends.cuda.matmul.allow_tf32 = False` or IEEE-FP32 Triton `syrk`):
```python
def scholqr2_refresh(C_bf16: torch.Tensor, Q_bf16: torch.Tensor):
    bsz, n, _ = C_bf16.shape
    c_sym = 0.5 * (C_bf16 + C_bf16.transpose(-1, -2))
    z = torch.bmm(c_sym, Q_bf16, out_dtype=torch.float32)               # 1 BF16 TC GEMM -> FP32
    qf = Q_bf16.float()
    d = (qf * z).sum(dim=-2)                                            # FP32 Rayleigh quotients
    order = d.argsort(dim=-1, descending=True)
    idx = order.unsqueeze(-2).expand_as(z)
    z_ord, q_ord = z.gather(-1, idx), qf.gather(-1, idx)
    d_ord = d.gather(-1, order)
    d_max = d_ord.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30).unsqueeze(-2)
    # Zero out C*q on non-positive Rayleigh columns so +tau*q cannot cancel negative BF16 eigenvalues:
    w_pos = (d_ord > 0.0).float().unsqueeze(-2)
    z_reg = w_pos * z_ord + (2e-4 * d_max) * q_ord
    z0 = z_reg / torch.linalg.vector_norm(z_reg, dim=-2, keepdim=True).clamp_min(1e-30)
    eye = torch.eye(n, device=z.device, dtype=torch.float32)
    g1 = torch.bmm(z0.transpose(-1, -2), z0) + (1e-5 * n) * eye         # run in IEEE FP32!
    r1, info1 = torch.linalg.cholesky_ex(g1, upper=True, check_errors=False)
    q1 = torch.linalg.solve_triangular(r1, z0, upper=True, left=False)
    g2 = torch.bmm(q1.transpose(-1, -2), q1) + (1e-7 * n) * eye         # run in IEEE FP32!
    r2, info2 = torch.linalg.cholesky_ex(g2, upper=True, check_errors=False)
    q2 = torch.linalg.solve_triangular(r2, q1, upper=True, left=False)
    r2_min = r2.diagonal(dim1=-2, dim2=-1).amin(dim=-1)
    bad = ((info1 != 0) | (info2 != 0) | (r2_min < 0.5)).view(bsz, 1, 1)
    return torch.where(bad, q_ord, q2).to(torch.bfloat16).contiguous(), order
```
2. **Where it runs:**
   - **Replay harness:** Chain every 10 ages across all 101 BF16 snapshots at $d \in \{256, 512, 1024, 2048\}$ **with no Householder QR repairs at all** (pure `PB4`), plus an ablation at $\tau \in \{2\times 10^{-4}, 5\times 10^{-4}\}\,d_{\max}$.
   - **Microbenchmark:** Compiled `b = 1` and `b = 12` across $n \in [256, 8192]$ and the 109M QR event region.
3. **Decision metric and threshold:**
   - **Stability:** `0 / 800` non-finite events at every width, fallback rate `bad.float().mean() < 0.5%`, and late-window $e_Q \le 2.50\times 10^{-3}$ at all widths and $n \le 8192$.
   - **Tracking:** Late-window $r_C$ interval within `0.008`, $r_C^w$ interval within `0.015`, and top-$1$ / top-$n/8$ overlap within `0.01` of `QR10` at all four widths.
   - **Speed:** 109M 96-factor refresh event $\le 140\text{ ms}$ ($\ge 4.2\times$ faster than Householder QR’s $594\text{ ms}$, and $\le 14.5\text{ ms/call}$ including transport at cadence 10—beating `warm10/QR50`’s $15.94\text{ ms/call}$).
4. **Probability it passes:** **0.85.**

---

#### Experiment 2: Row- & Spectrally-Capped QR-Tangent Warm Step with Order-2 Seed + 1 NS (`PB1` + `PB2` + `PA6b`)

1. **Exact spec** (6 BF16 GEMMs vs 7 today; exact tie-breaking at $d_i = d_j$; local row-$\ell_2$ cap `0.50` + deterministic 4-probe DCT-II spectral cap `0.65`):
```python
def qr_tangent_warm_refresh(C_bf16: torch.Tensor, Q_bf16: torch.Tensor) -> torch.Tensor:
    bsz, n, _ = Q_bf16.shape
    c_sym = 0.5 * (C_bf16 + C_bf16.transpose(-1, -2))
    b = torch.bmm(torch.bmm(Q_bf16.transpose(-1, -2), c_sym), Q_bf16, out_dtype=torch.float32)
    b = 0.5 * (b + b.transpose(-1, -2))
    d = b.diagonal(dim1=-2, dim2=-1)
    s = d.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30).unsqueeze(-1)
    denom = torch.maximum(d.abs().unsqueeze(-1), d.abs().unsqueeze(-2)).clamp_min(1e-4 * s)
    diff = d.unsqueeze(-2) - d.unsqueeze(-1)                            # d_j - d_i
    ones = torch.ones((n, n), device=b.device, dtype=torch.float32)
    sgn = torch.where(diff != 0.0, diff.sign(), ones.tril(-1) - ones.triu(1))
    sgn.diagonal(dim1=-2, dim2=-1).zero_()
    a = (b / denom) * sgn
    w = (0.50 / torch.linalg.vector_norm(a, dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
    a = w * a * w.transpose(-1, -2)                                     # skew-preserving row-l2 cap
    idx = torch.arange(n, device=b.device, dtype=torch.float32).unsqueeze(-1)
    v0 = torch.cos((idx + 0.5) * torch.arange(1, 5, device=b.device, dtype=torch.float32) * (math.pi / n))
    v0 = (v0 / torch.linalg.vector_norm(v0, dim=0, keepdim=True)).expand(bsz, n, 4)
    v1 = torch.bmm(a, torch.bmm(a, v0))
    v1 = v1 / torch.linalg.vector_norm(v1, dim=-2, keepdim=True).clamp_min(1e-20)
    spec = torch.linalg.vector_norm(torch.bmm(a, torch.bmm(a, v1)), dim=-2).amax(dim=-1, keepdim=True).sqrt()
    a = a * (0.65 / (1.05 * spec.unsqueeze(-1).clamp_min(1e-12))).clamp(max=1.0)
    ab = a.to(torch.bfloat16)
    k_step = (a + 0.5 * torch.bmm(ab, ab, out_dtype=torch.float32)).to(torch.bfloat16)
    y = torch.baddbmm(Q_bf16, Q_bf16, k_step, beta=1.0, alpha=1.0)
    gram = torch.bmm(y.transpose(-1, -2), y, out_dtype=torch.float32)
    gram.diagonal(dim1=-2, dim2=-1).sub_(1.0)
    return torch.baddbmm(y, y, gram.to(torch.bfloat16), beta=1.0, alpha=-0.5).contiguous()
```
2. **Where it runs:**
   - **Replay harness:** Chained as `warm10/QR50` across $d \in \{256, 512, 1024, 2048\}$ (directly comparable against R5’s ablation rows and R6’s uncapped/Frobenius-capped PB1).
   - **Microbenchmark:** Compiled `b = 1, 12` at $n = 256\text{–}8192$ and 109M warm event region.
3. **Decision metric and threshold:**
   - **Tracking:** Closes **$\ge 50\%$** of the `QR50 -> QR10` late-window **$r_C^w$ interval** gap at every width (vs $12\text{–}19\%$ for default warm; i.e., $r_C^w \le 0.56 / 0.62 / 0.66 / 0.72$ at $d = 256 / 512 / 1024 / 2048$), with `0 / 800` non-finite events and $e_Q \le 2.4\times 10^{-3}$.
   - **Speed:** $\le 0.95\times$ default warm cost at $n = 3072$ ($\le 2.34\text{ ms/factor}$ at `b12`) and $\le 1.10\times$ at $n = 768$.
4. **Probability it passes:** **0.70** (the main risk is R5’s $r_C = 0.64\text{–}0.80$ $3d/4d$ left factors hitting the $0.65$ spectral cap, which Experiment 3 resolves).

---

#### Experiment 3: Top-$k$ Procrustes Power Deflation + 1-NS QR-Tangent Warm Step (`PB5` + `PB1`/`PB2` + `PA6a`)

1. **Exact spec** (reuses 1st GEMM $Z = CQ$ for *both* exact FP32 Rayleigh sort and exact top-$k$ power step $Z_1$, applies M1-confirmed singularity-free Procrustes reflection $H_+$, and finishes with deflated QR-tangent + 1 NS = **$6 + \frac{2k}{n}$ GEMMs total**, cheaper than current 7-GEMM warm!):
```python
def topk_procrustes_warm_refresh(C_bf16: torch.Tensor, Q_bf16: torch.Tensor):
    bsz, n, _ = Q_bf16.shape
    k = min(64, n // 4)
    c_sym = 0.5 * (C_bf16 + C_bf16.transpose(-1, -2))
    z = torch.bmm(c_sym, Q_bf16, out_dtype=torch.float32)               # GEMM 1: exact C @ Q
    qf = Q_bf16.float()
    d = (qf * z).sum(dim=-2)
    order = d.argsort(dim=-1, descending=True)
    q_ord = qf.gather(-1, order.unsqueeze(-2).expand_as(qf))
    z1 = z.gather(-1, order[:, None, :k].expand(bsz, n, k))
    d_ord = d.gather(-1, order)
    s0 = d_ord.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30).unsqueeze(-1)
    q1, q2 = q_ord[..., :k], q_ord[..., k:]
    z1 = (d_ord[:, :k] > 0.0).float().unsqueeze(-2) * z1 + (2e-4 * s0) * q1
    z1 = z1 / torch.linalg.vector_norm(z1, dim=-2, keepdim=True).clamp_min(1e-30)
    eye_k = torch.eye(k, device=z.device, dtype=torch.float32)
    # FP64 k x k Gram (<0.05 ms at k<=64) bypasses TF32 inside fullgraph torch.compile:
    r1, _ = torch.linalg.cholesky_ex((z1.double().mT @ z1.double()).float() + (1e-5 * k) * eye_k, upper=True, check_errors=False)
    u1 = torch.linalg.solve_triangular(r1, z1, upper=True, left=False)
    r2, _ = torch.linalg.cholesky_ex((u1.double().mT @ u1.double()).float() + (1e-7 * k) * eye_k, upper=True, check_errors=False)
    u1 = torch.linalg.solve_triangular(r2, u1, upper=True, left=False)
    m = torch.bmm(q1.transpose(-1, -2), u1)
    us, _, vhs = torch.linalg.svd(m)                                    # [bsz, k, k], k <= 64
    rp = torch.bmm(us, vhs)
    u1_t = torch.bmm(u1, rp.transpose(-1, -2))
    s_k = 0.5 * (torch.bmm(m, rp.transpose(-1, -2)) + torch.bmm(rp, m.transpose(-1, -2)))
    vp = q1 + u1_t
    q2_new = q2 - torch.bmm(vp, torch.linalg.solve(eye_k + s_k, torch.bmm(vp.transpose(-1, -2), q2)))
    qm = torch.cat([u1, q2_new], dim=-1).to(torch.bfloat16)
    bm = torch.bmm(torch.bmm(qm.transpose(-1, -2), c_sym), qm, out_dtype=torch.float32)  # GEMMs 2, 3
    bm = 0.5 * (bm + bm.transpose(-1, -2))
    dm = bm.diagonal(dim1=-2, dim2=-1)
    sm = dm.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30).unsqueeze(-1)
    denom = torch.maximum(dm.abs().unsqueeze(-1), dm.abs().unsqueeze(-2)).clamp_min(1e-4 * sm)
    ones = torch.ones((n, n), device=z.device, dtype=torch.float32)
    sgn = torch.where((dm.unsqueeze(-2) - dm.unsqueeze(-1)) != 0.0, (dm.unsqueeze(-2) - dm.unsqueeze(-1)).sign(), ones.tril(-1) - ones.triu(1))
    sgn.diagonal(dim1=-2, dim2=-1).zero_()
    a = (bm / denom) * sgn
    w = (0.35 / torch.linalg.vector_norm(a, dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
    y = torch.baddbmm(qm, qm, (w * a * w.transpose(-1, -2)).to(torch.bfloat16), beta=1.0, alpha=1.0)  # GEMM 4
    gram = torch.bmm(y.transpose(-1, -2), y, out_dtype=torch.float32)   # GEMM 5
    gram.diagonal(dim1=-2, dim2=-1).sub_(1.0)
    return torch.baddbmm(y, y, gram.to(torch.bfloat16), beta=1.0, alpha=-0.5).contiguous(), order     # GEMM 6
```
2. **Where it runs:**
   - **Replay harness:** Chained as `warm10/QR50` across $d \in \{256, 512, 1024, 2048\}$ (specifically inspecting the $3d/4d$ left factors alongside the 8-factor mean).
   - **Microbenchmark:** Compiled `b = 1, 12` at $n \in [256, 8192]$ and 109M warm event region.
3. **Decision metric and threshold:**
   - **Tracking:** Closes **$\ge 75\%$** of the `QR50 -> QR10` late-window **$r_C$ interval** gap (via exact top-$k$ deflation on the $3d/4d$ factors), **$\ge 60\%$** of the **$r_C^w$ interval** gap, and matches `QR10`’s **top-1 and top-8 overlap** within `0.02` at all widths, with $e_Q \le 2.4\times 10^{-3}$.
   - **Speed:** $\le 1.15\times$ current warm cost at $n = 3072$ ($\le 2.83\text{ ms/factor}$ at `b12`, well inside the $2\times$ budget) and $\le 1.35\times$ at $n = 768$.
4. **Probability it passes:** **0.80.**

---

### 5. Disputes (D1–D6) and your strongest disagreement

#### Updated positions on D1–D6

- **D1 (Target generator):** Prefer **PB1 (QR tangent)** to track the QR10 trajectory across power-law tails (M3: chained $\rho_{\text{rms}} = 0.243$ vs $0.240$ QR10 and $0.253$ factor-1 Jacobi), unless R6 shows locally damped, capped full-gain Jacobi (`PB3`) beats PB1 on trained $r_C^w$ without instability (plausible given R5’s surviving `cap=None` factors hit $r_C = 0.036\text{–}0.052 < \text{QR10}$).
- **D2 ($V$ after refresh):** **Permute $V$ at sorted refreshes and otherwise leave it unchanged** as baseline (matching SOAP/QR10; M5 refutes $T\circ T$ mixing because it doubles stale variance $1.01 \to 2.01$). If testing a closed-loop fix for slow variance decay, test **clamped $d^+/d$ rescaling (`PB8b`, $[0.25, 4.0]$)**—never unclamped $d^+/d$ (M5: $3.9\times 10^4$ spike on stale $C$) or $T\circ T$.
- **D3 (Why Cholesky-QR failed):** Conditioning ($\kappa(Z)^2 u_{\text{FP32}} \gtrsim 1$ at synthetic $n \ge 6144$, and R5’s **near-null rank deficiency + $4\text{–}15\%$ BF16 indefiniteness** causing $791\text{–}800/800$ crashes at $d \ge 256$), *not* TF32 alone (M4 showed emulated TF32 gave $0.0043$ at $n=4096$, not $0.034$); however, forcing IEEE FP32 on $Z^T Z$ remains mandatory so TF32 does not cap conditioning at $\kappa \lesssim 45$.
- **D4 (Size of the engineering win):** Synchronously inside the ordinary step, Reviewer Z is right that non-GEMM slack is $\sim 1.5\text{–}2\text{ ms/call}$; **across the training step**, my post-$W$-write async pipeline (`PA12`) + `PA1` legitimately drops exposed optimizer wall time from $26.1\text{ ms}$ toward $\sim 9\text{–}11\text{ ms/call}$ with zero basis lag because Step 3 finishes updating $W$ before Steps 5–6 begin.
- **D5 (Loss predictions):** **Reviewer Y is right.** Without the pending matched `BF16 QR10` run and a closed-loop calibration from $r_C^w$ to CE, quantitative CE gap-closure claims ("60–80% with high confidence") are unsubstantiated.
- **D6 (Leaving $V$ unchanged at a warm step):** **Acceptable** for continuous, unsorted small-angle warm updates ($O(\theta^2)$ second-moment perturbation with $40\%$ EMA turnover every 10 steps at $\beta_2 = 0.95$), provided any refresh that reorders columns (`QRRepair`, `PB4`, `PB5`) permutes $V$ identically.

#### Single strongest disagreement with another reviewer

- **Reviewer and claim:** **Reviewer Y**, who ranked custom batched blocked Householder QR (`A3`/`PA4`) **#1 for Speed**, excluded stabilized Cholesky-QR (`PA1`) from Speed altogether, and ranked shifted Cholesky-QR (`B4`) **last (#5) for Tracking**, arguing it is *"already outside the 2× gate at 768 (0.171 versus 0.138 ms), inside it at 1536–3072, and numerically disqualified at 6144–8192."*
- **Why I disagree:**
  1. *Numerical "disqualification" at $n \ge 6144$ conflates naive 1-pass CholQR with shifted, column-equilibrated 2-pass `sCholeskyQR2`:* §2.2 failed at $n \ge 6144$ only because unshifted, non-equilibrated $Z^T Z$ squares $\kappa(C) \approx n$ past $u_{\text{FP32}}^{-1/2} = 4096$. With column equilibration and Gram shift $s_1 = 10^{-5}n$ (Fukaya et al. 2020) plus positive-Rayleigh damping (Experiment 1), $\kappa(G_1) \le 1 + (10^{-5}n)^{-1} \le 10^5 \ll u_{\text{FP32}}^{-1}$ at **every** $n \le 16384$, eliminating breakdown completely.
  2. *Blocked Householder QR (`PA4`) cannot match BLAS-3 Cholesky-QR (`PA1`) on GPUs:* Even a custom blocked WY Householder kernel with panel width $n_b = 64$ executes $n/n_b = 12\text{–}48$ sequential panel factorizations and trailing barriers per batch, capping GPU utilization at $\sim 10\text{–}35\text{ TFLOP/s}$, whereas `PA1` uses 3 full GEMMs + `potrf`/`trsm` at $\sim 80\text{–}150\text{ TFLOP/s}$ with zero custom CUDA kernels.
  3. *Per-factor $n = 768$ warm gating misses system wall time:* Even if `0.28 ms` (`sCholeskyQR2` at $n=768$) exceeds $2 \times 0.069 = 0.138\text{ ms}$, across all 60 $n=768$ factors at cadence 10 that is only $0.10 \times 60 \times 0.28 = \mathbf{1.68\text{ ms/call}}$, and across all 96 factors `PB4` costs $\sim 12\text{ ms/call}$—**faster** than current `warm10/QR50` ($15.94\text{ ms/call}$) while matching `QR10` globally!
- **Measurement that settles it:** Run **Experiment 1 (`scholqr2_refresh`)** vs blocked Householder QR ($n_b \in \{32, 64, 128\}$) in the §2.6(ii) microbenchmark across $n \in \{768, 1536, 2304, 3072, 6144, 8192\}$ ($b=12$) and chain `scholqr2_refresh` in Part 2’s trained replay ($d = 256\text{–}2048$). `PA1`/`PB4` wins decisively if it achieves `0/800` NaNs at all $n$ and widths, $e_Q \le 0.0025$, late-window $r_C^w$ within `0.015` of `QR10`, and $\ge 2\times$ faster 109M refresh-event time than blocked Householder QR.

---

### 6. Errors in the moderator checks

Four checks in R3 are mis-specified or fail to test the exact claim they adjudicate:

1. **M2 ("contradicts in magnitude ... 'explodes by $\sqrt{n}$' (Reviewer X)") tests `A` *with* global `0.01 s` damping still enabled, missing the conditional claim.**
   Reviewer X’s claim (Document 3, §B.0 Defect 3 & Pitfall P1) was explicitly conditional: *"if you fix Defect 2 [global `0.01 s` damping] without fixing Defect 3 [Frobenius cap], all $n/2$ planes activate and $\|A\|_F/\sqrt{2}$ explodes by $\sqrt{n}$."* M2 evaluated production `A` **with `0.01 s` global damping active** and only over $n = 128\text{–}1024$. Under $\lambda_i \propto 1/i$, `0.01 s` zeros out all pairs with $i, j \gtrsim 100$ regardless of $n$; summing $(0.1/\sqrt{n})^2$ over $j = 1\dots n$ and $i \le 100$ yields $\|A\|_F/\sqrt{2} \approx \sqrt{100 \cdot n \cdot (0.01/n)} = 1.0$ **independent of $n$**, restricting $A$ to skew rank $\le 200$ (hence Frobenius/spectral ratio $\approx \sqrt{200/8}\text{–}5$). Had M2 tested **relative-gap damping** (or PB1) across $n = 256\text{–}8192$ on §2.2’s $A_{\text{true}} \sim 0.2/\sqrt{n}$, all $\binom{n}{2}$ pairs activate and $\|A\|_F/\sqrt{2} \approx 0.2\sqrt{n/2}$ grows strictly as $\sqrt{n}$ while $\|A\|_2 \approx 0.4$ stays $O(1)$! Indeed, R5’s `damping=0.001` ablation ($r_C$ *worsening* by $+0.012\text{–}0.016$) empirically demonstrates this exact Frobenius-cap throttling when deeper modes are unfrozen.
2. **M4 fails to reproduce §2.2’s `0.034` baseline error by $10\times$, so it cannot explain what caused `0.034`.**
   In Document 1 §2.2, CUDA unshifted 1-pass Cholesky-QR at $n = 4096$ produced a BF16 Gram error of **`0.034`**. Yet in M4, *both* CPU `unshifted one pass, FP32` (`0.0023–0.0026`) and CPU `unshifted one pass, TF32` (`0.0027–0.0043`) sit near the $0.0024$ BF16 floor—**$8\times\text{–}13\times$ below `0.034`**! Moreover, on a misaligned matrix with $\kappa_2(Z) = 7.2\times 10^3$ without column equilibration, $\kappa_2(Z^T Z) = 5.18\times 10^7 > u_{\text{FP32}}^{-1} = 1.68\times 10^7$ cannot achieve max column error `2.8e-4` unless M4 either **applied column equilibration** (which §2.2’s naive Cholesky-QR did *not* use!) or used near-zero basis misalignment ($Q \approx U$) instead of §2.2’s 40-step BF16-accumulated rank-$n/4$ EMA $C$ + $r_C = 0.20$ perturbation (and M4 omitted $n \ge 6144$ altogether). To test Reviewer X’s P2 fairly, M4 must run on the **exact §2.2 $n = 4096$ tensor $(C, Q)$** that produced `0.034` and toggle only `allow_tf32` (`True` vs `False`).
3. **M3 ("Sorting columns made no difference to the QR tangent") tests only permutation-invariant metrics in a $V$-free toy, and tests `PB1` uncapped.**
   First, $A^{\text{QR}}_{ij} = B_{ij}\operatorname{sgn}(d_j - d_i)/\max(d_i, d_j)$ is algebraically permutation-equivariant: $(Q\Pi)\exp(\Pi^T A\Pi) = (Q\exp(A))\Pi$, so $r_C$ and $\rho_{\text{rms}}$ are *identically* invariant to sorting in an open-loop covariance toy with no $V$. Reviewers X, Y, and I proposed sorting + $V$-permutation (`PB9`) to preserve descending coordinate alignment with $V$ across eigenvalue crossings and match QR10’s ordered column prefixes—neither of which M3 measures. Second, M3 evaluated PB1 **with `no cap`**, whereas our Stage-1 B1 (`0.45 / spec_est`) and Reviewer Y’s B1 explicitly included a cap; M3’s $r_C = 0.45$ breakdown ($\rho_{\text{rms}} = 0.676$) thus tests an uncapped strawman rather than capped PB1.
4. **M5 ("no spike under any rule" on row 3) checks only upward step explosions and misses a 206-step downward step-freeze in its own numbers (plus M1’s conflation on Reviewer Y’s pitfalls).**
   In M5 row 3 (`true V₂ = 1e-6`, `V unchanged = 0.0395`), $|M_2^+/\sqrt{V_2^+}| \le 1$ indeed avoids an *upward* spike—but $\sqrt{V_2} = 0.199$ is **$199\times$ larger** than $\sqrt{V_2^{\text{true}}} = 10^{-3}$, throttling coordinate 2’s Adam step by $199\times$! Under $\beta_2 = 0.95$, decaying $0.0395$ down to $10^{-6}$ takes $\ln(10^{-6}/0.0395)/\ln(0.95) = \mathbf{206\text{ calls}}$ (over four full QR50 cycles), whereas clamped `d⁺/d` restores the scale in 1 step. Finally, M1’s last row labels decoupled weight-decay / $c_t$ placement *"the update-ordering concern in Reviewer Y’s pitfalls"*, conflating Reviewer Y’s weight-decay formula check (moot, since reference matches) with refresh-after-direction ordering (`PB10`, which remains a genuine method difference).
