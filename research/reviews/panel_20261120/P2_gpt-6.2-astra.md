# P2: GPT-6.2-astra

codex exec 0.167.2 (`codex --search exec -m gpt-6.2-astra -s read-only --skip-git-repo-check`, reasoning effort high, `--json`, empty working dir), 2026-11-20, about 15 min. Prompt: `prompts/P2_gpt-6.2-astra.md` on stdin (labels it saw: X = Fable, Y = Gemini 4 Carbon, Z = Gemini 4 Neon). Tool use: 52 `web_search` items, no command executions or file reads (event log). Verbatim final message (5,002 words). Event log: `transcripts/P2_gpt-6.2-astra_events.jsonl`.

---

## 1. What changed for me

**My biggest change is to my B4, shifted Cholesky-QR. I would now require it to demonstrate that it can serve as a *repair* on trained factors before spending a training run on it.** In stage 1 I treated conditioning mainly as a size-dependent risk, with an observed breakdown at \(n\ge6144\). R5 changes that assessment. The unshifted chain is already non-finite at age 10, with 800/800 non-finite factor-ages at each width \(d\ge512\). A second Cholesky pass cannot recover from a failed first pass. M4 also shows why a good post-storage Gram error is insufficient: increasing the operator ridge to \(10^{-3}d_{\max}\) leaves \(e_Q=0.0024\) while the reported maximum column distance to Householder reaches \(0.026\). Rank deficiency, orthogonality, and agreement with the ordered QR step are three different tests.

| My stage-1 claim | Revised assessment |
|---|---|
| **A4 / PA6a — one Newton–Schulz correction may suffice.** | **Confirmed more broadly than I expected for the current generator.** At all four widths, one correction preserves interval \(r_C\) and \(r_C^w\) to the reported precision, with \(e_Q=0.00211\)–\(0.00223\). At \(n=768,b=12\), the batch takes 0.64 ms versus 0.82 ms. This moves ahead of custom QR as an immediately actionable speed change. The result still needs a worst-direction check and full-event pricing. |
| **B1 / PB1 — the ordered QR tangent is a better model of the target.** | **Confirmed locally and on the chained toy.** M1 finds relative tangent errors of 0.002–0.02 at small perturbations. M3 gives chained \((r_C,\rho_{\rm rms})=(0.0767,0.243)\) against QR10’s \((0.0759,0.240)\). At larger misalignment, however, its uncapped one-shot \(\rho_{\rm rms}=0.676\), worse than 0.485 for the uncapped half-step. I retain PB1 as my first tracking test, with a bounded step. R6’s trained result is pending. |
| **B1 cap ablation — the Frobenius cap can suppress distributed motion.** | The mechanism survives, but the synthetic magnitude is now measured. M2’s Frobenius-to-spectral ratio is 2.5–5, with pre-cap \(\|A\|_2\le0.30\). On trained snapshots, doubling the Frobenius budget modestly improves both interval metrics. Removing it gives 59–82% non-finite factor-ages. The unusually low \(r_C\) among the survivors makes selective relaxation interesting, but cannot justify an uncapped chain. |
| **A3 / PA4 and A5 / PA7b — accelerate QR first and reassociate transport.** | QR still contributes 11.82 of 26.08 optimizer ms/call. R4 establishes the throughput gap; it does not establish my 50% QR-reduction target. I would benchmark concurrency before writing a blocked solver. M1 confirms the transport arithmetic savings of 8.3%, 0%, 7.5%, and 8.3%. Even eliminating all transport would save only about 0.51 ms/call. |
| **B2 / PB5 and B3 / PB6 — a panel or grouped rotation could buy tracking cheaply.** | R5 strengthens the case for testing a partial update on the large left factors, which reach \(r_C=0.64\)–0.80. It also sets a harder success criterion for a head panel. Directions below 1% of the largest eigenvalue can carry up to 41% of the variance. Improving top-1 alone would leave a material part of the problem unresolved. |
| **B5 / PB8a — squared-overlap mixing deserves a loss experiment.** | **Downgraded.** M5 confirms both its benefit on stale denominators and a case where it doubles leaked variance. There is also a published experiment with this very correction that I missed in stage 1. It generally hurt in that ablation. |
| **A6 / PB7 and my optimizer-ordering concern.** | Early QR10 remains a good causal intervention. R5 supplies a separate source of late error: even at \(d=1024\), phase-40 \(r_C\) is 0.400 versus 0.204 for QR10. A call-500 improvement cannot establish a 10k improvement. I withdraw the suggestion that the placement of decay or bias correction explains the reference gap. M1 verified that the reference uses the same equations. Refreshing before the weight update would still change the method. |

The practical distinction is between **tracking gained per refresh** and **tracking gained per millisecond**. QR20 has better interval metrics at every width. Measured from QR50, the current warm events nevertheless buy about 1.3–1.9 times as much interval-\(r_C\) reduction per millisecond of extra refresh work. The next method should beat that trade-off. Matching QR20’s loss by spending QR20’s time would answer a different question.

## 2. Prior work and implementation checks

I opened every source cited below. “Read” identifies an opened paper, abstract page, or manual as specified. I used no search-result-only source for a technical claim. Hardware speedups without a matching benchmark remain **[unverified]**.

### Refresh frequency, basis coordinates, and partial updates

**SOAP already establishes the refresh-frequency trade-off.** Vyas et al. compare refresh intervals, use the previous basis to initialize one orthogonal-iteration step, and continue adapting the diagonal moment between refreshes. Their frequency experiments explain why increasing the interval is a meaningful baseline. They do not establish a loss-equivalent replacement for four out of five QRs. Lin et al. subsequently derive KL-Shampoo and KL-SOAP from a different covariance-fitting objective. Their factor statistics and eigenvalue tracking matter when transferring a refresh result between optimizers. [Read, *SOAP: Improving and Stabilizing Shampoo using Adam*, Nikhil Vyas et al., 2024, revised 2025](https://arxiv.org/html/2409.11321v2) · [Read, author abstract, *Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization*, Wu Lin et al., 2025, revised 2026](https://arxiv.org/abs/2509.03378)

**The two 2026 FOAM papers address distinct costs.**

- **Nam and Ahn** adapt Shampoo’s inverse-root refresh frequency and damping using a proxy for stale-operator error. This is prior work for PB12’s principle. Their Appendix I gives an important negative result: on saved ViT trigger states, reaching residual tolerance 0.1 took eight QR iterations at median cost 3.93 times direct EVD. That argues against refining to a small diagonalization residual regardless of cost. Here the target is explicitly one QR iteration. Their analysis also gives examples where a small diagonalization residual coexists with a large inverse-root error. [Read, *FOAM: Frequency and Operator Error-Based Adaptive Damping Method for Reducing Staleness-Oriented Error for Shampoo*, Kyunghun Nam and Sumyeong Ahn, 2026](https://arxiv.org/html/2606.02365v1)
- **Yu et al.** remove the output-side factor and replace dense rotated \(V\) by row and column statistics. Algorithm 1 also checks an off-diagonal criterion at candidate refresh times and performs `eigh` when it fires. They report competitive validation loss with less state and runtime. This is relevant prior art for adaptive refresh and factored moment modeling. Its principal speedups change the two-sided coverage and dense moment representation specified here. Its row/column estimator also differs from PB8b’s post-hoc rescaling of an existing dense \(V\). [Read, *Factored One-sided Adam-Moment for Practical and Scalable SOAP*, Jaemyung Yu, Byeongho Heo, Sangdoo Yun and Dongyoon Han, 2026](https://arxiv.org/html/2611.16905v1)

**Milligan et al. are the closest prior work for M7 and PB6.** They store \(P=Q^\top S Q\), accumulate statistics in those coordinates, and transform \(P\leftarrow O^\top P O\) when \(Q\leftarrow QO\). They develop restricted-coordinate QR updates with unchanged complementary columns, including greedy block selection and repeated subspace updates. Their experiments show that a cheaper partial update can be spent on higher frequency. This establishes the basis-coordinate representation and the block-update family. My B3’s fixed/interleaved groups and B2’s global leading QR panel have different selection and completion rules. For full dense warm steps, M7 correctly charges the two products to transport \(P\). The stronger reason to try the representation is that a structured \(O\) can make this transport cheaper. [Read, *Reparametrizing Shampoo and SOAP for Subspace Basis Updates and BFloat16 Storage*, Alan Milligan, Zikun Xu, Simon Lacoste-Julien, Felix Dangel and Wu Lin, 2026](https://arxiv.org/html/2605.26327v1)

**EShampoo already has factor-dependent adaptive warm-started QR.** Eschenhagen et al. separate eigenvalue and eigenbasis updates and use a residual criterion to decide whether further QR work is warranted. This is more direct precedent for PB12 than a generic adaptive schedule. A trigger based specifically on the warm cap scalar still needs calibration: \(a\) measures how much a particular generator was clipped, while the desired decision concerns the benefit of paying for QR. [Read, *Purifying Shampoo: Investigating Shampoo’s Heuristics by Decomposing its Preconditioner*, Runa Eschenhagen, Aaron Defazio, Tsung-Hsien Lee, Richard E. Turner and Hao-Jun Michael Shi, 2025](https://arxiv.org/html/2506.03595v2)

### CholeskyQR2, shifts, and mixed precision

For ordered \(X=(CQ)_{:,\pi}\), ordinary Cholesky-QR forms \(R^\top R=X^\top X\) and solves \(YR=X\). The Gram matrix squares the singular-value condition number. CholeskyQR2 can restore working-precision orthogonality for sufficiently well-conditioned, full-column-rank input, provided the first pass succeeds. The familiar \(u^{-1/2}\) scale is a conditioning warning with dimension-dependent bounds. It does not predict a universal matrix size at which a kernel fails. [Read, journal abstract, *Roundoff Error Analysis of the CholeskyQR2 Algorithm*, Yusaku Yamamoto, Yuji Nakatsukasa, Yuka Yanagisawa and Takeshi Fukaya, 2015](https://www.maths.tcd.ie/EMIS/journals/ETNA/volumes/2011-2020/vol44/abstract_vol44_pp306-326.html?pages=306-326&vol=44)

Two modifications pooled under PA1 have different effects on \(Q\).

- **Gram shift.** Form \(R_0^\top R_0=X^\top X+sI\). Choose \(s\) large enough to dominate Gram-formation and factorization error, while keeping the first pass effective as a preconditioner. One sufficient bound in the paper scales as \(11[mn+n(n+1)]u\|X\|_2^2\), under explicit dimension and conditioning hypotheses. It is not a prescription to insert \(11nu\) into any normalized Gram. Complete shiftedCholeskyQR3 with **two unshifted passes**. In exact arithmetic all three right factors are upper triangular. If the final \(Y\) is orthogonal and each triangular diagonal is positive, uniqueness gives the positive-diagonal QR of the *original* \(X\). A Gram shift followed only by a polar polish lacks this triangular-factor argument.
- **Operator ridge.** Replacing \(X\) by \(((C+\mu I)Q)_{:,\pi}\) changes its ordered column spaces. In a two-dimensional block, the first-column angle changes from \(\arctan(b/d_1)\) to \(\arctan(b/(d_1+\mu))\). Accurate QR of the ridged matrix therefore reproduces a different step. Positive column equilibration, on the other hand, preserves the ordered column spaces. Also, \(\mu=\tau\max_j q_j^\top Cq_j\) does not imply \(\kappa(C+\mu I)\le1+1/\tau\): the largest diagonal may substantially underestimate \(\lambda_{\max}\), and a negative eigenvalue may lie near \(-\mu\).

A slightly indefinite \(C\) still gives \(X^\top X\succeq0\) in exact arithmetic. What threatens Cholesky is a small **singular value** of \(X\) relative to Gram error. For exactly rank-deficient \(X\), no sequence of invertible right solves can create the missing rank. Householder QR can return an orthogonal completion, though the trailing basis is nonunique. These facts explain why PA1 needs an actual fallback and why the frequency and cost of taking it belong in its benchmark. [Read, *Shifted Cholesky QR for Computing the QR Factorization of Ill-Conditioned Matrices*, Takeshi Fukaya, Ramaseshan Kannan, Yuji Nakatsukasa, Yusaku Yamamoto and Yuka Yanagisawa, 2020](https://arxiv.org/pdf/1809.11085)

Yamazaki et al. show that selectively doubling intermediate precision can change the orthogonality-error dependence from quadratic to linear in \(\kappa(X)\) under their assumptions. That supports a higher-precision diagnostic Gram. BF16 inputs, IEEE-FP32 Grams, and BF16 output in PA1 do not automatically inherit the paper’s guarantee. PyTorch 2.11’s `high` setting permits lower internal precision; the setting alone does not identify the kernel that produced a particular Gram. [Read, *Mixed-Precision Cholesky QR Factorization and Its Case Studies on Multicore CPU with Multiple GPUs*, Ichitaro Yamazaki, Stanimire Tomov and Jack Dongarra, 2015](https://pure.manchester.ac.uk/ws/files/47811215/mixed_precision_cholesky_qr.pdf) · [Read, *torch.set_float32_matmul_precision*, PyTorch contributors, version 2.11, 2026](https://docs.pytorch.org/docs/2.11/generated/torch.set_float32_matmul_precision.html)

### QR tangents, Jacobi tracking, and Rayleigh–Ritz

The mathematics behind PB1 is established QR differentiation. For \(X=D+E=(I+K)(D+\delta R)+O(E^2)\), with \(D\) positive diagonal, \(K^\top=-K\), and \(\delta R\) upper triangular, the lower triangle gives \(K_{ij}=E_{ij}/d_j\) for \(i>j\). Jacobi instead cancels \(E_{ij}+(d_i-d_j)K_{ij}\), which introduces an inverse gap. Plane-rotation tracking of time-varying covariance eigensystems also predates SOAP. The particular bounded, BF16, four-warm/one-QR application of the QR differential is a plausible implementation contribution. A claim of publication priority is **[unverified]**. [Read, author abstract, *Higher-Order Derivatives of the QR and of the Real Symmetric Eigenvalue Decomposition in Forward and Reverse Mode Algorithmic Differentiation*, Sebastian F. Walter, Lutz Lehmann and René Lamour, 2010](https://arxiv.org/abs/1009.6112) · [Read, *Plane Rotation-Based EVD Updating Schemes for Efficient Subspace Tracking*, Benoît Champagne and Qing-Guang Liu, 1998](https://www.ece.mcgill.ca/~bchamp/Papers/Jounal/IEEE-SP98.pdf)

For PB5, \(H=\operatorname{qf}(CQ_{:,1:k})\) spans the leading QR prefix on the same ordered, full-rank input. A Rayleigh–Ritz extraction additionally diagonalizes \(H^\top C H=S\Lambda S^\top\) and returns \(HS\). It preserves the subspace while changing its individual columns, which can matter to Adam. A subsequent unrestricted Jacobi update can even change that subspace. Orthogonal iteration and Rayleigh–Ritz on evolving matrices are established methods; “exact leading QR prefix” applies only at the stage where the algorithm actually preserves it. [Read, *Analysis of Subspace Iteration for Eigenvalue Problems with Evolving Matrices*, Yousef Saad, 2016](https://www-users.cse.umn.edu/~saad/PDF/ys-2014-1.pdf)

### Batched GPU QR and second-moment transport

**GPU QR.** MAGMA work demonstrates batched panel factorization, compact reflector updates, and substantial gains from reorganizing the small-matrix kernels. Haidar et al. report their optimized batched QR reaching about 167 GFLOP/s on their historical test setup. NVIDIA documents stream assignment for independent cuSOLVER operations, conditional on resources being available for overlap. Neither result prices FP32 **factorization plus explicit \(Q\)** at \(n=768\)–3072, batch 12, on an RTX 5090. The defensible figures for this machine remain the panel’s measured 30.8, 81.4, 143.5, and 219.4 ms per group. The claimed 2.5–4× concurrency gain is **[unverified]**. Low aggregate FLOP/s does not identify how much of a QR panel can execute alongside another panel. [Read, *Batched Matrix Computations on Hardware Accelerators Based on GPUs*, Azzam Haidar, Tingxing Dong, Piotr Luszczek, Stanimire Tomov and Jack Dongarra, 2014 report](https://www.osti.gov/servlets/purl/1361289) · [Read, *cuSOLVER, Release 12.8*, NVIDIA Corporation, 2025](https://docs.nvidia.com/cuda/archive/12.8.1/pdf/CUSOLVER_Library.pdf)

**PB8a has direct prior art and adverse experimental evidence.** Appendix D.2, Eq. 27 of *Purifying Shampoo* derives the squared-overlap transport of the diagonal raw second moment. Figure 6 reports worse performance overall with the correction, including regimes where one might have expected it to help. LDAdam gives a distinct precedent: it transports an estimate of *centered variance* with squared overlaps and adds the square of the transported mean, with bias corrections and other safeguards. PB8b’s clipped factor-ratio rescaling remains a heuristic for two-sided dense \(V\). I found no validation of that specific rule on this task **[unverified]**. [Read, *Purifying Shampoo*, Eschenhagen et al., 2025, Appendix D.2 and Fig. 6](https://arxiv.org/html/2506.03595v2#A4.SS2) · [Read, *LDAdam: Adaptive Optimization from Low-Dimensional Gradient Statistics*, Thomas Robert, Mher Safaryan, Ionut-Vlad Modoranu and Dan Alistarh, 2024 preprint, ICLR 2025](https://arxiv.org/html/2410.16103)

## 3. Ranking

I rank by expected value for **this two-sided BF16 optimizer**, including implementation effort and the risk of losing the repair path. The A ranks prioritize the 109M workload; PA13 rises for a workload dominated by \(n=8192\).

### A — speed

| Rank | Candidate | Reason |
|---:|---|---|
| 1 | **PA6 — fewer warm products** | Start with **one NS**. It has measured trained-replay parity on the interval metrics, a 22% small-batch-call saving at \(n=768,b=12\), and removes two cubic products at large \(n\). Polynomial variants have weaker evidence. |
| 2 | **PA2 — concurrent QR** | Cheapest direct test against the 11.82 ms/call repair increment. The five 768 groups alone cost 154 ms per QR event. |
| 3 | **PA8 — symmetric products** | Reduces arithmetic in both the every-call covariance update and warm Grams. The implementation must beat full GEMM after mirroring and EMA writes. |
| 4 | **PA4 — batched blocked Householder** | Best larger engineering investment if PA2 disappoints. It preserves a robust repair for near-null trained factors. |
| 5 | **PA9 — storage banks, coalescing, graphs** | Savings on the common 10.14 ms body recur every call. Price the three changes separately, including any gradient packing moved upstream. |
| 6 | **PA12 — overlap with next forward/backward** | The post-weight-write dependency permits overlap without an extra stale update. The gain depends on resource contention and the synchronization boundary. |
| 7 | **PA10 — epilogues and sub-batches** | Good for small-\(n\) launches and large-\(n\) workspace. R4’s 165–174 TFLOP/s leaves less sustained headroom than removing products; time 12 consecutive chunks. |
| 8 | **PA13 — size-aware schedule** | At 8192, existing Householder QR20 already costs 560 versus 588 ms/100 calls in factor work. This is a real crossover before any assumed PA1 speedup. Loss at that size is unknown. |
| 9 | **PA1 — stabilized Cholesky-QR** | High potential, but a repair needs trained-input reliability, ordered-prefix fidelity, and timing inclusive of fallback. The 72 ms single-pass extrapolation supplies none of these. |
| 10 | **PA11 — strided covariance** | Skipping half the covariance work has leverage. With \(\beta_C^2\), the nominal half-life is preserved but the independent-sample effective count drops from 39 to about 19.5. R5 cannot price the quality loss. |
| 11 | **PA5 — BF16-input \(CQ\)** | Simple engineering adjunct. Even a 6 ms saving per QR event is only 0.12 ms/call at QR50. Check ordering near ties. |
| 12 | **PA7 — cheaper \(M\) transport** | Useful when the rotation is already available. The measured total budget is small, and direct local transport differs from world transport on rounded, nonorthogonal bases. |
| 13 | **PA3 — R-only QR plus solve** | Retains Householder factorization cost and introduces \(R^{-1}\) sensitivity exactly where trained factors are weakest. Counting only the omitted `orgqr` overstates the saving. |
| 14 | **PA14 — FP16 basis / back-projection** | Finer basis storage may help a diagnostic. The current Gram error is already satisfactory; cast traffic and the speed and accuracy of FP16 accumulation remain **[unverified]**. |

### B — tracking

| Rank | Candidate | Reason |
|---:|---|---|
| 1 | **PB1 — bounded ordered QR tangent** | Best target-specific derivation and chained toy result. A trained-replay failure on weak directions or large-left factors would remove it from first place. |
| 2 | **PB2 — replace the global Frobenius cap** | Both R3 and R5 show that the current cap costs tracking; R5 shows the danger of removing it. Test a bounded spectral surrogate, especially on distributed rotations. |
| 3 | **PB5 — leading power panel** | Offers an interpretable same-input QR-prefix guarantee. Move it to first if R6’s full-basis generators fail on large left factors and the panel improves the weak-direction metric too. |
| 4 | **PB3 — full-gain/local-gap Jacobi** | Factor-1 Jacobi is the strongest challenger. It beats QR on toy raw \(r_C\), trails on whitened correlation, and outperforms the specified atan variant. R5’s doubled production step is a separate algorithm. |
| 5 | **PB6 — grouped QR rotations** | Structured rotations offer the best large-\(n\) arithmetic and cheap transport. The open question is how much coupling crosses the chosen block boundaries. |
| 6 | **PB7 — front-loaded QR** | Best experiment for separating an early trajectory loss from continuing refresh error. It matches BF16 QR10 during the common prefix only; a later CE improvement requires training evidence. |
| 7 | **PB12 — adaptive QR trigger** | Worth testing by factor because R5 localizes the largest deficits. Compare triggers at equal QR milliseconds and include the cost of computing and dispatching the decision. |
| 8 | **PB9 — sort and permute** | Correct bookkeeping for an ordered generator. A consistent relabeling alone cannot improve \(r_C\); M3 saw no standalone benefit. |
| 9 | **PB4 — PA1 every ten ages** | A successful unshifted, accurate QR would be the target up to signs. A ridged power step has a different target, and one-pass Cholesky already exceeds \(2\times\) warm at 768 before stabilization. |
| 10 | **PB8 — change \(V\)** | Both corrections have counterexamples. The published negative result for mixing and the absence of moment histories in R5 put this behind improving the basis itself. |
| 11 | **PB11 — higher-precision basis** | The warm chain has *lower* mean Gram error than BF16 QR10 while tracking much worse. Precision is worth revisiting if an improved generator exposes a storage limit. |
| 12 | **PB10 — refresh before update** | Lets the current gradient select its own basis and changes the QR10 reference trajectory. It would require a separate closed-loop comparison. |

**What would change the top rank.** For width \(d\), define weak-direction gap closure as
\[
H_w(d)=\frac{r^w_{\rm default}(d)-r^w_{\rm candidate}(d)}
                  {r^w_{\rm default}(d)-r^w_{\rm QR10}(d)}
\]
using the late interval means. If R6’s factor-1 Jacobi or relative-gap variant is finite throughout, beats PB1 by at least 0.10 in \(H_w\) at three widths including 2048, and preserves QR-prefix agreement at comparable cost, I would put it first. A raw-\(r_C\) win alone would leave the ranking unchanged. None of these R6 outcomes is assumed.

## 4. The three experiments I would run first

The probabilities below are my estimates for the stated joint criteria. The snippets are specifications; I did not execute code.

### 1. PB1 with a bounded step that distinguishes concentrated from distributed motion

Use production QR at ages divisible by 50. At the four intervening refreshes use the function below. The control changes only `cap_kind` from `"fourth"` to `"fro"`; both use the same pivot floor, column order, rounding, and two NS corrections. Apply the returned permutation to \(V\) and retain production \(M\) transport.

```python
def f(C, Q, cap_kind="fourth"):
    def mm(x, y):
        return torch.bmm(x, y, out_dtype=torch.float32)
    def cols(x, p):
        return x.gather(2, p[:, None, :].expand(-1, x.size(1), -1))

    Z = mm(C, Q)
    score = (Q.float() * Z).sum(1)
    p = score.argsort(dim=1, descending=True, stable=True)
    q = cols(Q, p)
    z = cols(Z, p)
    B = mm(q.mT, z.bfloat16())
    B = 0.5 * (B + B.mT)
    d = score.gather(1, p)
    floor = (1e-4 * d.abs().amax(1, keepdim=True)).clamp_min(1e-30)
    L = torch.tril(B / torch.maximum(d, floor)[:, None, :], diagonal=-1)
    K = (L - L.mT).bfloat16()
    if cap_kind == "fourth":
        H = mm(K.mT, K)
        bound = (0.5 * H.square().sum((1, 2), keepdim=True)).sqrt().sqrt()
    else:
        bound = (0.5 * K.float().square().sum((1, 2), keepdim=True)).sqrt()
    a = (0.5 / bound.clamp_min(1e-12)).clamp(max=1.0)
    Y = (q.float() + mm(q, (a * K.float()).bfloat16())).bfloat16()
    I = torch.eye(Q.size(1), device=Q.device, dtype=torch.float32)
    for _ in range(2):
        E = (mm(Y.mT, Y) - I).bfloat16()
        Y = (Y.float() - 0.5 * mm(Y, E)).bfloat16()
    return Y.contiguous(), p
```

For a real skew matrix with plane magnitudes \(\theta_i\), the new bound is \((\sum_i\theta_i^4)^{1/4}\ge\max_i\theta_i\) in exact arithmetic. It equals the production bound for a single active plane and grows as \(p^{1/4}\), versus \(\sqrt p\), for \(p\) equally active planes. It avoids trusting a fixed probe to see a coherent spike. The fourth-moment arm uses eight square products versus seven for the Frobenius arm. Storage rounding still warrants a Gram check.

**Where and decision.** Run both arms on all 101 snapshots per width and in the compiled microbenchmark. Require zero non-finite factor-ages, \(H_w\ge0.35\) at **every** width, no increase over default in late interval \(r_C\), mean \(e_Q\le0.003\) at each width, and cost \(\le2\times\) warm at each production size \(768\)–3072, batch 12. To measure target fidelity, also branch production QR from the same input and report \(1-\|\widehat Q_{c,k}^{\top}\widehat Q_{{\rm QR},k}\|_F^2/k\), \(k=n/8\), with each ordered prefix orthonormalized in diagnostic FP32. Require improvement over default averaged over the 3\(d\) and 4\(d\) factors. **Probability the fourth-moment arm passes: 0.40.** Report both batch sizes over 256–8192. A separate all-size deployment criterion of \(\le0.75\times\) QR at both batch sizes is substantially harder, particularly at \(256,b=1\).

### 2. PA6a — qualify one NS correction for the production warm step

```python
def f(C, Q):
    def mm(x, y):
        return torch.bmm(x, y, out_dtype=torch.float32)

    B = mm(torch.bmm(Q.mT, C), Q)
    B = 0.5 * (B + B.mT)
    d = B.diagonal(dim1=-2, dim2=-1)
    s = d.abs().amax(-1, keepdim=True).clamp_min(1e-30)[:, :, None]
    h = (d[:, :, None] - d[:, None, :]) / s
    A = -0.5 * (B / s) * h / (h.square() + 0.01**2)
    bound = torch.linalg.vector_norm(A, dim=(-2, -1), keepdim=True) / 2**0.5
    a = (0.5 / bound.clamp_min(1e-12)).clamp(max=1.0)
    Y = (Q.float() + mm(Q, (a * A).bfloat16())).bfloat16()
    I = torch.eye(Q.size(1), device=Q.device, dtype=torch.float32)
    E = (mm(Y.mT, Y) - I).bfloat16()
    return (Y.float() - 0.5 * mm(Y, E)).bfloat16().contiguous()
```

**Where and decision.** Keep warm10/QR50, the current permutation policy, and all production casts. R5 has already answered the mean-quality question. Use its chains for the remaining diagnostic, \(\max_{f,t}\|Q_{f,t}^\top Q_{f,t}-I\|_2\), including phase 40. Require that maximum to stay below 0.07, all outputs finite, both late interval metrics within 0.003 of default at every width, and at least **8 ms saved per complete 109M warm event**, inclusive of transport. That last threshold is 0.64 ms per average optimizer call. Price the larger factors separately. **Probability: 0.80.** I would qualify it again if PB1 changes the distribution of applied rotations.

### 3. PA2 — price concurrent, explicit-\(Q\) Householder before writing PA4

**Exact timing target.** Preserve production \(CQ\), Rayleigh scores, sort, gather, FP32 Householder factorization, explicit \(Q\), BF16 store, \(V\) permutation and \(M\) transport. Change only the solver dispatch. At each production size compare the stock batch-12 call, 12 single-matrix calls on one stream, and the same 12 calls distributed round-robin over \(S\in\{2,4,8\}\) streams. Use a separate solver handle and nonaliasing workspace per concurrent stream, a common input-ready event, and a join before consuming \(Q\). Hold the ordered \(Z\) fixed across arms. Include workspace preparation outside steady-state timing and report peak memory. The one-stream loop separates concurrency from a change in the single-matrix solver path.

**Where and decision.** Use the microbenchmark with the production determinism and precision settings, 10 warm-up batches and 30 timed batches per configuration. Select \(S\) by size, then validate that selection in 30 complete QR events, interleaved with the stock control. Require a **20% reduction in median whole-event time**, no p95 regression, zero failures, and pre-cast normalized QR residual \(\|Z_\pi-QR\|_F/\max(\|Z_\pi\|_F,10^{-30})<10^{-5}\). Compare ordered leading prefixes and post-cast \(e_Q\) on identical trained inputs as well. A 20% win on 601 ms is about 2.4 ms/call at QR50. **Probability: 0.45.** The result would tell me whether PA4 needs a better panel kernel, better explicit-\(Q\) formation, or simply better scheduling.

## 5. Disputes and strongest disagreement

- **D1 — QR tangent versus Jacobi.** PB1 is my first choice for QR10. On the chained toy, Jacobi wins raw \(r_C\) by 0.0025 over PB1, while PB1 is nearer QR10 in whitened correlation and wins the one-shot QR-prefix comparison. R6 can reverse that judgment on trained factors.
- **D2 — transport \(V\).** Use permutation-only as the reference. Squared mixing assumes negligible old-coordinate cross moments. Factor-ratio rescaling assumes the marginal variances explain each relevant entry of \(V\). M5 supplies a failure case for each assumption.
- **D3 — Cholesky failure.** The conditioning of \(Z\) and actual Gram arithmetic both matter. M4 weakens TF32 as the sole explanation for 0.034; R5 makes near-null directions the immediate production concern. Matched GPU inputs with IEEE and `high` Grams would distinguish the mechanisms.
- **D4 — engineering headroom.** Ordinary-step slack and exposed refresh time are separate quantities. With unchanged kernels and only the next 283.4 ms forward/backward available, even perfect overlap hides at most \(0.08(51.47)+0.02(283.4)=9.79\) ms/call. Reaching 12 ms from 26.08 requires speeding up work as well as hiding it.
- **D5 — loss predictions.** No numerical CE recovery fraction is justified. R5 has one seed, QR10-driven covariances, no moment histories, and no feedback through \(W\). A paired compiled BF16 QR10 arm at calls 500, 2000 and 10k remains the necessary attribution control.
- **D6 — leaving \(V\) unchanged.** It is a defensible default after a small rotation. It is first-order accurate if the old raw second moment is diagonal. In general, for \(T=I+K\) the new diagonal contains the first-order term \(\operatorname{diag}(K\Sigma-\Sigma K)\), which the stored \(V\) cannot determine.

**My strongest disagreement is with Reviewer Z’s B6b claim that ridged CholeskyQR2 reproduces BF16 QR10 identically and therefore has zero warm-approximation loss gap.** The two-dimensional angle calculation already disproves identity for nonzero \(\mu\); M4 observes the discrepancy while the stored Gram error stays flat. What remains open is *practical equivalence*. Settle that with same-input, sign-aligned QR-prefix comparisons over trained factors, including near-null cases, followed by chained \(r_C^w\), fallback-inclusive GPU time, and paired CE against compiled BF16 QR10. A fast, measurably different step could still be an excellent optimizer.

## 6. Errors and scope limits in the moderator checks

1. **M5 does not implement Reviewer Z’s specified rescaling rule.** In the first row the raw ratio is about \(0.01/1.01=0.0099\). Z clips each ratio to \([0.25,4]\), adds a floor, and uses exponent \(0.5\) or \(1\). For a single rotating axis even the smallest allowed multiplier gives \(1.01(0.25)=0.2525\); the specified floor can make it larger. Likewise, that clipped rule can amplify \(M/\sqrt V\) by at most \(2\times\) for one axis or \(4\times\) for two, holding \(M\) fixed and \(\alpha\le1\). It cannot turn 198 into \(3.9\times10^4\). M5 is a good test of **unclipped** marginal rescaling. Its exact recoveries and extreme spike should be relabeled. The reported ratios also precede the next gradient EMA; they are not measurements of the next weight update.

2. **M1 needs an ordered tie convention.** The confirmed formula is \(K_{ij}=B_{ij}/d_j\) below the diagonal of the *ordered* matrix, followed by skew completion. R1’s compact `sgn(d_j−d_i)` gives zero at an exact tie under ordinary `sign`. QR can move there. Specify an incumbent-order tie break, positive diagonal convention for \(R\), and resolved pivots when testing the differential. Likewise, M3’s “sorting made no difference” is specific to its implementation and reported metrics. Permutation-invariant diagnostics cannot by themselves settle state-label effects.

3. **M2/M3 do not establish an uncapped production policy or an isolated causal explanation.** M2 uses exact polar retraction. Production applies two polynomial corrections with intermediate BF16 storage; those corrections can diverge. M2 isolates synthetic cap and gain effects well. Its measured ratio of 2.5–5 weakens a claimed \(\sqrt n\) *magnitude on those inputs* while leaving the algebraic worst-case scaling intact. M3’s one-shot counterexamples weaken the reviewers’ quantitative predictions; the inputs and sizes also differ from the original \(n=768\) timing fixture.

4. **M4 is a sensitivity study rather than a reproduction of the CUDA failure.** It stops at 4096, emulates operand rounding on CPU, and does not establish identical \(Z\), GPU accumulation order, or solver path. Report the smallest eigenvalue of the *computed* Gram, the singular values of a higher-precision reference \(Z\), factorization status, orthogonality before and after storage, and the definition of “column distance.” Also distinguish first failures from propagated non-finite states: after a chain fails at age 10, its later factor-ages are evidence of an unusable chain, not independent first-pass breakdowns.

5. **M7 is an exact-arithmetic identity with an orthogonality condition.** For the left factor, \(\bar G\bar G^\top=Q_L^\top G(Q_RQ_R^\top)G^\top Q_L\). Equality with \(Q_L^\top GG^\top Q_L\) requires an orthogonal right basis, plus consistent frame transport and EMA initialization. The \(1.8\times10^{-3}\) result measures a discrepancy. Also, seven products is the count for the stated **dense** warm implementation; block rotations can change it.

I accept the checked NS recurrence, polynomial Gram identities, short-side transport counts, and 8192 timing crossover. The most consequential correction to the evidence is M5’s missing clipping. The most consequential limit on inference is still the absence of closed-loop BF16 QR10 loss and trained results for the R6 generators.