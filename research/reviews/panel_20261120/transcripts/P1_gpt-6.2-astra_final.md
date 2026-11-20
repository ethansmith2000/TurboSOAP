I would attack the **QR implementation first for speed**, and the **choice of rotation first for loss**. A QR repair costs an extra 591 ms when it fires. The current warm rotation, meanwhile, approximates a step toward an eigenbasis with a coefficient proportional to an inverse *gap*. The first-order rotation made by ordered QR has a different coefficient, proportional to the inverse *leading column’s Rayleigh quotient*. That difference gives a direct replacement to test on your snapshots.

The derivations below are my own. Timings and empirical observations come from your prompt. Speed projections are estimates, and none of the proposed loss improvements are established.

## Where the time goes

Using the shared-window event means, the BF16 optimizer costs

\[
10.14 + 0.08(61.61-10.14)+0.02(601.18-10.14)
=10.14+4.12+11.82=26.08\ \mathrm{ms/call}.
\]

Thus the **QR increment accounts for 45%** of optimizer time, the work common to all calls for 39%, and the warm increment for 16%. Halving each component separately would save 5.91, 5.07, and 2.06 ms/call, respectively. Even eliminating the entire optimizer would improve measured whole-step speed by only \(309.5/(309.5-26.08)=1.092\times\), assuming the rest of the step stays fixed. The *achieved* whole-step speedup over the eager reference is \(380.5/309.5=1.229\times\).

For a matrix \(m\times n\), write \(P=m^2n+mn^2\), and count a multiply-add as two FLOPs. The ordinary projection and unprojection each cost \(2P\). The two full covariance products together cost another \(2P\). The ordinary step therefore has **six GEMMs and \(6P\) leading FLOPs**. Moment transport adds four rectangular GEMMs and \(4P\). For a square factor of order \(d\), warm uses **seven** dense products, two for \(B\), one for \(QA\), and two per polar correction. That's \(14d^3\) leading FLOPs. Its speed advantage over QR comes from the kernels and their parallelism, despite this substantial arithmetic bill.

Two details change the priorities by size. At batch 12, five 768 groups alone consume about \(5(30.8)=154\) ms per QR event. They are worth optimizing alongside the 219 ms 3072 group. For warm, a whole small batch takes about 0.8 ms, whereas at 8192 a *single factor* takes 45.5 ms amortized at batch 12. Eliminate launches at the small end and dense products at the large end. At 6144 and 8192, batch 1 even has better warm time per factor than batch 12 in your table.

## A. Speed

### A1. Remove ordinary-step packing and fuse the GEMM boundaries

1. **Update.** Keep the ordinary equations. Give same-shape parameters and gradients stable views into contiguous group storage where the model permits it. Read those views directly and write the FP32 master weights in place. Fuse the output of the second projection with the FP32 moment updates and formation of the BF16 normalized direction. Fuse the final FP32-output GEMM with the weight update where the backend supports the epilogue. Preserve every specified BF16 rounding point, including the rounded projected gradient used for its square.
2. **Cost.** Still six products and \(6P\) FLOPs per matrix. The two covariance products and the first three projection/unprojection products have BF16 outputs. The final unprojection has FP32 output or its equivalent inside the epilogue. Savings come from eliminated copies, launches, and large intermediate reads and writes. For example, physically stacking and then copying back the FP32 parameters alone entails two copies, each reading and writing \(4mn\) bytes, or \(16mn\) bytes per matrix. Credit that saving only if the compiled region actually materializes both copies.
3. **Expected effect.** High confidence in the priority of measuring this, low confidence in a specific speedup. A 10% reduction of the common 10.14 ms is about 1 ms on *every* call. Twelve full graphs establish capture coverage. They do not establish which buffers or library calls Inductor eliminated. I would compare the existing graphs, stable-address CUDA graph replay where applicable, and a custom epilogue only for the boundaries that still materialize.
4. **Failure modes.** Gradient hooks, clipping, accumulation, or model kernels can force a copy elsewhere. A slow custom GEMM can cost more than its epilogue saves. In-place writes must respect all old-state reads. Changing when an intermediate rounds changes the experiment.
5. **BF16 and \(\beta_C\).** Keep BF16 operands, FP32 accumulation, and the existing covariance EMA. There is no intended interaction with its 0.95 decay. Keep `emulate_precision_casts=True` for the control.
6. **Cheap test.** Use the event-region microbenchmark with production strides and aliasing. Compare stored states and FP32 parameter deltas from identical inputs. Reject if the common event does not improve beyond timing variation, or if savings move into gradient preparation outside the measured region. The latter requires a short in-training timing window.

### A2. Use symmetric products for covariance and polar Grams

1. **Update.** Compute one triangle of each \(GG^\top\) and \(G^\top G\), apply the EMA there in FP32, round once, and mirror into the full BF16 tensor. In warm, compute one triangle of each \(Y^\top Y\), form the residual, and mirror for the correction GEMM. Use a tiled symmetric kernel so diagonal tiles and the final mirror do not dominate. The two products forming \(Q^\top C Q\) are general GEMMs in the supplied evaluation order.
2. **Cost.** For the two covariance factors the number of computed dot-product terms falls from \(P\) to
   \[
   \tfrac12m(m+1)n+\tfrac12n(n+1)m=P/2+mn.
   \]
   This saves \(P-2mn\) FLOPs, about one sixth of the ordinary step's dense arithmetic. For each warm Gram, \(d^3\) dot-product terms fall to \(d^2(d+1)/2\). Across two Grams this saves about one full GEMM equivalent out of seven. Inputs are BF16, Gram outputs FP32, covariance storage BF16. Mirroring is quadratic traffic.
3. **Expected effect.** Most promising for large warm factors and large ordinary covariance products. The arithmetic ceilings are about 17% of ordinary and 14% of warm FLOPs. Wall-time savings can be much smaller. At small \(d\), I would favor fusing residual formation and mirroring over dispatching an additional library operation.
4. **Failure modes.** A highly tuned full GEMM can beat a triangular kernel. Changing the reduction tree can change rounding. Mirroring the EMA presumes the old state is symmetric, which should be verified once at entry.
5. **BF16 and \(\beta_C\).** Apply \(\beta_C C+(1-\beta_C)GG^\top\) before the storage cast. Rounding the product first would introduce an extra error every call. Symmetric storage does not guarantee positive semidefiniteness.
6. **Cheap test.** Time both covariance products and both Gram corrections at actual shapes inside their event regions. Reject size by size if event time does not fall. Compare EMA results and Gram residuals with the original implementation.

### A3. Implement ordered QR as a genuinely batched, blocked Householder operation

1. **Update.** Retain the existing \(Z=CQ\), score contraction, sort, and gather. Factor each ordered \(Z\) in panels, accumulating a panel's reflectors as \(H=I-YTY^\top\). Apply \(H\) or \(H^\top\) to the trailing matrix by two rectangular GEMMs around a small triangular multiply. Form the explicit basis by applying the panels in the appropriate reverse order to the identity. Batch independent factors at each panel index. Compare with a small number of concurrent library QR calls if a custom implementation is expensive.
2. **Cost.** One FP32 \(d\times d\) product for \(CQ\), \(O(d^2)\) contraction and gather, sort, QR factorization, explicit Q, and one BF16 cast. For the leading count, a reflector on a trailing \(j\times j\) square costs about \(4j^2\) FLOPs for a dot and a rank-one update. Summing gives \(4d^3/3\) for factorization and approximately the same for forming Q using its trailing structure. Including \(CQ\) gives roughly \((2+4/3+4/3)d^3\). Blocking reorganizes the work. Keep the panel, norms and reflector arithmetic FP32. Check the actual matmul precision of the trailing updates.
3. **Expected effect.** Highest speed priority. Halving the measured QR increment would save 5.91 ms/call. That is a target, not a prediction. The near absence of a batch-12 gain for QR gives a specific opportunity at small sizes. At 3072–8192, panel dependency and single-factor utilization matter more.
4. **Failure modes.** Launching 12 streams need not help if one QR already occupies the GPU. Poor panel kernels, workspace pressure, or an expensive explicit-Q phase can erase the gain. Handle rank deficiency and agree on sign and tie conventions when validating. Sign differences alone need not affect the optimizer if coordinates are handled consistently.
5. **BF16 and \(\beta_C\).** Factor the same FP32-expanded state and round only the final Q. The BF16 orthogonality floor gives little reason to demand bitwise agreement between two stable QR implementations. Ill-conditioned input still makes *trailing directions* sensitive. The short EMA makes those directions change between repairs.
6. **Cheap test.** Split existing QR time into \(CQ\), scores/gather, factorization, explicit Q, and cast. Sweep panel width and concurrency in the microbenchmark. Replay with sign-aligned column and ordered-prefix comparisons. Reject if end-to-end QR time fails to improve or prefix agreement degrades beyond ordinary FP32 implementation differences.

### A4. Drop the second polar correction where one suffices

1. **Update.** Return \(Y_1\) on warm events for the factor groups that pass replay. Keep the two-correction arm as the control. A static choice by size/group avoids a device-to-host decision on every refresh.
2. **Cost.** Five dense products instead of seven. Remove one BF16-input/FP32-output Gram, one BF16-input/FP32-output correction product, its residual arithmetic, and a basis cast. To see what is lost, write \(E=Y^\top Y-I\). In exact arithmetic the correction is \(Y'=Y(I-E/2)\), hence
   \[
   E'=(I-E/2)^2(I+E)-I=-3E^2/4+E^3/4.
   \]
   For orthogonal input \(Q\) and applied skew \(K\), the initial \(E=K^\top K\). A real skew matrix has paired singular values, so the present cap gives \(\|K\|_2\le\|K\|_F/\sqrt2\le0.5\). Thus \(0\preceq E\preceq0.25 I\). At the extreme eigenvalue the first correction leaves \(-0.04297\), appreciably above the storage floor. Many distributed rotations can have a much smaller spectral norm at the same cap.
3. **Expected effect.** The compute-dominated estimate is \(2/7\) of warm arithmetic, about 0.70 ms/factor at 3072 or 13 ms at 8192 if products cost equally. Lower confidence at 768. Removing launches can still help there.
4. **Failure modes.** BF16 input Q is already nonorthogonal. A small normalized Frobenius Gram error can conceal a bad singular direction. One correction is least attractive when motion concentrates in a single plane.
5. **BF16 and \(\beta_C\).** The extra cast can itself perturb Q, while an extra correction can reduce prior storage error. Replay *chains* with the actual ten-age spacing; a single synthetic refresh cannot select the policy.
6. **Cheap test.** Record spectral Gram error in addition to the normalized Gram error, including the fourth consecutive warm event. Reject the one-correction policy for groups with accumulating spectral outliers, poorer ordered-prefix agreement, or no timing gain.

### A5. Reassociate moment transport on the short side

1. **Update.** For \(m>n\), form \(S_R=Q_R^\top Q_R^+\), then compute \((Q_L^+)^\top[Q_L M]S_R\). For \(n>m\), form \(S_L=(Q_L^+)^\top Q_L\) and apply the two right-side basis multiplications separately. Keep the existing association for square matrices unless benchmarking favors another. This uses associativity even when the bases are imperfectly orthogonal.
2. **Cost.** Four products in both versions. For \(m>n\), the original costs \(2(2m^2n+2mn^2)\) FLOPs, and the new order costs \(2(2m^2n+mn^2+n^3)\). The saving is \(2n^2(m-n)\). Forming *both* overlaps blindly can increase the cost because \(m^3+n^3-(m^2n+mn^2)=(m+n)(m-n)^2\). Try BF16 output for the overlap first, then FP32 output with a single cast. Both change the rounding path.
3. **Expected effect.** Modest. Even eliminating the measured 5.06 ms of transport altogether would save only about \(0.1(5.06)=0.51\) ms/call. The stronger opportunity occurs if the refresh supplies its own small coordinate rotation, as in B3 below.
4. **Failure modes.** The short-side GEMM can be more efficient than its replacement despite doing more FLOPs. Forming a nearly identity overlap in BF16 changes the moment error.
5. **BF16 and \(\beta_C\).** This preserves the real-arithmetic transpose transport, including its sensitivity to a nonorthogonal basis. With \(\beta_1=0.95\), a transport error survives ten calls with multiplier \(0.95^{10}=0.599\).
6. **Cheap test.** Microbenchmark the four production shapes and compare with FP32 transport of the *stored* bases on artificial moments. Reject by shape if slower or if the relative error grows materially. Replay lacks moments; loss needs a short run.

### A6. Reallocate a fixed QR budget toward the early trajectory

1. **Update.** As a concrete budget-neutral test, replace the four warm events at ages 10, 20, 30, 40 of every 50-age interval through age 500 by QR. This buys 40 extra early QRs. Convert 40 later scheduled QRs to warm, for example ages \(600,800,\ldots,8400\). Every ten-age event still refreshes. A speed-oriented second point uses the same later substitutions with only the first 250 ages on QR10.
2. **Cost.** The first policy exchanges 40 events each way and has the original total arithmetic by event class. The second buys 20 early QRs and removes 40 late ones. Using shared-window means, its estimated saving is \(20(601.18-61.61)/10000=1.08\) ms/call. Early and late timings need not be identical.
3. **Expected effect.** High confidence this tests a consequential timing question. Low confidence in the late loss outcome. It spends accurate refreshes where the observed loss gap opens fastest. This is a schedule alternative, not a cheap per-event QR replacement.
4. **Failure modes.** Removing a later QR creates a 100-age interval between repairs. Hard late factors may deteriorate, and the persistent late gap may require continued accuracy. A schedule tuned to a single seed may overfit.
5. **BF16 and \(\beta_C\).** After 50 ages only \(0.95^{50}=0.077\) of an old covariance contribution survives. A late basis cannot count on the covariance that originally determined it. Compare against matched BF16 QR10.
6. **Cheap test.** Replay the first 1,000 ages, including the first enlarged repair intervals. Reject the later substitutions if tracking degrades sharply in those intervals. The early-loss hypothesis needs a paired 500-call training run; the budget trade needs the full horizon.

## B. Tracking the QR10 trajectory

For a clean local target, take the same rounded \(C,Q\) at a snapshot, sort by the incumbent QR scores, and call the reordered orthogonal basis \(Q_p\). Write \(B=Q_p^\top C Q_p=D+E\), with \(E\) off-diagonal. Choose signs for positive \(R\) diagonal. To first order,
\[
D+E=(I+K)(D+T)+O(E^2)=D+KD+T+O(E^2),
\]
where \(K\) is skew and \(T\) upper triangular. Consequently
\[
\boxed{K_{ij}=E_{ij}/d_j\quad(i>j),\qquad K_{ji}=-K_{ij}.}
\]
This expansion needs positive, sufficiently resolved pivots and small relative perturbations. It follows because QR of \(CQ_p=Q_p B\) has orthogonal factor \(Q_p\,\mathrm{qr}(B).Q\), up to signs. The BF16 Q makes this a local model.

For two ordered directions with \(d_1>d_2\), the exact QR first-column angle is \(\arctan(B_{21}/d_1)\). The present warm angle is approximately \(+\tfrac12 B_{21}/(d_1-d_2)\) when damping and the cap are inactive. Depending on the spectrum it can **undershoot or overshoot** QR. At equal positive diagonals it returns zero, while the QR angle is still \(\arctan(B_{21}/d_1)\). Increasing the present step size alone cannot fix this discrepancy.

For all B candidates I would report a **cost gate** beside quality. At batch 12 the 2× warm ceilings are 0.138, 0.662, 2.04, and 4.92 ms/factor at 768, 1536, 2304, and 3072. I would also use \(0.5\times\) measured QR as a concrete interpretation of “well below QR” and report batch 1 separately. At batch 1 and 256, **the existing warm implementation itself misses that second gate** (0.79 versus 0.89 ms). FLOP counts cannot certify it there.

### B1. Use the ordered QR tangent instead of the inverse-gap tangent

1. **Update.** Compute \(B\) as now, sort its diagonal descending with the QR tie policy, and permute both axes and Q consistently. For the first controlled experiment use
   \[
   L_{ij}=\mathbf1_{i>j}\,B_{ij}/\max(d_j,\delta),\quad K=L-L^\top,\qquad
   \delta=0.01\max_j|d_j|
   \]
   with an absolute positive floor. Use the existing cap and retraction on \(Q_p(I+K)\). Permute \(V\) on both sides of the matrix. For an explicit cap ablation, estimate \(\|K\|_2\) with four power iterations on \(K^\top K\) from a fixed non-axis-aligned vector, and scale by \(\min(1,0.4/\max(\widehat{\|K\|_2},10^{-12}))\). Measure spectral Gram outliers for this ablation.
2. **Cost.** Same seven square products, BF16 inputs and FP32 accumulation as current warm. Add sort and quadratic gathers. The cap ablation adds eight matrix-vector products per factor, not eight GEMMs of cubic cost. All denominator, skew and reduction arithmetic should be FP32. The cheap diagonal score differs slightly from production QR's \(Q\odot(CQ)\) contraction owing to intermediate rounding; record permutation agreement.
3. **Expected effect.** My first choice for a full-basis approximation. It matches the local ordered QR generator before limiting and repairs the positive equal-diagonal stall. I expect better *same-input QR agreement* for resolved directions. The existing global cap may mask nearly all the gain, hence the explicit ablation. No defensible CE prediction.
4. **Failure modes.** Off-diagonal mass may be too large for first order, especially in trailing columns. The floor changes weak-direction dynamics. A power estimate can underestimate the norm. With tied scores, small BF16 perturbations can select a different QR ordering.
5. **BF16 and \(\beta_C\).** The pivot floor limits amplification of BF16 noise in a weak column. Ten calls replace about 40% of the covariance history; expanding around an almost diagonal basis may be a poor approximation in precisely the factors that need help. Stratify by pre-refresh error.
6. **Cheap test.** Branch both methods from identical replay states and compare signed-aligned column overlaps, prefix overlaps, permutation agreement, and Gram tails; then chain both. Record the cap multiplier and estimated spectral norm before and after clipping. Reject if it only improves \(r_C\), or if the new cap yields poor Gram tails or misses the timing gates.

### B2. Match the leading QR panel and give the tail a smooth orthogonal completion

1. **Update.** Compute \(Z=CQ\) and order as in QR. In the ordered basis form only \(X=Q_p^\top Z_{p,1:k}\), and take its tall Householder QR. Let \(H\) be the implicit full orthogonal product of its \(k\) reflectors. Return \(Q_pH\), with the first \(k\) signs aligned to the positive-diagonal convention. Use the stable reflector sign and apply its column-sign correction separately. For a diagonal positive input this gives the old basis up to signs, including its untouched tail. Start with \(k=d/8\). Permute \(V\).
2. **Cost.** One full \(CQ\) (BF16 inputs, FP32 output if desired), one \(d\times d\) by \(d\times k\) product, FP32 tall QR, and implicit right application of \(k\) reflectors to Q. The latter costs at most \(4d^2k\) FLOPs by a dot and update per row per reflector; the tall QR costs about \(2dk^2-2k^3/3\) by summing its shrinking trailing updates. No full square QR or explicit \(H\). For the \(Q_p^\top Z\) product choose FP32 or deliberately round \(Z\) for BF16 tensor cores and measure both. Batch the panel application.
3. **Expected effect.** With orthogonal Q and exact arithmetic the first \(k\) QR columns agree up to sign: their QR depends only on the first \(k\) input columns. This is the most interpretable quality/speed knob. It can also reduce large-\(d\) warm arithmetic substantially. Its narrow factorization can still be latency-bound at 768.
4. **Failure modes.** Leading variance directions need not dominate the Adam update. The trailing completion does not perform their ordered power iteration. Explicitly forming a full QR completion defeats the cost argument. If the stored Q is nonorthogonal then \(Q_p Q_p^\top Z\ne Z\); a right orthogonal update also preserves existing singular-value defects before storage rounding.
5. **BF16 and \(\beta_C\).** Stored-basis drift is a real risk across the four warm events. The leading set can turn over as the short EMA changes, so sort before selecting it. Track turnover near the \(k\)-boundary.
6. **Cheap test.** Sweep \(k/d=1/16,1/8,1/4\). Compare to the QR10 *prefix* at several ranks inside and outside the panel, and inspect normalized off-diagonal correlations in the tail. Reject panel sizes that miss either speed gate or improve the leading subspace while materially degrading the bulk.

### B3. Perform exact ordered QR rotations within small coordinate groups

1. **Update.** Compute \(Z=CQ\) and sort as for QR. Partition ordered coordinates into groups \(J\) of size \(b\). Compute \(B_J=Q_J^\top Z_J\), \(H_J=\mathrm{qr}(B_J).Q\) with consistent signs, and return \(Q_J H_J\). Start with \(b=16,32\). Compare contiguous groups with a fixed interleaving that places separated ranks in the same group. For \(b=2\), this is just a Givens rotation with cosine and sine proportional to \((B_{11},B_{21})\). It moves in your equal-diagonal example.
2. **Cost.** One full \(CQ\), plus \((d/b)\) products of shape \(b\times d\) by \(d\times b\) and \((d/b)\) products of shape \(d\times b\) by \(b\times b\). Together the small products cost \(4d^2b\) FLOPs. Small FP32 QR with explicit Q costs about \((8/3)db^2\) in leading arithmetic. Use BF16-input GEMMs with FP32 local \(B_J\), FP32 QR, and BF16-stored Q. With two-element groups, direct FP32 `hypot` and column rotations avoid the solver dispatch altogether. The known \(H_J\) also permit moment transport in coordinate blocks without forming world coordinates, exact for ideal orthogonal bases before rounding.
3. **Expected effect.** Most attractive at 4096–8192: only one cubic GEMM and no full retraction. It reproduces QR exactly for a block-diagonal ordered \(B\) with those groups and consistent signs. On dense factors, the fraction of omitted coupling determines quality.
4. **Failure modes.** Contiguous grouping can miss a strong leading-to-tail coupling. Interleaving can miss neighboring directions. Many tiny QR dispatches can erase all gains; a single batched small kernel is important. Nonorthogonality can accumulate as in B2.
5. **BF16 and \(\beta_C\).** FP32 local rotations avoid dividing by noisy eigenvalue gaps. They still respond to a BF16-rounded \(C\). Keep the full repair on its original cadence during the comparison, especially if group membership changes.
6. **Cheap test.** Measure \(\sum_J\|\mathrm{offdiag}(B_J)\|_F^2/\|\mathrm{offdiag}(B)\|_F^2\) offline to explain success or failure. Compare chained QR-prefix agreement, normalized correlations, spectral Gram error, and grouped-kernel time. Reject if captured coupling is small and QR agreement fails to improve even with interleaving.

### B4. Stabilize Cholesky-QR by shifting the power operator

1. **Update.** Form \(Z_\mu=(C+\mu I)Q_p=Z_p+\mu Q_p\), using the *original* sort scores. Compute \(R^\top R=Z_\mu^\top Z_\mu\), then solve \(Q^+R=Z_\mu\) by a right triangular solve. Sweep \(\mu/\|C\|_F\) over \(0,10^{-4},10^{-3},10^{-2}\); avoid an explicit inverse. Form the Gram with verified full FP32 multiplicand precision. Shifting \(C\) matters here. Adding a shift only to the Gram yields \(Q^{+\top}Q^+=I-\nu R^{-\top}R^{-1}\) in exact arithmetic and can leave visibly short columns.
2. **Cost.** One square FP32 \(CQ\), one symmetric FP32 Gram (about \(d^3\) FLOPs with triangular computation), one Cholesky (about \(d^3/3\), from summing trailing triangular updates), and one triangular solve (about \(d^3\), from \(d\) right-hand sides each costing about \(d^2\)). Add \(O(d^2)\) shift, norm, scores, gather, and BF16 cast. A full GEMM Gram costs \(2d^3\). Full-precision Gram time must be measured separately from the supplied “high” setting.
3. **Expected effect.** At zero shift, successful accurate Cholesky-QR gives the ordered QR target up to signs, since \(Z=Q^+R\) and \(Q^{+\top}Q^+=I\). For orthogonal Q and PSD \(C\), the shifted Gram condition number is \(((\lambda_{\max}+\mu)/(\lambda_{\min}+\mu))^2\). Shifting damps the local QR angle by changing its denominator from \(d_j\) to \(d_j+\mu\). Your existing timing is already outside the 2× gate at 768 (0.171 versus 0.138 ms), inside it at 1536–3072, and numerically disqualified at 6144–8192. **This is a conditional size-specific candidate.**
4. **Failure modes.** A small shift may leave an unusable Gram; a large one suppresses the desired rotation. The rounded \(C\) can be indefinite. A successful Cholesky status alone gives no orthogonality guarantee. Do not count a NaN run as a fast refresh.
5. **BF16 and \(\beta_C\).** Shifting cannot restore tail information lost in the BF16 covariance. A fixed absolute shift would change strength substantially over training; tie it to the current factor scale. Check Gram error both before and after basis storage.
6. **Cheap test.** Replay each scale choice with factorization status, worst singular-value defect, and same-input QR agreement. Reject a size if no shift passes all three of numerical reliability, trajectory quality, and timing. I would not put this on the training critical path before that sweep.

### B5. Transport diagonal second-moment history with the known rotations

1. **Update.** Always apply the same permutations to both coordinate axes of \(V\) that the refresh uses for the bases. For an actual rotation, let \(T_L=Q_L^{+\top}Q_L,\ T_R=Q_R^\top Q_R^+\). Then a gradient becomes \(g'=T_L g T_R\) for ideal orthogonal bases. Under a diagonal **raw second-moment** approximation in the old coordinates,
   \[
   V'=(T_L\odot T_L)V(T_R\odot T_R).
   \]
   This follows by expanding \(g_{ij}'{}^2\) and dropping cross terms between distinct old coordinates. Test first with B3's known block rotations after permutation. For a pair rotated by \(\theta\), each updated diagonal variance is a mixture with weights \(\cos^2\theta,\sin^2\theta\).
2. **Cost.** With known rotations of block size \(b\) on both sides, two collections of block applications cost \(2mnb+2mnb=4mnb\) FLOPs, plus squaring the coefficients, permutation, and one BF16 storage cast. Use FP32 coefficients and weighted sums for small blocks. For dense *known* rotations, the two products cost \(2P\); inferring both dense overlaps from the world bases costs another \(2(m^3+n^3)\). I would avoid that latter version on the larger factors.
3. **Expected effect.** Zero effect on open-loop \(r_C\). Potential loss improvement if large rotations leave diagonal variances attached to the wrong directions. Pure permutations are exact for diagonal history. Keep a permutation-only arm: QR10 itself uses that approximation, so extra mixing could move the optimizer away from the desired trajectory.
4. **Failure modes.** Correlated coordinates have missing cross terms, and repeated mixing can flatten genuinely anisotropic \(V\). The product of squared orthogonal entries conserves total variance but discards correlations, so conservation alone is a weak test. Explicit coordinate rotations differ slightly from overlaps of stored Q's.
5. **BF16 and \(\beta_C\).** Round the mixed moment once. Here \(\beta_2=\beta_C=0.95\); about 60% of the previous refresh's history remains ten ages later. Consequently stale denominators and over-mixed denominators can both matter through successive refreshes.
6. **Cheap test.** The microbenchmark can price the block kernels; neither supplied harness can evaluate moment fidelity. The smallest informative extension is a short saved per-step gradient stream for several actual matrices, replaying \(M,V\) and comparing the world update to BF16 QR10. A paired 500-call training run is needed for CE. Reject if mixing increases direction error or gives no reproducible loss gain over permutation alone.

## What the quality measurements establish

**Use two QR comparisons.** First, branch a candidate and production QR from the *same* \(C,Q\) at every saved snapshot. That measures whether the proposed refresh implements the target map. Second, chain both schedules from age zero. That measures accumulated trajectory error. For each, compare sorted leading-prefix projectors and sign-aligned ordered columns. For orthonormal \(U,V\) with \(k\) columns,
\[
\|UU^\top-VV^\top\|_F^2=2k-2\|U^\top V\|_F^2
\]
by expanding the square and cycling the trace. Orthonormalize diagnostic prefixes in FP32 or account for their Gram defects. The full-square subspace overlap is vacuous. Compare several proper prefixes, and distinguish a near-tied eigenvalue cluster's subspace from the labeling of columns within it.

**Keep \(r_C\) as one diagnostic.** It is invariant to column permutations and sign flips, strongly weights high-energy directions, and cannot see stale \(V\). An exact eigenbasis can beat QR10 on \(r_C\) while giving a different next Adam update. The normalized correlation and floored inverse-variance metrics already in your harness help with tail directions. Report distributions by shape, side, age, and time since QR. A mean of 0.38 and a maximum of 0.975 warrant finding out *which* factors produce the tail. In a gradient replay I would prioritize the cosine **and norm ratio** of the world update relative to BF16 QR10, immediately after a refresh and over the next ten calls. Cosine alone misses a denominator-induced step-size error.

**Treat call 500 as a discriminator.** The large early gap survives in the FP32-state warm pilot. That makes BF16 storage an insufficient explanation for the early loss difference. The short-control fractions are \((5.018046-5.001514)/(5.018046-4.877601)=11.8\%\) at 500 and \((3.998679-3.993939)/(3.998679-3.985905)=37.1\%\) at 2000. Both denominators use the unmatched FP32 reference. BF16 QR20 also has better CE than warm at those two evaluations, so the evidence favors testing what a higher-quality refresh buys, as well as its frequency. I would evaluate CE at equal calls **and** equal training wall time; the two comparisons answer different questions. The persistent late gap makes a call-500 success insufficient for selecting a final method.

## Other pitfalls that affect interpretation

- **The Frobenius cap is a dimension-dependent motion budget.** If there are \(p\) independent planes with equal skew coefficient \(x\), then \(\|K\|_F/\sqrt2=\sqrt p\,|x|\), so after clipping each plane gets at most \(0.5/\sqrt p\). The spectral norm in this example is only \(|x|\). A large factor can therefore be severely under-rotated even when every individual rotation would be safe. The shared damping scale further suppresses gaps in a long weak tail. Log the *applied* motion, not just the proposed skew.
- **Refresh age and update age differ.** At age 10 the parameter uses the *old* basis; the first parameter update using the new one is at age 11. A replay's post-refresh number at age 10 describes the basis available for age 11. Moving refresh ahead of the update changes how the current gradient influences its own basis and requires transporting history before the EMA. It is a different experiment.
- **BF16 casts alter more than orthogonality.** Near a well-scaled entry, BF16 unit roundoff is about \(2^{-8}\). The covariance increment is \(0.05(P_{ij}-C_{ij})\), so an entry can stick when this is below roughly half a local storage ulp, with the exact threshold depending on the mantissa. The same issue affects moments. Symmetry survives a symmetric rounding path; PSD need not. Also, the QR ordering score \(q^\top Cq\) is a literal Rayleigh quotient only for unit-norm \(q\); BF16 rounding perturbs norms and can break close ties. QR's “FP32 island” also includes a matmul that may use TF32 under your settings.
- **The optimizer equations should match the reference literally.** Your decay multiplies the adaptive update by \(1-\eta\lambda\) as well as shrinking the old weight. Compared with \(W(1-\eta\lambda)-\eta c_tU\), the difference is \(+\eta^2\lambda c_tU\). Likewise, applying \(c_t\) outside a denominator containing \(\epsilon\) corresponds to a bias-corrected denominator with \(\epsilon/\sqrt{1-\beta_2^t}\). These are small but checkable differences. Check the skipped first update, age used for correction, and clipping against the reference before attributing a residual CE gap.
- **Timing attribution needs matched arms.** Isolated warm plus transport is 54.90 ms versus a 51.47 ms event increment; isolated QR plus transport and permutation is 599.18 versus 591.04 ms. Attribute gains using whole compiled regions and report primitive times separately. A rough BF16 QR10 counterfactual is \(10.14+0.1(591.04)=69.24\) ms/call, but it is not a measured matched control. Synchronizing every host event, periodic evaluation or logging at multiples of 10 or 50, clock changes, and cross-launch differences can bias event classes. Include an uninterrupted whole-window timing with synchronization only at its boundaries, and report tails plus successful-factor counts.

## Ranking

| Rank | A — speed | B — QR tracking and loss |
|---|---|---|
| 1 | **A3, batched blocked QR.** The measured repair increment costs 11.82 ms per average call. | **B1, ordered QR tangent.** Tests the actual QR generator with nearly the existing warm machinery. |
| 2 | **A1, ordinary data movement.** Savings recur on every call; inspect materialized copies first. | **B2, leading QR panel.** Gives an exact-arithmetic guarantee on a tunable part of the target. |
| 3 | **A4, one polar correction.** Easy to falsify in replay and removes two cubic products. | **B3, grouped QR.** One cubic product and a direct way to test how much coupling must be retained. |
| 4 | **A2, symmetric products.** Applies to both ordinary and warm work; most promising toward large \(d\). | **B5, second-moment transport.** Directly tests a loss mechanism invisible to every covariance metric. |
| 5 | **A6, time-dependent QR budget.** Can trade later repairs for early accuracy with a specified total cost. | **B4, shifted Cholesky-QR.** Close to the target when stable, but timing and conditioning restrict its usable range. |

**Top three pitfalls**

1. Optimizing a gap-based diagonalization step under a global motion cap when the desired QR map uses ordered pivots and can move strongly at zero gap.
2. Selecting by average \(r_C\) while missing column identity, tail directions, and the denominator history that actually scales the update.
3. Attributing the loss or speed difference to warm refreshes before obtaining the matched compiled BF16 QR10 arm.

**Five measurements I would want next**

1. The matched BF16 QR10 CE and timing, paired through call 500 and 2000, followed through 10k for the selected comparison.
2. Same-input QR agreement and chained QR10 agreement by factor and age, including ordered prefixes, column turnover, and tail correlations.
3. Warm cap multiplier, spectral norm of applied skew, pivot/gap distribution, and spectral Gram error after zero, one, and two corrections.
4. A short per-step gradient replay with \(M,V\), reporting world-update cosine and norm ratio through the ten calls after refresh.
5. A whole-event device/host timing breakdown covering packing, covariance, \(CQ\), QR factorization, explicit Q, warm GEMMs, and moment transport, reconciled against uninterrupted whole-step timing.