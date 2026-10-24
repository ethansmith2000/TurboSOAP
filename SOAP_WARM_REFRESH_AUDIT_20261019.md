# SOAP warm-refresh implementation audit

Date: 2026-10-19. Scope: the tested J1 control, named highest_residual20, with warm refreshes every20 matrix-update ages and QR repair every200. The fifteen implementation files recorded in the [source audit](../manifold_opt/soap_warm_refresh_audit_20261019_evidence/before.json) match frozen seed181 byte-for-byte. This document describes the implemented algorithm, distinguishes exact-arithmetic motivation from approximation, and lists questions for joint review. It does not change code or numerical policy.

The currently running trial195 is pure hold/QR200 and has no warm step. Its frozen source is untouched. TransportMuon is a separate polar-factor tracker and is outside this audit. The generic SOAP class, the original May implementation and the measured J1 research hook must not be treated as interchangeable.

## 1. State and coordinates

For one matrix parameter \(W\in\mathbb R^{m\times n}\), maintain:

- Uncentered gradient-Gram EMAs \(C_L\in\mathbb R^{m\times m}\) and \(C_R\in\mathbb R^{n\times n}\).
- Square bases \(Q_L,Q_R\), intended to be approximately orthogonal and to approximately diagonalize those factors.
- First moment \(M\) and elementwise second moment \(V\), both \(m\times n\), expressed in the current two-sided basis.
- Matrix update age t and refresh/transport counters.

All of these tensors are stored in FP32 in the measured J1 recipe. There is no J1 generator EMA, per-column normalization state, acceptance controller or persistent second-moment cross-covariance.

Projection and reconstruction are

$$P_Q(X)=Q_L^T XQ_R,\qquad P_Q^{-}(Z)=Q_LZQ_R^T.$$

The superscript minus denotes the implemented back-projection; it is an inverse only for exact orthogonal square bases.

The factors called covariance in the code are not centered covariances, per-example Fisher estimates or Hessians. They are EMAs of the clipped minibatch gradient's matrix Grams:

$$C_L\leftarrow\gamma C_L+(1-\gamma)gg^T,\qquad
C_R\leftarrow\gamma C_R+(1-\gamma)g^Tg.$$

The trainer clips the complete model gradient before the optimizer sees g. Its tested values are global clip1, covariance beta \(\gamma=.99\), Adam betas \((.9,.999)\), epsilon1e-8, matrix LR .0005 and weight decay .05. Covariance bias correction is not applied. The compared LLM recipe applies two-sided SOAP to hidden attention/MLP matrices, with tied embedding/head and norms on auxiliary AdamW; the stated LR is the peak value under the declared schedule.

Sources: [projection helpers](soap.py#L478), [covariance accumulation](soap.py#L652), [research step](soap_reference.py#L80), [frozen recipe](../optimizer_replay_results/llm_soap_sparse_screen_seed181_frozen_20261015/config.json).

## 2. Initialization and actual cadence

On the first gradient-bearing call, the matrix branch allocates zero M/V/C, accumulates both Grams, and initializes each basis by eigendecomposition of the symmetrized covariance. It tries FP32 eigh, retries in FP64 on RuntimeError, casts the result to FP32 and reverses the ascending eigenvector order. This call does not perform the matrix parameter update or its weight decay. The auxiliary AdamW branch still steps.

Thus external optimizer call k corresponds to matrix age \(t=k-1\). Later calls with missing gradients do not advance that parameter's age.

For J1, the refresh is performed after the current parameter update and covariance accumulation:

- Ages20,40,...,180: warm refresh, first appearing on calls21,41,...,181.
- Age200: QR repair instead of warm, on call201.
- The same pattern repeats, with nine warm refreshes and one QR repair per200 update ages.

The new basis is used by the following gradient/update, not retroactively by the update that triggered its construction. QR20 refreshes every20 ages. Hold200 uses only the QR event at every200 ages; it still updates covariance/M/V on intervening steps.

## 3. Exact order of an ordinary matrix update

Given the previous bases, increment t and compute

$$z=P_Q(g),$$
$$M\leftarrow\beta_1M+(1-\beta_1)z,\qquad
V\leftarrow\beta_2V+(1-\beta_2)z^{\odot2},$$
$$U=P_Q^{-}\left(\frac{M}{\sqrt V+\epsilon}\right),\qquad
a_t=\eta_t\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}.$$

The literal parameter update is

$$W^\star=W-a_tU,\qquad W\leftarrow(1-\eta_t\lambda)W^\star.$$

This preserves the reference's epsilon/bias convention and its decay-after-update ordering. It is slightly different from writing decay on the old W and then subtracting the unscaled update. There is no matrix Nesterov term here.

Then accumulate covariance using the same already-computed g, and perform the basis refresh if due. A second gradient is not computed at the updated W.

## 4. One-factor J1 warm refresh: complete equations

Repeat this independently for each pair \(C,Q\), where both are \(d\times d\). Building both new bases does not itself mutate M/V; those are handled once afterward.

### 4.1 Normalize covariance scale

$$s=\max(|\operatorname{tr}C|/d,10^{-12}),\qquad \bar C=C/s.$$

This normalization is local to direction computation; it does not replace the stored C.

### 4.2 Express covariance in the old coordinates

$$B=\operatorname{sym}(Q^T\bar CQ),\qquad
\operatorname{sym}(X)=\tfrac12(X+X^T).$$

Both dense products use FP32 operands under the highest float32-matmul context. This is not the BF16 covariance-prefix variant retained elsewhere in the repository.

### 4.3 Construct the damped Jacobi generator

Let

$$b_i=B_{ii},\quad \Delta_{ij}=b_i-b_j,\quad
\tau=.01\max(\max_i|b_i|,10^{-12}).$$

Elementwise, compute

$$\Omega_{ij}=
-\frac{B_{ij}\Delta_{ij}}{\max(\Delta_{ij}^2+\tau^2,10^{-24})},$$
$$K=\tfrac12(\Omega-\Omega^T),\qquad A_{\rm raw}=.5K.$$

The outer .5 is the basis step size; the inner .5 is antisymmetrization. They have different purposes. When B is exactly symmetric, Omega is already skew, so \(K=\Omega\); the effective step is .5Omega, not .25Omega.

The diagonal entries \(b_i\) approximate eigenvalues only near a diagonal basis. Their differences are not actual eigengaps in general.

### 4.4 Apply one tensor-wide rotation cap

$$r=\|A_{\rm raw}\|_F/\sqrt2,\qquad
\alpha=\min\left(1,\frac{.5}{\max(r,10^{-12})}\right),\qquad
A=\alpha A_{\rm raw}.$$

There is no upward normalization of a small step. For a real skew matrix, \(\|A\|_2\leq\|A\|_F/\sqrt2\), so this caps that upper bound at .5 in exact arithmetic. It is not an estimate of the actual spectral norm, nor a per-entry angle cap. The raw_bound diagnostics refer to the undoubled J1 raw generator.

### 4.5 Form the near-orthogonal candidate

$$Y=Q+\operatorname{FP32}
\left(\operatorname{BF16}(Q)\operatorname{BF16}(A)\right).$$

Q remains FP32 in storage; the candidate product explicitly casts both operands to BF16. The FP32 addition preserves the old Q rather than replacing it with a BF16 copy. Product arithmetic is the PyTorch/backend BF16 matmul path, not a promise of exact products.

### 4.6 Retract with two residual corrections

Starting from \(Y_0=Y\), for j=0,1:

$$E_j=Y_j^TY_j-I,$$
$$D_j=\operatorname{FP32}
\left(\operatorname{BF16}(Y_j)\operatorname{BF16}(E_j)\right),$$
$$Y_{j+1}=Y_j-\tfrac12D_j.$$

The Gram product and residual subtraction are computed in highest-context FP32; the correction product is BF16 and the update subtraction is FP32. Store \(Q_{\rm new}=Y_2\) in FP32.

In exact arithmetic each correction is \(Y\mapsto1.5Y-.5YY^TY\), a cubic Newton-Schulz polar correction. The implemented residual form has different finite-precision behavior from evaluating the two large terms directly. Two corrections do not constitute an exact polar decomposition.

Code chain: [arm installation](benchmarks/llm_sparse_screen_kernel.py#L23) → [LiveKernel](benchmarks/native_damped_kernel.py#L9) → [PolicyKernel](benchmarks/native_residual_named_kernel.py#L22) → [prefix extraction](benchmarks/common_y_kernel.py#L8) from [warm_split](benchmarks/precision_atoms_kernel.py#L8), followed by [residual_retract_two](benchmarks/residual_retraction.py#L4). The prefix extractor stops before warm_split's own first Gram statement; its old retraction body is not executed as well. Compiled callables use default mode, fullgraph=True, dynamic=False, scoped highest precision and restored outside flags.

## 5. What the direction is trying to accomplish

This is a damped pairwise Jacobi correction, motivated by reducing

$$F(Q)=\|\operatorname{offdiag}(Q^T\bar CQ)\|_F^2.$$

For an exact orthogonal Q and skew A, the exponential perturbation \(Q\exp(uA)\) gives

$$B'(0)=BA-AB.$$

Close to diagonal B, its off-diagonal change is approximately \(\Delta_{ij}A_{ij}\); cancelling \(B_{ij}\) suggests \(-B_{ij}/\Delta_{ij}\). Our step damps that inverse and halves it before applying the global cap.

In exact arithmetic the full objective's directional derivative is \(2\sum_{ij}B_{ij}\Delta_{ij}A_{ij}\). The chosen sign is a nonpositive direction for this infinitesimal objective. That does not certify descent after a finite step, BF16 rounding, approximate retraction, or drift from exact orthogonality.

Tikhonov damping drives the proposed rotation to zero when a diagonal gap vanishes. This prevents inverse-gap explosions, but can also stall at a non-diagonal B with equal diagonal entries. For example \(B=\left[\begin{smallmatrix}1&.9\\.9&1\end{smallmatrix}\right]\) gives zero J1 rotation despite substantial off-diagonal mass and distinct true eigenvalues1.9/.1. This is an algebraic property of the implemented direction, not an observation that the training runs reached this state.

## 6. First-moment transport after both bases are built

Reconstruct the currently represented world momentum using the old bases:

$$M_{\rm world}=Q_{L,\rm old}M Q_{R,\rm old}^T.$$

Then store

$$M_{\rm new}=Q_{L,\rm new}^TM_{\rm world}Q_{R,\rm new}.$$

The measured J1 recipe scopes all four rectangular products to highest-context FP32. There is no first-moment decay or gradient incorporation inside the refresh; that already happened in the ordinary step. There is no transport on a held-basis ordinary step.

For exact orthogonal square bases this preserves represented world momentum. Approximate Q makes the back-project/project operation approximate. The actual-M diagnostic reconstructs before/after the real transport and requires relative error <=1%; a Gram guard alone does not establish that bound.

Source: [refresh transaction](soap_reference.py#L44), [transport helper](soap.py#L783).

## 7. Diagonal second-moment handling

On a warm refresh, V is carried unchanged. This is deliberate approximation, not exact second-moment transport. V contains only coordinatewise squares; exact rotation of its historical covariance generally needs cross-coordinate moments that are not stored.

On QR repair, V is permuted along each axis by that factor's column-sorting permutation. It is not otherwise rotated. Even this reference policy does not reconstruct a full historical second moment for the new coordinates.

The optional squared-overlap variance transport in the generic class is a different research policy and is not selected in J1's permutation mode. Physical-M fidelity, diagonal-V history fidelity and learning quality must be assessed separately.

## 8. What QR repair actually computes

For each factor, use the unnormalized current covariance C:

$$\pi=\operatorname{argsort}_{\downarrow}
       \operatorname{diag}(Q_{\rm old}^TCQ_{\rm old}),$$
$$Q_{\rm new}=\operatorname{qr}(C Q_{\rm old}[:,\pi]).Q.$$

This is one warm-started orthogonal/subspace-iteration step, not a full eigendecomposition. It still depends on the old basis. In this ResearchSOAP path there is no additional sign canonicalization or overlap-based column alignment beyond this sorting. Transport M and permute V as above, then install both new Qs.

The reference QR work executes under the ordinary high/TF32 context; highest precision is separately scoped around M transport. The J1 prefix/Gram uses highest context. Precision therefore belongs in the comparison specification, not merely in a common FP32-storage label.

## 9. Numerical checks and what they do not prove

The frozen sparse-screen contract requires finite source state, declared two-sided coverage/cadence and:

- Absolute Gram error \(\|Q^TQ-I\|_F\leq.05\).
- Normalized Gram error \(\|Q^TQ-I\|_F/\sqrt d\leq.005\).
- Actual represented world-M transport error <=1% at declared disposable qualification events.
- Own compiled-versus-eager basis relative discrepancy <=1e-4, movement discrepancy <=.1 and movement cosine >=.995 at declared probes.
- Exact initial comparisons, full disposable/primary reset prefixes, evaluation restoration and registered-workspace release.

In seeds180/181 the own-eager J1 probes are calls21/41/221 of the disposable phase. Actual-M checks include each policy's first refresh and J1's first age200 QR repair. Geometry/finite-state/counter checks are at the declared common guard calls, not a full matrix audit on every training call. Scalar loss/gradient finiteness is checked during the trajectory.

These are numerical/execution qualifications, not proof of closeness to QR updates or equivalent learning. The eager reference uses the same chosen mixed algorithm; it is not an exact mathematical polar/eigenbasis oracle. There is no J1 online line search, acceptance test, adaptive basis step controller or error-triggered fallback in this measured path. Failed hard checks abort the experiment; QR200 is scheduled repair, not an adaptive rescue.

## 10. Complete refresh cost

For each d-by-d warm factor, the core has seven dense matrix products: two to form B, one BF16 Q@A candidate product, and two products for each of the two retraction corrections. Four products use highest-context FP32 operands (two B products/two Grams); three explicitly use BF16 operands (candidate/two corrections). Add normalization, symmetrization, pairwise gap/damping work, reductions, casts and allocation/traffic.

For each m-by-n parameter, the refresh also transports M with four rectangular matrix products. Ordinary gradient projection, Adam-style arithmetic, reconstruction and covariance accumulation remain part of the optimizer even when no refresh happens. There is no cheaper asymptotic complexity from dense J1: its factor work remains cubic.

Nine warm and one QR event occur per200 steady update ages. If O/W/R are complete ordinary/warm/QR optimizer-step costs, the cadence estimate is \(.95O+.045W+.005R\); QR20 is \(.95O+.05R\). This explains why refresh latency, average optimizer latency and complete training throughput are different quantities. Include actual diagnostics/setup in their separately reported all-in costs.

## 11. Nearby variants that must not be conflated with J1

The experimental damped20 branch uses an off-diagonal-energy generator without J1's inverse-gap denominator, then limits its scalar step with a diagonal-curvature surrogate. Its raw generator is .5 times the skew form of \(-B_{ij}\Delta_{ij}\). The selected scalar is the minimum of the same global cap scale and the surrogate optimum, clipped to[0,1]. It uses the same candidate BF16 product and two residual corrections. This is not simply extra damping applied to J1; the curvature surrogate is not a full-Hessian or descent certificate. See [generator and prefix](benchmarks/tracking_curvature_kernel.py#L13).

Jacobi2 doubles J1's already capped generator: \(A_2=2A_1\), with effective bound1 rather than.5. It does not merely double the raw direction before reapplying the original .5 cap. Cubic3 uses that same doubled candidate with three residual corrections. The original two-correction version failed a geometry/M qualification; the three-correction version recovered numerical admissibility but did not show a consistent learning advantage over cheaper J1. See [doubling](benchmarks/llm_jacobi2_kernel.py#L7), [third correction](benchmarks/llm_cubic3_kernel.py#L7), [learning result](../optimizer_replay_results/soap_llm_cubic3_learning_20261014/REPORT.md).

## 12. Difference from the original May warm implementation

The historical source used the direction \(CQ-Q\operatorname{diag}(Q^TCQ)\) (or optional Brockett), a skew/eigengap helper, column RMS/EMA normalization, a step controller, BF16 basis storage and a different degree-five polar polynomial. It could recalibrate and fall back, reindexed V on warm events and round-tripped M every gradient-bearing call. These are substantive algorithm/arithmetic differences, not just implementation names.

For an exact square orthogonal Q and symmetric C, the tangent-style raw direction satisfies \(Q^TD=\operatorname{offdiag}(Q^TCQ)\), which is symmetric. Its skew projection is zero before nonuniform normalization and finite-precision effects. This explains why importing its name/intuition into the current skew-Jacobi implementation is misleading; it does not explain the historical training trajectory or negate its reported H200 timing.

The [historical audit](../optimizer_replay_results/soap_historical_audit_20261008/README.md) records which source/defaults are recovered and which original run settings remain unknown. It must not be read as an exact reproduction of that run.

## 13. Questions for our joint audit

1. Does a diagonal-gap approximation miss important collective rotations or stall when diagonal entries are similar?
2. Does setting tau from the largest diagonal over-damp useful small-eigenvalue directions?
3. Does one tensor-wide Frobenius cap suppress distributed movement increasingly with factor dimension? This is a hypothesis, not a width law.
4. Do warm carry-V history changes help or hurt useful adaptation relative to more frequent QR/permutation?
5. Do mixed candidate/correction products preserve the useful rotation, especially when the cap makes movement tiny?
6. Does warm motion earn its complete cost versus simply holding the basis? Paired180/181 currently favor hold200 in their short one-layer setting.
7. Which original/H200 speed benefits remain after matching timing boundaries, precision, correction work, transport and implementation?

No tuning or experiment is implied by this list. The first discussion should verify the implemented equations and intended coordinate-history semantics before changing any of them.
