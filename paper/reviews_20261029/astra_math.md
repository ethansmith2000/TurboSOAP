# Independent mathematical and implementation-fidelity review

Reviewed 2026-10-29. Snapshot: `paper/reviews_20261029/snapshot/DRAFT.md`, SHA256 `56f71b53c88af7c6213aaf82a630750f037ad488c896a92e7ca521b3f597d694`. Section and manuscript line references below refer exclusively to this snapshot. Repository paths are relative to `/workspace/TurboSOAP`.

**Disposition: minor revision for precision reproducibility and diagnostic definitions.** I found no substantive equation, sign, indexing, or update-order error in §§3.1–3.3 relative to `bf16_soap.py`. The scoped loss/time/memory conclusion is defensible on the supplied evidence. Two medium-priority reporting omissions and one low-priority mathematical wording issue should be corrected using existing sources. None requires another experiment, a different comparator, or an optimizer change.

This review used source inspection, algebra, read-only standard-library parsing of saved results, source SHA256 verification, and official PyTorch documentation. All four production-source hashes in the snapshot configuration match the files inspected. No GPU use, Torch execution, training, dependency installation, production/manuscript mutation, or consultation of other reviewers' new reports occurred. Only this report was written. Historical review conclusions were not used as a substitute for independent checking.

## Findings requiring textual correction

### M1 — Medium: the precision specification omits two harness settings needed to reproduce it

**Location:** §3.1, lines 84–98 and 132–144; §3.3, lines 180–195; §4, lines 264–268. Companion `REPRODUCIBILITY.md`, optimizer example at lines 39–60 and precision paragraph at lines 69–73; `configuration.json`, optimization/runtime fields.

**Classification:** reproducibility omission and potentially misleading precision scope, not evidence that the completed runs used the wrong precision.

The equations carefully distinguish BF16 output rounding from FP32 accumulation, but the optimizer class alone does not enforce the entire runtime policy. The training runner explicitly sets `torch.set_float32_matmul_precision('high')` and `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False` in `benchmarks/bf16_storage_screen.py:184–187`. Neither setting appears in the standalone example or the configuration record. The synthetic profile specifies `high` in §5.4, but this does not document the main training setting.

This matters in two distinct ways. BF16 GEMM can otherwise permit reduced-precision intermediate reductions. Also, FP32 tensor inputs/outputs do not by themselves imply strict FP32 internal multiplication: `high` permits faster lower-precision implementations. In particular, `qr_basis` casts C/Q to FP32 and calls ordinary `bmm` (`bf16_soap.py:58–64`), inheriting the surrounding policy. The reference likewise uses ordinary FP32-tensor products (`soap_reference.py:54–55`). The FP32 QR/eigh solver description remains appropriate; it should not imply that every product preparing their inputs uses the `highest` matmul setting. PyTorch documents both distinctions in its [2.11 matmul-precision API](https://docs.pytorch.org/docs/2.11/generated/torch.set_float32_matmul_precision.html) and [2.11 numerical-accuracy notes](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms).

**Recommended correction:** add near §3.1's product definitions:

> The reported CUDA runs disable BF16 reduced-precision reductions and set FP32 matmul precision to `high`. FP32 denotes tensor/solver precision; FP32-tensor matrix products inherit that backend setting rather than requiring strict IEEE FP32 multiplication. Geometry diagnostics temporarily use `highest`.

Add those two exact settings to the reproducibility example and machine-readable configuration. `bf16_soap.py:195–198` already supplies the compilation precision-cast option internally, so that option does not need an additional user setup step. Retain the original stock-beta QR10 comparator; this correction does not justify replacing it or weakening the observed comparison.

**Unresolved question:** is the standalone example intended to reproduce the reported precision policy, or merely demonstrate construction? If the latter, explicitly label the missing harness setup. Existing code settles what the experiments did; no new measurement is needed.

### M2 — Medium: define the numerical diagnostics, especially the two different “physical-M” quantities

**Location:** §5.4, lines 382–389; §5.5, lines 400–412; §4, lines 277–283. Companion `CLAIMS.md`, actual-M/Gram and local-update rows.

**Classification:** incomplete metric definition. The quoted numbers match the saved evidence; the concern is what a reader may infer from them.

The manuscript gives normalized Gram error and physical-M error without formulas, denominators, or a precise statement of the reconstructed reference. Moreover, the training and synthetic-profile “physical-M” metrics differ in both reference and reconstruction precision:

* In training, `momentum_before` forms the newly rounded moment in the old bases, then reconstructs its world representation with FP32 products under `highest` (`benchmarks/bf16_storage_screen.py:91–104`). `diagnostics` compares the stored, transported moment reconstructed in the new bases against that value, again under `highest` (`:107–129`). This is a local before/after refresh discrepancy within the candidate, not deviation from stock SOAP, total accumulated history error, or a bound on the adaptive direction.
* In the QR profile, `next_probe` reconstructs `world_m` as `torch.bmm(new, moved)` while both tensors are BF16 (`benchmarks/bf16_qr_attribution.py:52–67`). Thus this reconstruction itself has BF16 output rounding. `relative` subsequently converts those rounded outputs to FP32 and compares original versus reuse (`:48–49`, `:90–91`). This is not the strict-FP32 reconstruction used by the training diagnostic. The fixed right identity makes the omitted right multiplication harmless, but does not remove the extra BF16 rounding.

The main-run geometry files reproduce the manuscript maxima: seed 204, `0.00992422737` physical-M and `0.00236585326` normalized Gram; seed 205, `0.01035554428` and `0.00236707583`. Sources: `reports/bf16_replicate_20261027_seed204/runs/bf16/geometry.json` and `reports/bf16_full_horizon_20261026_seed205/runs/bf16/geometry.json`. The profile's `0.00090631208` maximum comes from `rows[*].numerical.state_relative.world_m` in `reports/bf16_qr_attribution_20261027/result.json`. These are correctly rounded in the draft, but should not be read as interchangeable errors.

**Recommended correction:** define in §5.5, with all stored tensors promoted to FP32 for these diagnostic products:

\[
e_Q=\frac{\|Q^\top Q-I\|_F}{\sqrt d},\quad
P_{\rm before}=Q_L R_b(M_f)Q_R^\top,\quad
P_{\rm after}=Q_L^+M^+(Q_R^+)^\top,
\]
\[
e_M=\frac{\|P_{\rm after}-P_{\rm before}\|_F}
{\max(\|P_{\rm before}\|_F,10^{-12})}.
\]

State that maxima are over sampled factors/matrices and calls, and that these are local refresh diagnostics. In §5.4 replace “maximum physical-M difference” with “maximum relative difference between the BF16-rounded reconstructed world moments of the original and reuse probes.” Specify that the profile relative difference uses the original output's Frobenius norm, floored at `1e-30`, as denominator. Its next-update comparison is also original-versus-reuse, not comparison to an exact optimizer.

Keep the existing statement that finite screen crossings are descriptive. No acceptance threshold or recalculation from unavailable tensor checkpoints is required.

**Unresolved question:** do the authors intend the identically named quantities to be directly comparable across §§5.4 and 5.5? The saved code does not support that interpretation. Distinct labels and definitions resolve it using current evidence.

### M3 — Low: “an orthogonality floor” is too absolute as a mathematical statement

**Location:** §5.5, lines 408–409: “BF16-rounded bases have an orthogonality floor.”

**Classification:** overgeneralized wording, not a failure of the measured numerical diagnostics.

BF16 representation does not impose a universal positive lower bound on orthogonality error: identity and signed permutation matrices are exactly representable and exactly orthogonal. The draft's own §3.3 stall example uses such a basis. Generic dense QR bases do acquire rounding error, but its magnitude depends on their entries and on the chosen error norm. A fresh QR rounded to BF16 provides an empirical comparison, not a proven lower bound on every possible basis.

The initial observations in the two saved main-run `geometry.json` files contain `fresh_bf16_rounded_qr_normalized_gram`, computed by `benchmarks/bf16_storage_screen.py:120–123`. Across those observations it ranges approximately from 0.001757 to 0.001790. This directly supports a qualified description without a universal-floor claim.

**Recommended correction:**

> Rounding the observed dense bases to BF16 introduces measurable orthogonality error even after a fresh FP32 QR. This provides an empirical rounding baseline, not a universal lower bound.

**Unresolved question:** was “floor” intended only as shorthand for this observed baseline? If so, the proposed wording fully addresses the issue; no additional experiment is needed.

## Mathematical checks that passed

1. **Projection, moments, bias correction, and decay (§3.1).** The two BF16-output gradient products, FP32 moment temporaries, BF16-rounded normalized direction, and final FP32-output product match `bf16_soap.py:85–98`. Epsilon remains inside the uncorrected denominator, with the scalar bias correction outside; replacing this with a textbook bias-corrected-variance denominator while leaving epsilon unchanged would be a different convention. Decay applies to the already updated parameter, matching both `bf16_soap.py:94–95` and `soap_reference.py:104–108`.

2. **Cold start and call/age distinction (§§3.1, 3.3).** Call 1 initializes covariance and descending eigenvector bases, leaves moments zero, and skips the matrix update (`bf16_soap.py:212–231`). Age is call minus one. Thus the first warm event is call 11 and first QR repair is call 51. A 10,000-call run has 9,999 matrix updates, with 800 warm and 199 QR events. The description does not incorrectly place the repair on call 50. The auxiliary update exception is explicitly disclosed.

3. **Warm generator and residual corrections (§3.2).** The sign, half step, normalized diagonal gap, damping, skew-Frobenius cap, BF16 cast placement, and both residual iterations match `bf16_soap.py:35–55`. Since B is symmetric and h antisymmetric, A is skew-symmetric in exact arithmetic. For an orthogonal Q and a small motion, the off-diagonal first-order change is `(d_i-d_j)A_ij`; the stated negative sign reduces that component in the local diagonal approximation. This observation does not prove global descent, diagonalization, or convergence, and the draft correctly makes none of those claims. The residual step targets Gram error, not covariance diagonalization. Define `sym(X)=(X+X^T)/2` on first use as a small notation improvement.

4. **QR reuse identity (§3.3).** The score is exactly `diag(Q^T C Q)` in exact arithmetic without requiring orthogonality. Gathering columns of `CQ` is algebraically equivalent to multiplying C by the correspondingly gathered Q. The three-versus-one product count agrees with `soap_reference.py:54–55` and `bf16_soap.py:60–64`. The draft appropriately separates that identity from finite-precision order changes and from measured whole-training speed.

5. **First- and second-moment history (§3.3).** The first moment is rounded before four BF16-output transport products (`bf16_soap.py:68–74`, `:96–110`), as the equations state. QR permutes both variance axes; warm refresh leaves the rounded variance unchanged. Neither is a full covariance-of-gradients transport. Even without arithmetic rounding, transpose projection through a nonorthogonal new basis need not preserve the reconstructed world moment. These approximation limits are already acknowledged and are not new defects to be counted against the completed loss comparison.

6. **Stall counterexample (§3.3).** At Q=I and the stated C, B=C has equal diagonals, so h and A vanish; Y remains I and both Gram residuals vanish. All entries are exactly BF16-representable, so the stated rounding boundaries do not invalidate the example. C has distinct eigenvalues 1.5 and 0.5: this is a coordinate-diagonal stall, not necessarily an eigenvalue degeneracy. QR of C moves away from I, although one QR power step is not an exact eigendecomposition. The draft correctly confines the example to disproving unconditional warm diagonalization and does not claim a measured training incidence.

7. **Storage arithmetic (§3.4).** The six-tensor formula `2mn+2m²+2n²` is correct. The declared shapes and multiplicities give 622,854,144 elements: 2,491,416,576 FP32 bytes versus 1,245,708,288 BF16 bytes. The reported candidate total is 32 bytes higher because four groups each retain FP32 age and LR scalars (`bf16_soap.py:229–230`, `:236–237`). The manuscript's approximate halving and distinction from total GPU allocation are accurate.

## Scientific interpretation and remaining scope

The equations do not establish convergence, covariance alignment, exact moment transport, or preservation of the Adam direction. The draft already says so in §§3.3, 5.5, and 6. In particular, a small first-moment reconstruction error cannot bound `M/(sqrt(V)+epsilon)` when V changes coordinates approximately; a small Gram error does not establish that `Q^T C Q` is diagonal. These are limitations already acknowledged, not newly discovered factual errors or grounds for imposing a numerical acceptance veto.

The completed two-seed loss comparison, short same-precision cadence controls, and synthetic QR attribution are scoped appropriately. I found no formula-based justification for demanding an optimized FP32 QR10 training comparator, declaring equivalence, or attributing the combined speedup solely to warm tracking. Preserve the observed endpoint gaps and early loss differences, the original implementation as primary comparator, and the limits on the short controls. Address M1–M3 through documentation and wording; the existing evidence is sufficient for this review's recommended revision.
