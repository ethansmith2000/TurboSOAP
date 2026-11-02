# Fixed-input QR attribution profile

Bounded approved profile after the completed QR20/QR50 controls. No training,
optimizer changes or hyperparameter selection. Existing dimensions768/1536/
2304/3072, paired batch1 and12 (the real repeated-shape group size). Every pair
uses identical frozen generated inputs, FP32 QR arithmetic/output, matmul
precision high, eager dispatch and the same FP32 solver. Synthetic covariance
has spectrum1..4 plus a fixed low-rank perturbation; C/Q are BF16-rounded and
promoted to FP32 before timing, matching the production QR island's inputs.
This does not reproduce a trained covariance distribution.

Original QR sorting/refinement computes diag(Q^T C Q), sorts, then QR(C Qsorted):
three dense products. Reuse computes Z=CQ, columnwise sum(Q*Z), sorts Z columns,
then QR(Zsorted): one dense product. CPU checks compare the original primitive
to actual ResearchSOAP refresh and reuse to production qr_basis, with a
float64 independent oracle and variance/moment transport checks.

Measure each complete primitive, products/sort alone, and the same fixed-input
QR solver. Two warmups then ten rounds, alternating execution order, CUDA
events and synchronized wall time. Save raw samples and medians. Component
medians are informative, not an additive decomposition or whole-optimizer claim.
Pair within batch/shape; batching and FP32 product reuse are distinct from
BF16 arithmetic/storage, compilation and whole-training efficiency.

Outside timing: ordering disagreement/score band, sign-aligned and raw basis
differences, normalized Gram (strictFP32 evaluation), and actual BF16 moment/
variance transport plus a next ordinary matrix_step on a shared gradient.
The probe isolates one factor, with a fixed32D right identity. Report its
physical moment, variance and following-update differences/cosine; finite
differences are diagnostics, no invented loss threshold. Near ties can reorder
columns and do not justify a bitwise-equivalence claim. Assert finite outputs,
nonnegative variance and exact input hashes before/after. Save scalars only.

Prepare/test/freeze before GPU execution. Managed soap-bf16-qr-profile, shared
gpu-claim, no restart/retries/successors,24GiB allocation cap. Queue wait up to
7,200seconds is separate. Claimed execution uses TERM after590seconds and
KILL10seconds later, hard maximum600seconds including imports/verification,
input generation, all timing/diagnostics and final copying. Stop automation on
operational failure. No production optimization follows from this profile.
