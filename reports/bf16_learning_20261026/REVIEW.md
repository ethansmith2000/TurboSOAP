# Completed 2,000-call pilot review

All five arms completed in64.94minutes; supervisor exit status0. Repeated audit
verified577 frozen source hashes, input/support-file hashes, all10,000 training
calls and observation/evaluation/capture counts. Initial weights, gradients and
full-panel CE match. All compiled arms stayed at12 graphs with zero graph breaks.

Final validation CE: stock QR10 **3.985905**, existing warm **3.995753**, compiled
FP32 storage **3.991560**, full BF16 **3.993939**, BF16 with FP32 bases **3.995611**.
Full BF16 trails QR10 by+.008034 and is within the small spread among warm
variants. It is not established as a better-learning optimizer. The early gap
against QR10 is larger (about+.124 at500 calls) and narrows by2000, similarly to
the existing warm method. Keep the full trajectory in view.

Mean step times, calls101..2000: QR10 **382.773ms**, original warm **337.172ms**,
compiled FP32 **316.281ms**, full BF16 **313.162ms**, BF16/FP32-Q **315.376ms**.
Full BF16 saves18.19% against original QR10 and7.12% against original warm;
only.99% relative to compiled FP32 storage in this run. Persistent matrix-state
memory halves2.320->1.160GiB; peak allocation drops9.394->8.219GiB. The main
timing gain combines batching, compilation, efficient repair and changed
arithmetic. General BF16 storage is prior work, and optimized/sparse QR
comparators remain necessary for wider performance claims.

All precision-specific guards passed. Full BF16 fails the original absolute
Gram guard (~.13), passes normalized Gram (maximum.002364), and peaks at.9924%
actual-M error at the first warm refresh. Sampled ages500..1990 peak at.5323%,
with no increasing late pattern observed. These sampled checks do not certify
unobserved calls or convergence. FP32 bases improve margin (overall.5326%) but
retain1.608GiB state and did not establish better CE.

Coordinate offdiagonal maxima near.96 at age1990 occur for all warm variants,
including original FP32 warm. Means are about.39, versus QR10 about.21 at that
observation. These observations confirm that near-orthogonal bases and small
M round-trip errors do not guarantee close covariance diagonalization. They
are measurements on each arm's own covariance, not a causal isolation or a
reason to silently alter the fixed update policy.

Advance full BF16 to one prospective10,000-call pair against stock QR10 at fresh
model seed205. Select it for memory savings and successful sampled late checks;
do not select a quality winner from endpoint differences of a few thousandths.
The new pair keeps data/order, LR, betas and numerical gates fixed. Full-horizon
quality, another paired seed and fair efficient/sparse QR controls remain open.

Evidence: `comparison.json`, `completion_audit.json`, `review_summary.json`,
per-arm scalar events/geometry and the frozen source bundle. The next protocol
is [BF16_FULL_HORIZON_20261026.md](../../BF16_FULL_HORIZON_20261026.md).
