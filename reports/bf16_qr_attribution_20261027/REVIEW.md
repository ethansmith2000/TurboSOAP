# Same-state QR product reuse

The bounded profile completed with supervisor exit0 in36seconds of supervised
wall time (32.36seconds inside the runner), well within600seconds. All frozen
source, support, manifest, service and production pins were rechecked. Eight
shape/batch pairs have the expected ten samples per component; finite state,
nonnegative V, preserved input hashes and24GiB allocation checks pass.

For batch12, original/CQ-reuse median CUDA milliseconds are768:31.210/30.914,
1536:82.053/80.535,2304:145.854/141.566,3072:225.660/215.218. Time reductions
.95%,1.85%,2.94%,4.63%. Removing two dense products improves products/sorting
more, but the QR solver dominates the complete primitive. Batch1 reductions
range3.33–7.19%. Do not attribute batching, BF16 or compiled-training effects
to this matched FP32 eager primitive comparison.

The fixed synthetic inputs are BF16-rounded C/Q promoted to FP32, not trained
states. Near-tie score order changes occur: up to160 of36,864 columns in the
largest batch, within2.68e-5 of max score. Sign-aligned basis relative difference
reaches.0932; raw coordinate-M difference reaches.877 due to signs/order. V is
permuted accordingly (relative coordinate difference up to.0404). Neither those
coordinate differences nor small Gram prove physical update equivalence.

The one-sided BF16 transport/next-update probe, with a held32D right identity,
has max physical-M difference.0009064 and next-update difference.0011161
(about.112%), cosine at least.9999994. These are descriptive finite measurements,
not a loss threshold. All raw samples and detailed numerical records are in
result.json. No bitwise or trained-trajectory equivalence claim; no production
change or new training followed. Current primary training speed comparison
continues to name the original stock implementation.
