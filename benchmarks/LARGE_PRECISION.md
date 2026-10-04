# Large-factor precision and compilation screen

Use a depth-one square projection at widths 2048 and 4096 with GELU, 64 activation
rows and four repeated synthetic input/target batches. Both square factors must
be active. This changes architecture from the earlier 4x MLP: width 4096 here
means factors of 4096, whereas width 1536 there already included factors of 6144.
Report model width, matrix shape and actual factor dimensions separately. Reducing
depth reduces aggregate memory/work; it does not shrink each factor's solver cost.

Run two fresh identical-seed timing repeats in forward/reverse arm order. Each
uses cold initialization plus a natural 200-update cycle, including all Adam,
covariance, projection, first-moment transport and permutation/carry-v work.
Do not force counters. QR10/QR20 and warm10+QR200 are distinct schedules; record
event counts and raw synchronized model/backward, optimizer and whole-step CUDA
and wall intervals. Initialization, diagnostic checks and compilation preflight
are outside the 200-update timing summary. Only one reset occurs per warm cycle.
Keep between-repeat ranges; these are short latency probes, not full-trainer
throughput, convergence or time-to-quality measurements.

Hold LR .0005, covariance beta .999, Adam (.9,.999), decay .05, epsilon 1e-8,
average rotation cap .1 and warm basis LR .5. Check geometry in strict FP32 after
every refresh and before any later reset. Reject nonfinite parameters/moments or
normalized Gram error >= .05; retain failures and omit cost summaries. This is
a failure screen, not validation of eigenbasis tracking or training quality.

Explicit arms separate FP32 matmul `highest` versus `high`, BF16 covariance
products with FP32 accumulation/storage, BF16 autocast inside the warm gauge,
compilation of the existing warm tensor kernel, and optimizer-only inference
mode. Model forward operations are BF16 in every arm; parameters and persisted
optimizer tensors are FP32. NS2 remains a failure control; NS6 is an experimental
retraction candidate with all six iterations charged to its timing. No production
defaults change. BF16 covariance/TF32 may benefit QR as well as warm; avoid claiming
algorithmic advantage by comparing differently optimized implementations.

`high` permits TF32 or other reduced internal precision; it does not identify the
actual kernel. BF16 gauge autocast is deliberately more aggressive than BF16
covariance products and is separately guarded. Do not conflate dtype of storage,
matrix multiplication and accumulation. See the installed-version
[matmul precision documentation](https://docs.pytorch.org/docs/2.11/generated/torch.set_float32_matmul_precision.html).

Compile only `warm_six(cov, q)` using Inductor, fullgraph=True, dynamic=False.
Preflight on separate noncommuting inputs, record first-call wall cost and eager
output difference; reject relative output error > .01. That tolerance is a coarse
failure guard. Exclude preflight from step timing and retain its cost for eventual
amortization. Compiler caches can make later first calls cheaper. This does not
measure whole-optimizer, model compilation, CUDA graphs or distributed execution.
[PyTorch's optimizer compilation recipe](https://docs.pytorch.org/tutorials/recipes/compiling_optimizer.html)
supports treating compilation as its own measured implementation axis.

Inference mode wraps only `optimizer.step`, including initialization. Training
forward/backward remains outside. Record inference-state tensors and test another
backward followed by an ordinary optimizer step after the screen. Inference
tensors have stricter reuse/mutation rules; this is a compatibility experiment,
not a drop-in recommendation. Never reuse the state after the exit probe, even
if it fails partway. See [inference mode](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad_mode.inference_mode.html).

Freeze with `PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python
benchmarks/benchmark_large_precision.py --prepare --bundle <new-directory>`.
Run the frozen source under the private supervisor and one lifetime GPU claim,
without wait/queue/restart. Budget: 900 seconds internal, 1050 seconds external;
44 cases, at most 8844 initialization/update calls. No weights, checkpoints or
moment snapshots. Retain compact evidence under 20 MiB; disposable compiler
cache expected below 2 GiB, keep separate from evidence. Require 5 GiB free before
launch. Broader widths/activation workloads, rectangular factors and target-model
confirmation follow only after interpreting numerical and timing evidence.
