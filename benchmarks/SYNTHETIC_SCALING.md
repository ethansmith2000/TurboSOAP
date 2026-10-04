# Synthetic SOAP runtime scaling

This mock trainer separates dimension-dependent cost from real CIFAR/OpenWebText
learning. It uses actual ResearchSOAP, including covariance accumulation,
projections, Adam moments, m reexpression, QR v permutations and warm retraction.
It makes no claim about synthetic loss, equal-quality schedules or actual LLM
throughput. Existing optimizer defaults are unchanged.

The bias-free block has square and 4x expansion/contraction matrices with GELU.
Default widths 192/768/1536 give factors up to 768/3072/6144; activation rows
64/512 vary model work independently. One-sided and two-sided results are
separate algorithmic configurations. Eager BF16 model operations, FP32 parameters
and optimizer state/products with TF32 permitted; strict FP32 Gram diagnostics.
No attention, compiler, input pipeline or distributed communication is included.

Three fresh rounds alternate QR/warm order. Each warms the model path three
times, then measures fresh optimizer-state initialization (including eigh)
separately. Initialization is not included in steady cadence estimates. Force
step counters to sample ordinary, QR-refresh, warm-refresh and warm QR-reset
events. Two event warmups are discarded and five samples retained. This also
forces bias-correction age: these states are timing probes, not valid learning
histories. Each sample synchronizes and measures model/backward, optimizer,
whole-step CUDA elapsed and whole-step wall latency. Elapsed CUDA intervals
include host dispatch gaps. Retain raw samples and between-round ranges.

Inspect moments, parameters and basis geometry after initialization and after
each event type, before another reset can repair drift. Stop a case on a state
geometry guard failure (nonfinite state or normalized Gram error above .05),
record its per-factor diagnostics and continue the independent grid cases.
Exclude any method/configuration with a rejected round from cadence estimates
and ratios. Nonfinite loss or an unexpected execution error stops the job.
This is a guard, not geometry equivalence.
Record allocated state bytes and peak allocated bytes excluding diagnostics.
Verify all requested factor dimensions are active; never silently skip large
factors to fit a speed benchmark.

Estimate QR10 with event weights ordinary/QR .9/.1; QR20 .95/.05. Warm10+QR200
uses ordinary/warm/reset .9/.095/.005. Per-round event means are weighted, then
round estimates summarized. QR20 reuses QR event samples. CPU tests verify these
weights against natural complete 200-update cycles. Estimates are **not measured
sustained schedule throughput**. Any target speed claim requires natural cycles
in the intended trainer and matched precision/compile settings and loss quality.

Freeze a new bundle from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python benchmarks/prepare_synthetic_scaling.py --bundle <new-directory>
```

Optional `--widths`, `--tokens`, `--rounds` and `--maximum-seconds` define an
explicit new bounded grid. Do not silently extend a running job. Run its frozen
`source/benchmarks/benchmark_synthetic_scaling.py --bundle <directory>` with one
lifetime GPU claim and private supervisor, no queue or automatic restart. Default
internal budget 1200 seconds and external timeout 1500 seconds. Check disk before
launch. No weights, checkpoints or moment snapshots; compact evidence under
20 MiB. All tensors are discarded after timing.

```bash
PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python benchmarks/summarize_synthetic_scaling.py --bundle <directory>
PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python benchmarks/plot_synthetic_scaling.py --bundle <directory>
```

Covariance beta .999, Adam (.9,.999), LR .0005 and carry/permutation v policy are
fixed here. Beta/cadence/rotation-cap/LR interactions belong in learning studies;
this map cannot establish those choices. Do not combine synthetic times and CIFAR
loss into an unmeasured time-to-quality curve. No automatic grid expansion or
additional real-data training is part of the speed benchmark.

The [initial completed map](../../optimizer_replay_results/soap_synthetic_scaling_v3_20261002/README.md)
contains explicit numerical rejections. A separate natural-cycle mock check also
rejects warm, including one configuration that passed the short event probe.
Treat surviving event ratios only as conditional costs. Validate retraction
stability and include any repair cost before claiming a useful training speedup.

For the next shallow width-2048/4096 natural-cycle test, see
[precision and compilation](LARGE_PRECISION.md). This uses square factors;
the original width-1536 block already had 6144-dimensional factors. Preserve
that distinction when interpreting size effects. Production defaults stay fixed.
