# Reproducing the draft and inspecting the experiment

## Paper artifacts: CPU only

Run from the repository root:

```bash
python3 paper/build_artifacts.py --check
```

This checks five manuscript tables against saved scalar evidence, confirms the
two copied figure hashes, verifies four optimizer-source pins, checks shared
initialization/reference identities and repeated-prefix endpoint CE agreement,
and validates the generated evidence/configuration records. It reconstructs
tables from saved scalar summaries, without independently aggregating raw event
timings or checking every frozen harness source. The frozen campaign audits
provide broader execution checks. It uses the standard library and does not
import Torch or touch a GPU.

To regenerate those tables and records:

```bash
python3 paper/build_artifacts.py
```

The independent-review context is generated separately:

```bash
python3 paper/review_context.py
python3 paper/review_context.py --check
```

This standard-library calculation reconstructs the event-class optimizer-time
accounting, pilot/control contrasts, late observed gap bands and baseline-only
LR selection from saved summaries. Its input hashes are in
`review_context.json`. Event accounting is descriptive, not a causal
partition of cadence, compilation, arithmetic and storage.
It also extracts the saved coordinate residuals and compilation parity probe.
The new manuscript Figure 2 is rendered on CPU with:

```bash
/venv/main/bin/python paper/plot_review_context.py
python3 paper/build_artifacts.py
```

The figure's input/output hashes are in `review_figure.json`. Both original
report SVG exports remain preserved; Figure 2 now uses a separately named
review-derived plot. The QR primitive profile remains in its table and archives.

To regenerate the underlying report plots on CPU in the existing environment:

```bash
/venv/main/bin/python benchmarks/summarize_bf16_completion.py
```

The latter recreates report outputs with Matplotlib and can change SVG creation
metadata. The draft deliberately retains byte-identical copies of its selected
figures. If adopting freshly generated plots, copy both chosen SVGs into
`paper/figures/`, rebuild the paper evidence, and review the new artifact hashes.
The paper build itself does not modify that completed report or any raw result.

## Optimizer entry point

The existing implementation is `bf16_soap.CompiledSOAP`. Use it for dense,
contiguous, FP32 master matrices with gradients on every step, beside auxiliary
AdamW. Optimizer steps run outside autocast; model forward may use BF16 autocast.
All grouped matrices advance synchronously. No new wrapper or alternate
algorithm is needed for the draft.

```python
import torch
from bf16_soap import CompiledSOAP

# Match the recorded training harness, before compilation or CUDA matmuls.
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.use_deterministic_algorithms(True)
torch.set_num_threads(4)

matrix_optimizer = CompiledSOAP(
    matrix_groups,                  # existing matrix-only parameter groups
    lr=0.0015,
    betas=(0.95, 0.95),
    shampoo_beta=0.95,
    eps=1e-8,
    weight_decay=0.05,
    precondition_frequency=10,
    hard_reset_interval=50,
    state_dtype=torch.bfloat16,
    basis_dtype=torch.bfloat16,
    compute_dtype=torch.bfloat16,
    compile_kernels=True,
)
```

The complete benchmark grouping, auxiliary optimizer and LR multiplier are in
`benchmarks/bf16_storage_screen.py`, `benchmarks/cifar_optimizers.py`, and
`benchmarks/owt_restart.py`. The optimizer's standalone default LR does not
implement the training schedule: the harness updates each group's LR per call.
Group-supplied LR/decay values take precedence where the API permits them.

The six persistent tensors are BF16. Current directions use unrounded FP32
moment temporaries; initial eigendecompositions and QR repairs have FP32 solver
islands. The exact update order and transport approximations are specified in
the manuscript. FP32-storage and FP32-basis options are development context,
not alternate replications of the main fullBF16 method.

The “high” setting applies to FP32-tensor matrix products, including products
preparing QR inputs; it permits lower-precision internal multiplication without
changing output dtype. The QR/eigh solvers use FP32 tensors. Saved geometry
diagnostics temporarily select “highest” and restore the training setting.
The class does not set these global backend flags itself. See the versioned
[PyTorch matmul-precision documentation](https://docs.pytorch.org/docs/2.11/generated/torch.set_float32_matmul_precision.html)
and [BF16 reduction notes](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms).

## Exact recorded recipe

[configuration.json](configuration.json) records the model, data revisions,
optimizer settings, coverage, schedule and production pins. Main facts:

| Setting | Recorded value |
| --- | --- |
| Model |109,396,992 parameters; width768, depth12, heads12; FF hidden1536 |
| Matrix coverage |48 dense two-sided matrices;96 factors; four shape groups |
| Memory parameters |Matrix betas(.95,.95), covariance.95; auxiliary(.9,.999) |
| Learning rates |Matrix peak.0015, auxiliary peak.0006 |
| Schedule |10,000 calls;200-call linear warmup; cosine to.1 peak ratio |
| Other optimization |Epsilon1e-8; decay.05 on declared groups; global clip1 |
| Batch |32 rows, microbatch8, accumulation4 |
| Tokens |1,024 per row;1,023 targets;327,360,000 targets per full run |
| Initial call |Matrix initialization without matrix update; auxiliary still updates |
| Seeds |Model initialization204/205; identical fixed data cohort/order |
| Validation |1,024 rows,1,047,552 targets; development research panel |
| Runtime |PyTorch2.11.0+cu128, CUDA12.8, RTX5090; one GPU per experiment |

The main-run historical service wrappers set the following before starting
Python. These are recorded execution settings, not additional optimizer
constructor defaults:

| Execution setting | Recorded value |
| --- | --- |
| cuBLAS deterministic workspace | CUBLAS_WORKSPACE_CONFIG=:4096:8 |
| CPU threads | OMP_NUM_THREADS=4; OPENBLAS_NUM_THREADS=1; MKL_NUM_THREADS=4; torch.set_num_threads(4) |
| Compiler workers | TORCHINDUCTOR_COMPILE_THREADS=2 |
| Compiler cache | Each campaign's own compiler_cache directory, reused sequentially within the campaign |
| Other launch environment | HF_HUB_OFFLINE=1; PYTHONDONTWRITEBYTECODE=1 |
| PyTorch algorithms | Deterministic algorithms enabled; FP32 matmul high; BF16 reduced-precision reductions disabled |

The seed205 frozen wrapper is
`soap_bf16_full_20261026_seed205_v1/service/soap-bf16-full205.sh`;
the seed204 wrapper is
`soap_bf16_replicate_20261027_seed204_v1/service/soap-bf16-repl204.sh`.
Their production optimizer pins and shared storage-screen worker match;
campaign-specific runners are separately pinned in their manifests.

The matrix peak LR was calibrated on seed204 stock QR10 only: 500 calls per
rate at 0.0005/0.0015/0.003, with the full 10k horizon. Lowest call-500
validation CE selected 0.0015; warm candidates inherited it without a separate
LR search. The record is `reports/owt_restart_20261026/calibration.json`.
Both arms are sequential on the same claimed GPU; there is no controlled or
logged clock/thermal-state comparison. The covariance equations describe the
API-level weighted update and output dtype rather than a kernel-level
rounding audit.

Recorded data source: `Skylion007/openwebtext`, revision
`79d93d786212f7344586290adb811d4ae6a1762c`; GPT-2 tokenizer
`openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
The validation split precedes tokenization and packing. Full split/packing,
EOS and causal-boundary policies are retained in the configuration and source
data provenance. Training selects320,000 rows with selection seed2205;
validation selects1,024 rows with seed4205. Row order is fixed after selection.
No new dataset was prepared for the draft.

Frozen selected-input SHA256:

```text
train.npy       09e8fdeb7cf65d342b4909b8a8e9869390d66ab1673c9eb8e47912657c0684c8
validation.npy  47ae2f8f9fd457d4abe59dd3f7db04471df5e968ecb9d81dcfcb7680f4d6fbce
```

Local data bytes can be checked independently with `sha256sum`; paper table
regeneration requires only the saved JSON evidence. Original worker startup
audits verified the data before GPU work. No trained checkpoints are supplied.

## Main evidence and frozen execution records

The repository-relative scalar records are indexed in [evidence.json](evidence.json).
On this instance the original bundles are under `/workspace/optimizer_replay_results/`:

| Study | Bundle | Managed program |
| --- | --- | --- |
| Original seed204 reference |`soap_owt_restart_20261026_v4` |`soap-owt-learning` |
| Full seed205 pair |`soap_bf16_full_20261026_seed205_v1` |`soap-bf16-full205` |
| Full seed204 +201-call replay |`soap_bf16_replicate_20261027_seed204_v1` |`soap-bf16-repl204` |
| Pure-BF16 QR20/QR50 |`soap_bf16_refresh_controls_20261027_seed204_v1` |`soap-bf16-controls204` |
| Same-state QR profile |`soap_bf16_qr_profile_20261027_v1` |`soap-bf16-qr-profile` |

The later BF16 bundles use `manifest.json` for source/input/support pins,
`progress.json` for progress, and `service/` for recorded launch wrappers.
The original seed204 reference instead uses `provenance.json` and
`learn_status.json`; its launcher and supervisor configuration are the
top-level `learning_launcher.sh` and `learning_supervisor.conf`.
All listed bundles retain `PROTOCOL.md`, scalar results and audit records.
Worker source is in each bundle's `source/` directory.
Relevant readable protocols are `BF16_FULL_HORIZON_20261026.md`,
`BF16_REPLICATION_20261027.md`, `BF16_REFRESH_CONTROLS_20261027.md`, and
`BF16_QR_ATTRIBUTION_20261027.md` at the repository root. Historical numerical
guard wording is preserved in frozen protocols; later interpretation treats
finite crossings as diagnostics rather than a loss acceptance criterion.

## Historical launch commands and GPU admission

The completed launch scripts remain under `/opt/supervisor-scripts/` and in
bundle `service/` directories where recorded. Inspect them without executing:

```bash
sed -n '1,180p' /opt/supervisor-scripts/soap-bf16-repl204.sh
sed -n '1,180p' /opt/supervisor-scripts/soap-bf16-controls204.sh
sed -n '1,180p' /opt/supervisor-scripts/soap-bf16-qr-profile.sh
```

For example, the recorded controls launch has this command shape:

```text
gpu-claim run --owner TurboSOAP --job bf16-qr20-qr50-seed204-controls
  --count 1 --wait --timeout 7200 --
  timeout --signal=TERM --kill-after=10s 1790s
  /venv/main/bin/python -u <frozen-bundle>/source/benchmarks/bf16_refresh_controls.py
  --bundle <frozen-bundle>
```

The actual shell script also has environment settings and an outer admission
watchdog. The placeholders above describe the archived
launch, not a command to rerun against completed output directories. Fresh
training reproduction needs a new bundle/admission decision and the preserved
runner's preparation checks; completed bundles reject output reuse. None is
launched by the paper tools. All GPU work on this instance must follow
`/workspace/GPU_QUEUEING.md`, use shared gpu-claim and supervisor, and respect
other projects. There is no dependency on a particular historical GPU index.

## What this package reproduces

Scalar evidence supports exact reconstruction of the paper's endpoint tables,
timing aggregation records, figures and scope labels. It does not supply model
checkpoints, guarantee bitwise CUDA trajectories on another software/hardware
stack, or turn reused prefixes into independent runs. First-ever compilation
and whole-run time-to-quality require their own matched accounting. No new
experiment, sweep, schedule, production optimizer change or GPU job was created
for the draft.

The current manuscript package and bundle paths are local working artifacts,
with no public artifact URL assigned. An external release should carry the
indexed scalar summaries, frozen sources, manifests/provenance, wrappers,
input/row-ID records and raw event logs required for the advertised audits.
The current paper checker reconstructs scalar summaries; it does not by itself
authenticate historical execution or redistribute the selected dataset.
