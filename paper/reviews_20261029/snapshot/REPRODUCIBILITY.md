# Reproducing the draft and inspecting the experiment

## Paper artifacts: CPU only

Run from the repository root:

```bash
python3 paper/build_artifacts.py --check
```

This checks five manuscript tables against saved scalar evidence, confirms the
two copied figure hashes, verifies four optimizer-source pins, checks shared
initialization/reference identities and repeated-prefix identity, and validates
the generated evidence/configuration records. It uses the standard library and
does not import Torch or touch a GPU.

To regenerate those tables and records:

```bash
python3 paper/build_artifacts.py
```

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

Their `manifest.json` files pin source snapshots, data and support artifacts;
`PROTOCOL.md`, `progress.json`, scalar results and audit records describe the
actual campaigns. Worker source is in each bundle's `source/` directory.
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
