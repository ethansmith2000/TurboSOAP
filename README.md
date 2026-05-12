# TurboSOAP

__SOAP but its really fast__

Blog: https://open.substack.com/pub/ethansmith2000/p/turbosoap?r=jsutr&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true

TurboSOAP is an optimizer that keeps the core SOAP update,
but replaces most repeated eigendecompositions with warm, matmul-heavy basis
tracking.

Standard SOAP periodically computes eigenbases for per-axis covariance factors.
TurboSOAP still projects gradients into those bases, applies Adam-style moments
there, and projects the update back, but the default refresh path uses a
warm-started tangent update with a Newton-Schulz retraction. This makes the
basis refresh much friendlier to accelerators than repeatedly calling dense
LAPACK eigensolvers.

## Installation

`soap.py` contains the optimizer.

```bash
pip install torch
```

Then either keep `soap.py` next to your training script or put this directory
on `PYTHONPATH`.

```python
from soap import SOAP
```

`TurboSlimSOAP` is also exported as a backward-compatible alias for `SOAP`.

## Quick Start

Use it like any other `torch.optim.Optimizer`.

```python
import torch
from soap import SOAP

model = ...
optimizer = SOAP(
    model.parameters(),
    lr=3e-3,
    weight_decay=0.01,
)

for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
```

The first `step()` initializes SOAP state and does not update parameters. This
matches the SOAP convention because the preconditioner needs an initial
covariance estimate before it can apply the projected Adam update.

## Current Defaults

The default constructor is tuned for the fast basis-tracking path:

```python
SOAP(
    params,
    lr=3e-3,
    betas=(0.95, 0.999),
    shampoo_beta=0.999,
    eps=1e-8,
    weight_decay=0.01,
    precondition_frequency=10,
    max_precond_dim=10000,
    merge_dims=False,
    precondition_1d=False,
    normalize_grads=False,
    basis_stage1_method="tangent",
    basis_step_size=0.01,
    basis_substeps=1,
    basis_direction_normalizer="col",
    basis_direction_normalizer_beta=0.99,
    basis_step_controller="correction_ratio",
    basis_target_correction_ratio=0.99,
    basis_backtrack_factor=0.7,
    basis_growth_factor=1.5,
    basis_compute_dtype="bfloat16",
    basis_storage_dtype="bfloat16",
    covariance_compute_dtype="bfloat16",
    basis_direction_preconditioner="eigengap",
    basis_metric_frequency=5,
    basis_fallback_diag_err_threshold=0.3,
    basis_fallback_method="qr",
    basis_warmup_steps=250,
)
```

Most users should start with only `lr`, `weight_decay`, and perhaps
`precondition_frequency`.

## What It Does

For each preconditioned tensor axis, TurboSOAP maintains:

- `GG`: an exponential moving average of gradient outer products.
- `Q`: an orthogonal basis used to approximately diagonalize each covariance
  factor.
- `exp_avg`: the Adam first moment, stored in the SOAP basis.
- `exp_avg_sq`: the Adam second moment, also stored in the SOAP basis by
  default.

Each training update follows the SOAP pattern:

```text
gradient -> project into Q -> Adam update in basis -> project back -> parameter update
```

When `Q` changes, TurboSOAP transports the first moment through the old basis
and reprojects it into the new one. The second moment is reordered by estimated
eigenvalue so it stays aligned with the refreshed basis.

## Basis Refresh Modes

Choose the refresh behavior with `basis_stage1_method`.

- `"tangent"`: the default fast path. It follows the projected covariance
  tangent direction and retracts with one Higham cubic Newton-Schulz step.
- `"brockett"`: a warm tracker with Brockett ordering pressure. This can put
  more emphasis on ordering the leading spectrum.
- `"qr"`: reference-style SOAP refresh using one power iteration from the
  current basis followed by QR.
- `"eigh"`: full eigendecomposition refresh.
- `"none"`: keep the initialized basis fixed.

Cold start always uses `eigh` to create the initial basis. During
`basis_warmup_steps`, refreshes are forced through `basis_fallback_method`
before switching to the selected warm tracker.

## Useful Configurations

Fast default-style training:

```python
optimizer = SOAP(
    model.parameters(),
    lr=3e-3,
    precondition_frequency=10,
)
```

More conservative, closer to reference SOAP:

```python
optimizer = SOAP(
    model.parameters(),
    lr=3e-3,
    precondition_frequency=10,
    basis_stage1_method="eigh",
    basis_compute_dtype="float32",
    basis_storage_dtype="float32",
    covariance_compute_dtype="float32",
    basis_warmup_steps=0,
)
```

Use QR refreshes without the tangent tracker:

```python
optimizer = SOAP(
    model.parameters(),
    basis_stage1_method="qr",
    basis_fallback_method="qr",
    basis_warmup_steps=0,
)
```


For very large tensors, lower `max_precond_dim` to skip expensive axes, or set
`merge_dims=True` to merge small dimensions before building covariance factors.
`merge_dims=True` currently only supports `adam_second_moment="elementwise"`.

## Diagnostics

When `basis_track_stats=True`, the optimizer writes aggregate basis diagnostics
after each `step()` to:

```python
optimizer.latest_basis_stats
optimizer.just_gathered_basis_stats
```

`basis_metric_frequency` controls how often metric refreshes are run. Metric
refreshes are used for accept/reject decisions, fallback checks, controller
updates, and stats collection. Intervening refreshes reuse the current step
scale without recomputing diagnostics.

Example:

```python
optimizer.step()

if optimizer.just_gathered_basis_stats:
    print(optimizer.latest_basis_stats)
```

## Important Options

- `shampoo_beta`: covariance EMA beta. `-1` means use `betas[1]`.
- `precondition_frequency`: how often to refresh the basis after initialization
  and warmup.
- `max_precond_dim`: axes larger than this are not preconditioned.
- `basis_compute_dtype`: dtype for warm refresh matmuls.
- `basis_storage_dtype`: dtype used to store and apply `Q`.
- `covariance_compute_dtype`: dtype for gradient outer products before fp32 EMA
  accumulation.
- `basis_step_controller`: `"correction_ratio"` adapts per-factor step scales;
  `"fixed"` uses `basis_step_size` directly.
- `basis_fallback_diag_err_threshold`: if metric refreshes leave too much
  diagonalization error, fall back to `basis_fallback_method`. Use `0` to
  disable this threshold fallback.
- `adam_second_moment`: second-moment denominator shape. `"elementwise"` is
  standard SOAP; `"axis_product"`, `"axis_geomean"`, `"axis_logblend"`,
  `"axis_sum"`, `"axis_max"`, and `"scalar"` are memory/performance variants.

## Notes

- Weight decay is decoupled and applied after the adaptive update.
- Gradients can optionally be RMS-normalized with `normalize_grads=True`.
- `data_format="channels_last"` is supported for 4D tensors when dimension
  merging is enabled.
- The Apache 2.0 license is included in `LICENSE`.

## References

- SOAP: <https://arxiv.org/abs/2409.11321>
- Shampoo: <https://arxiv.org/abs/1802.09568>

