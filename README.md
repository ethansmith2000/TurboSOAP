# TurboSOAP

TurboSOAP is SOAP with a warm, matmul-oriented eigenbasis tracker. It retains
the standard SOAP update:

```text
gradient -> project into covariance bases -> elementwise Adam -> project back
```

The implementation is intentionally narrow. Each covariance basis is
initialized with `eigh`, follows a damped Jacobi/double-bracket flow between
refreshes, and is periodically reset with QR or `eigh`.

See the [long-form method write-up](BLOG.md) for the motivation, equations,
state-transport details, limitations, and experiment plan.

## Installation and use

`soap.py` contains the optimizer and only requires PyTorch.

```python
from soap import SOAP

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

`soap_block.py` provides the experimental `BlockSOAP` variant. It stores each
covariance factor in its current basis and updates the basis with disjoint exact
2x2 rotations or small Cayley blocks, avoiding dense tangent retraction on
ordinary refreshes:

```python
from soap_block import BlockSOAP

optimizer = BlockSOAP(
    model.parameters(),
    lr=3e-3,
    betas=(0.95, 0.99),
    shampoo_beta=0.999,
    precondition_frequency=10,
    basis_pairs_per_refresh=0,  # one complete parallel matching
    basis_pair_selection="round_robin",
    basis_block_size=16,        # damped near-identity Cayley blocks
    basis_block_rotation_cap=0.25,
    basis_block_seed=0,
)
```

`round_robin` pair selection avoids a device synchronization and eventually
visits every coordinate pair. `top_correlation` greedily targets the strongest
current correlations but synchronizes once per active factor refresh.
With `basis_block_size > 2`, the tracker uses reproducibly shuffled blocks and
batched Cayley updates to correct several within-block correlations while
preserving a near-identity coordinate change. The refresh count and
`basis_block_seed` determine the schedule, so unrelated random-number use does
not change it. By default the update compensates for the probability that a
pair appears in a sampled block, then applies the configured block spectral cap.

Run `python benchmarks/benchmark_basis_refresh.py` to compare one-factor dense
and block refresh time and diagonalization progress across factor sizes.

The first `step()` observes a gradient, initializes covariance statistics and
their eigenbases, and does not update parameters. `TurboSlimSOAP` is an alias
for `SOAP`.

## Basis tracker

For one covariance factor, define

```text
B = Q.T @ C @ Q
```

TurboSOAP minimizes the off-diagonal covariance energy. Its skew generator is

```text
Omega_ij = B_ij (B_jj - B_ii)
```

and the implemented damped Jacobi generator is

```text
Omega_ij = B_ij (B_jj - B_ii) / ((B_ii - B_jj)^2 + tau^2)
```

where `tau` is controlled by `basis_jacobi_damping`. A tracker refresh applies

```text
Q <- retract(Q + Q @ (basis_lr * Omega))
```

after clipping excessive rotation. Small directions are never normalized
upward, so the update naturally vanishes in a diagonal basis.

## Main options

- `precondition_frequency`: optimizer updates between warm basis refreshes.
- `basis_lr`: damped Jacobi step size.
- `basis_rotation_cap`: maximum average column rotation in one refresh; `0`
  disables clipping.
- `basis_rotation_cap_mode="spectral_bound"`: use a conservative row-sum bound
  on the rotation spectral norm instead of the original average-column cap.
- `basis_jacobi_damping`: relative pairwise eigengap damping.
- `basis_stall_pair_threshold`: enable an exact largest-pair fallback when the
  dense flow stalls near equal Rayleigh quotients. Default: `0` (disabled,
  because checking the condition synchronizes the device).
- `basis_reset_frequency`: hard-reset interval in basis refreshes; `0` disables
  periodic resets.
- `basis_reset_method`: one-step orthogonal iteration (`"qr"`) or exact
  eigendecomposition (`"eigh"`).
- `basis_ns_iterations`: Newton--Schulz iterations in the warm retraction.
- `max_precond_dim`: skip covariance factors above this axis size.
- `precondition_mode="smaller_side"`: for 2D tensors, keep only the smaller
  covariance factor and basis to reduce optimizer state and update cost.
- `merge_dims`: merge adjacent dimensions before constructing factors.
- `basis_track_stats_frequency`: collect tracker diagnostics every N refreshes
  to reduce synchronization overhead.
- `transport_second_moment_on_warm`: apply squared-overlap diagonal variance
  transport on warm rotations. Default: `False`; repeated diagonal projection
  discards cross-coordinate second moments, so this is experimental.

The default basis state, covariance EMA, and Adam moments are fp32. Gradient
outer products may be computed in bf16 on CUDA before fp32 accumulation.

## State transport

The first moment is transported only when the basis changes: it is projected
back through the old basis and into the new one. Warm gauge steps keep columns
continuous, so the elementwise second moment remains attached to those column
labels. Periodic hard resets transport it through squared basis overlaps,

```text
R = Q_old.T @ Q_new
variance_new = variance_old transformed along each mode by R**2
```

which handles reset permutations, signs, and nontrivial rotations under the
diagonal-covariance approximation.

## Diagnostics

On basis refreshes, `latest_basis_stats` reports averages over active factors:

- `diag_err`: off-diagonal energy relative to `||C||_F`.
- `corr_err`: RMS off-diagonal correlation.
- `orth_err`: normalized `||Q.T @ Q - I||_F`.
- `basis_drift`: normalized basis change.
- `hard_reset`: fraction of factors hard-reset in that refresh.

`just_gathered_basis_stats` indicates whether the current optimizer step
collected diagnostics. This can be less frequent than basis refreshes when
`basis_track_stats_frequency > 1`.

## References

- SOAP: <https://arxiv.org/abs/2409.11321>
- Shampoo: <https://arxiv.org/abs/1802.09568>
