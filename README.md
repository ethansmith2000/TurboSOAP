# TurboSOAP

TurboSOAP is SOAP with a warm, matmul-oriented eigenbasis tracker. It retains
the standard SOAP update:

```text
gradient -> project into covariance bases -> elementwise Adam -> project back
```

The implementation is intentionally narrow. Each covariance basis is
initialized with `eigh`, follows a damped Jacobi/double-bracket flow between
refreshes, and is periodically reset with QR or `eigh`.

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
- `basis_rotation_cap`: maximum average column rotation in one refresh.
- `basis_jacobi_damping`: relative pairwise eigengap damping.
- `basis_reset_frequency`: hard-reset interval in basis refreshes; `0` disables
  periodic resets.
- `basis_reset_method`: one-step orthogonal iteration (`"qr"`) or exact
  eigendecomposition (`"eigh"`).
- `basis_ns_iterations`: Newton--Schulz iterations in the warm retraction.
- `max_precond_dim`: skip covariance factors above this axis size.
- `merge_dims`: merge adjacent dimensions before constructing factors.

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
performed a basis refresh.

## References

- SOAP: <https://arxiv.org/abs/2409.11321>
- Shampoo: <https://arxiv.org/abs/1802.09568>
