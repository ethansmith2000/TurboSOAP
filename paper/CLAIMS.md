# Claim-to-evidence map

This is an authoring control document, not an extra experiment or a numerical
acceptance policy. [evidence.json](evidence.json) records source paths, SHA256
hashes and calculated values. `build_artifacts.py --check` verifies tables,
source pins, figures and generated records without importing the optimizer.

| Manuscript claim | Evidence | Allowed interpretation |
| --- | --- | --- |
| Two-seed endpoint CE gaps and perplexity ratios | `reports/bf16_replicate_20261027_seed204/comparison.json`: `candidate`, `cached_quality_reference`; `reports/bf16_full_horizon_20261026_seed205/comparison.json`: `results` | Two model-initialization seeds, fixed data order; no equivalence or confidence-interval claim |
| Larger early gap at call500 | Same saved validation arrays | Observed evaluation values; no inference about unobserved steps |
| Seed20417.36% step-time reduction | Replication comparison: `paired_prefix_timing` | Freshly matched calls101..200 only; cached full reference used for learning quality |
| Seed20518.66% step-time reduction | Full-horizon comparison: `results[*].mean_step_ms` | Freshly paired calls101..10000; timing excludes diagnostics/evaluation/startup |
| Matrix state approximately halved | Seed205`results[*].matrix_state_bytes`; six-tensor element count in builder | Matrix-state tensor bytes; masters and auxiliary AdamW remain FP32 |
| Lower peak allocated GPU memory | Seed205`results[*].peak_gpu_bytes` | Observed allocated tensor memory including diagnostic work, not total device reservation |
| Short QR20/QR50 cadence context | `reports/bf16_refresh_controls_20261027/comparison.json`, audits and cached warm prefix | Seed2042k prefix on full10k LR horizon; warm timing across launches, no final-horizon ranking |
| QR product reuse saves.95–4.63% at batch12 | `reports/bf16_qr_attribution_20261027/result.json`: rows/timing | Synthetic matched FP32 eager primitive; no whole-training or BF16-speed attribution |
| Near-tie orderings and local update differences | Same profile: rows/numerical | One-factor synthetic probe, not trained-state or bitwise equivalence |
| Actual-M/Gram observations | Full-horizon and replication results, `runs/bf16/geometry.json` | Descriptive finite-state diagnostics, no loss-quality threshold |
| Warm step can stall at equal diagonals | `reports/review_20261026/equal_diagonal_counterexample.json`; formula in `bf16_soap.warm_basis` | Algebraic/CPU counterexample, not measured training incidence |
| Development-family context | `reports/bf16_learning_20261026/comparison.json` | Same seed and reused prefix; not independent replication or pure BF16-only attribution |

All report paths above are relative to the repository root. The corresponding
readable reviews are in the same directories. Frozen campaign manifests
identify worker source snapshots, inputs and support hashes. Main-run audits
include startup reproduction, event/observation/evaluation counts and capture
stability; short controls additionally check final model/matrix/auxiliary
finite state. These operational checks are distinct from method selection.

## Positioning supported by the draft

- A fully specified dense BF16-state implementation and scoped empirical
  quality/time/memory tradeoff against the named stock-beta reference.
- Short-control evidence that intermediate tracking improves this prefix's
  loss over QR50 at extra work, while QR20 is slightly better at the endpoint.
- A transparent local comparison of redundant QR products, including
  finite-precision ordering changes.

## Claims that need additional evidence if later desired

Statistical equivalence; general optimizer superiority; superiority over
optimized FP32 QR10, adaptive or partial-basis methods; final-horizon QR20/QR50
ranking; pure BF16-only or warm-kernel-only causal speed attribution; exact
history transport; standalone warm convergence; broad hardware/data/model
generalization. None is part of this draft's claim, and this list authorizes
no new experiment.

## Primary references checked for this draft

- SOAP, arXiv2409.11321v2, §4 and implementation context.
- Milligan et al., arXiv2605.26327v1, §§3.1–3.2: persistent basis-coordinate
  representation, selected subspace updates and BF16 storage.
- Eschenhagen et al., arXiv2506.03595v2, §4: approximation-error-driven basis
  refinement/scheduling.

Metadata and versions were checked against arXiv on2026-10-28. Bibliography
entries and manuscript links use explicit versions. The review notes under
`reports/review_20261026/` retain the earlier detailed comparison; statements
there about then-pending experiments are historical, not current run status.

## Diagnostic and runtime clarifications after mathematical review

The current draft defines training physical-M error as a local before/after
refresh discrepancy, reconstructed in FP32 at the highest matmul setting.
It does not measure deviation from stock or cumulative history error. The
synthetic profile instead compares original/reuse outputs and includes BF16
rounding in world-moment reconstruction; those metrics are not interchangeable.
The observed BF16-rounded fresh-QR Gram error is an empirical baseline, not
a universal positive floor. Exact definitions and denominator clamps are in
Sections 5.4–5.5.

The reported training harness uses FP32 matmul precision “high” and disables
BF16 reduced-precision reductions. Solver tensor dtype does not imply strict
FP32 internal multiplication for every product preparing the solver input.
The reproduction example and generated configuration now record these existing
settings. This documentation revision changes no frozen evidence or source pin.

## Descriptive context after scientific review

`review_context.py` regenerates `review_context.json` from saved summaries
and the original LR calibration. Seed205 event-class accounting places 68.38%
of observed optimizer-time saving in refresh events and 31.62% in ordinary
events. These shares do not causally isolate cadence from changed refresh
execution. The four warm-policy pilot arms already have similar early gaps
across precision modes; the compiled-mode endpoint contrast is not a pure
BF16-storage penalty. Neither full-horizon QR20/warm ordering is established.

LR selection was baseline-only, as recorded in the original calibration.
Clock/thermal equivalence and exact vendor-internal covariance rounding were
not audited. Synthetic QR timings do not establish an upper bound on the
performance of an untrained optimized reference.
