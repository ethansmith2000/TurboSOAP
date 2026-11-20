# Newton–Schulz retraction coefficients for the warm step (2026-11-20, exploratory)

**Question.** Can better polynomial coefficients give a cleaner one-step retraction?
- The warm step forms Y = Q(I + A) with A skew, then applies Newton–Schulz (NS) corrections.
- For exact Q, every singular value of Y is √(1 + θ²) ≥ 1. So a one-sided polynomial could in principle do better
  than standard NS.

**Key fact.** NS polynomials act only on singular values, so the polar factor (the retracted basis) is fixed by the
seed Y. Coefficients change only the orthogonality error left after the corrections, not which basis the step
moves to.

**Scripts** (CPU/GPU, small; run from this directory):
- `ns_coeffs.py`: Lawson minimax odd-polynomial fits.
- `ns_taylor.py`: third-order NS, (15/8, −10/8, 3/8).
- `ns_chain.py` and `ns_chain2.py`: four chained retractions from a BF16 basis with production-style rounding.
- `stack_check.py`: an Inductor fusion check of `torch.stack` in `group_step`.

**Variants:**
- **Pinned cubic.** y ← y + c₃·yE with c₃ = −1/(s(s+1)) and s = √(1 + θ_bound²). This makes p(1) = p(s) = 1 and
  reduces to standard NS as θ → 0.
- **exp2 seed.** Q(I + A + ½A²) has Gram I + ¼A⁴. Here I is kept in FP32 and only Q·bf16(A + ½A²) is rounded.

**Result** (`ns_chain2.py`): spectral orthogonality error, maximum over four chained refreshes, cap 0.5. The BF16
floor for an exactly orthogonal Q is about 0.0048 spectral (0.0024 normalized Frobenius).
- The full output, including cap 0.25 and rank-8 generators and the [Frobenius, spectral] pairs, is in
  `ns_chain2_output.txt`.
- It was rerun on CPU during the close-out and reproduces the table below.

| Scheme | Products per side (excl. B = QᵀCQ) | Spread | Decay | Rank-1 |
|---|---|---|---|---|
| Production: Q + QA, then 2 NS | 5 | 0.0043 | 0.0038 | 0.0101 |
| 1 NS (C2) | 3 | 0.0051 | 0.0065 | 0.043 |
| Pinned cubic | 3 | 0.0061 | 0.0114 | 0.0049 |
| Third-order NS | 4 | 0.0050 | 0.0047 | 0.0105 |
| **exp2 seed + 1 NS** | 4 | **0.0043** | **0.0043** | **0.0047** |
| exp2 seed, no NS | 2 | 0.0105 | 0.0105 | 0.0205 |

**Conclusion.**
- **exp2 + 1 NS matches or beats production's orthogonality with one fewer product** (6 instead of 7 per side in
  total), about 14% of the warm step's cost.
- **The minimax cubic lets p(1) drift**, which hurts the common case. Pinning p(1) = 1 fixes that but is worse on
  spread-out generators.
- **None of this changes tracking quality.** It does not change the close-out's cost conclusion either: see
  `../../../CLOSEOUT_20261120.md`.
