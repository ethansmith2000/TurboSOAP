# Moderator priors (pre-registered 2026-11-20, before any brief was sent)

These are the moderator's own candidate ideas, written down before any reviewer saw the P1 brief. **They are not
shown to reviewers.** They exist so the syntheses can say which reviewer ideas are new and which the moderator
already had, and so the moderator's own ideas get the same scrutiny as the reviewers'.

Items 1–11 were set by the main TurboSOAP session in its task brief. Items M1–M3 are observations the moderator
(this session) made while reading `bf16_soap.py` to write the brief; they are also recorded before sending and are
likewise withheld from the P1 brief.

## (A) Faster refresh events and a faster method

1. **Maintain B = QᵀCQ in the basis frame between refreshes.** Keep an EMA of ḠḠᵀ, using the Ḡ = Q_LᵀGQ_R that
   the ordinary step already computes, instead of (or alongside) C. That would remove the two n×n products that form
   B at each warm refresh. Compare the basis-coordinate storage in Milligan et al. (2026) [verify].
2. **Transport M with the small rotation**, i.e. M⁺ = R_LᵀMR_R with R = QᵀQ⁺ known from the warm step, instead of
   the four full products of reconstructing M in world coordinates and projecting back.
3. **One Newton–Schulz correction instead of two.**
4. **Skip factors with a small residual** (no refresh where the off-diagonal mass of B is already small).
5. **Symmetric / syrk kernels and fusion** (B and the Gram YᵀY are symmetric; GGᵀ is a syrk).

## (B) Make the warm refresh match QR more closely

6. **Exact 2×2 Jacobi angles** (θ_ij from tan 2θ = 2B_ij/(d_j − d_i), as in a Jacobi sweep) instead of the
   first-order formula.
7. **Cayley or truncated-exponential retraction** instead of I + A followed by Newton–Schulz.
8. **Two warm steps per refresh.**
9. **Adaptive warm-vs-QR choice** from a measured residual (e.g. r_C of B, which the warm step already forms).
10. **Approximate rotation of V** (the second moment) at warm refreshes, instead of leaving it unchanged.
11. **Front-loaded schedules**: refresh (or QR) more often early in training, where the call-500 gap is largest.

## Moderator observations made while writing the brief (also withheld)

- **M1, the warm step is a half step.** For |h_ij| ≫ 0.01 the code's A_ij → −½·B_ij/(d_i − d_j). To first order,
  B⁺ = (I+A)ᵀB(I+A) ≈ B + BA − AB gives B⁺_ij ≈ B_ij + A_ij(d_i − d_j) = ½·B_ij. So one warm refresh removes only
  about half of the first-order off-diagonal mass, before the cap. A full first-order step (factor 1 instead of ½)
  is the most direct change, and it is not in the refresh study's ablation grid (damping, cap, NS count, 2 steps).
  Prediction: a step factor in (0.5, 1] lowers replay r_C at small cost; the risk is overshoot where off-diagonal
  entries are comparable to the gaps.
- **M2, prior 1 may not save products at a warm refresh.** If B is stored, it must be rotated after each basis
  change (B⁺ = RᵀBR: 2 products, or first-order B + BA − AB: 2 products). That cancels the 2 products saved by not
  forming QᵀCQ. The saving is real only if the rotation of B is approximated more cheaply, or B is consumed by
  something other than the warm step (adaptive skip, diagnostics). QR repair also needs CQ, which equals QB only
  for orthogonal Q. To be checked in P1_SYNTHESIS.
- **M3, the optimizer's share of step time.** At 109M the per-call optimizer mean is 26.08 ms of a 314 ms step
  (≈ 8%). QR repair events contribute about 12.0 ms/call of that (2% × 601 ms), ordinary events 9.1 ms/call, and
  warm increments about 4.1 ms/call. So at this scale the QR repair, not the warm step, is the largest refresh
  cost, and the whole optimizer bounds any end-to-end saving at about 8%. At larger n the warm step's share grows
  (warm/QR 0.41 at n=8192).
