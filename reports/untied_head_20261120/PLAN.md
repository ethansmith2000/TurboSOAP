# Untied head/embedding study: does one-sided SOAP beat AdamW on E or H? (plan, awaiting approval)

Side study. The paper keeps its tied model. Runner: `benchmarks/untied_head_study.py`; data prep:
`benchmarks/untied_head_prepare.py`. Status on 2026-11-20: CPU self-test passed. **No study runs have
been launched.** Before approval the only GPU work was a 150-call timing probe of A0 and A2 on one
claimed GPU (about 3 min per arm). The brief asked for a measured untied step, so I ran it; it waited
behind refresh-study's `pb-timing` job. The probe also caught a bug: FlashAttention backward was
nondeterministic under `warn_only`. The runner now uses strict determinism as the paper does, and the
probe was rerun.

## Fixed setup (identical in every arm)

| item | setting |
|---|---|
| model | paper's 109M architecture with `tie_embeddings=False` → **147,994,368 params** (E 50257×768 and H 50257×768 separate, both N(0, 0.02) init) |
| data | **the paper's exact frozen rows**: regenerated from the local OWT cache with the paper's selection seeds 2205/4205; `train.npy`, `validation.npy` and both row-id files match the sha256s in `reports/owt_restart_20261026/provenance.json`. 320,000 train rows in fixed order, 1,024 validation rows (1,047,552 targets) |
| batch / schedule | 32×1024 rows (4 microbatches of 8), 10,000 calls, 200-call linear warmup then cosine to 10% at 10k; global clip 1.0; BF16 autocast + FlashAttention; FP32 CE |
| hidden block matrices | production TurboSOAP `bf16_soap.CompiledSOAP`: BF16 M/V/C/Q, warm every 10 calls, QR every 50, β1=β2=β_C=0.95, peak LR 1.5e-3, wd 0.05 |
| norm gains | AdamW (0.9, 0.999), peak LR 6e-4, wd 0 (paper) |
| E/H shared | β1 = 0.9 (paper's aux), eps 1e-8, same schedule multiplier; **E wd 0, H wd 0.05** (the repo's grouping rule: an untied `lm_head` is a decayed `nn.Linear`. Note that in the tied paper model the shared matrix is registered as `token_embedding.weight` and gets wd 0) |
| E/H one-sided SOAP | FP32 `soap.SOAP(precondition_mode="smaller_side", max_precond_dim=10000)`. It keeps one 768×768 covariance factor and basis on the hidden side, and per-element Adam M/V on the full 50257×768 rotated gradient; the vocab side has no factor. β_C = 0.95; basis = QR(C·Q_old) (one power step + QR) **every 10 calls**, QR10-like; `normalize_grads=False` (normalization is per element as in Adam, but the realized update RMS can differ, see the probe); fp32 covariance; first call only observes |

Why a QR refresh every 10 calls rather than warm10/QR50 for E/H: a 768² refresh costs about 1 ms. Refreshing by QR gives the most accurate basis and keeps the warm tracker out of the comparison. `soap.SOAP`'s warm tracker is also a different algorithm (gauge flow) from CompiledSOAP's. Note that `soap.SOAP`'s default `basis_reset_frequency=50` would mean QR every 500 calls, so it is set to 1. This is QR10-like, not literally the paper's QR10 control. `soap.SOAP`'s reset is QR(C·Q_old), with columns aligned to the old basis and V transported by squared overlaps; the paper's ResearchSOAP sorts columns by Rayleigh quotient and permutes V.

CPU self-test (passed): with no factor, `soap.SOAP` equals AdamW one call late (max diff 2e-7). The one-sided update matches a hand-written reference (6e-8). The structure checks (on a 500×32 CPU tensor) give GG = [None, d×d] with per-element V. The real shapes (one 768² factor, V of shape 50257×768) were exercised in the GPU probe's diagnostics. QR runs every 10 calls. The schedule is identical to `owt_restart.multiplier`. In tiny end-to-end runs of all four arms, the initial-parameter and call-1 gradient hashes are identical.

## Statistic half-lives (in updates)

| statistic | β | half-life | pooling |
|---|---|---|---|
| E/H momentum | 0.9 | 6.6 | per element |
| E/H second moment | 0.95 / 0.999 | **13.5 / 693** | per element (in SOAP: per element of the rotated gradient) |
| E/H SOAP factor C (768×768) | 0.95 | 13.5, refreshed into the basis every 10 calls (basis age 0–9) | pooled over all vocab rows (~7.1k active E rows per batch; all H rows) |
| hidden M / V / C | 0.95 | 13.5 each; warm basis every 10 calls, QR every 50 | per matrix |

Rare-row context, from the 327.7M training targets: the top 1k tokens make up 66% of targets, ranks 1k–10k 23%, and ranks 10k+ 11%. 164 tokens never appear. A rank-10k token appears about once every 3.5 calls, and a rank-30k token about once every 15 calls, which is one β2=0.95 half-life. So with β2 = 0.95, the V of rare E rows is mostly forgotten between appearances.

## Arms (2×2 factorial on the E and H optimizers)

| arm | E | H |
|---|---|---|
| A0 baseline | AdamW | AdamW |
| A1 | AdamW | one-sided SOAP |
| A2 | one-sided SOAP | one-sided SOAP |
| A3 | one-sided SOAP | AdamW |

H effect = A1−A0, E effect = A3−A0, interaction = A2−A1−A3+A0.

## Hyperparameter selection: at the full 10k horizon, seed 204, equal budget per optimizer

- **H family** (E held at base AdamW 6e-4 / 0.999): H ∈ {AdamW, SOAP} × β2 {0.95, 0.999} × peak LR {3e-4, 6e-4, 1.2e-3, 2.4e-3}.
- **E family** (H held at base AdamW 6e-4 / 0.999): E ∈ {AdamW, SOAP} × β2 {0.95, 0.999} × peak LR {6e-4, 1.2e-3, 2.4e-3, 4.8e-3}. This grid is shifted up one step because untied input embeddings usually prefer higher LR. 6e-4 is in both grids.
- That gives 8 cells per (group, optimizer), the AdamW baseline included, and **31 unique runs**. The (base, base) cell is shared by both families.
- **Pre-registered edge rule:** if a (group, optimizer)'s best LR is the largest or smallest LR tried, add the next factor-2 LR at both β2 values. At most 2 steps past the original grid, and never above a diverged LR.
- Selection metric: validation CE at call 10,000. A diverged run counts as +∞.
- **Combine:** for each arm, run the best E setting for its E optimizer together with the best H setting for its H optimizer at seed 204 (4 runs; a combination may already exist as a tune cell).
- **Arm configuration (pre-registered):** each arm's argmin of seed-204 CE over *every* finished full-horizon cell with that (E, H) optimizer pair. That is the combination plus all tune cells of the pair: A0 has 15 AdamW/AdamW cells + 1 combination, A1 and A3 have 8 + 1, A2 has the combination only. The baseline gets the most tuning, so the comparison is conservative for SOAP.
- **Confirm:** each arm's chosen configuration at **fresh seeds 205 and 206**, giving 8 runs. Seed 204 is reported but flagged as the selection seed.
- The automatic launcher halts before combine/confirm if any cell is missing a result, or if a whole (group, optimizer) family has no finite cell.
- **LR cap (pre-registered):** if after the edge rounds a setting's best LR is still the largest (or smallest) LR tried, the chain continues. Every verdict that uses such a setting, in the arm or in A0, is then stamped **"LR-capped, provisional"** in `summarize`, and I propose a follow-up extension.
- **Pre-registered verdict** (fresh seeds, paired by seed, versus best-tuned A0): *better* if mean Δ ≤ −0.005 CE and both seeds are negative; *worse* if ≥ +0.005 and both are positive; otherwise *no resolvable difference*. Noise reference: a single seed pair — the paper's BF16 TurboSOAP differs by 0.003 CE between seeds 204 and 205 at 10k (3.4224 vs 3.4194). A0's three seeds will give this study's own spread, reported next to the verdict.

## Metrics

- Validation CE at calls 0, 50, 100, 200, 500, 1k, 1.5k, 2k, 2.5k, 3k, 4k, 5k, 6k, 7k, 7.5k, 8k, 9k and 10k (the paper's eval calls), overall and by target-token frequency bucket.
- Step time, split into forward/backward plus per-optimizer parts (matrix / E / H / gains), with diagnostic calls excluded. Peak GPU memory and optimizer state bytes.
- Diagnostics at calls 1, 2, 11 and every eval call. Per group and per frequency bucket (top1k / 1k–10k / 10k–30k / 30k+ / unseen):
  - unit-LR update RMS with the decoupled decay removed, for all rows and for rows active this call;
  - active-row fraction;
  - row-norm mean and CV, plus row-norm spread over seen rows (CV, p90/p10, max/median);
  - 768-side spectra (effective rank, top-1 share) of the gradient, the update and the weights;
  - for SOAP groups only: the factor's eigen-spread, the basis off-diagonal ratio and orthogonality.

## Cost (measured step time)

Timing probe: 150 calls on one claimed RTX 5090 (GPU 2), strict determinism, with other projects on neighbouring GPUs. The first, nondeterministic probe is archived in `/workspace/data/untied_head_20261120/probe_v0_nondeterministic/`.

| probe arm | step ms (mean / median) | fwd+bwd ms | matrix opt ms | E opt ms | H opt ms | peak GiB | E+H opt state |
|---|---|---|---|---|---|---|---|
| A0 (AdamW E, AdamW H) | 315.8 / 300.9 | 287.8 | 25.8 | 0.83 | 0.83 | 8.81 | 588 MiB |
| A2 (SOAP E, SOAP H) | 318.1 / 300.9 | 282.7 | 25.6 | 4.65 | 4.85 | 8.82 | 598 MiB |

- **One-sided SOAP costs about +3.9 ms per group per step** (+7.8 ms for E and H together, about 2.5% of a step) and +5 MiB of state per group. Memory is unchanged at about 8.8 GiB peak.
- Strict determinism costs about 5 ms/step compared with the first probe (310.8 ms).
- A0 and A2 have identical initial-parameter and call-1 gradient hashes. Validation now counts 1,047,552 targets.
- Early-learning signal, not a verdict: at the shared base LR 6e-4/0.999, A2 is behind A0 (CE 8.42 vs 7.12 at call 50, 6.54 vs 6.00 at call 150). The probe diagnostics locate this in the **head**:
  - At call 50, the head gradient is nearly rank-1 on the 768 side (89–90% in the top direction).
  - AdamW's head update has RMS 0.77 and is also nearly rank-1 (effective rank 1.6, 93% top direction). Sign-like per-element normalization puts a step of about √768 along the dominant direction.
  - One-sided SOAP rotates that signal into a single coordinate. Its head update has RMS 0.25 (effective rank 8.4, 59% top direction), so it moves far less along the dominant direction.
  - The gap shrinks as the gradient spreads out: at call 150, head update RMS is 0.20 (SOAP) vs 0.25 (AdamW).
  - The embedding update is similar under both optimizers (RMS 0.27 vs 0.29; effective rank 550–680).
  - So SOAP-H may want a higher LR. The H grid plus edges reaches 9.6e-3 (16× base) and the E grid reaches 1.92e-2.
- **Per 10k run ≈ 0.92 GPU-h**: 10k × 0.316 s ≈ 53 min, plus about 2–3 min of evals, diagnostics and compile.
- **Plan A (recommended, matches the target-horizon rule):** 31 tune + about 4 (0–16) edge + 4 combine + 8 confirm ≈ 47 runs ≈ **43 GPU-h**. Stages are sequential (tune → edge rounds → combine → confirm). Wall time is about 11.5 h with 4 GPUs, about 9.5 h with 6, about 7 h with 8, plus any queueing behind other projects.
- **Plan B (cheaper, flagged short-horizon):** tune at 2,000 calls on the full 10k schedule (31 × ≈0.19 GPU-h ≈ 6 GPU-h), then 4 combine + 8 confirm at 10k ≈ **17 GPU-h**, about 4 h with 6 GPUs. The LR/β2 choices would come from a 2k proxy, which violates the target-horizon rule.

## Scheduling

- Everything goes through `gpu-claim run --owner untied-head --job <name> --wait --timeout 259200`, using `untied_head_study.py launch --auto --max-concurrent N`.
- The launcher submits nothing while refresh-study has a waiting claim or an exclusive (timing) claim. In that case it withdraws its own not-yet-started claims, and it logs each defer/resume transition to `launcher.log`.
- It keeps at most one waiting claim of its own at a time, so it never queues ahead of other projects in bulk.
- Runs that already finished are skipped, so the launcher can be restarted safely.

## Caveats to carry into the report

- Coordinate-wise tuning: each group is tuned with the other group at base AdamW. A2 combines bests that were tuned independently, which assumes a weak E–H interaction. If the interaction exceeds the threshold, offer a local A2 check.
- β1 (0.9), β_C (0.95) and the basis cadence (QR10) are fixed, not tuned. The head init is 0.02, not zero.
- One model size (148M) and one horizon (10k calls, 327M targets); horizon transfer is untested. The validation panel is the paper's development panel, not an untouched test set.
- Step times come from a shared host, with other projects on neighbouring GPUs.

## Outputs

- Runs: `/workspace/data/untied_head_20261120/runs/<name>/` (`config.json`, `events.jsonl`, `diagnostics.json`, `result.json`, `stdout.log`, and `final_weights.pt`: FP32 model weights at the end of each completed 10k run, about 0.6 GB per run and about 28 GB in total, kept for later per-token and row analyses; no optimizer state).
- Tables: `reports/untied_head_20261120/` (`tables.md`, `summary.json`, `launcher.log`).

## Tune-stage run list (31 runs, seed 204, 10k calls; name = `E-<opt><peak LR>b<β2 digits>_H-…_s<seed>`; rows 1–16 vary H, 17–31 vary E, row 6 is the shared base cell)

 1. `E-adamw6.0e-4b999_H-adamw3.0e-4b95_s204`
 2. `E-adamw6.0e-4b999_H-adamw6.0e-4b95_s204`
 3. `E-adamw6.0e-4b999_H-adamw1.2e-3b95_s204`
 4. `E-adamw6.0e-4b999_H-adamw2.4e-3b95_s204`
 5. `E-adamw6.0e-4b999_H-adamw3.0e-4b999_s204`
 6. `E-adamw6.0e-4b999_H-adamw6.0e-4b999_s204`
 7. `E-adamw6.0e-4b999_H-adamw1.2e-3b999_s204`
 8. `E-adamw6.0e-4b999_H-adamw2.4e-3b999_s204`
 9. `E-adamw6.0e-4b999_H-soap3.0e-4b95_s204`
10. `E-adamw6.0e-4b999_H-soap6.0e-4b95_s204`
11. `E-adamw6.0e-4b999_H-soap1.2e-3b95_s204`
12. `E-adamw6.0e-4b999_H-soap2.4e-3b95_s204`
13. `E-adamw6.0e-4b999_H-soap3.0e-4b999_s204`
14. `E-adamw6.0e-4b999_H-soap6.0e-4b999_s204`
15. `E-adamw6.0e-4b999_H-soap1.2e-3b999_s204`
16. `E-adamw6.0e-4b999_H-soap2.4e-3b999_s204`
17. `E-adamw6.0e-4b95_H-adamw6.0e-4b999_s204`
18. `E-adamw1.2e-3b95_H-adamw6.0e-4b999_s204`
19. `E-adamw2.4e-3b95_H-adamw6.0e-4b999_s204`
20. `E-adamw4.8e-3b95_H-adamw6.0e-4b999_s204`
21. `E-adamw1.2e-3b999_H-adamw6.0e-4b999_s204`
22. `E-adamw2.4e-3b999_H-adamw6.0e-4b999_s204`
23. `E-adamw4.8e-3b999_H-adamw6.0e-4b999_s204`
24. `E-soap6.0e-4b95_H-adamw6.0e-4b999_s204`
25. `E-soap1.2e-3b95_H-adamw6.0e-4b999_s204`
26. `E-soap2.4e-3b95_H-adamw6.0e-4b999_s204`
27. `E-soap4.8e-3b95_H-adamw6.0e-4b999_s204`
28. `E-soap6.0e-4b999_H-adamw6.0e-4b999_s204`
29. `E-soap1.2e-3b999_H-adamw6.0e-4b999_s204`
30. `E-soap2.4e-3b999_H-adamw6.0e-4b999_s204`
31. `E-soap4.8e-3b999_H-adamw6.0e-4b999_s204`
