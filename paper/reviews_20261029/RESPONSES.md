# Manuscript review responses — 29 October 2026

This is a rolling response record. The four original reviews address the
immutable snapshot identified in snapshot_manifest.json; changes below are to
the current working manuscript. Findings are checked against code and saved
artifacts before adoption. No review authorizes a new experiment or source change.

## Astra mathematical review — minor revision

Report: [astra_math.md](astra_math.md). No substantive equation, sign, indexing
or update-order error was found. Three reporting findings are accepted:

| Finding | Verification | Response |
| --- | --- | --- |
| M1: omitted runtime precision settings | Training harness sets FP32 matmul precision high and disables BF16 reduced-precision reductions; geometry temporarily selects highest. PyTorch 2.11 documents the distinction between tensor dtype and internal multiplication precision. | Added the existing settings to Section 3.1, the reproduction example, and generated configuration. Clarified the scope of the FP32 solver description. |
| M2: underdefined and nonidentical physical-M diagnostics | Training reconstructs before/after candidate transport with FP32 products at highest, denominator clamp 1e-12. The synthetic profile compares original/reuse BF16-rounded world reconstructions, denominator clamp 1e-30. | Added formulas and local scope in Section 5.5 and distinct profile wording in Section 5.4. Reported values stay unchanged. |
| M3: universal orthogonality floor wording | Exactly representable orthogonal matrices refute a universal positive bound; the saved fresh-QR probes provide an empirical rounding comparison. | Replaced the floor claim with an observed rounding baseline. |

Also defined sym on first use. Claims and reproduction notes reflect these
clarifications. The original manuscript, first PDF proof and old build receipt
remain preserved. All five generated tables, twelve source records, two frozen
figures and four optimizer source pins still check.

Precision sources verified by the parent:
[matmul API](https://docs.pytorch.org/docs/2.11/generated/torch.set_float32_matmul_precision.html),
[numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms).

## Sol evidence review — minor revision

Report: [sol_evidence.md](sol_evidence.md). All headline numbers and five
tables checked. All four findings are accepted:

| Finding | Verification | Response |
| --- | --- | --- |
| 1: incomplete execution recipe | Frozen storage worker enables deterministic algorithms, four Torch threads, high FP32 matmul and disabled reduced BF16 reductions. Both frozen main service wrappers record the same workspace/thread/compiler settings and bundle-local caches. | Extended the reproduction example, recipe and generated runtime record. Astra's precision clarification already addresses part of this finding. |
| 2: overly broad source-hash identity | Production optimizer and shared storage-worker hashes match; full-horizon campaign runner hashes differ between the manifests as seed/audit handling evolves. | Narrowed Section 1's identity claim and explicitly identified separately pinned campaign runners. |
| 3: original reference archive filenames | Original reference uses provenance.json, learn_status.json and top-level learning launcher/config files. Later bundles use manifest.json, progress.json and service directories. | Corrected archive navigation in the reproduction guide. |
| 4: CPU-check scope | Builder checks repeated-prefix endpoint CE agreement and reads summary means; it does not independently recompute all raw timings or every frozen source pin. | Stated the exact check scope and delegated broader execution checks to frozen audits. |

Adopted two presentation suggestions: disclose the larger observed early CE
gap in the abstract and add short scope captions above each table. Added the
local working-artifact/public-release distinction to answer the artifact-access
question without implying an external release already exists.

## Fable scientific review — textual revision

Report: [fable_science.md](fable_science.md). Fable independently confirmed
the displayed evidence, formulas and primary-reference metadata, recommending
minor-to-moderate textual revision without experiments. The interpretive
findings are addressed with qualifications:

| Finding | Response and scope |
| --- | --- |
| F1: most time saving is cadence | Accepted descriptive event accounting: 47.28/69.15 ms per call, or 68.38%, lies in refresh events; 21.87 ms, or 31.62%, in ordinary events. These recombine exactly to the recorded mean. The refresh category changes both frequency and execution, so its full share cannot be labeled a pure cadence effect. Added this distinction to Section 5.2 and surfaced refresh-work accounting in the abstract. |
| F2: early loss gap is chiefly cadence | Added the four warm-policy pilot arms' similar call-500 gaps, including the FP32-state arm, and QR20 context. This refutes attributing the early gap uniquely to BF16. The data do not identify a pure cadence share or a precision share at 10k calls. The 0.002379 CE compiled-mode endpoint contrast also changes basis precision/correction arithmetic, so it is labeled a within-family contrast. The review's “all five arms” wording is narrowed to four warm arms; stock is the zero-gap reference. |
| F3: QR20 remains a live alternative | Added symmetrical uncertainty about both full-horizon orderings and the descriptive 4.93% prefix time difference. Did not adopt “within measurement noise”: there is no calibrated noise interval for these cross-launch means. |
| F4: unnamed numerical lineage | Named Jacobi-type first-order motion and the cubic Newton–Schulz polar map; added verified Forsythe–Henrici and Björck–Bowie primary references. The dense simultaneous motion and BF16 policy remain distinct from a classical sequential Jacobi algorithm. |
| F5: original SOAP also warm-starts QR | Expanded the SOAP relationship paragraph, including its low-precision discussion and solver limitation. Clarified that campaign betas explicitly override local constructor defaults. |
| F6: early basis accuracy and V correction context | Added the Imagewoof ViT early-basis finding as qualitative context, not proof of our mechanism; included the EShampoo second-moment-correction observation with its different-setting qualification. No adaptive schedule is introduced. |
| F7: memory reduction units | Clarified the smaller fractional peak-allocation reduction and reported 12.49%; absolute reductions are not compared with “less”. |
| F8: late gap plateau | Replaced continued-narrowing language with the observed approximately 0.007–0.012 CE band after call 2000. |
| F9: seed204 short timing window | Added that it lies in LR warmup but contains two complete 50-call event cycles. |
| F10: optimized-reference speed bound | Declined the bound. A synthetic primitive speed ratio is not an upper bound on a differently optimized training implementation, and the evidence supplies no such comparator. Existing pilot and QR-profile limits remain visible. |
| F11: LR selection | Confirmed from the original calibration: seed204 stock QR10 only, three rates, 500 calls on the full 10k LR horizon, lowest call-500 CE. Warm candidates inherited the chosen rate without their own search. |
| F12: retain numerical screens | Retained the diagnostic interpretation. Kept Astra's more precise empirical-rounding-baseline wording instead of restoring a universal orthogonality floor. |

New descriptive values and source hashes are reproducible with
`python3 paper/review_context.py --check`. No original numerical result is
rewritten. Vendor-internal covariance rounding was not audited, so the draft
states a tensor/API contract; clock and thermal state were not controlled or
logged. Sequential execution on a claimed GPU does not establish clock parity.

Primary sources were checked by the parent, including SOAP §§7.2.1/7.3.1/8,
Purifying Shampoo §4.4/Appendix D.2, Forsythe–Henrici's paper scan, and
Björck–Bowie's publisher metadata and paper scan. Five bibliography entries
now appear in the current manuscript; the snapshot's three-entry bibliography
is preserved.

## Opus manuscript review — focused revision

Report: [opus_manuscript.md](opus_manuscript.md). Opus also supports the narrow
case study after textual revision and finds no numerical or equation error.
Its main interpretive findings overlap with Fable and are addressed above.
Additional responses:

| Findings | Response and limits |
| --- | --- |
| M1/M3: early gap and late plateau | Existing Fable revision brings the FP32 warm pilot into Section 5.1 and states the late 0.007–0.012 CE band. The association is descriptive; neither a pure cadence effect nor a final-horizon BF16 penalty is identified. |
| M2: optimizer-only timing | Added seed205 optimizer means 95.23→26.08 ms (3.65-fold ratio). The net 68.38% refresh/31.62% ordinary accounting is equivalent to the review's QR-saving-minus-warm-cost decomposition, without labeling its entire QR share as a pure frequency effect. |
| M4: interpretation and damped regime | Added the exact relative damping weight before the cap and the one-power-step interpretation of QR repair. Used coordinate-diagonal gaps, not eigenvalues. Small relative weighting is not an absolute-motion bound; the prevalence of small gaps is unmeasured. |
| Mo1: diagonalization diagnostic | Added its formula and the seed205 96-factor means/maxima at ages 1990/9990, including the original QR10 comparison. These values are extracted by the CPU context checker. |
| Mo2: screen interpretation | Metric definitions were already added after Astra; added that the 1% M crossing is only at the first sampled warm refresh, with later sampled maxima at most 0.5424%. Historical flags remain unchanged. Rejected the claimed universal dimensional impossibility for BF16 bases: identity and signed permutations pass at any dimension. The “fresh QR” comparison depends on which stored basis is re-orthogonalized and is not a universal rounding floor. |
| Mo3: FP32/high/TF32 | Existing precision revision records the actual requested high mode and backend flags. Did not relabel every FP32 product as verified TF32 execution; no kernel-level GEMM algorithm trace is available. |
| Mo4: synchronized timing | Documented phase-boundary synchronizations and host-wall intervals. Their magnitude and unequal effect versus an asynchronous loop are unmeasured; no claim that the reference is penalized by a known amount. |
| Mo5: steady-cost break-even | Added the explicitly hypothetical approximately 8,134 stock-call equality using mean step rates, excluding startup/other untimed work. It is not time-to-quality evidence or a measured shortened-schedule result. |
| Mo6: controls context | Added the original pilot reference row and call-500 CEs, labeling mixed precision/implementation and cross-launch timing. Both full-horizon QR20/warm rankings remain open. |
| m1–m4: implementation details | Added the 32-byte scalar accounting, full event/update counts, scoped zero-error parity observation and precision-cast option. Refresh/QR definitions remain explicit. |
| m5: local profile placement | Retained the short profile section/table for transparent attribution; removed its bars from the main Figure 2. The frozen original figure is preserved. |
| m6: prior work | Verified and added 4-bit Shampoo v3, which already combines quantized bases and Björck correction. Classical Jacobi/Björck lineage was added after Fable. |
| Figure suggestions | Figure 2 now shows CE relative to stock and amortized optimizer event costs from the pilot/controls. Figure 1 retains the full-horizon overview and explicit paired-gap panel; the body states the late band. No frozen report figure was overwritten. |

The local open questions are answered as far as evidence allows: LR selection
was baseline-only; global precision flags are documented; spectra/small-gap
incidence, the cause of the first-refresh discrepancy, optimized-reference
training, and shortened-horizon time-to-quality remain unknown. No hypothesis
was promoted to a measured result. The descriptive title is retained.

## Closure

All four independent model reviews are complete and their findings have been
verified, adopted, qualified or declined above. All agree the existing evidence
supports a narrowly scoped implementation case study after textual revision.
These are model reviews, not external human peer review or venue acceptance.
The reviewed working PDF leaves author/affiliation and venue choices open.
No production optimizer, frozen result or experiment protocol was modified,
and no GPU experiment or scheduler was started for this review.
