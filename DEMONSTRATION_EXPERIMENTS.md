# Author metadata and Overleaf handoff — 2026-10-29

The paper now lists Ethan Smith and the supplied contact email, with no
organization, affiliation or venue. [Overleaf upload ZIP](paper/TurboSOAP-Overleaf.zip),
[compiled ten-page PDF](paper/TurboSOAP-Overleaf.pdf), [editable LaTeX](paper/overleaf/main.tex).
The standalone project has separate sections, vector figures and a live BibTeX
bibliography. It compiles from its ZIP without Python, Pandoc or repository
paths. Overleaf edits should be kept as the editing copy; the export script
refuses to replace an existing source folder. All scientific tables/equations
and frozen evidence are preserved. The pre-author reviewed archive remains a
historical snapshot; the new export has a separate manifest and validation.

The next discussion can resume TransportMuon from its saved README and roadmap,
beginning with a state/evidence audit before selecting further experiments.
This handoff used no GPU and launched no experiment or automation.

# Manuscript reviewed and typeset — 2026-10-29

All four independent model reviews (Astra, Sol 6.1, Fable 5.1, Opus 5.5) are
complete. [Reviewed working PDF](paper/TurboSOAP-review.pdf), [finding responses](paper/reviews_20261029/RESPONSES.md),
[artifact guide](paper/ARTIFACTS.md). The nine-page draft adds precise runtime
and diagnostic definitions, descriptive event-cost accounting, the FP32 warm
pilot context, a fuller short-control table, and verified numerical lineage.
No substantive equation or headline-number errors were identified. Causal
attribution, full-horizon QR20 ranking, optimized-reference training and
shortened-schedule time-to-quality remain unestablished.

The frozen review snapshot, original report figures and completed experiments
are preserved. A new Figure 2 is generated from existing scalar data; all
production source pins and raw numerical flags remain unchanged. Both Claude
reviews completed with supervised permissions. No training, GPU reservation,
scheduler, source-code release, commit or publication occurred. Author and
venue choices remain editorial inputs. The reviewed working artifact is saved
separately at /workspace/soap-paper-reviewed-20261029.tar.gz; prior archives
remain unchanged.

> Paper draft available: [manuscript](paper/DRAFT.md), [claim map](paper/CLAIMS.md), [reproduction guide](paper/REPRODUCIBILITY.md). The completed study and frozen evidence remain unchanged; no further GPU work was launched.

> **Completed and closed, 2026-10-27:** All four ledger stages are completed. The scheduled monitor is disabled (app receipt verified; no next run). [Final report](reports/bf16_completion_20261027/REPORT.md) and `/workspace/soap-bf16-completed-backup-20261027.tar.gz` are saved. No remaining GPU work in this plan.

# Completed SOAP demonstration experiments

The user-authorized bounded plan is complete: two full-BF16 model-initialization
comparisons, two short sparse-QR controls and one same-state QR attribution
profile. [Final report](reports/bf16_completion_20261027/REPORT.md),
[scalar results and provenance](reports/bf16_completion_20261027/summary.json),
[automation ledger](automation/remaining_experiments_20261027.json).

One109M-parameter Transformer, OpenWebText with fixed order, seeds204/205,
stock matrix betas(.95,.95)/covariance.95, full10k LR horizon. No model, data,
LR or beta sweep. FullBF16 M/V/C/Q and matrix arithmetic, FP32 master weights
and accumulation/solver islands, compiled shape batches, warm10/QR50.

Final stock/fullBF16 CE: seed2043.411152/3.422389, seed2053.409272/3.419354.
Mean paired CE gap+.010660, approximately1.0–1.13% higher perplexity. Show the
full learning curves and larger early gap. This is an observed approximation,
not exact matching or statistical equivalence. Fixed data order and a research
validation panel limit scope; no untouched test-set claim.

Matrix state is halved2.320→1.160GiB. Newly matched whole-step savings17.36%
(seed204100-call prefix) and18.66% (seed2059,900 calls) use different windows.
Time includes auxiliary updates/packing/casts; evaluation, diagnostics and
startup belong to elapsed accounting. Baseline is the original stock
implementation. Compilation/storage/batching/cadence change together, so do
not attribute all improvement to the warm kernel.

Pure fullBF16 QR20/QR50, seed2042k calls, use the full10k LR horizon and matched
initialization/validation. Compare with cached warm prefix: final CE
3.990383/3.998679 versus warm3.993939. Warm shows a modest gain over QR50 and
slightly worse loss than QR20 at similar-to-QR50 contextual timing. No final-
horizon cadence superiority claim. Cached warm timing is not a fresh paired
measurement. The actual capture/event/finite-state audits passed.

The fixed-input QR profile separates original three-product sorting/refinement
from one-product CQ reuse at768/1536/2304/3072 and batch1/12, identical FP32
precision within pairs. Batch12 primitive time falls.95–4.63%; solver dominates.
Near-tie permutations/signs differ; max synthetic next-update difference.112%.
No bitwise/trained equivalence claim, no new trained efficient-FP32-QR10 baseline.

Finite M/Gram measurements remain diagnostics, per user steering. Preserve old
flags; do not invent a loss acceptance threshold. Actual finite state,
reproducibility, source/capture and resource checks retain operational meaning.
See the completed prior-work review: general warm reuse and BF16 storage are
established. A paper may focus on the concrete dense kernel/precision policy
and the measured quality/time/memory tradeoff, with these limitations visible.

The additional controls/profile used at most1,500seconds supervised wall time
including startup/any queue, below their1,800+600second execution caps and
2,700second total authorization. No retries, new full runs, sweeps or additional
GPU work remain. Final report, research notes and compact backup close this
plan. Any broader study is a new decision rather than an automatic successor.
