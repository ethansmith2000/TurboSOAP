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

# Paper first draft prepared — 2026-10-28

The [working manuscript](paper/DRAFT.md) now includes an abstract, related-work
positioning, explicit dense update/rounding/repair/history rules, method and
experimental design, generated result tables, figures, limitations and a
pilot appendix. See [outline](paper/OUTLINE.md), [claim map](paper/CLAIMS.md),
[reproducibility guide](paper/REPRODUCIBILITY.md) and [package index](paper/README.md).
The first draft is3,423 words. Author names/affiliations, venue formatting and
an author review of framing/notation remain editorial work.

Five tables regenerate from12 saved evidence records; both figures match the
completed report byte for byte. CPU validation checks those artifacts, data/
method configuration, local links and four production source pins. The main
claim remains a measured quality/time/memory tradeoff against original stock-
beta QR10, with the loss gap, timing scopes and comparator limitations visible.
Finite numerical screen crossings are diagnostics. No experiment or production
optimizer change was made. The ledger remains completed and its monitor disabled.

This writing work is backed up separately at
/workspace/soap-paper-draft-20261028.tar.gz. The prior completed-experiment backup
and all frozen result records remain unchanged. The paper's source/figure/
evidence inventory and validation receipt are in paper/.

> **Completed and closed, 2026-10-27:** All four ledger stages are completed. The scheduled monitor is disabled (app receipt verified; no next run). [Final report](reports/bf16_completion_20261027/REPORT.md) and `/workspace/soap-bf16-completed-backup-20261027.tar.gz` are saved. No remaining GPU work in this plan.

# Bounded demonstration measurements complete — 2026-10-27

All approved GPU work is complete and audited. [Final compact report](reports/bf16_completion_20261027/REPORT.md),
[summary/provenance](reports/bf16_completion_20261027/summary.json), and PNG/SVG
loss/control/profile figures are saved. Fixed full BF16 final paired CE gaps
at seeds204/205: +.011237/+.010082, mean+.010660 (about1.0–1.13% higher
perplexity). Matrix state2.320→1.160GiB. Newly matched step savings17.36%
(seed204,100-call prefix) and18.66% (seed205,9,900 calls); scopes remain separate.
Finite moment/Gram screen crossings remain diagnostics; raw flags are unchanged.

Short full-BF16 controls at2k calls: QR20/warm10-QR50/QR50 CE
3.990383/3.993939/3.998679. Warm's small improvement over held QR50 comes with
extra work; QR20 is slightly better at this prefix. Cached warm timing is not
newly paired, and these controls do not establish final-horizon superiority.
Same-state fixed-input CQ reuse reduces batch12 QR primitive time.95–4.63%;
solver time dominates. Near-tie ordering changes occur; max synthetic next-
update difference.112%, not bitwise equivalence or trained-trajectory evidence.

The specific dense compiled precision policy supports a modest quality/time/
memory demonstration against original stock SOAP, not a general superiority
claim or novelty of warm reuse/BF16 storage. Two fixed-order initialization
seeds do not establish statistical equivalence. No efficient-FP32-QR10 full
training comparison was added. No production optimizer changes or further GPU
experiments are part of this completion. Drafting can use the bounded evidence.

The final ledger, schedule-disable receipt and compact backup record closure.
Historical ACTIVE/launched/draft statuses below are superseded by this record;
the ledger and live supervisor state remain authoritative.

# Short controls audited; fixed-input QR profile launched — 2026-10-27

The QR20/QR50 pair completed in24.4minutes with supervisor exit0 and repeated
frozen source/input/support/event/capture/finite-state audits passed. See
[control review](reports/bf16_refresh_controls_20261027/REVIEW.md). Seed204 at
2k calls: QR20/warm10-QR50/QR50 CE3.990383/3.993939/3.998679. Complete shared
cycles101..2000 take329.641/314.142/313.551ms; warm is cached across launches,
so the small warm-versus-QR50 timing difference is not a fresh paired result.
These prefix results support a modest quality/cost tradeoff, not final-horizon
superiority; no new method or schedule is selected.

The approved [same-state QR attribution profile](BF16_QR_ATTRIBUTION_20261027.md)
passed four CPU checks and is now submitted as soap-bf16-qr-profile under one
shared gpu-claim. Bundle /workspace/optimizer_replay_results/soap_bf16_qr_profile_20261027_v1.
Fixed synthetic input at existing factor dimensions, matched FP32 QR precision,
batch1/12; original versus CQ reuse, numerical update/state observations and
synchronized timings. Hard600second claimed execution cap including startup,
no training, production changes or retries. Its result and final synthesis are
the only remaining bounded tasks. Ledger/live supervisor state is authoritative.

# Full BF16 replicated; bounded refresh controls launched — 2026-10-27

Both full BF16 model seeds are now complete. The repeated frozen measurement
and source/input audits pass. [Seed204 review](reports/bf16_replicate_20261027_seed204/REVIEW.md)
and [two-seed scalars](reports/bf16_replicate_20261027_seed204/monitor_review.json):
stock/fullBF16 final CE204=3.411152/3.422389,205=3.409272/3.419354. Mean paired
CE gap+.010660 (~1% perplexity), with half matrix state. Seed204 newly matched
100-call prefix saves17.36% step time; seed205 paired9,900-call comparison
saves18.66%. These timing scopes differ. Two initialization seeds with fixed
data order do not establish statistical equivalence. Finite numerical screen
crossings remain diagnostics, consistent with latest user steering.

The approved [fullBF16 QR20/QR50 controls](BF16_REFRESH_CONTROLS_20261027.md)
passed17 CPU checks and are now launched sequentially under one shared GPU0
claim, program soap-bf16-controls204. Bundle
/workspace/optimizer_replay_results/soap_bf16_refresh_controls_20261027_seed204_v1.
Two2,000-call controls, seed204, full10k LR horizon, full validation, unchanged
production math. Compare to the cached warm prefix; no new warm/full training.
585 frozen Python sources, pinned production/input/support hashes; actual
cadence tested, eight expected training graphs. Hard combined execution cap
1,800seconds includes compilation/startup (TERM1790s, KILL10s later); no retry.

Automation ledger owns this launch. Do not resubmit an active stage. The QR
attribution profile remains pending until controls complete and are audited.
Only the previously authorized profile/report remain after these controls.

# Bounded experiment completion automation enabled — 2026-10-27

The user explicitly authorized an automation to detect completed runs and proceed
through necessary remaining experiments. A T3 recurring task is enabled in this
thread every10 minutes; first check2026-10-27 08:10:11 UTC. See
[runbook](EXPERIMENT_AUTOMATION.md),
[atomic state ledger](automation/remaining_experiments_20261027.json) and
[schedule receipt](automation/schedule_receipt_20261027.json). The current frozen
replication supervisor remains unchanged and does not itself spawn successors.
The newly authorized app-owned monitor performs the review and next launches.

Scope: review seed204 replication and synthesize seed204/205 evidence; run fixed
full-BF16 QR20 and QR50 controls for2k calls each at seed204; run one small same-
state QR attribution profile; save final compact report/backup and disable the
schedule. At most one TurboSOAP GPU job queued/active, shared gpu-claim and
supervisor, no retries or policy tuning. Additional GPU execution after current
replication is capped45 minutes; controls reserve at most30 minutes combined,
profile10 minutes. Numerical errors stay descriptive. Failed/invalid experiments,
ambiguous ownership, exceeded limits or new methodological choices stop follow-
ups and return to the user. No broad benchmark suite or new full training runs.

The monitor expires2026-10-27 15:58:27 UTC if still unfinished. It disables
itself on completion/failure/expiry. Atomic ledger reservations prevent duplicate
submissions; all raw results and historical flags are preserved. Later preparation
includes meaningful CPU checks, frozen protocols/source and input/production
hash verification before GPU work. T3 scheduling/provider and workspace access
must remain available for agent review/launch turns; supervised GPU jobs keep
their own lifecycle independently. Prior notes saying no automatic successor
describe the earlier standalone launch scope and are superseded by this new
user authorization at the coordinator level.

# Full BF16 seed204 replication launched — 2026-10-27

The user approved the proposed replication. [Frozen protocol](BF16_REPLICATION_20261027.md).
Fixed full BF16 M/V/C/Q warm10/QR50 at model seed204,10k calls, same stock betas,
data/order and full10k schedule as seed205. No optimizer or training-step source
changes. Reuse the completed `primary_stock_qr` seed204 quality reference (final
CE3.411151532) and run201 fresh stock replay calls first, requiring exact
reproduction at0/50/100/200 and pinned initial weights/gradients/loss/norm/CE.
Paired timing uses calls101..200 of both new phases on the same claimed device;
cached full-control timing remains contextual.

Numerical-error role is explicitly **diagnostic**, per user preference. Record
all old moment/Gram screen flags, with no controller/rejection based solely on
their finite values. Finite state, reference reproduction, capture and resource
checks retain their operational roles. Runner/audit helpers now accept the
specified model seed and candidate; production optimizer and training-step
files match the completed seed205 experiment byte for byte.31 CPU checks pass,
including advisory diagnostics with strict reference reproduction.

Managed program `soap-bf16-repl204`, one shared GPU claim, stock replay then
full BF16. Bundle
`/workspace/optimizer_replay_results/soap_bf16_replicate_20261027_seed204_v1`,
583 frozen Python sources, source/input/reference hashes. Expected55–65minutes
after admission,90-minute phase/two-hour execution limits,24GiB allocation,
no retry/automatic successor. Read live supervisor/claim state and `progress.json`
for current status. Completion copies compact audited evidence to
`reports/bf16_replicate_20261027_seed204/`. No model/optimizer checkpoints.
Short QR20/QR50 controls remain later planned work, not part of this launch.

# FP32 basis run completed and decision criterion clarified — 2026-10-27

The FP32-basis seed205 confirmation completed in57.69minutes, exit0; repeated
source/input/support, replay-reproduction and full candidate audits pass. See
[review](reports/bf16_hybrid_20261026_seed205/REVIEW.md) and
[summary](reports/bf16_hybrid_20261026_seed205/review_summary.json).
Final CE stock QR10/full-BF16/BF16-FP32-bases:
3.409272/3.419354/3.422055. FP32 bases improve max actual-M error to.5155% from
1.0356%, but do not show better loss on this seed; they use1.608GiB state versus
full BF161.160GiB and stock2.320GiB. Keep the single-seed scope; no significant
quality ranking is established. Matched100-call prefix time falls16.73% for
FP32 bases. Full BF16's18.66% timing comes from a different paired9900-call
window and should not be treated as a precise head-to-head timing comparison.

The user explicitly clarified that loss performance matters more than the
arbitrary1% engineering bound. I gave that screen too much decision weight.
For method selection going forward, finite moment/Gram errors are descriptive
diagnostics; loss, efficiency and actual stability are primary. Historical
thresholds/flags and raw artifacts remain unchanged. Full BF16's old false
screen flag does not establish bad learning, and the1% threshold has no validated
loss-quality meaning. Nonfinite/invalid-state, reproducibility and resource
checks still serve their actual operational purposes.

Current recommendation: return to fixed full BF16 as the leading precision
option and replicate it at seed204 for10k calls, reusing the completed stock204
quality control plus a short timing replay. This costs about one additional
GPU-hour and would yield two full model-initialization comparisons. No new run
was launched in this review turn; both completed supervisors are inactive.
The revised [demonstration plan](DEMONSTRATION_EXPERIMENTS.md) keeps the scope
to one OWT model, two model seeds and cheap refresh/profile controls. Stock
betas and the measured loss gaps remain explicit.

# Full BF16 reviewed and FP32 basis confirmation launched — 2026-10-26

The10k-call seed205 stock QR10/full BF16 pair completed in119.93minutes,
supervisor exit0. Repeated source/input/support hash, event, evaluation and
capture audits passed. [Completed review](reports/bf16_full_horizon_20261026_seed205/REVIEW.md)
and [scalar summary](reports/bf16_full_horizon_20261026_seed205/review_summary.json).
Final CE QR10 **3.409272**, full BF16 **3.419354**, delta+.010082 (1.013% higher
perplexity). Step380.548->309.533ms,18.66% less time; matrix state2.320->1.160GiB.
Full BF16 is **numerically unqualified under the frozen guard**: actual-M error
exceeded1% for three matrices at the first warm refresh, maximum1.0356%. Sampled
late maximum.5424% does not erase that failure. Original absolute Gram also
fails as expected. Preserve all results and thresholds; do not claim exact
matching, equivalence or qualification from the completed measurement audit.

At the user's request to review, launch next and keep demonstrations modest,
launch the existing BF16 M/V/C with FP32 bases at seed205 for10k calls. It passed
the seed204 pilot with.5326% max actual-M error and both geometry guards. Reuse
the completed stock205 control for learning quality; run201 stock replay calls
first and require exact validation reproduction at0/50/100/200. Compare timing
on calls101..200 of both new phases on the same claimed device; cached full-
control wall time is contextual. No optimizer code/math change, only selection
of the existing FP32-basis precision option. [Frozen next protocol](BF16_HYBRID_CONFIRM_20261026.md).
28 CPU checks passed. Managed program `soap-bf16-hybrid205`, admitted onGPU0
after normal queueing;581 frozen Python sources. Bundle
`/workspace/optimizer_replay_results/soap_bf16_hybrid_20261026_seed205_v1`.
Expected55–65minutes after admission, oneGPU, no retry or successor. Live state
is in supervisor/claim status and the bundle's `progress.json`; this note is a
historical launch record once the job completes. Compact evidence will copy to
`reports/bf16_hybrid_20261026_seed205/` automatically. No tensor checkpoints.

[Draft demonstration experiments](DEMONSTRATION_EXPERIMENTS.md): one109M OWT
model, two fixed model seeds using existing quality controls, shortQR20/QR50
controls and a small symmetric QR/profile audit. Roughly2–2.5 additional GPU-
hours including the current candidate; later items are drafts, not queued jobs.
Report measured quality/time/memory tradeoffs and qualification failures. Broad
dataset/model sweeps are outside the narrow matching-to-stock demonstration.

# Pilot reviewed; full-horizon seed205 pair launched — 2026-10-26

The five-arm2,000-call stock-beta pilot completed in64.94minutes, supervisor
exit0, with all source/input/support hashes and event/diagnostic audits repeated.
See [review](reports/bf16_learning_20261026/REVIEW.md),
[curves and comparison](reports/bf16_learning_20261026/review.html), and
[scalar summary](reports/bf16_learning_20261026/review_summary.json).
Final CE QR10/original-warm/compiled-FP32/full-BF16/BF16-FP32-Q:
3.985905/3.995753/3.991560/3.993939/3.995611. Full BF16 halves matrix state and
saves18.19% measured whole-step time against original QR10,7.12% against existing
warm, and.99% against compiled FP32 storage. These are implementation comparisons,
not claims against optimized/adaptive/sparse QR. All declared precision-specific
guards passed; BF16 still fails the original absolute Gram guard. Its overall
actual-M maximum.9924% remains at the first warm refresh; sampled late maximum
.5323%. Coordinate offdiagonal maxima near.96 occur in all warm variants at
age1990, so orthogonality/M checks do not certify close diagonalization.

At the user's explicit request to review and launch the next experiment, advance
the fixed full BF16 policy to one10,000-call pair versus original stock QR10 at
fresh model seed205. Selection is for memory benefit and stable sampled late
transport, not a CE ranking from one pilot. Same data/order, betas and full10k
LR schedule. [Frozen full-horizon protocol](BF16_FULL_HORIZON_20261026.md).
New initializer is the only recipe change; optimizer source/mathematics remain
the successful pilot's.26 CPU checks pass. Bundle
`/workspace/optimizer_replay_results/soap_bf16_full_20261026_seed205_v1`,579 Python
sources, input/support hashes. Managed program `soap-bf16-full205`, sequential
QR10 then BF16 on one shared claim; expected2–2.5hours,90-minute arm/3-hour pair
limits,24GiB allocation. No retry/automatic successor. Check live `progress.json`
and supervisor state rather than inferring current status from this launch note.
Completion copies audited compact evidence to
`reports/bf16_full_horizon_20261026_seed205/`. No model/optimizer checkpoints.
Full-horizon quality, additional paired seeds and efficient/sparse QR controls
remain open; existing completed v1 evidence is preserved.

# Five-arm BF16 learning comparison launched — 2026-10-26

User authorized the proposed2,000-call stock-beta comparison. See the
[frozen learning protocol](BF16_LEARNING_20261026.md). Five fresh sequential
arms: stock QR10, original warm v1, compiled FP32 storage, full BF16 storage,
and BF16 M/V/C with FP32 bases. Same seed204/data order and full10k LR schedule;
1,024-row validation panel, sparse early/late actual-M and alignment diagnostics.
No optimizer mathematics changed from the successful storage screen.24 CPU
checks passed, including cadence, factory, observation-preservation and existing
BF16/capture tests; receipt in `reports/bf16_learning_20261026/cpu_checks.xml`.

Managed program `soap-bf16-learning`, admitted on GPU1 via shared gpu-claim.
Bundle `/workspace/optimizer_replay_results/soap_bf16_learning_20261026_v1`,
577 frozen Python sources, pinned input hashes and startup reference receipts.
At launch the stock QR10 arm is first; each arm must reproduce the pinned
initial weight/gradient hashes and initial full-panel CE before training.
Expected60–75minutes, 30-minute arm/two-hour campaign limits,24GiB allocation,
no retry/autorestart/automatic successor. Numerical guard failures are recorded
as unqualified; nonfinite/startup/resource/capture failures stop the campaign.

Read the bundle's `progress.json` and live supervisor state before acting: this
launch note is historical once the job finishes. Per-arm logs are under
`runs/<arm>/events.jsonl`, portal log `/var/log/portal/soap-bf16-learning.log`.
Completion automatically audits results and copies compact scalar evidence to
`reports/bf16_learning_20261026/`. No model/optimizer tensor checkpoints. No
result or method-selection claim yet; existing completed v1 evidence stands.

# BF16 storage / compiled optimizer screen completed — 2026-10-26

At the user's request, implemented the opt-in `bf16_soap.CompiledSOAP` branch:
batched full optimizer regions per repeated matrix shape, BF16 M/V/C/Q storage
and matrix arithmetic, FP32 master weights/accumulation/solver islands, plus an
FP32-basis alternative. Existing v1 optimizer sources remain untouched. See
[precision contract, bounded protocol and completed results](BF16_STORAGE_20261026.md)
and [compact audited evidence](reports/bf16_storage_20261026/comparison.json).

On a fresh 201-call stock-beta OWT screen, all-state BF16 halves matrix state
2.320->1.160 GiB and reduces measured optimizer/whole-step time 50.19%/7.74%
against existing FastSOAP. Peak allocation 9.394->8.218 GiB. Early validation CE
5.768391 versus 5.768373, one seed and a 128-row panel, is a smoke test only.
Compiled FP32 storage already saves 6.15% whole-step time; full BF16 adds 1.70%
relative to that arm, so attribute most speed to the combined implementation.
All compiled variants hold 12 graphs with no graph breaks through call201.

Full BF16 bases fail the original absolute Gram guard (~.13 vs .05), but pass
the prospectively declared BF16 normalized guard (.002363 <= .005). Actual-M
error .9924% is close to the 1% limit. BF16 M/V/C with FP32 bases uses 1.608 GiB,
saves 6.58% whole-step time and lowers actual-M error to .5045%; both original
geometry and BF16 guards pass. Keep both experimental options. Longer matched
learning and paired replications are needed before replacing FP32 v1 evidence.
Stock betas remain the primary comparison. General BF16 storage is established
prior work, including Milligan et al.; the prior novelty/efficient-QR/sparse-control
review requirements below remain open. All screen processes exited; no next
GPU job was launched. v1/v2 startup failures and final v3 sources are preserved.

# Independent pre-freeze review completed — 2026-10-26

See the [reconciled review](reports/review_20261026/REVIEW_SYNTHESIS.md),
[GPT methodology review](reports/review_20261026/gpt_review.md), and
[GPT math review](reports/review_20261026/gpt_math_review.md). Both independent
GPT reviewers read the requested full papers and audited actual code/results;
Claude Opus/Sonnet attempts failed at the provider layer and produced no review.
The parent independently checked papers, the linked authors' implementation,
saved timing/quality curves and a CPU stall counterexample. No production
optimizer/trainer code changed and no GPU job was launched by this review.

Preserve tested v1 as an immutable reference. Before final performance claims,
qualify the existing algebraic CQ-reuse QR optimization under the CURRENT
recipe, apply it symmetrically to control/repair, and measure complete cadence
cost. It removes redundant products in exact mathematics but can change
finite-precision ordering; no new speedup is measured. A single early-QR phase
is the most evidence-led optional schedule candidate because the current stock
gap peaks at+.118CE at500 and falls to+.007 at10000. Choosing500 or1000 is a
prospective design choice; no switch-point sweep or trial is launched. Shared
fusion/batching is conditional on profiling. Persistent P, block updates,
adaptive repair and BF16 state are separate branches requiring new validation.

The papers materially narrow novelty: our specific dense matmul warm step,
repair/retraction and state/precision policy are distinct from stored-P subspace
QR and adaptive decomposition, but warm starts/Jacobi/hybrid refresh alone are
not novel claims. Exact superiority/priority remains unestablished. QR repair
is substantive because the warm generator can stall at equal diagonals despite
nonzero coupling; the CPU example does not prove incidence in training.
Late actual-M/residual diagnostics belong in future confirmation runs.

The primary stock-beta QR10 comparison remains primary; sparse QR20/50 controls
provide context rather than a retroactive requirement to beat every schedule.
Methodology corrections for paper labels: per-arm elapsed uses a warmed
campaign compiler cache and excludes earlier preparation/calibration and final
cleanup; "full state hash" covers listed tensors, not every Python scalar,
CPU RNG or buffer. The shared validation set is not an untouched test set.
The49-test XML is v3 CPU evidence; v4 adds cleanup and passed GPU qualification.
See synthesis for full scope. Existing measurements and source remain unchanged.
If adopting a new version, explicitly re-plan replications rather than pooling
old/new timings. The review is completed; no successor job is queued.

---

# Pre-freeze independent review in progress — 2026-10-26

The user requests independent Claude/GPT optimization/methodology brainstorming
and a close comparison with arXiv2605.26327 and2506.03595 before final freeze.
Review artifacts and exact delegated-task IDs are in
[reports/review_20261026](reports/review_20261026/parent_findings.md).
Claude Opus5.5 and Sonnet5.5 both failed immediately with "Provider turn failed"
and produced no review. Two independent GPT reviews were commissioned instead:
GPT6-astra math/optimization review is complete; GPT6.1-sol methodology/novelty
review is still running. Do not describe these as completed Claude reviews.

The parent has read the full primary method/experiment sections and checked the
linked Purifying Shampoo implementation at a pinned revision. See the
[parent findings](reports/review_20261026/parent_findings.md),
[math review](reports/review_20261026/gpt_math_review.md),
[timing decomposition](reports/review_20261026/parent_timing_analysis.json), and
[CPU stall counterexample](reports/review_20261026/equal_diagonal_counterexample.json).
No optimizer/trainer implementation changed and no GPU job was launched.

Provisional priorities: retain v1 as immutable reference; qualify the already
known CQ-reuse QR primitive under the current recipe and apply it symmetrically
before freezing a performance headline; consider one early-QR burn-in policy
because the stock warm CE gap peaks at+.118 at500 and falls to+.007 at10000.
The1000-call burn-in's~41s extra cost/~9.73% retained step saving is only an
arithmetic estimate from old per-policy mean costs, not measured learning or a
new speed claim. Current warm-only surcharge has about2.31% complete-step
headroom even if eliminated entirely under fixed other costs, so do not launch
an unlimited warm-kernel tuning grid. Fused common ordinary work and batching
need measured whole-policy costs and state tests. Persistent basis-coordinate
covariance, partial/block updates and adaptive repair are distinct new policies,
not automatically safe drop-ins. Both papers substantially overlap broad warm,
Jacobi and cheap/expensive refresh ideas; the concrete dense mixed-precision
update/repair/state policy remains the narrower positioning question.

The current warm generator can stall at equal diagonal entries despite
nonzero offdiagonal error; this was verified on a2x2 CPU example. It is not
proof of incidence or harm in training, but QR repair is substantive and pure
warm/global diagonalization claims are unsupported. Geometry guards alone do
not measure tracking quality, and late actual-M/diagonalization observations
should be added sparsely to future confirmation runs. The user's primary
matched-interval QR10 comparison remains primary; sparse controls supply
context rather than an undisclosed requirement to outperform every schedule.
Final reconciliation awaits the second review. No new experiment is approved
or queued by this documentation entry.

---

# Primary comparison anchor and paper-closing proposal — 2026-10-26

The user favors official stock SOAP betas as the primary comparison anchor and
asks whether method development is nearing completion and a paper is now the
right focus. Use matrix beta1=.95, beta2=.95, effective covariance beta=.95,
QR10 control and the frozen FastSOAP warm10/QR50 candidate on OpenWebText.
Retain the other beta profile as a secondary robustness result. Pin the tested
upstream revision and complete numerical/grouping/training recipe; stock betas
are a reproducible reference, not a guarantee that every other recipe is
suboptimal. The current .0015 matrix LR was selected by the recorded bounded
baseline calibration. Do not silently change global optimizer defaults,
auxiliary betas, source algorithms, or completed experiments.

Recommendation: freeze the current TurboSOAP method and shift effort to a
bounded evidence/positioning package. This is a proposed plan, not authorization
to launch extra training jobs, and no successor is launched by this note.

Proposed closing experiments:

1. Two additional paired training replications of stock QR10/warm10QR50,
   for three paired seeds total. Freeze the selected LR, horizon, numerical
   policy and validation protocol. Pair initialization and data order within
   each replication; document whether data order also varies across seeds.
   Counterbalance QR-first/warm-first order. Keep the existing validation set.
2. Stock-beta QR20 and QR50 contextual controls on the existing seed, with
   the same recipe and complete-cost accounting. QR50 shares warm's repair
   cadence and measures the incremental value of intervening warm updates;
   QR20 tests a simpler intermediate cost/quality tradeoff. This screen plus
   the two paired replications is six additional full runs. A competitive
   sparse policy would need fair bounded LR calibration/replication before
   claiming method superiority; an unfavorable outcome should narrow the
   claim rather than trigger an indefinite tuning loop. These secondary
   controls do not replace the primary matched-interval QR10 comparison.
3. A bounded implementation/precision attribution check: confirm an efficient
   QR control and separate warm reuse from mixed-precision and compilation
   benefits in complete costs. Numerical parity alone does not certify a
   strong speed baseline. Reuse applicable saved diagnostics; avoid rerunning
   missing historical experiments merely to recreate context.

A second model shape/scale is conditional on claiming generality; the narrow
109M OpenWebText result can anchor a focused technical report. Wider-factor
advantages, effective-rank mechanisms, the unresolved old H200 refresh timing,
additional beta/cadence grids, and new moment-update rules are deferred research,
not automatic requirements for closing this narrower method. TransportMuon
remains a separate extension requiring a modern efficient fresh-polar baseline
and matched warm validation before a joint SOAP/Muon paper claim.

Working paper title: "Warm Basis Tracking for Faster SOAP". Suggested structure:
problem/cost motivation; frozen update and state semantics; skew/Jacobi motion,
retraction and QR repair; stability/state limitations; matched-token and measured
elapsed-time learning; sparse-refresh and precision attribution; reproducibility
and limitations. Treat diagonal-second-moment handling as an approximation, and
avoid claiming dense warm operations improve cubic asymptotic complexity.

Current supportable claim is workload-specific: on one109M OpenWebText seed,
stock-beta warm10QR50 reduces complete-step time10.81% and optimizer time44.03%
while increasing final CE by .006976 (~.700% perplexity). First observed
threshold timing is favorable, but sampled panels do not establish exact
continuous crossing times or statistical equivalence. Existing published
historical10x refresh-event figures are not current end-to-end training speedups.
Broader claims await the closing evidence above. Acceptance/novelty is not
established by this plan.

Primary sources checked on2026-10-26:
- Official SOAP defaults and update semantics:
  https://github.com/nikhilvyas/SOAP/blob/main/soap.py ; tested pinned fixture at
  a1e553530fde97d0e6b307d7c82ac6d38b072340 in tests/reference/provenance.json.
- Milligan et al., "Reparametrizing Shampoo and SOAP for Subspace Basis Updates
  and BFloat16 Storage" (2026): https://arxiv.org/abs/2605.26327 . Their partial
  subspace QR updates and low-precision storage are directly relevant prior
  work. Our warm full-basis motion and repair/state/precision policy must be
  distinguished explicitly; a head-to-head requirement depends on the claims.
- Eschenhagen et al., "Purifying Shampoo" (2025):
  https://arxiv.org/abs/2506.03595 . Their adaptive basis-update criterion based
  on warm-started QR is related scheduling/approximation work.
- Vyas et al., "Improving SOAP Using Iterative Whitening and Muon":
  https://nikhilvyas.github.io/SOAP_Muon.pdf . Iterative orthogonalization and
  low-precision implementation issues are relevant to algorithm positioning.
This is a bounded initial related-work check, not an exhaustive novelty review.

---

> **Completed and audited — 2026-10-26:** All four10,000-call primary arms finished,40,000 calls/1.30944B target presentations combined, with matched initial parameters/gradients and verified input/source hashes. Stock betas at selected matrix LR .0015: QR10 CE3.411152 versus warm10/QR50 CE3.418128 (+.006976, ~.700% higher perplexity); mean complete step380.056 versus338.956ms (10.814% lower), optimizer94.556 versus52.920ms (44.033% lower), elapsed65.281 versus58.454min. Previous beta profile at selected matrix LR .0005: QR CE3.442197 versus warm3.451100 (+.008903, ~.894% higher perplexity); complete step380.905 versus338.419ms (11.154% lower), optimizer94.594 versus52.000ms (45.029% lower), elapsed65.461 versus58.389min.
>
> All180 primary geometry observations/17,280 factor checks,384 actual-M checks,12 full-state evaluation checks and four registered releases pass; final allocated bytes zero for every arm.572 frozen files and all four input files rehashed successfully. First observed CE<=3.5 occurs at43.864 versus48.980min for stock warm/QR, and46.638 versus52.345min for previous warm/QR; these are sampled observations, not precise continuous crossing times. Stock's warm upper observed bracket precedes QR's prior observation; repeated panels and this single paired seed are not independent replications. Stock is better here under its selected LR, but beta1 and selected LR differ, so the profile comparison does not isolate covariance/beta2 causally. No universal quality equivalence or default promotion. [Completion audit](reports/owt_restart_20261026/completion_audit.json) and [full comparison](reports/owt_restart_20261026/comparison.json) are copied into this repository for recovery. Our supervisor exited normally and its GPU claim was released; no successor launched. A fresh model-seed replication of the stock QR/warm pair is the proposed next decision, not a launched experiment. Earlier ACTIVE labels below are historical.

> **OpenWebText campaign qualified and launched — 2026-10-26:** [Restart protocol](OWT_RESTART_20261026.md). User authorizes continuing with the existing shared OpenWebText dataset, including official SOAP beta settings. New 109.397M/768x12 full two-sided comparison: stock matrix betas(.95,.95)/covariance .95 and previous(.9,.999)/covariance .99, each QR10 versus FastSOAP warm10/QR50, identical data/model seed204. Separate auxiliary AdamW stays(.9,.999). Full horizon10,000 calls/327.36M targets per arm, global32/physical8/sequence1024 blocks yielding1023 targets each. Six baseline-only500-call LR calibrations (.0005/.0015/.003 per profile), using the same full schedule, precede four fresh main arms; selected LR shared within each QR/warm pair. This is a new OpenWebText experiment, not a FineWeb Trial213 replay or a full upstream training-recipe reproduction.
>
> 49 CPU tests PASS, including pinned upstream stock-beta parity. Four61-call GPU qualification arms PASS with exact cross-arm initial weight/gradient/validation agreement, geometry/actual momentum checks and zero registered tensor/storage leaks; worst Gram .017224, worst M relative .001083, peak9.398GiB. A separate real-batch compiled/eager CE execution screen PASS (gradient relative .1385%, cosine .99999904, CE difference2.384e-7). Deterministic CUDA and explicit post-training cuBLAS workspace release were qualified; earlier startup/release failures and code fixes remain preserved in versioned bundles. No thresholds relaxed.
>
> Active managed campaign: `soap-owt-learning`, lifetime-claimed physical GPU1, frozen bundle `/workspace/optimizer_replay_results/soap_owt_restart_20261026_v4`, log `/var/log/portal/soap-owt-learning.log`. Calibration and primary outcomes remain pending. Exact selected input IDs and hashes, all571 Python sources, protocol, checks and scalar logs retained; no weights/checkpoints. One-seed/profile-comparison/order/shared-node limits apply.12h external bound,6h per-arm bound,24GiB allocation bound, no restart or undeclared successor. Check live status before action. Other projects/services remain untouched; older ACTIVE labels below are historical.

> **Restart direction — 2026-10-26:** The user prefers OpenWebText for consistency with other projects and is interested in stock SOAP covariance/beta2 settings as well as the recent profile. FineWeb was selected for a particular reference-recipe comparison; it is not required for warm-basis research. Prepare the next comparison on OpenWebText. This supersedes the proposed immediate FineWeb Trial213 replay; the historical records below remain unchanged. No new training run is launched by this note.
>
> Official SOAP defaults are `betas=(.95,.95)` and `shampoo_beta=-1`, which ties covariance decay to beta2 and therefore gives effective covariance beta `.95`; refresh interval defaults to 10. The pinned upstream fixture at `tests/reference/soap_upstream.py` (revision in `tests/reference/provenance.json`) agrees with the official source checked on 2026-10-26. The latest historical Trial213 profile instead used beta1 `.9`, adaptive beta2 `.999`, and covariance beta `.99`. Comparing these complete profiles does not isolate a covariance-only or beta2-only effect. Stock beta values also do not by themselves make a blocked/normalized or smaller-side implementation identical to upstream SOAP.
>
> Proposed preparation: one fixed OpenWebText model/data/validation/execution recipe, with QR10 and warm10/QR50 at each of the two beta profiles, preceded by local numerical and complete-step qualification and bounded baseline learning-rate calibration. Exact model, token budget, seeds, learning rates and full-run scope remain to be frozen. Do not silently inherit the old OpenWebText config's smaller-side coverage or the FineWeb port's hardcoded beta2. The original machine lost outbound internet, only Git repositories were rescued, and external run bundles/results/data were not recovered here. Historical "active" means last saved status, not confirmed current liveness or completion. The rebuilt shared GPU coordinator is documented at `/workspace/GPU_QUEUEING.md`.

> **Trial213 active — fresh model-seed replication and sparse drift diagnostics, 2026-10-24:** [Protocol](../optimizer_replay_results/soap_seed_replication213_20261024/PREPARATION.md), [startup audit](../optimizer_replay_results/soap_seed_replication213_20261024/startup_audit.json), [diagnostic plan](../optimizer_replay_results/soap_seed_replication213_frozen_20261024/basis_diagnostic_plan.json). User authorizes the next paired run. QR10 FIRST, warm10+QR50 second, 5,000 updates /655.36Mtargets each, on claimedGPU7/private `manifold_seed_replication213`, trainer 31488. Model initialization seed204 replaces203; optimizer-vector seed203 and data/order/recipe stay fixed. This is a second model initialization, not a fresh data-order seed or a large seed ensemble.
>
> Same Shampoo covariance beta .99, adaptive variance/bias correction .999, raw momentum .9, LR .032/WD .0002/clip1, full28,332-update schedule/177warmup,185.68M parameters/width512/depth32/block512 normalized SOAP. Original transformer+compiledFP32CE, BF16forward/FP32state/recompute, physical16x8/global128/sequence1024/validation8. First50EIGH and QR/warm cadences unchanged. Additive harness overrides model seed and adds observation; backward, evaluation, state hash and data methods AST-identical to212. Existing optimizer/model/helpers untouched. Reuse exact210–212eight data files/640,000uniqueIDs/5,000batch hashes/2,048validationwindows; no downloads/copies. Earlier212 results preserved as context.
>
> Sparse measurements sample ten512x512 factors: embedding right/readout left, query left/right atlayers0/16/31, firsttile layer16fc1 left/right. Windows50->100,1950->2000,4950->5000; observe start/after49updates/after50updates+QR. Measure covariance Frobenius change, current-versus-held-start-basis relative offdiagonal defect, current/held Gram, and PSD participation ratio. Held-basis reference uses each actual arm's covariance trajectory, not a trained hold control;49->50 measurement also changes covariance, not a same-C QR isolation. These purposive samples do not measure the full factor population, direct eigendirection rotation or significant-PC count.
>
> At most20MiB C/Q RAM references retained for50updates perwindow, registered and released atwindowend; scalar results only saved. DetachedFP64 metric copies leave training precision/storage unchanged. All observation calls check RNG/modes/recompute/gradientidentity/precision/counters; first50 andlast5000 also check full model/gradient/optimizer hashes, in addition to evaluation hashes.18events/180factorobservations total;94validations,4evaluation+4diagnostic full-state checks,24finite/Gram observations and2registered releases. Diagnostics cost charged to elapsed, separately recorded; union of guards+diagnostic calls excluded from clean cadence cost with repairs included. No threshold relaxation or diagnostic-based fallback/policy change.
>
>65 frozen CPU checks PASS, including exact all-state transitions with diagnostics on/off and reference release; ten sampled factors resolve in the full1,026-factor model.339frozen inputs/330sources/84pins/568currentPython/565priorPython/33,030prior evidence files/185old private configs/dirtyHEAD-status/originalmasterbytes/data preserved. Startup PASS: new204 parameter+repeated-gradient hashes differ from203 and are locked for exact matching in secondarm; initialuniformCE and firstactualEIGH finite/all-factor geometry pass. First actual basis diagnostic50, formal speed gates80, later QR/warm/quality/diagnostic preservation and release remain pending. No trained-quality claim from startup.
>
> Expected6.5–7hours combined,8-hour internal/8h1min external limits;24GiBGPU/12GiBRSS/1GiBcache/100MiBcompactbundle. No modelweights/checkpoints/tensor snapshots/resetbanks/resume/retry/duplicate/automaticextension/successor. Exact stale private manager PID/socket files were archived after confirming absent process/command and refused socket, then the manager was started with all186programs initially stopped and all185old configs noautostart/restart. Other projects/services untouched. Monitor existing213 only. Longer horizon/actual factor-width scaling and modernMuon remain later directions if the practical result repeats.

> **Trial212 completed and audited — 2026-10-24 review:** [Report](../optimizer_replay_results/soap_covariance_beta212_20261022/REPORT.md), [analysis](../optimizer_replay_results/soap_covariance_beta212_20261022/analysis.json). All three5,000-update arms finished with Shampoo covariance beta=.99 and adaptive beta2=.999 fixed: final CE QR10 3.631892, warm10/QR50 3.635688, QR50-only 3.639754. Warm improves hold by.004065 CE (hold perplexity~.407% higher) but trails QR10 by.003796 CE (~.380% higher perplexity), with5.615% lower clean complete-step time and60.705% lower optimizer time. Step times QR2.259407s /warm2.132536s /hold2.126308s; per-arm elapsed205.087 /194.773 /194.542min. Warm adds2.457ms optimizer cost over hold; hold saves13.83seconds all-in. No claim that the tiny hold timing difference is reliable.
>
> The covariance-memory hypothesis is not clearly amplified here. Warm's endpoint advantage over hold is.004065 at.99 versus.004892 at.999. Mean1,500+ advantage is slightly larger (.004078 versus.003895), but3,000+ is smaller (.003172 versus.005100); warm wins34/36 and19/21 panels respectively at.99. Thus the beta effect varies across training stage. Faster QR's relative quality advantage increases. Single seed203/fixed LR/changed job order/numerical variation preclude a general or causal conclusion. Direct covariance/basis drift was not logged: EMA turnover does not itself measure eigendirection movement, and Gram only checks orthogonality. Adaptive beta2 tuning remains separate.
>
> Predeclared CE3.8 first observed warm2,400/5804.30s, hold2,500/6032.41s, QR2,400/6091.11s. Warm's adjacent sampled elapsed bracket ends before the others' lower observations (only4.03s separation from hold); locally favorable sampled target evidence, no precise continuous crossing/interpolation or universal time-to-loss win. Early CE5 favors QR;3.6/3.4/3.2 unreached.
>
> Completion audit PASS:15,000updates /120,048microbackwards /1.966Btargets /141validations /6full-state checks /36geometry observations /36,936factors (worstGram.007936<.05) /3speedgates /3registered releases.32graphs, zero new clean optimizer graphs.335inputs /327sources /84pins /565currentPython /562priorPython /30,241priorfiles /184old configs /dirty HEAD-status /old master bytes /all exact210/211 data hashes preserved.9.9083hours, peak8.923GiB. Trainer1777416 dead,GPU7 free. Private manager already absent/refused connection; stale PID/socket files preserved, all185configs noautostart/restart. No weights/checkpoints/tensor artifacts, retries or successor. Older ACTIVE212 labels below historical.
>
> [Next proposed decision](../optimizer_replay_results/soap_covariance_beta212_20261022/NEXT_DECISION.md): fresh paired-seed QR10/warm10QR50 at.99/.999, with sparse predeclared scalar covariance-change/diagonalization diagnostics if testing the geometric mechanism; hold optional. Prioritize replication over a broad beta/cadence/precision grid. Extra.9 CPU stress remains retained and unqualified. No new job launched by this review. Before a future managed launch, recheck liveness and resolve only the exact stale private manager runtime files; do not touch other services or evidence.

> **Trial212 active — covariance memory isolated, 2026-10-22:** [Protocol](../optimizer_replay_results/soap_covariance_beta212_20261022/PREPARATION.md), [startup audit](../optimizer_replay_results/soap_covariance_beta212_20261022/startup_audit.json). The user prioritizes Shampoo beta over adaptive beta2: high covariance memory may make a 50-step hold nearly sufficient. New experiment lowers only Shampoo covariance decay from .999 to .99; adaptive variance beta2 and its bias correction stay .999, raw momentum stays .9. This is a covariance-beta sensitivity experiment, not a joint beta2 change. It supersedes the previously proposed fresh-seed replication as the immediate next task; replication remains on the roadmap.
>
> One claimed GPU7/private `manifold_covariance_beta212`, trainer 1777416. Three fresh 5,000-update / 655.36M-target arms: QR50-only first, warm10+QR50 second, QR10 third. All three use covariance beta .99 with the same adaptive beta2 .999, model, seed203, data, LR .032 / WD .0002, full 28,332-update schedule / 177 warmup, global128 / physical16 x accumulation8 / sequence1024. Same 185.68M-parameter / width512 / depth32 / blocked512 normalized model, original transformer and compiled FP32 CE. Existing210/211 .999 curves supply context without rerunning them. No hyperparameter tuning or schedule compression; 5,000 updates is 17.65% of the horizon.
>
> The new optimizer inherits the qualified step body and projected update, replacing only the covariance callable. Exact .999 all-state regression passes all three policies. Independent .99 checks cover zero and nonzero resident states, padded/one-sided blocks, initialization, ordinary/warm/QR and postrepair transitions. All 58 frozen CPU checks pass. An extra .9 padded warm CPU stress case missed the existing oracle tolerance after97; the failed XML and diagnostic are retained, .9 is unqualified and is not launched. Initial zero-variance power directions can be numerically unidentifiable even at unchanged .999; zero-state parameter/M/C/V checks remain, and power direction is compared on nonzero resident states. No optimizer or tolerance change was made to accommodate those directions.
>
> At covariance beta .999, the old coefficient after50 steps is95.12%; at .99 it is60.50% (new coefficient sums4.88% versus39.50%). This motivates the hypothesis but does not bound matrix change or eigenvector drift. Report full warm-minus-hold loss/time curves at both betas, with matched QR10 at .99 for quality context. Fixed LR, one seed, numerical and order/host variation limit causal/general claims. A larger repeated warm benefit would support the hypothesis locally; no universal equivalence or precise time-to-loss interpolation.
>
> Startup PASS: parameters and repeated initial real-data gradients match210/211 exactly; initial CE10.8249054, first EIGH finite across1,026 factors with Gram below .05. 335 frozen inputs /327 sources /84 pins /565 current Python /562 prior Python /30,241 prior evidence files /184 old private configs /dirty HEAD-status /original master bytes and exact eight-shard data/640,000 IDs/5,000 batch hashes preserved. No downloads or data copies. Actual call80 speed gates, QR/warm guards and later learning remain pending.47 validations /2 full-state checks /12 geometry observations and registered release per arm; guard thresholds unchanged.
>
> Expected about10 hours combined;12-hour internal /12h1min external watchdog,24GiB GPU /12GiB RSS /1GiB compiler cache. Retain compact source/config/provenance/IDs/scalar CE/logs/guards/releases; RAM-only states/transient hash copies. No weights/checkpoints/tensor snapshots/resetbanks/retry/automatic extension/successor. Monitor existing212 only, preserve all prior evidence, other projects untouched. Adaptive beta2 tuning, fresh-seed replication, full convergence, width and modernMuon questions remain separate follow-ups.

> **Trial211 completed and audited — 2026-10-22:** [Report](../optimizer_replay_results/soap_long_sparse211_20261022/REPORT.md), [comparison](../optimizer_replay_results/soap_long_sparse211_20261022/analysis.json). The5,000-update QR50-only control ends CE3.637591 versus completed210 warm10/QR50 CE3.632700 and QR10 CE3.632795. Warm gains.004892 CE over hold (hold perplexity about.4904% higher), with lower loss at all21 observations from3,000..5,000 (mean advantage.005100);33/36 from1,500 onward. Early results vary, including warm worse at177. Same seed and fixed validation set; repeated panels are not independent training replications. Initial gradients match exactly, but a pre-warm50-step CE gap+.003075 demonstrates numerical trajectory variation. Small later benefit is promising, not statistically replicated superiority.
>
> Hold clean cadence-weighted whole step2.126889s versus warm2.130122s, only.152% lower. Optimizer80.277 versus82.715ms,2.438ms extra for warm; per-arm all-in193.601 versus194.357min, hold saves45.33seconds (.389%). Thus warm appears to add a modest quality benefit at almost the same complete cost, while retaining the previous roughly5.74% step-time advantage over QR10. Predeclared3.8 first observations both at2,500; adjacent elapsed brackets overlap. No precise time-to-loss or universal equal-time win. All policies still beta2/covariance=.999; beta2 sensitivity remains untested and is recorded separately.
>
> Completion audit PASS:5,000 updates /40,016 microbackwards /655.36M targets /47 validations /2 full-state evaluation checks /12 geometry observations /12,312 factors (worst Gram.007210<.05) /actual speed gate /registered release with zero live registered tensors/storages.26 graphs, zero new clean optimizer graphs;331 inputs /324 sources /84 pins /562 current Python /561 prior Python /28,454 prior evidence files /dirty HEAD-status /old master bytes /exact210 data IDs and all SHA checks preserved. Job3.228hours, peak8.923GiB. Trainer1735830 dead, GPU7 free, private manager off after184 terminal programs. No weights/checkpoints/tensor snapshots or successor queued; older ACTIVE211 labels below historical.
>
> [Recommended next decision](../optimizer_replay_results/soap_long_sparse211_20261022/NEXT_DECISION.md): a fresh paired seed of warm10/QR50 versus QR50-only, hold first, before broader claims. Keep the same5,000-update prefix and unchanged full schedule. Then a targeted matched beta2 comparison with independently qualified parameterization can test the user's memory/cadence hypothesis. No new job is launched by this review. Full convergence, width/effective-rank and modernMuon work remain separate roadmap items.

> **Trial211 active — longer QR50-only control, 2026-10-22:** [Protocol](../optimizer_replay_results/soap_long_sparse211_20261022/PREPARATION.md), [startup audit](../optimizer_replay_results/soap_long_sparse211_20261022/startup_audit.json), [beta2 design note](../optimizer_replay_results/soap_long_sparse211_20261022/BETA2_DESIGN_NOTE.md). User authorizes trying the longer hold control and notes beta2 sensitivity. One 5,000-update / 655.36M-target arm on claimed GPU7, private `manifold_long_sparse211`, trainer1735830. Keep beta2 and covariance decay at .999, same 185.68M-parameter / 512-width / 32-layer blocked512 model, normalized update, source 28,332-update schedule / 177 warmup, global128 / physical16 x accumulation8 / sequence1024, original transformer plus compiled FP32 CE. First50 EIGH unchanged; then QR at100/150/.../5000, holding the basis between refreshes while covariance, moments, variance and parameter updates continue each step.
>
> Compare against completed210 warm10/QR50 and QR10 without rerunning them. Exact same seven training shards, validation shard, 640,000 unique training windows, 5,000 batch hashes and 2,048 validation windows; no new data download or copy. Same47 validation observations, two full-state evaluation checks, twelve sparse finite/Gram checks including101 after the first QR. Report full loss curves, observed crossing brackets and complete cost. If hold matches or beats warm at similar or lower cost, intermediate warm tracking has not earned its work at this beta and recipe. This contextual control shares warm's outer QR cadence; the primary QR10 comparison uses the same inner refresh interval as warm. One seed and 5,000 updates remain insufficient for convergence, equivalence or broad claims.
>
> Startup PASS: exact initial parameters and repeated real-data gradient hash match210, initial CE10.8249054, first actual EIGH finite across1,026 factors with maximum Gram error .007210 < .05. All30 frozen CPU checks pass; 331 inputs / 324 sources / 84 reference pins / 561 prior Python files / 28,454 prior evidence files / dirty HEAD-status / original master bytes and all data hashes preserved. Single additive training harness; qualified optimizer/model/helpers untouched. Actual speed gate80, clean cadence cost, later loss, QR refresh and release checks remain pending. Expected about3.3 hours, internal4-hour / external4h1min watchdog; 24GiB GPU / 12GiB RSS / 1GiB compiler cache. No weights, checkpoints, tensor snapshots, retries or successor queue. Monitor existing211; preserve completed210 and earlier evidence. Older ACTIVE labels below are historical.
>
> **Beta2 is a separate follow-up:** at .999 the nominal EMA horizon is about1,000 updates (half-life693); at .99 about100 (half-life69). After50 steps the old-covariance coefficient is95.12% versus60.50%; these weights do not bound eigenvector drift. A shorter memory may increase the value of intermediate tracking. Here covariance decay, projected-gradient variance and bias correction all use the same hardcoded .999, so changing beta2 also changes adaptive scaling. Keep211 fixed for attribution. A later matched comparison at another beta, e.g. .99, needs additive parameterization, an independent full-state oracle, the same beta for all policies and declared LR/schedule calibration. No beta sweep is launched now.

> **Longer, faster trial210 completed and audited — 2026-10-22:** [Report](../optimizer_replay_results/soap_fast_learning210_20261021/REPORT.md).5000updates/655.36Mtargets per arm on the qualified185.68M/512x32/blocked512normalizedSOAPport. Warm10/QR50 endsCE3.632700 versusQR103.632795 (−.000096): essentially matched sampled endpoint, not significant quality superiority. Warm's earlygap narrows; all36sampled panels from1500..5000 are within+.01CE (max+.009897,mean+.003068). One seed/fixedwarm-firstorder/17.65%fullhorizon: latercatchup supported, fullconvergence/equivalence/widthclaims unproven. Earlier1000stepscreenmisses remain preserved.
>
> Clean cadence-weighted whole iteration2.130122s warm versus2.259753s QR (5.737%lower), optimizer82.715vs209.760ms (60.567%lower). Optimizer difference corresponds to5.622%ofQRwholestep withFBfixed;FBdiff2.588ms accounts for~.114percentagepoints. Per-arm all-in194.36vs204.17min,9.81minless for essentially matched final loss; job6.644hours/peak8.923GiB. Actualtraining uses originaltransformer+compiledFP32CE/physical16accum8/global128/validation8; both actualspeedgates pass. Candidate repairs, cold50EIGH/compile/eval/guards/releases charged.
>
> Denseactualsamples support promisinglaterloss/time tradeoff:20of21QRpanels3000..5000haveanobservedwarm pointwithnogreaterloss/time (3600exception), descriptive only. PredeclaredCE3.8firstobservedwarm2500/6009.6s vsQR2400/6036.2s;adjacenttimebracketsoverlap,so no preciseinterpolatedtime-to3.8win. EarlierCE5favorsQR. No statisticallyreplicatedclaimorautomaticpromotion.
>
> Independentcompletionaudit PASS:10000updates/80032microbackwards/94validation/4fullstatechecks/22finite-Gram checks(22572factors,max.007731<.05)/2speedgates/2registeredreleases;32graphs/0newcleanoptimizergraphs.329inputs/323sources/84pins/561currentPython/560priorPython/25706priorfiles/dirtyHEAD-status/oldmasterbytes/all8dataSHA/640000uniqueIDs/1000prefix preserved. Trainer1723622dead,GPU7free,private manageroff after183terminalprograms. No weights/checkpoints/tensor snapshots or successorqueued. All ACTIVE210 labels below historical.
>
> [Recommended next decision](../optimizer_replay_results/soap_fast_learning210_20261021/NEXT_DECISION.md): one5000-updateQR50-only contextual control withsamefastbackend/data/schedule/panels, then freshpairedseedreplication of a useful survivor. Existing207sparsecontrolis1000steps anddoesnotestablishlaterholdquality. No newjob or cadencegrid bythisstatusreview. ModernMuon/H200/width/stage-aware directions remain followups; LLMfocus/PINNsdeferred.

> **Trial210 actual warm-arm speed gate PASS — 2026-10-21:** [Progress audit](../optimizer_replay_results/soap_fast_learning210_20261021/progress_speed_gate.json). Existing5000-update paired job continues on claimedGPU7/private `manifold_fast_learning210`,trainer1723622; no new job. Warmcall80observations over51–80: median actual forward/backward1.94963s and ordinary completeiteration1.99618s, passing<=2.6s FB requirement. These are actualtraining measurements after first50EIGH, not merely the fixed-stateprofile. QRrefresh iterations have their additional cost; no fullcadence mean claimed yet.
>
> Throughcall119, exacthasheddata/cadence/finite loss+norm/source checks pass. All5finite/geometry checks(5130factorobservations)pass; QRrepair100Gram.00001747 and subsequentwarm110Gram.007425<.05. Full-state validation50preservation passes. No new cleanoptimizergraphs after100; firstrepair100startupgraphs charged as declareddiagnostic.329frozeninputs verified and onlyoneprivateresearchjobactive. Both5000-updatecurves, QRarmactualspeedgate, laterstates/repairs/releases/finaltime-to-lossremainpending. Continue EXISTING210 unchanged; no retry/duplicate/successor/defaultpromotion or checkpoint change.

> **Faster, longer FineWeb trial210 active — 2026-10-21:** [Protocol](../optimizer_replay_results/soap_fast_learning210_20261021/PREPARATION.md). User authorizes proceeding beyond the short pilot and prefers training faster than~3s/step. [Execution209](../optimizer_replay_results/soap_loss_speed209_20261021/REPORT.md) qualified original transformer + compiled FP32 cross-entropy, physical16/accum8/global128: FB2.084s versus eager3.12–3.27s (1.50–1.57x), gradient difference.6831%/cosine.9999767/CEdifference<.000001, selected peak8.749GiB. Compiled-block208gradient-screen miss remains preserved and unpromoted.
>
> One claimed GPU7/private `manifold_fast_learning210`, trainer1723622. Two fresh5000-update arms, warm10/QR50 FIRST then QR10,655.36M targets/arm. Same185.68M/512x32/blocked512 normalized SOAP model/update/ordinary precision math; full28332-update schedule/177warmup unchanged. Physical batch changes execution only; global128 and refresh cadence stay fixed. First50actualEIGH per arm still charged. This is17.65%of the reference horizon, not a full convergence/paper/distributed/width claim.
>
> Seven SHA-pinned training FineWeb shards + separate validation shard; five new200MB shards downloaded after purpose/storage/retention declaration, total shared input1.600GB.640000 unique disjoint windows per arm. Original206first1000batch hashes and2048validation windows preserved; fresh unique append afterward. Validate0/50/177/350/500/750/1000, then every100updates through5000 (47observations/arm), physical8 validation. Primary full loss curves and actual sampled time-to-loss brackets; earlier+.01screen retained as report-only context here, not a convergence verdict or statistical equivalence. No precise interpolation/automatic promotion.
>
> Startup PASS: same full initial parameter hash as205/206, exact repeated initial real-data gradients, expected initial validationCE10.8249054, first actual EIGH finite/all1026-factor Gram.007210<.05. Early actual FB about1.95s median over first8updates; first50EIGH initialization is not steady-state. Formal actual median FB over51–80 must<=2.6s in EACH arm before continuing beyond80; gate remains pending. Sparse finite/geometry guards at1/50/60/100/110/500/1000/2000/3000/4000/5000, full-state evaluation checks50/5000, registered releases. Later warm/QR/quality/resource/release results pending.
>
>24frozen CPU model/optimizer/data/execution checks PASS;329inputs/323sources/84reference pins/560priorPython/25706prior evidence files/dirty HEAD-status/original master bytes/data preserved,561currentPython. Expected6–7hours combined,8h/8h1min internal/external watchdog;24GiBGPU/12GiBRSS/1GiBcompilecache. RAM-only states/transient hashes, no weights/checkpoints/reset banks/tensor snapshots. No retry or successor queue; monitor EXISTING210, never duplicate/restart. Other projects untouched. Longer quality and actual speed are being tested; no claim from the local profile alone.

> **Faster execution209 qualified — 2026-10-21:** [Report](../optimizer_replay_results/soap_loss_speed209_20261021/REPORT.md). Originaltransformer/blockcheckpointmath unchanged; compiledFP32tokenCE+physical16/accum8/global128measuresFB2.083864s vs eager3.274946open/3.120500close (1.50–1.57xfaster). loss16isfastestpassingcase;32/64areslower. Selectedgradientrelative.6831%/cosine.9999767/CEdiff-.000000954,withinunchanged2%/.999/.002screens;8.749GiBGPU.208compiledblockscreenmisspreserved,notpromoted.
>
>209independentcompletionaudit PASS:114.38s,27fullglobalbackwards/18timings/0newtimedgraphs/3graphs,all226gradienterrorsretained,712registeredtensors/storagesincluding742723584-byteCPUreferenceallreleased,parameters/residentoptimizerstate/countersunchanged.328inputs/322sources/84pins/560workingPython/559priorPython/25214priorfiles/oldmasterbytespreserved. Trainer1720772dead,GPU7free;noweights/checkpoints/tensorfiles. Profilingisfixednonzerodiagnosticstate,nottrainedqualityorcomplete-stepvictory.
>
>Next210freezesactual5000-updatepairedQR10/warm10QR50learningwiththisexecution,warmfirst,samefirst1000batchesplusfreshuniqueFineWebwindows,global128/source28332schedule/177warmupunchanged;validationphysical8. EarlyactualFBspeedcheckbeforelongbudget,anddenseafter1000validationfordirectsampledtime-to-loss. Userauthorizeslongerandpreferstrainingfasterthan~3s. Nooptimizerredesign,compiledtransformerpromotion,broadgridorautomaticclaimfromlocalfidelity.

> **Execution profile208 completed — 2026-10-21:** [Report](../optimizer_replay_results/soap_training_speed208_20261021/REPORT.md). Realbatch/global128, fixed nonzero diagnosticpoint with actual resident optimizer state: eagerFB3.322s(open)/3.161s(close), compiledblocks+CEB16=1.345s. Four compiledblockpaths have~2.56%globalgradientdifference,loss+.000020CE/cosine~.99967; they miss prospective2%gradientengineering screen. No threshold relaxation/default promotion; local difference neither proves training degradation nor equivalence. CEcompilation reduces memory, enabling a distinct loss-only/largerbatch209test with originaltransformer path.
>
>208completionaudit PASS:121.20s,33fullglobalbackwards/22timings/0newtimedgraphs/4totalgraphs,10.496GiBGPU/5.47GiBaggregateRSS,all226gradientsnonzero,parameter/optimizer/counters unchanged,712registeredtensors+storages(including742723584-byteCPUreference)allreleased.327inputs/321sources/84pins/559workingPython/556priorPython/24414priorfiles/oldmasterbytes preserved. No optimizersteps,trainingcurve,weights/checkpoints/tensorfiles. Trainer1719560dead,GPU7free; older208active labels historical. Next209originaltransformer+compiledCEphysical16/32/64 atglobal128, unchangedscreens/bounds; then qualifiedactualtraining/longerfreshdata learning. User prioritizes fasterthan~3s/step and meaningful longerhorizon; no newoptimizer/update/cadencegrid.

> **Trial207 completed; faster training now prioritized — 2026-10-21:** [Report](../optimizer_replay_results/soap_reference_sparse207_20261021/REPORT.md), independent completion audit PASS. QR50-only1000-update control finalCE4.077399 vs completed206warm4.079469/QR104.064316. Warm improves some intermediate panels but has no established late benefit over hold at the same QR repair cadence. Short1000-step/3.5%-horizon evidence is not a convergence verdict; user explicitly questions pilot length and supports longer representative training.
>
> QR50 optimizer79.432ms/backward3248.453ms/whole3327.912ms; across-job backward variation prevents attributing complete-step differences to refresh.64.24min/10.496GiB,1000updates/16032microbackwards/7validation/2state-preservation/8finite-Gram observations(8208factors)/registeredrelease/25graphs/0newclean-rowgraphs PASS.324inputs/318sources/84pins/556workingPython/553priorPython/22720priorfiles/oldmasterbytes/dirtyHEAD-status/data preserved. Trainer1679328dead,GPU7free,no weights/checkpoints/tensor snapshots. Older ACTIVE207 labels below historical.
>
> User authorizes proceeding with a preference for training faster than~3seconds/step. Next bounded208profiles/qualifies compiled transformer/loss execution, larger physicalbatch with global128fixed, and reduced activation recomputation within resource limits. Keep model/optimizer/data/tokenbatch/update rules fixed; verify numerical behavior and count compile/setup/memory/complete-step cost. Then choose execution and freeze a longer fresh-data quality comparison (5000–10000 discussed, not yet launched). OrdinarytrainingFB dominates current cost; no more optimizer-only microbenchmarks as the main speed answer. Existing206/207evidence retained; no reruns to recover context, default promotion, automatic precision/cadence grid or update redesign. LLM focus;Muonsubsequent/PINNsdeferred.

> **Sparse QR50 contextual control207 active — 2026-10-21:** [Protocol](../optimizer_replay_results/soap_reference_sparse207_20261021/PREPARATION.md). User requested206review then next launch. [Completed206](../optimizer_replay_results/soap_reference_learning206_20261021/REPORT.md) missed late+.01CE engineering screens (+.052193/+.019596/+.015153) while optimizer-only whole-step saving was3.732%. No longer warm extension or default promotion. One fresh QR50-only control is the next diagnostic; completed206QR10/warm arms are reused, not rerun. This is the active task; older ACTIVE labels below are historical.
>
> Single claimedGPU7/private `manifold_reference_sparse207`,trainer1679328,1000updates/131.072Mtargets. Same185.68M/512x32/blocked512 normalized model/optimizer math,source28332-update/177-warmup schedule,global128 via16x8/seq1024/FlashAttention/BF16/recompute,seed203 initial parameters as206. First50EIGH unchanged; after that QR at100/150/.../1000 and basis held between, while covariance/momentum/variance/normalization/updates continue each call. Comparing against warm10/QR50 isolates intermediate warm tracking at the same repair cadence; this contextual control intentionally differs from the primary matched-NQR10 control. Additive update body AST-identical except refresh dispatch; interval10 CPU qualification reproduces qualifiedQR10fullstate through111updates.
>
> Exact206data plan/window IDs/1000batch hashes/2048validationwindows reused; no new download/copy/tokenization. Evaluate0/50/177/350/500/750/1000; preserve full training-state hashes around50/1000, mode/RNG/counters/gradientidentity on every evaluation. Sparse1026-factor finite/Gram<=.05 checks at1/50/60/100/101/110/500/1000;101 prospectively excludes first-postQRordinary column-layout specialization from clean timing. Cadence-weighted cost.98ordinary/.02QR. Report full paired CE curves and actual times; sparse crossing brackets do not establish precise equal-time benefit. If sparse matches/beats warm at same/lower cost, warm tracking has not earned its work here; if warm improves on sparse, report incremental value without erasing its gap versusQR10. No new threshold, automatic extension, cadence grid or update redesign.
>
> Startup PASS: full initial parameters and exact repeated real-data gradient hashes match206; initial validationCE10.8249054 and first actualEIGH all-state/1026factorGram.006684<.05 pass. Later cadence/quality/full-state/guard/release checks remain pending. Six new working checks/24frozen policy+model+optimizer+data checks PASS;324inputs/318sources/84referencepins/553priorPython/22720prior evidence files/dirtyHEAD-status/originalmasterbytes preserved,556currentPython. Expected60–70min,90/91-minuteinternal/externalwatchdog,24GiBGPU/12GiBRSS/1GiBcompilecache. RAM-only states/transient hashes, no resetbanks,weights/checkpoints/tensorsnapshots. No retry/successorqueue;monitor EXISTING207,never duplicate/restart. Otherprojects untouched;LLM focus/PINNsdeferred.

> **Trial206 completed and audited — 2026-10-21:** [Report](../optimizer_replay_results/soap_reference_learning206_20261021/REPORT.md). Real FineWeb, same qualified185.68M/512x32/blocked512 normalized SOAP port: QR10 versus warm10/QR50,1000 actual updates and131.072M targets per arm. Final validation CE4.064316 versus4.079469 (warm+ .015153,1.527%higher perplexity). Gaps at500/750/1000 are+.052193/+.019596/+.015153: all miss the prospective+.01 engineering screen, despite narrowing. This threshold is not a universal acceptable loss or statistical equivalence margin. No automatic longer warm extension or default promotion.
>
> Optimizer209.354vs82.017ms (60.824%lower), complete iteration3411.632vs3277.965ms (3.918%observed lower). Optimizer difference alone gives3.732%of QR whole-step cost with backward fixed;6.328ms backward variation and fixed arm order prevent attributing all observed saving to refresh. QR/warm per-arm wall65.58/62.84min includes asymmetric preflight/compile reuse; total128.48min,10.496GiB peak. Sparse validation gives overlapping sampled time-to-loss brackets, not a demonstrated equal-time advantage. At50 both use EIGH and already differ by-.000284CE from numerical trajectory variation; no full-trajectory bitwise repeatability claim.
>
> Completion audit PASS:2000updates/32032microbackwards,14validation observations/4full-state evaluation checks,14finite/geometry checks covering14364factors (maxGram.007539<.05),2registered releases,31graphs/0newclean-row graphs,321inputs/315sources/84reference pins/553workingPython/20123prior files/dirtyHEAD-status/oldmasterbytes/data hashes preserved. Trainer1656607dead;GPU7free. All179private programs terminal;private manager retained idle for the user-authorized successor. No weights/checkpoints/reset banks/tensor snapshots. All ACTIVE206 statements below are historical.
>
> User requests review then launch next. Next is one bounded QR50-only contextual control using the identical1000batches,initial weights,fullschedule and validation windows. First50EIGH unchanged; then QR at100/150/.../1000 and basis held between, while moments/covariance/normalization/updates continue every call. This isolates intermediate warm tracking against the same QR repair cadence. Reuse completed206 QR10/warm arms; no rerun, cadence grid, update redesign or long warm extension. Warm beating sparseQR50 would support incremental tracking but would not erase its gap against QR10. LLM focus;PINNs deferred.

> **Real-data quality screen206 active — 2026-10-21:** The user authorized proceeding after the completed cost205 review. [Protocol](../optimizer_replay_results/soap_reference_learning206_20261021/PREPARATION.md). One claimed GPU7/private `manifold_reference_learning206` attempt, trainer1656607; no automatic restart or queued successor. This is the next active task; older ACTIVE labels below are historical.
>
> Same qualified 185.68M-parameter, width512/depth32/blocked512 normalized PyTorch SOAP port as205. QR10 versus warm10 with QR repairs every50 updates,1000 actual updates/131.072M targets per arm, global128 via16x8/seq1024/forced Flash Attention. Model/update/moment/normalization math unchanged; seed203 initial parameter hash matches205. The source28332-update schedule/177-update warmup stays fixed: this is an early prefix, not a full convergence or paper reproduction run.
>
> Three SHA-pinned GPT-2 FineWeb shards at revision `50d1422b27e1a928440c26a8829f3f827f44ac56`,600003072bytes, saved to a new shared cache; existing OWT untouched. Two training shards provide128000 unique disjoint1025-token windows, same order and batch hashes across arms. Separate validation shard provides2048 fixed windows/two1024-sequence panels, evaluated at0/50/177/350/500/750/1000. Every window yields1024 valid next-token targets; token0 is not padding. Correct token-mean CE/window geometry is a documented adaptation from the upstream padded-row/reduction code. No paper-trajectory or semantic-deduplication claim.
>
> Prospectively, considering a longer study requires warm-minus-QR validation CE<=+.01 at500/750/1000 and optimizer-only clean cost reduction>=2% of QR whole-step time with backward held fixed. These are engineering screens, not established tolerances, statistical equivalence or time-to-loss proof. Report all panels/per-sequence deltas, actual elapsed/training times and sampled crossings of CE8/7/6/5/4 with brackets; no precise interpolation from sparse panels. Fixed arm order/compile reuse and varying backward times remain limitations. Cost205 showed61.06%lower optimizer time but only~3.74%optimizer-attributable complete-step headroom; this new run tests quality on the exact port, which older full-factor learning results did not establish.
>
> Eight new working CPU checks and18frozen data/model/optimizer checks PASS.321frozen inputs/315sources/84reference pins verified;550existing Python files/20123prior evidence files/dirty HEAD-status/original master bytes preserved. Startup PASS: exact repeated initial real-data gradient hash, expected initial validation CE10.8249054, initial parameters matching205 and first actual EIGH finite/all1026factor Gram check (.006684<.05). Remaining first50, later validation/full-state hashes, warm/repair/late quality/release checks remain pending. All1026-factor Gram<=.05 guard at sparse calls1/50/60/100/110/500/1000. Full training-state hashes around validation50/1000; every evaluation restores modes/RNG/counters/gradient identity.165/166-minute internal/external watchdog,24GiB GPU/12GiB RSS/1GiB compiler cache. RAM-only states and transient hashes; no reset banks, tensor snapshots, weights or checkpoints. Shared pinned data retained for review/further scoped comparisons. Monitor EXISTING206; never duplicate or restart. LLM focus;PINNs deferred.

> **Completed reference-style cost205 — 2026-10-20:** [Report](../optimizer_replay_results/soap_reference_cost205_20261020/REPORT.md) and independent completion audit PASS. Actual251calls per arm/502updates/8032microbackwards, including50EIGH calls per arm;39.61minutes combined,10.496GiB peak. QR10 versus warm10/QR50 clean optimizer208.283vs81.097ms (61.06%lower),whole step3402.358vs3227.815ms (5.13%observed lower). Optimizer reduction is127.187ms,or3.738%of the QR whole step;the later warm arm's47.357ms shorter backward region contributes another1.392percentage points. Fixed arm order and backward variability prevent attributing the full observed5.13%to refresh. Optimizer-only/common-backward arithmetic gives modest~3.74%headroom,not an independently measured causal/production gain. QR refresh-update optimizer1672.149ms versus warm80.553ms (~20.8x);candidate QR repairs are included in cadence means. Large refresh-event speed and modest whole-step benefit are consistent.
>
> PyTorch base-width185.68M/512x32/blocked512/normalized reference adaptation,forced Flash Attention,BF16forward/recompute,synthetic global128via16x8. No dataset read or quality/convergence/paper-reproduction/distributed/width/fidelity result. Clean counts135ordinary+14QR control,135ordinary+11warm+3QR candidate. All12finite/geometry checks (12312factor observations),2registered releases,initial hashes,316inputs/312sources/550workingPython/17542priorfiles/reference pins/dirtyHEAD-status/old doc suffixes PASS.31graphs total,0new clean-row compiles. Worst Gram.007349<.05. Identical initial weights/preflight repeated gradient do not imply bitwise whole-trajectory repeatability: first50scalar differences are small but nonzero,retained in report. Both actual first50EIGH phases and all startup/diagnostic/I/O/release work charged to total. No weights/checkpoints/reset snapshots/tensor files. Trainer1565547dead,GPU7free,private manageroff after178terminal programs. No successorqueued;older ACTIVE205 labels below historical.
>
> Next bounded preparation is pinned real-data/validation and prospective matched loss-versus-update/time criteria on this exact port,if the few-percent cost opportunity warrants a quality screen. Earlier full-factor learning is not quality evidence for this implementation. Full3.7B-token paired reproduction requires a separate longer budget;no automatic learning launch,cadence grid,update redesign or default promotion. LLM focus;PINNs deferred.

> **Actual reference-style complete-step cost205 active — 2026-10-20:** Corrected [preflight204](../optimizer_replay_results/soap_reference_preflight204_20261020/REPORT.md) and independent audit PASS:10working+10frozenCPU oracle/model/logging checks, exact initial Flash backward hash, full-shape EIGH/ordinary/QR/warm and4x1026factor guards/releases;70.54s,7.107GiB. Spent203 logging failure remains frozen; no optimizer/model math changed. Its active labels below are historical, as are ACTIVE204 labels.
>
> [Actual cost205 protocol](../optimizer_replay_results/soap_reference_cost205_20261020/PREPARATION.md) launched ONCE on claimedGPU7/private `manifold_reference_cost205`,trainer1565547. Base512x32/FF2048/185680896parameters/226matrices,forced PyTorch Flash Attention,BF16forward/activation recomputation,synthetic global128x1024 via16x8. Full-state CPU port oracle covers111updates; port preserves raw momentum,tiedbeta2,current-gradient order,one-sided groups,normalization/scaling and actual first50EIGH. QR10 then warm10/QR50 each run251actual calls from matching full initial parameter hash.502updates/8032microbackwards,32,899,072targets per arm (65,798,144total; corrected frozen-protocol count label documented before launch). Clean means call>100 excluding diagnostics110/251;candidate repairs150/200/250 provide3cleanQRsamples. Diagnostic110 may compile the first post-repair warm column-layout specialization;all clean timing rows require0newgraphs. All switches,copies,normalization and parameter updates charged to complete optimizer cost;first50EIGH/startup/checks/logs/releases charged to all-in.
>
> Startup205 PASS: control full initial hash matches204,first EIGH all-state/1026factor Gram guard passes,first repeated actual EIGH updates and finite losses/norms progress. Remaining EIGH/QR/warm/repair/geometry/release/final timing checks pending;no measured saving or quality result yet.316inputs/312sources,550currentPython/17542priorfiles/dirtyHEAD-status/original docs preserved.45/46minute watchdog,24GiBGPU/12GiBRSS/1GiBcache, RAM-only model/optimizer/gradients;no reset snapshots,weights/checkpoints,retry or successorqueue. Expected30–40minutes,not a promised speedup. Monitor EXISTING205,never duplicate/restart. PyTorch base-width/synthetic cost adaptation,not JAX/FineWeb reproduction,convergence,distributed,width or fidelity claim. LLM focus;PINNs deferred.

> **Corrected reference GPU preflight204 active — 2026-10-20:** Spent203 remains frozen after a logging payload collision following its successful initial Flash repeat and first EIGH geometry check. No optimizer/model math changed. Fresh [204 protocol](../optimizer_replay_results/soap_reference_preflight204_20261020/PREPARATION.md) corrects two logging payload keys and adds a CPU logging regression;10working+10frozen tests pass. Same seed203,512x32 model,physical8x16global128,forced Flash Attention,highest ordinary/QR and high warm,thresholds and900/960s resource limits.316inputs/312sources,550currentPython/14989priorfiles pinned. Single GPU7/private attempt, no snapshots/weights/checkpoints or automatic retry. Monitor existing204;never relaunch203/204. Complete250+call two-arm cost job remains pending successful preflight review; no quality result. Earlier ACTIVE203 statements below are historical.

> **Reference-style port GPU preflight203 active — 2026-10-20:** New additive `reference_llama.py`, `reference_blocked_soap.py`, full-state CPU oracle tests and managed cost harness implement the pinned base-width adaptation. Existing optimizers/evidence remain unchanged. Nine working and nine frozen CPU checks pass:111-update histories across EIGH50/QR/warm/repair, padded/one-sided groups, raw momentum/covariance/variance/parameter updates, recomputation, zero-output initialization and185680896parameters/226matrices. This is a PyTorch base-width port, not general width scaling or a JAX/paper trajectory reproduction.
>
> [Preflight203](../optimizer_replay_results/soap_reference_preflight203_20261020/PREPARATION.md) launched ONCE on claimedGPU7/private supervisor, trainer1562939. Forced PyTorch Flash Attention, synthetic global128 via16x8, BF16forward/recomputation; exact repeated initial backward PASS (loss10.8249054,norm.06249973,matching full gradient hash). Compiled EIGH/ordinary/QR/warm shape/memory/geometry/release checks remain pending. Artificial counters after first EIGH are only execution/compilation checks; actual first50EIGH training calls are reserved for a separately frozen complete-step cost comparison after preflight review. No quality result, default promotion or learning job. 900/960s watchdog,24GiBGPU/12GiBRSS/1GiBcache, RAM-only/no model snapshots/checkpoints/weights/retry/queue.316inputs/312sources,546oldPython/14125priorfiles preserved. Monitor existing203;never duplicate/restart. LLM focus;PINNs deferred.

> **Batched512 refresh qualification202 PASS — 2026-10-20:** [Completed report](../optimizer_replay_results/soap_block512_trial202_20261020/REPORT.md), independent completion and generated-code audits pass. New additive batched primitive reproduces the existing scalar warm policy per factor and the pinned reference's regular QR/variance permutation; existing optimizer sources unchanged. Compiled PyTorch batched QR versus warm refresh-only host ratios are3.39–3.56x for1factor,27.55–27.63x for8dense factors,and48.50–49.49x for64dense factors; padded8case27.73–28.13x. Factors remain512x512, so this is a batching/backend result, not wider-matrix fidelity or model-width scaling. No complete-step, learning, native-JAX reproduction or optimal-QR claim. QR highest / warm high, precision switches outside timed regions.
>
> All96geometry observations/1944factor checks,source/RNG/warm-variance identity/registered releases PASS; worst absoluteGram.006884 versus unchanged.05 bound,worst scalar-policy relative basis discrepancy2.14e-5 versus prospective1e-4 bound.5working+5frozenCPU checks pass;12startupgraphs/0timed newgraphs;6generated warm modules each have7products/3BF16-input directFP32-output products.53.79seconds all-in,.857GiB peak.318inputs(308sources+6incidental compactbytecode+4otherinputs),543oldPython/12279priorfiles/referencepins/dirtyHEAD-status/oldmasterbytes preserved. No weights/checkpoints/tensor files or retry. Trainer1482622dead,GPU7claimfree,private manageroff;no successorqueued.
>
> [Prepared complete-step gate](../optimizer_replay_results/soap_block512_trial202_20261020/NEXT_COST_PROTOCOL.md): next implement/independently qualify the source-faithful blocked512 baseline and512x32 architecture; preserve raw-coordinate momentum,tiedbeta2,current-gradient ordering,first50EIGH,one-sided groups,normalization and scaling. Declare efficient attention/physical batch/accumulation and charge all ordinary work,normalization,refresh/repair/checks/startup to whole-step cost. Full model/optimizer port is not yet implemented or qualified. Only after measured complete-step benefit should a separate learning budget be considered. Existing201small overall gain and202large refresh-only ratios measure different workloads and portions of training. LLM focus;PINNs deferred;no update redesign/default promotion or cadence grid.

> **Trial201 completed; independent reference implementation pinned — 2026-10-20:** [Batch-128 report](../optimizer_replay_results/soap_batch_cost_trial201_20261020/REPORT.md) and independent completion audit PASS. QR10 versus warm10/QR50: clean cadence-weighted optimizer 77.667 versus 36.454 ms (53.06% lower), complete iteration 2,314.779 versus 2,277.467 ms (1.61% lower). This single-GPU, 16-microbatch serial-accumulation probe completed all 552 calls / 8,864 backwards in 22.30 minutes, peak 16.90 GiB. All numerical, both exact 75-update prefixes, registered release and source/evidence checks pass; 8 disposable / 0 measured new compile graphs. CPU reset snapshot and all registered state released; trainer dead, GPU7 claim free, private research manager off. No weights/checkpoints or successor job. All earlier ACTIVE201 passages below are historical.
>
> The QR refresh surcharge is only 2.563% of this measured complete step, giving a configuration-specific ceiling if that surcharge were free and ordinary work stayed fixed. Global batch alone does not determine per-device amortization: distributed execution, larger physical microbatches and efficient attention can change the balance. This cost pilot gives no convergence, full-reference, width-fidelity or universal production conclusion. A long run of the unchanged trial201 configuration is not justified yet.
>
> [Read-only independent reference audit](../manifold_opt/soap_reference_implementation_20261020_evidence/REPORT.md) pins the official `fineweb` branch at `a5be0b1eb4d4c6ee885bddaf02beac0198d164a1` (main also pinned), 65 source/config/dependency files. The actual target is 512-by-32 Llama, batch128/seq1024, FineWeb, blocked512 and normalized SOAP: raw-coordinate first momentum without basis transport, covariance beta tied to .999, current-gradient covariance/refresh before the update, first50 EIGH then QR10, one-sided embedding/readout and spectral/RMS normalization. These are baseline rules to preserve, not an update redesign. All active factor dimensions remain512 across the published widths, so model width increases block count rather than individual basis size. Warm512 qualification and complete-step measurement with efficient batched QR, a declared attention backend and feasible physical batch are the next engineering gate before another learning budget. Resolve the missing unconditional attention-module import and pin data revision/split before a reproduction. No upstream execution/install/download/optimizer change, new GPU job or automatic promotion. LLM focus; PINNs deferred.

> **Active LLM batch-cost feasibility — 2026-10-20:** [Trial201](../optimizer_replay_results/soap_batch_cost_trial201_20261020/ACTIVE_STATUS.md) is running on claimed GPU7 under private supervisor, trainer PID 1476810. PINNs are deferred. This short cost probe adopts batch 128 / sequence 1024 from the [independent scaling study](https://arxiv.org/html/2512.05620), keeping our existing 109M-parameter, width-768, depth-12 OpenWebText model and unchanged trial199/200 full-factor SOAP math. It is a partial-recipe feasibility check; a full reference model/blocked-normalized optimizer port and learning comparison remain pending.
>
> QR control startup **PASS**: two identical accumulated backward observations; 75 disposable updates with 48 physical-momentum checks; exact 75-update reset prefix across model, optimizer, gradient, RNG and scalar state. All observed numerical, source and release checks pass. The warm arm runs later in this same job; its qualification remains pending. Each global update averages 16 microbatches of 8, clips once, and updates optimizer/covariance once. QR10 and warm10/QR50 each receive 75 disposable and 201 measured updates. The bounded probe totals 552 optimizer calls and 8,864 microbatch backwards, including preflight; it measures complete iteration cost and reports sample counts and startup overhead.
>
> Six working and six frozen CPU tests pass; 305 sources / 315 frozen inputs, 540 pre-existing Python files, 10,577 prior evidence files, dirty HEADs and original document bytes are preserved. Limits: 30-minute internal / 31-minute external watchdog, 24 GiB GPU, 12 GiB RSS, 100 MiB compact evidence, 512 MiB compile cache, minimum 5 GiB shared disk free. The 595,123,200-byte CPU reset snapshot is deliberately retained until the final arm's measured update 75, then released. No disk model snapshots, weights, checkpoints, retry or queued successor. Monitor the existing job; do not duplicate or restart it. The completed trial200 review remains linked below.

> **Completed matched-N10 result and LLM-only next scope — 2026-10-20:** [Trial200 review](../optimizer_replay_results/soap_deep_matched_warm_trial200_20261020/REPORT.md) and independent audit PASS:35000primary/35075total,109.524min,18.279GiB. FinalCE3.443959 vs QR10/1993.435068 (+.008890,~.893%higherperplexity);cleanoptimizer36.206vs77.379ms (53.21%lower),iteration176.593vs217.652ms (18.86%lower). Every matched-update panel is behind QR10; report-only+.01 screen misses first5,passesendpoint. Warm10/QR50 endpoint/cost is worse than completedwarm20/QR200 andholdQR200 in thisseed;more frequentrepair has not established practical improvement. All1760guards/480M/192reports/exact75prefix/7eval+observer/14banks/releases PASS;8disposable/0primarygraphs.311inputs/302sources/537oldPython/8956priorfiles/pins/80stats/dirtyHEAD-status/olddocs preserved. Actual elapsed panels retained;no inferred precise equal-time loss from sparse evaluations. Trainer1459996dead/GPU7free/private manager off. User directs LLM focus;PINNs deferred as too esoteric. Next bounded work is independent-LLM recipe selection and measured resource/cost feasibility before a long learning run;keep published update math fixed,defer moment-update redesign,and choose complete time-to-quality success criteria prospectively. No new learning job or default promotion. Earlier ACTIVE200 labels below historical.

> **Independent SOAP use cases — 2026-10-20:** [Primary-source shortlist](../manifold_opt/soap_independent_use_cases_20261020_evidence/shortlist.json) expands future benchmark selection beyond the canonical paper: [Wen et al.](https://arxiv.org/abs/2509.02046) independent LLM benchmark/tuned recipes; [Qiu et al.](https://arxiv.org/html/2512.05620) scaling with512blocked,spectrally normalized SOAP (AppendixH uses betas(.9,.999),so our beta pair alone is not evidence of an unrepresentative recipe); [Wang et al.](https://arxiv.org/html/2502.00604v2) PINN/PDE applications with frequency ablation choosing interval2; [Khona et al.](https://arxiv.org/html/2607.20548) large-scale SOAP variant requiring current-gradient inclusion plus per-stepQR,with KL covariance changes; [Bolatov et al.](https://arxiv.org/html/2609.23055) smaller diffusion tasks with task-dependent quality and measured runtime overhead. Distinguish ordinary/full-factor SOAP,blocked/normalized SOAP and KL-SOAP before selecting a comparator; preserve the selected baseline's update rules and only replace basis refresh. Data/FLOP-equivalent gains in the LLM scaling papers are not measured wall-time gains and may exclude optimizer cost. Feasibility/exact smaller-model recipe selection pending; no new job,default,update rule or running200 protocol changed. User's scope remains representative complete time-to-quality benefit,not update-mechanism redesign.

> **User-directed scope and representative benchmark decision — 2026-10-20:** Keep SOAP's parameter-update/second-moment rules fixed; defer the proposed update-mechanism audit/redesign. Practical objective: useful complete wall-time-to-loss improvement on a recognizable, source-backed SOAP training recipe. Current200 continues unchanged; review its completed matched-N10 result first. Next bounded preparation should select one public reference recipe and estimate model/data/token-batch/training-horizon/QR-cost and hardware feasibility, then compare native QR and warm at the same N with identical remaining math; occasional QR remains optional. Include a cheap sparse-QR contextual control and only the baseline calibration needed for a credible reference. Do not launch another cadence/precision grid or extrapolate width benefits without evidence. Official [SOAP example](https://github.com/nikhilvyas/SOAP) uses betas(.95,.95), covariance tied to beta2 by shampoo_beta=-1, and N10. [Original paper](https://arxiv.org/html/2409.11321v2) uses C4,seq1024,widths1024/1408 for main360m/660m non-embedding models,token batches~2m(N10) or256k(N80); its smaller-batch SOAP beta2=.99 result does not imply one universal beta. [Author's124M example](https://github.com/nikhilvyas/modded-nanogpt-SOAP) supplies a smaller public recipe on FineWeb. Current768x12/109M model is a useful engineering case, but betas(.9,.999),covariance beta.99,~8k-token batch and286M-token horizon differ materially. Step interval is not portable across batch sizes: ourN200 is~1.64M input tokens between refreshes, versus~20M in the original2m/N10 setting; these are not equivalent optimization dynamics. Larger token batches can also amortize QR and shrink warm's end-to-end saving, so do not assume the reference setup will favor warm. Judge complete time-to-quality,not refresh-only speed; choose practical success criteria before any future learning launch. If representative bounded cases show no useful tradeoff, narrow the claim to demonstrated basis-tracking/kernel results and stop expanding SOAP-specific experiments. Source/feasibility selection pending; no new recipe,hyperparameter,update rule,default,job or running protocol changed here.

> **Active matched-N warm comparison — 2026-10-20:** [Trial200](../optimizer_replay_results/soap_deep_matched_warm_trial200_20261020/ACTIVE_STATUS.md) launched ONCE on lifetime-claimed GPU7/private `manifold_deep_matched_warm_trial200`,trainerPID1459996. Same seed193,width768/depth12/109.397M model,full-OWT batches/panels,35000 primary updates/286.44M targets as completed/audited QR10 control199 (not rerun). Warm every10 ages; QR replaces every fifth refresh (repair50);699QR/2800warm per matrix. K5 is an initial optional repair policy, not a necessity/optimum; fixed/adaptive/no repair remain open. FastSOAP warm math/source unchanged from197/198,FP32 stored state,caller high,BF16 operands/directFP32 outputs,two corrections; native control retains highest momentum transport, so compare full implementations without isolated basis attribution. 302sources(299 unchanged+3 additive),311inputs,27working+27frozenCPU/fullmodel initial-reset/pins PASS;537oldPython/8956priorfiles preserved. GPU startup PASS: exact QR10 initial backward/panels,75disposable calls/96actual-M/192own reports/8compilegraphs,exact75-call full-state/scalar/RNG prefix,CPU snapshot released,and first260 read-only observation/restoration/workspaces/both banks/correct N10 metadata pass. Prospective full loss curve/endpoint/perplexity and complete cost/time; historical+.01 panel screen report-only, no automatic promotion. 75disposable+35000primary calls,480actual-M/192own factor reports/1760guards/7eval+observer/14banks planned. 140/141min,24GiB GPU/12GiB RSS,RAM-only reset/diagnostics,no weights/checkpoints/retry/queue. Monitor EXISTING200,never duplicate/restart. Earlier active199/next200 statements below historical.

> **Completed QR10 reference — 2026-10-20:** [Trial199 review](../optimizer_replay_results/soap_deep_qr10_trial199_20261019/REPORT.md) and independent audit PASS. 35000 primary/35075 total calls,133.003min,18.277GiB peak. Final CE3.435068 vs QR20/1933.437160 (−.002092 CE,~.209% lower perplexity); small one-seed difference, QR10 slightly worse at10k. Clean optimizer77.379 vs47.821ms (+61.81%),iteration217.652 vs188.184ms (+15.66%). All1758guards/432actual-M/exact75prefix/7eval+observer/14banks/releases pass;308inputs/299sources/534existingPython/8613priorfiles/11refs/80stats/dirtyHEAD-status/originaldocs preserved. Trainer1444357 dead/GPU7 free/private manager off. QR10 is now the control for matched-N warm10; historical+.01 QR20 screen report-only, not a universal quality limit. Prepare a separate warm10 candidate, initially repair50 (optional starting choice; fixed/adaptive/no repair remain open), then compare full loss curves/endpoints and complete cost/time. No warm10 result or automatic promotion. Earlier active199 statements below are historical.

> **Active QR10 reference — 2026-10-19:** [Trial199](../optimizer_replay_results/soap_deep_qr10_trial199_20261019/ACTIVE_STATUS.md) launched ONCE on claimed GPU7/private `manifold_deep_qr10_trial199`, trainer PID1444357. New native QR10 control for the primary matched-N comparison; completed QR20/193 is contextual and is not rerun. Same seed193, width768/depth12/109.397M parameters, 35000 primary calls/286.44M targets and fixed full-OWT batches/panels. 75 disposable +35000 primary calls, original numerical/source/restoration/release guards. 299 sources (296 unchanged+3 additive),308 frozen inputs,13 working+13 frozen CPU tests and full-model initial/reset PASS;534 old Python files/8613 protected files preserved. GPU startup PASS: exact native initial backward/panels, exact75-call full-state/scalar/RNG prefix, CPU snapshot released, first260 read-only observation/restoration/workspaces/both bank releases and QR10 age metadata pass. Monitor EXISTING199; never duplicate/restart/queue. 170/171min limits,24GiB allocated, RAM-only reset/diagnostics; no saved weights/checkpoints. User's [N/K/J design](../TurboSOAP/SOAP_REFRESH_POLICY.md) is a rough sketch: repairs are optional, fixed/error-triggered/none remain possibilities. Next warm10 may start with QR replacing every fifth refresh (repair50), separately frozen after review. Historical +.01 every-panel QR20 screen is report-only for this new control; compare whole loss curves/endpoints and complete cost/time. Older ACTIVE/NEXT labels below are historical where inconsistent.

> **User-directed primary comparison — 2026-10-19:** [Matched refresh/repair/check policy](SOAP_REFRESH_POLICY.md) supersedes the emphasis on beating QR200 hold. Hold remains a secondary sparse-refresh ablation. Primary: same basis-refresh interval N for QR and warm; warm may replace every Kth refresh with QR, and optionally check a declared metric every J refreshes. Current198 already compares N20/K10 warm against QR at N20. Proposed next pair: N10 for both, warm K5 (QR every50steps), then separately qualify monitored/adaptive repair. K counts refresh opportunities; QR replaces the scheduled warm computation. Declare old-basis versus candidate checks, repair granularity/counter semantics, error metric and J; charge checks/synchronization/discarded candidates/repairs to complete cost. Orthogonality and tracking metrics are distinct, and thresholds need training relevance. Compare loss versus updates and complete elapsed time. No new job, controller, default, threshold change or old evidence mutation in this update.

> **Completed and audited — 2026-10-19:** [FastSOAP deeper-model trial198](../optimizer_replay_results/soap_deep_fast_trial198_20261019/REPORT.md) completed 35,000 primary/35,245 total calls in 103.11 min,18.276 GiB peak. Final validation CE 3.442869 versus native QR20 3.437160: +.005709 CE (~.5725% higher perplexity). Own-cadence clean optimizer24.674 versus47.821 ms (48.40% lower), iteration165.062 versus188.184 ms (12.29% lower). Original +.01 absolute-CE screen misses at1k(+.022130)/3.5k(+.024306)/10k(+.016597), passes20k/30k/35k. This is a prospective screening convention, not numerical failure or a universal unacceptable-loss threshold. All1765guards/480actual-M/192own reports/exact245prefix/7eval+observer/14banks/releases pass;8disposable/0primary compile graphs. 305inputs/296sources/6978priorfiles/native+hold+qualification refs/80cache stats/dirtyHEAD+status/originaldocs preserved. Fast versus hold195: endpoint+.001031CE and2.13%more iteration cost; warm motion has not demonstrated a later quality/cost advantage over hold in this seed. J1 is the existing warm candidate; FastSOAP optimizes that J1-derived path, not an unrelated baseline. No old-J1 full-depth learning control, so precision/refactor effects are not isolated. Trainer1425140dead/GPU7claimfree/private manageroff; no next job launched. Retain measured speed/loss tradeoff and original screen label separately. Earlier ACTIVE/PENDING198 passages below historical.

> **Running, startup verified — 2026-10-19:** [FastSOAP deeper-model trial198](../optimizer_replay_results/soap_deep_fast_trial198_20261019/ACTIVE_STATUS.md),GPU7/trainerPID1425140,matchedmodel/data seed193. Initial full-model backward and all initial evaluation panels exactly match nativeQR20 control193. Disposable245calls/96actual-M/192own-policy factor reports/8compilegraphs pass; primary245-call full-state/scalar/RNG reset prefix is exact, including QR200 and postrepair warm220/240. CPUreset snapshot released. First260 observation passes24factors/24gradientrows/12read-only adaptive previews, source/restoration/workspace and bothhistory-bank releases; warm/QR age metadata correct. 305frozen inputs/296working sources/6978priorfiles/dirtytrees/originaldocs preserved. Continue EXISTING35000-update warm20/QR200 candidate; fixed learning/cost screen remains pending. 140/141min,24GiB/RAM-only/no weights/checkpoints/retry/queue; neverduplicate/restart. Older ACTIVE/PENDING/NEXT passages below historical where inconsistent.

> **Active — 2026-10-19:** [FastSOAP deeper-model trial198](../optimizer_replay_results/soap_deep_fast_trial198_20261019/ACTIVE_STATUS.md) launched ONCE on lifetime-claimedGPU7, trainerPID1425140, private`manifold_deep_fast_trial198`. Model/data seed193 matches completednative193/hold195; same768x12/35000primary/286.44Mtargets. Fastwarm20/QR200, FP32state/globalhigh/BF16operands-directFP32output/two corrections; ordinary projections/covariance native. 296sources(293parent unchanged+3additive)/305inputs/34working+34frozenCPU/full-model initial reference and actual environment PASS. Exact initial native backward passes;245prefix/firstwarm+QR+postrepair warm/192own reports/480actual-M/1765guards/7eval+observer/14banks pending. Practical everypostwarmup+.01CE and lowercleanoptimizer+iteration;140/141min,24GiB/RAM-only/no savedweights/checkpoints/retry/queue. Monitor EXISTING198,neverduplicate.

> **Completed and audited — 2026-10-19:** [FastSOAP qualification197](../optimizer_replay_results/soap_fast_qualification197_20261019/REPORT.md) passes80factor/2016blockgeometry/84physical-M/16registered-release checks,5100synthetic block calls/113.90s/.421GiBpeak. Complete warm optimizer steps~8.22vsJ1~11.28ms (~27%faster); cadence-weighted block2.1245vsJ12.2658ms (6.23%lower),QR204.0387/hold2001.8238ms. Factor savings18.6-28.8%; no monotonic width/learning/deployed-speed claim. EightgeneratedCUDAmodules confirm7products/3BF16-input-directFP32-output products each. WorstM.08253%<1%,FastfactorGram.015783<.05; numerical/source/RNG/release pass. 297inputs/293sources/old195+196evidence preserved. Failed196 remains spent;197only fixed probegradmode, candidate/math/limits/cachelimit unchanged. User-authorized matchedloss198 now follows; prior ACTIVE197/196 passages historical.

> **Active corrected qualification — 2026-10-19:** [Fresh FastSOAP197](../optimizer_replay_results/soap_fast_qualification197_20261019/ACTIVE_STATUS.md), ONE claimedGPU7/private`manifold_fast_soap_qualification197`, trainerPID1423329. 293sources/297inputs/41working+41frozenCPUchecks;292parent sources unchanged, one harness correction: all factor probes use no_grad like optimizer steps. Seed196, FastSOAP source, compiler cache limit, original geometry/M/finite/source/release and cost/resource criteria unchanged. [Failed196](../optimizer_replay_results/soap_fast_qualification196_20261019/REPORT.md) preserved after mixed grad-mode probes exhausted J1 control specializations; all80factor guards passed but complete optimizer comparison/M coverage unfinished. No relaunch of196. Monitor EXISTING197; no LM loss/production-speed claim. Prior ACTIVE196 passages historical.

> **Active — 2026-10-19:** [FastSOAP CUDA qualification196](../optimizer_replay_results/soap_fast_qualification196_20261019/ACTIVE_STATUS.md) launched ONCE on lifetime-claimedGPU7, trainerPID1422541, private `manifold_fast_soap_qualification196`. 293sources/297inputs/41working+41frozenCPUchecks and actual environment/claim verified. Synthetic factors768/1536/2304/3072 and one real-shaped four-matrix block; nativeQR20/holdQR200/J1/Fast,3rotating blocks x425calls=5100optimizer calls,84physical-M checks. Original geometry/M/finite/source/release guards; report-only eager differences. 1140/1200s,24GiBGPU/12GiBRSS/100MiBcompact/512MiBcode cache,RAM-only/no weights/checkpoints/retry/queue. CUDA qualification and timing pending; no LM loss/production-speed result. Monitor EXISTING196, never duplicate.

> **Completed and audited — 2026-10-19:** [Deeper sparse QR200 trial195](../optimizer_replay_results/soap_deep_sparse_trial195_20261019/REPORT.md) completed35000primary/35225total calls in100.62min,18.278GiBpeak. FinalCE3.441838 vs native1933.437160 (+.004678), but+.01margin fails at1000(+.048237)/3500(+.025535)/10000(+.015742). Clean own-cadence optimizer21.199 vs47.821ms (55.67%lower), iteration161.615 vs188.184ms (14.12%lower). Practical screen FAILS, numerical/restoration/release PASS. 1764guards/432actual-M/exact225prefix/7eval+observer/14banks;302inputs/289originalPython sources/11native refs/80cache stats preserved; no weights/checkpoints. Trainer1403531dead/claimreleased. Gap narrows late but late-switch schedule remains untested. Retain nativeQR20. User-directed fast warm qualification196 follows; previous ACTIVE195 passages are historical.

> **Fast candidate prepared — 2026-10-19:** At the user's request, [FastSOAP](FAST_SOAP_CANDIDATE.md) implements one compiled warm kernel, caller-owned global `high`, FP32 persistent state, relative Jacobi scaling without a trace reduction, and BF16-input/direct-FP32-output CUDA products. Two corrections, warm20/QR200 defaults, original update ordering and V policy remain. Highest M transport becomes caller-precision transport. All34 CPU checks pass, including compiled execution, independent policy arithmetic, geometry/history/replay and a fake CUDA operation trace; actual CUDA qualification, speed and learning are pending. This is an opt-in candidate; complete-step dispatch remains eager. Qualified source and trial195 frozen inputs are unchanged. Continue and review EXISTING195, then qualify/time this candidate and prospectively prepare a matched deeper-model loss comparison. This user-directed implementation work supersedes older passages that paused warm implementation work; it does not promote a default or alter past evidence.

> **Implementation audit — 2026-10-19:** [SOAP warm-refresh equations and source audit](SOAP_WARM_REFRESH_AUDIT_20261019.md) specifies the tested J1 warm20/QR200 path: exact update ordering, damped Jacobi generator, global rotation cap, mixed retraction, first/second-moment history, QR repair, precision, costs and numerical limits. It distinguishes damped/cubic3/original May variants and lists questions for joint review. Fifteen implementation files match frozen181. This is a documentation-only audit; running trial195 remains native-precision hold/QR200 and is unchanged.

> **Running, startup verified — 2026-10-19:** [Deeper sparse-refresh trial195](../optimizer_replay_results/soap_deep_sparse_trial195_20261019/ACTIVE_STATUS.md), GPU7/trainer PID1403531, matched model/data seed193. Initial full-model backward and all initial evaluation panels exactly match native QR20 control193. All48 disposable first-QR200 physical-M checks pass; the primary225-call full-state/scalar/RNG prefix is exact and the CPU reset snapshot is released. First260 observation covers blocks1/6/12 with24 factors/24 gradient rows/12 read-only adaptive previews; source/restoration/workspaces/both history-bank releases pass and QR200 refresh-age metadata is correct. 302 frozen inputs/289 working sources/11 native refs/dirty trees/original document suffixes preserved. Continue EXISTING35,000-update native-precision hold/QR200 trajectory; learning/cost screen remains pending. 140/141min,24GiBcap, RAM-only/no weights/checkpoints/retry. Never duplicate/restart; older active/pending/next passages below are historical.

> **Active — 2026-10-19:** [Deeper sparse-refresh trial195](../optimizer_replay_results/soap_deep_sparse_trial195_20261019/ACTIVE_STATUS.md) launched ONCE on lifetime-claimed GPU7, trainer PID1403531, private `manifold_deep_sparse_trial195`. Native-precision hold/QR200 only; same width768/depth12/model-data seed193/35,000 primary updates/286.44M fresh targets and fixed panels as completed QR20 control193, which is not rerun. Covariance/moments remain per-step; no warm motion or BF16 source projections. 289 frozen sources (286 parent unchanged+3 additive),302 prepared inputs,23 working+23 frozen CPU checks and fresh full-shape CPU initial reference pass; actual process/environment/claim verified. Initial native backward/evaluation and own225-call reset prefix including first QR200 transport remain pending. Original numerical/restoration/release guards,1,764 guard observations/432 actual-M checks,7 layer observations/14 RAM banks;35,225 total calls,140/141min,24GiB cap, RAM-only/no saved weights/checkpoints/retry. Every postwarmup validation panel must remain <=native+.01 and clean own-cadence optimizer/iteration costs lower. Monitor EXISTING195; never duplicate/restart. Modern Muon/H200/width directions remain follow-ups. Prior ACTIVE/NEXT passages below are historical unless consistent with this update.

# Current research direction — 2026-10-19

The user agrees that warm basis/polar reuse remains a promising research direction, with competitiveness judged by held-out progress per complete training time. Warm reuse can either define a competitive standalone optimizer or accelerate the update computation of a stronger existing optimizer. Neither general loss equivalence nor a width-dependent advantage is established. This update supplements the method principles below; the current roadmap and frozen readouts supersede their older pending experiment plans.

## Current evidence and immediate sequence

Short width-2048, depth-1 LLM comparisons found warm tracking close to QR in loss, with about 14% lower average optimizer cost. However, pure hold/QR200 achieved a better measured quality/cost tradeoff than intervening warm motion in paired seeds180/181. The deeper width-768, depth-12 native baseline193 now covers 35,000 updates and 286.44M fresh targets. BF16 projection trial194 passed numerical checks but failed its prospective quality/cost screen. Retain native QR20; next investigate sparse-refresh learning transfer on this deeper native-precision testbed, keeping precision and cadence effects separate. Current evidence does not justify restarting warm tuning or promoting a default.

## Width, matrix structure and historical speed hypothesis

The user's hypothesis is that wider matrices may devote a smaller fraction of their dimensions to important principal directions, making warm tracking cheaper or more faithful in the directions that matter. Distinguish entrywise sparsity, effective rank, dominant-subspace stability, eigenvalue separation, physical-M preservation, adaptive-update agreement and learning quality. Low-rank instantaneous gradients do not guarantee a low-rank covariance EMA. Near-degenerate weak eigendirections may move substantially even when the important subspace is stable. These are mechanisms to test, not a dimension law.

Future width attribution should keep depth, architecture proportions, data/token batch, schedule, optimizer coverage and numerical policy controlled. Record actual factor dimensions and aspect ratios, effective rank or spectral concentration, dominant-subspace drift and useful update/learning effects. MLP expansion and model width are separate axes. Dense warm operations remain cubic; increasing width does not by itself prove an increasing asymptotic speed advantage.

The user confirms the original width2048/depth24, batch32/sequence1024 run used an H200 and refresh interval10, recalling about3,900ms QR versus390ms warm on refresh steps. PyTorch/CUDA versions remain unknown. The original published report gives398ms warm, approximately1,000ms every fifth recalibration refresh, and approximately4,000ms QR. These are historical refresh-event observations, not present average optimizer means. The current width2048 implementation measured roughly314ms QR versus193ms warm per refresh, only about1.6x; averaging alone therefore does not explain the old/new gap. Preserve that unresolved question.

Separate refresh-only work, complete refresh optimizer steps, average optimizer time and complete training time. Charge cadence, periodic recalibration/QR repairs, casts, corrections, transport, state traffic and setup consistently. The original and current policies differ in direction/controller, precision, basis storage, correction work and history handling. The previously recalled historical covariance beta .999 is provisional; current .99 is confirmed. Hardware (H200 versus current RTX5090) and library versions are possible contributors, not established causes. Do not infer that current14% is the original method's ceiling or rerun completed experiments merely to reconstruct context.

## Modern Muon comparison requirements

When TransportMuon resumes after the deeper SOAP work, compare against an efficient modern fresh orthogonalization baseline, including an appropriate Polar Express or CANS implementation, rather than relying solely on the original five-step Muon kernel. Keep momentum/Nesterov conventions, logical matrix splits, auxiliary Adam, output scaling, precision and distributed execution comparable; allow a fair bounded learning-rate calibration. Pin source revisions and execution policy before launch.

- NorMuon adds neuron-wise adaptive normalization after orthogonalization. Warm polar computation could potentially accelerate that stage while retaining the normalizer, but errors entering its second-moment statistics require new qualification and learning checks.
- Aurora jointly addresses row-energy imbalance in tall matrices and orthogonality. A warm extension must track its intended balanced target; the existing warm polar step is not automatically a drop-in Aurora solver. Its reported benefits with MLP expansion motivate shape-aware comparisons, not a general width claim. Its evidence that polar precision can affect learning argues against assuming approximation error is free.
- Public ANVIL II changes the momentum/spectral update and appears in a broader speedrun recipe. Attribute optimizer effects separately from architecture, kernels, data schedules and weight averaging. ANVIL III is described by its authors as proprietary; retain its reported results as external context, not a locally reproduced or fully specified baseline. Do not infer its update rule from ANVIL II or compare fitted headline compute savings directly with our measured wall times.

Begin with one strong reproducible fresh baseline and its matched warm variant. First qualify arithmetic, geometry, target alignment, state and complete component cost; then test learning at matched tokens and comparable measured time. Include anchors, controllers, normalization, extra state and communication/overlap in the relevant cost accounting. Expand to further modern variants only when the first comparison resolves a useful question. Historical TransportMuon results do not establish competitiveness against these newer variants, and compatibility remains a hypothesis.

## Sources and scope

Sources reviewed on2026-10-19: [original TurboSOAP report](https://ethansmith2000.substack.com/p/turbosoap), [historical source audit](../optimizer_replay_results/soap_historical_audit_20261008/README.md), [paired sparse screen](../optimizer_replay_results/soap_llm_sparse_screen_paired_20261016/REPORT.md), [deeper baseline193](../optimizer_replay_results/soap_deep_learning_seed193_20261019/REPORT.md), [trial194](../optimizer_replay_results/soap_deep_projection_trial194_20261019/REPORT.md), [GaLore](https://arxiv.org/html/2403.03507v2), [Q-GaLore](https://arxiv.org/html/2407.08296v1), [NorMuon](https://github.com/zichongli5/NorMuon), [Aurora](https://arxiv.org/html/2606.27715v1), [ANVIL II](https://hyperstition.cc/training-nanogpt-in-39-9-seconds), [ANVIL III disclosure](https://hyperstition.cc/cutting-pretraining-costs-by-62-percent). External claims are source- and workload-specific; they are not project measurements.

This is a documentation update. No experiment, implementation change, new default, threshold relaxation, automatic sweep or checkpoint retention change is introduced. Frozen evidence and all earlier document bytes remain preserved below.

---

# SOAP research framework — current decisions (2026-10-15)

This is the master method document for the current SOAP work. The [roadmap](../manifold_opt/optimizer_research_roadmap.md) owns execution order, the [handoff](../manifold_opt/NEW_AGENT_HANDOFF.md) owns continuation navigation, and frozen reports own experimental observations. This section supersedes the dated execution instructions retained under **Historical method record** below. No experiment or default optimizer change is introduced by this documentation update.

## Objective and design principles

Maintain useful preconditioning coordinates and optimizer history cheaply enough to improve heldout training progress per complete wall time, with little quality penalty relative to practical QR SOAP. Closing the gap is sufficient; outperforming QR is not required. The May implementation is a source of separable ideas, not a recipe to reproduce1:1 or an objective to assume correct. Sparse QR, warm tracking and hybrids are competing solutions; no basis update is intrinsically preferred.

Judge the complete policy: ordinary covariance/projection/update work, refresh/retraction, momentum and V handling, precision/casts, compilation/setup, repair/fallback, memory and required checks. A faster refresh atom or lower covariance residual is not by itself a training improvement. Keep two-sided SOAP coverage fixed for the primary comparison; smaller-side coverage, reduced covariance cadence and changed attention backends are separate interventions.

## Decisions after the three reviews

The [review reconciliation](../optimizer_replay_results/soap_review_reconciliation_20261015/RECONCILIATION.md) changes priorities. Pause cubic3 confirmation and further warm tuning. Retain J1 as a control; test pure hold/QR200 and QR40 against QR20/J1 before assuming intervening warm motion earns its cost. The initial planned screen uses two fresh paired seeds, four3500-call arms each, with unchanged model/recipe and prospectively counterbalanced order. It is not yet prepared or launched; the roadmap specifies the preparation and follow-up branches.

Cubic3 recovered a real numerical failure and improved tracking, but [seed179](../optimizer_replay_results/soap_llm_cubic3_learning_20261014/REPORT.md) does not establish a consistent learning gain over cheaper J1. Severe left-factor cap suppression is observed; equivalence to hold, the cause of the suppression and a causal dimension law are unproved. Preserve real first-warm failures. A measured QR burn-in is a policy change, distinct from disposable compilation warmup/reset.

## Fidelity and evidence hierarchy

- Finite/arithmetic/reset/geometry and momentum checks constrain implementation fidelity; they are not a learning objective. Keep original numerical gates in qualifications. Absolute Gram0.05 does not imply physical-M error≤1%; the frozen concentrated-M negative control already demonstrates this.
- The physical-M round trip measures preservation of currently represented world momentum. It is not an exact past-gradient EMA oracle. Diagonal V lacks cross-moment history; overlap transport adds an approximation and may diffuse history through closed loops. The [completed October10 oracle](../optimizer_replay_results/soap_direction_history_cpu_20261010/REPORT.md) establishes mechanisms, not learner rankings, and must not be rerun for context.
- Tracking and adaptive-preview improvements are explanatory diagnostics. Correlated72/72 or36/36 signs do not establish independent evidence of learning gains. Use paired trajectories and practical quality/cost effects; report uncertainty and mixed panels. Two seeds are a finite screen, not an equivalence proof.
- Compare matched-update learning and comparable measured training time. Interpolate only within a shared observed horizon; do not infer CE gains from a local slope or attach synthetic timings to another workload. Qualify efficient QR before broad speed claims. More layers, coverage, longer horizons and the intended attention backend require their own continuation checks.

## Mechanisms retained for conditional work

First qualify covariance-product reuse in QR, then profile ordinary covariance/projection/pointwise traffic on the surviving policy. Selected BF16/TF32 arithmetic, cast removal, batching and long pure default/fullgraph regions are candidates; complete-cost and learning checks determine value.

Only if motion provides consequential learning value should alternative objectives/directions advance. Candidate ideas include relative damping, linearized ordered-QR motion, structured pair/block rotations and known relative M transitions. Account for dense covariance-basis products and history costs. Entrywise bounds do not certify dense-generator safety; inverse/relative-coordinate shortcuts must match actual mixed arithmetic and inherited drift. Scalar EMA can cancel under saturated clipping; row/column EMA and one-sided SOAP change the algorithm and need a cost case. Shifted Cholesky QR remains speculative.

Use a bounded trained-history diagnostic only when an observed ambiguity blocks a design choice. Do not automatically add correction-count, EMA/cap, overlap-V or precision grids. Retain compact provenance and scalar evidence; no default checkpoints/final weights. Preserve dirty trees and frozen successes/failures, declare deliberate state purpose/retention, and check shared resources before launches.

## Historical method record

The complete previous document follows unchanged; its old latest/next-step statements are historical.

---

# SOAP research: basis maintenance under a compute budget

> Latest: [width1536 learning attempt](../optimizer_replay_results/soap_wide_learning_round_20261007/README.md) STOPPED: exact historical model/optimizer anchors fail despite matched initialization/input/RNG.875 durable/at least1000 logged updates, firstQR only; numerical checks pass. GPU3 released; private supervisor stopped. Next: bounded fresh-process numerical execution diagnostic, no automatic21k retry or relaxed gate.

The new six3500-update campaign was stopped during its firstQR arm on the
historical100-update exact-state failure. Initial model/source/recipe/sampled
input/augmentation/RNG/software/GPU all match old1536; four retained anchors
100/200/400/420 fail model/fulloptimizer/metrics equality. CE shifts are small
but do not relax the exact gate.875updates/five observations/44guards/28targetedM
samples persist; progresslog confirms at least1000updates, unlogged tail unknown.
All recorded numerical checks pass. No Fro arm, completed3500 control, timing
qualification or learningwinner. Raw screen stillrunning; execution_status.json
records termination. No finalowned-release/memory/affinity-restoration record;
ownthreads died, no other process masks changed, GPU3 claim released/private
supervisor stopped.167 oldPython files/sixteen earlier inventories intact;
237 CPU tests pass. Workspace-root pytest collection was interrupted, correctly
scoped tests and historical evidence checks pass. No historical GPU rerun.

Read-only audit finds unchanged common frozen source and initial/data streams;
top-level config branch_steps420→3500 plus declared observation/flush/allocation/
owned-lifecycle changes. Drift appears before the420partial boundary. Old smoke
and new smoke gradient parity differ slightly, both pass smoke limits. Compiler
startup telemetry differs (old AOT1hit/2miss, new3hits; InductorGPUbenchmark62
versus12), but no recorded selected code/runtime-kernel fingerprint establishes
a cause. Do not claim cuDNN/compiler causality or optimizer instability.

Next bounded task: separately declare fresh-process numerical execution
fingerprints at actual1536, before another long learning run. Proposed allocation
420/3500 × flush10/25, two fresh processes each, fixed25 discarded compilation
smoke plus25 measured updates/process (400total). Record backendflags, actual
compiled code/runtime-kernel provenance and model/gradient/firstcovariance/basis/
optimizer/RNG/input fingerprints. No modelgraph hooks, shared cache edits, new
controller, automatic horizon extension or old-gate relaxation. Proposed
internal600/external660s,24GiBallocated/<25MiBexpected100MiBcap; verify/freeze
beforelaunch. This diagnostic is NOT yet prepared/running. Preserve this aborted
campaign and all earlier qualified/failing scopes. See NEXT_TASK.md.


> Latest: [actual width2048 pilot](../optimizer_replay_results/soap_wide2048_round_20261007/README.md) completed; feasibility/numerics and owned-adapter teardown pass, QR repeat exact. GPU3 released; private supervisor stopped. Next: longer matched width1536 learning past LR warmup with its own three-repeat timing qualification. No wall-quality winner or historical rerun.

Actual shallow ViT2048/depth1 (50,796,644params/eight factors through8192)
completed1260 primary updates plus25 discarded compilation smoke in235.230s.
All63 guards/72 targetedmoment samples/144factor rows retained. Max physicalM
.0780062%, selected-update error0, Gram .000687319. QR repeat model/fulloptimizer/
RNG/input/probe/chunkloss records exact; compiler stable and observers complete.
Owned hooks restored before optimizer deletion; every matrix/split optimizer
weakref dead immediately without explicit GC/cache clearing. Processpeaks5.144/
5.584/5.143GiB, cleanup baseline .732386GiB each; cleanup .00044–.00057s.
Peaks include shared model/data/compiler/offline work. Earlier1536 peaks may
include adapter cycles awaitingGC; no retroactive isolated-memory comparison.

Observed completeoptimizer QR48.582/48.624 versusFro41.337ms; model28.113/28.130
versus28.105ms. Trainingwall33.721/33.699 versus30.618s; CE4203.332654 versus
3.346387. All within5epoch LRwarmup; only2QR/1Fro arms. QRpair spreads .066%wall/
.085%optimizer descriptive, not three-repeat qualification. No fullstrictQR
shadow, wall endpoint/learningwinner/dimensioncausality. Old failures preserved.
All32 warmfactor samples bindcap.5; tracking still weak beforefirstQR reset,
stronger afterward. Retained role/age-matched widthcomparison is descriptive,
with differing weights/gradient/covariance histories. No EMA/controller change.
230 CPU tests/five focused cleanup checks pass;161 originalPython files and
fourteen previous inventories verified. Compact evidence only, no checkpoints/
weights/tensors/traces; GPU3 released/private supervisor stopped.

Next bounded task: new width1536/depth1 matched3500-update learning, threefresh
interleaved repeats per method, current QR20/Frocap.5/cubic2/.99 covariance/
strictM/currentV/both factors on original200epoch recipe. Preserve four existing
100/200/400/420 anchors, observe875/1750/3500; qualify its OWN unchanged3%wall/
2%optimizer gates before any wall-quality interpretation. Freeze warmup/setup,
probe ages, observations and runtime/storage/memory limits beforelaunch; owned
teardown throughout. See soap_wide2048_round_20261007/NEXT_TASK.md. No width/EMA/
seed sweep, automatic wall endpoint or completed experiment rerun.


> Latest: [actual width1536 pilot](../optimizer_replay_results/soap_wide_pilot_round_20261007/README.md) completed; feasibility/numerics pass and QR repeat is exact. GPU3 released; private supervisor stopped. No wall-quality claim. Next: owned-adapter lifecycle cleanup, then one shallow width2048 pilot. Preserve prior warmed90 failure; no reruns.

Actual shallow ViT1536/depth1 completed1260 measured updates plus25 discarded
compilation smoke in134.530s. All63 guards/72 targeted moment samples pass,
QR repeat/model/fulloptimizer/RNG/input/probe scalars exact; compiler stable and
placement restored. Actual8factors through6144. Max physicalM .0741428%,
selectedupdate error0. Full strict reference shadow prospectively not collected;
all older QR/reset discrepancies remain separately preserved.

Observed QRoptimizer22.920/22.958 versusFro17.871ms, training17.504/17.650 versus
15.324s; CE4203.287511 versus3.374298. Only2QR/1Fro arms, within original5epoch
LR warmup: no wallqualification/learningwinner/widthranking. All32 warmfactor
samples bindcap.5. Residual barely drops before firstQR reset, more afterward.
Scalar generator amplitude normalization cancels when homogeneous cap binds.
No EMA or controller changed. Per-arm allocated peaks2.970/4.240/5.010GiB include
process state; CPU lifecycle audit confirms adapter cycles may awaitGC, without
attributing actualGPUbytes. Restore ownedhooks before a future memory comparison.
225 CPU tests pass;155 originalPythonfiles/twelve earlier inventories verified.
Compact evidence only, no weights/checkpoints/tensors/traces; claimreleased.

Next bounded task: add explicit restoration/removal of owned per-instance
probe and Fro adapter hooks when discarding an arm, and verify object release on
CPU without global GC changes. Then freeze one actual shallow width2048/depth1
QR/Fro/QR420-update pilot (factors through8192), same recipe, both factors,
strict M/current V/.99 covariance, and bounded diagnostics/runtime/storage/memory.
Do not infer algorithm memory from the present per-arm process peaks. No width
sweep or EMA intervention. After feasibility, longer matched width1536 learning
must pass the5epoch LR warmup and qualify its own three repeats per method before
any wall-quality interpretation. No rerun of completed1536/90-second campaigns.


> Latest: [warmed90-second campaign](../optimizer_replay_results/soap_warmed_budget_round_20261007/README.md)
> completed: warmup/reset/numerics pass, but **QR wall repeat gate fails**.
> Endpoints remain exploratory. GPU3 released; private supervisor stopped.
> Next: shallow width1536 actual-model fixed-work pilot. Do not rerun this campaign.

[Warmed90-second QR/Fro campaign](../optimizer_replay_results/soap_warmed_budget_round_20261007/README.md)
completed on 2026-10-07: 28,950 measured updates plus 2,400 disposable warmup
updates, global128 compilation/parity smoke, 669.7s all-in. Every arm has the
same fixed400 warmup on its own policy, then restores original weights and
complete RNG and starts a fresh optimizer at measured update1. Warmup/reset
hashes and repeats pass exactly; no measured recompilation. All18 historical
fixed anchors, optimizer hashes, repeated probes, 1,449 guards and 1,008 targeted
M samples pass. The 564 full strict QR/reset discrepancies remain separate.

NEW wall qualification FAILS. QR fixed3500 wall spread9.724% and optimizer
8.621% exceed unchanged3%/2% limits. Fro spreads .064%/.124% pass. All six first
crossings meet the declared90-second/<=1s overshoot rule. QR endpoints4800/4450/
4800 updates, CE1.592796/1.614801/1.592796; Fro4950/4975/4975, CE1.619056/
1.628469/1.628469. These endpoints stay exploratory because the campaign repeat
gate fails. No wall-learning winner or old-endpoint promotion. Earlier full3500
qualification retains its own execution scope; it does not qualify this new run.

Retained identical3500 audit: slow QR has optimizer main CPU36.293s versus
32.863/32.808s, model main CPU13.450 versus12.176/12.196s, autograd native
runtime16.491 versus14.886/14.893s. Main queue wait11.486 versus11.343/11.465s;
GC about.014s; measured compiler counters stable. Sparse own-GPU SM clock ranges
are similar. This documents additional CPU time across phases, not an identified
clock/cache/instruction cause. Warmup duration itself is close across QR arms
(7.507–7.539s); no evidence supports silently adding burn-in or trimming the slow
repeat. No experiment was rerun to erase this failure.

Before launch221 CPU tests and seven focused checks passed;147 old Python files
and ten prior inventories are intact. Frozen execution source/config/protocol,
raw warmup/reset records, metrics/logs/observers and derived audit/figure retained;
about15MiB, no weights/checkpoints/tensor snapshots or new traces. GPU3 released;
private supervisor stopped. No queued claimant or other-project changes.

Next bounded work follows user steering: shallow actual-model ViT with embedding
width1536, initially one block, matched QR20/Fro .5/cubic2, strict M/both factors,
current V and covariance.99. CPU meta inventory gives28,660,324 parameters and
8 SOAP factors, actual factor widths1536/4608/6144. Covariance+basis FP32 storage
alone is828MiB (algebra, not measured peak); moments, activations, casts and
workspaces are additional. Start with a prospectively bounded fixed-update
pilot through natural warm/reset cycles and measured complete optimizer/model
cost; no equal-wall quality claim until that setup passes its own repeat gates.
A second declared larger width and longer learning follow feasibility/numerics.

Larger matrices being easier to refresh or needing dimension/magnitude-specific
step scaling remains a hypothesis. Record offline normalized generator and
tracking diagnostics before choosing an intervention. Scalar/row/column EMA
normalization would be a separate bounded ablation with reductions, state updates,
casts and memory charged; no full-tensor EMA without a stated benefit/cost case.
The current small-factor recipe and all prior failures remain preserved.

[Full-horizon co-location qualification](../optimizer_replay_results/soap_colocated_round_20261007/README.md)
completed on 2026-10-07: six 3,500-update QR/Fro arms, 21,000 primary updates
in 460.9s. Full qualification passes. All 18 historical observations,
full optimizer-state hashes, repeated trajectories/probes and six plain/profile
replays are exact. All 1,050 guards and 1,008 targeted M checks pass. Maximum
sampled physical M drift is .10039%; selected-basis errors are zero. The 564 full
strict-reference QR/reset disagreements remain separate: 492 QR and 72 Fro reset
samples; warm full-reference checks pass.

Main/autograd both CPU116. All three repeats pass unchanged limits: QR/Fro wall
range/mean .890%/.375%, optimizer .339%/.126%. Mean training wall is 66.218/63.810s;
optimizer events 9.760/9.144ms. Fro saves 3.64% training wall and 6.31% optimizer
event time at fixed work, but CE remains 1.738019/1.862894. This is a local
instrumented timing qualification, not a learning winner, uninstrumented
throughput result, device comparison or retroactive qualification of old endpoints.

Observer/placement records, stable compiler counters and corrected GPU trace
attribution pass. Original main/autograd masks are restored. Execution source,
config and protocol remain frozen. Before launch, 214 CPU tests passed; all 142
previously existing Python files still match preflight hashes. No weights,
checkpoints or tensor files were saved. Compact evidence is about 14.5MiB,
below the 30MiB estimate and 100MiB cap. GPU3 claim released; private supervisor
stopped. No queue or changes to other projects.

Next bounded work: declare a NEW 90-second measured-training-wall comparison
of QR20 versus Fro .5/cubic2, covariance .99, seed139, current V carry/reindex,
strict M and both factors, main/autograd CPU116/116. Include interleaved fresh
3,500-update controls with three repeats per method; require unchanged 3% wall /
2% optimizer gates in that campaign before interpreting its new budget endpoints.
Keep the recipe/LR and full online optimizer costs; declare stop/evaluation rules
and bounds before launch. No .999 wall claim, old endpoint promotion, radius/V
sweep or seed expansion follows from timing qualification alone.

Updated 2026-10-06. This document describes the current research motivation and
comparison protocol. Optimizer defaults are unchanged; proposed studies are not
claims of measured improvement.

Latest continuation: [Autograd co-location diagnostic](../optimizer_replay_results/soap_dispatch_round_20261006/README.md)
completed on 2026-10-07: six 1,000-update QR arms, 6,000 primary updates in 186.17s.
All numerical controls, twelve historical observations, six historical optimizer
hashes at update 875, probes and six plain/profile replays are exact. Both placements
pass the unchanged repeat gates: near/same wall 1.223%/.449%, optimizer .322%/.211%.
Main/autograd on CPU116/116 versus CPU116/117 lowers model/backward events 35.69%
and total training wall 18.66%; all three pairs agree and model ranges separate.
Mean wall falls from 23.253 to 18.913s; mean model time from 12.089 to 7.774ms.

Observed autograd runtime roughly halves, from 8.43–8.62 to 4.26–4.28s. Same-core
queue wait rises to about 3.21–3.23s, while net execution is faster. Other observed
stable threads contribute <.03s. Snapshot intervals include offline work and omit
threads created or destroyed between snapshots; they are not complete phase
accounting. Corrected profiler model kernels remain 3.540–3.555ms/update with exact
operation signatures. This supports a local host-placement effect; the precise
cache or microarchitecture cause remains unresolved. All 300 guards and 720
targeted probes pass. The 714 full strict QR reference disagreements remain
separately preserved. Before launch, 213 CPU tests and 10 focused tests passed.
No weights, checkpoints or tensor files were saved. Defaults and prior failed
wall endpoints retain their scope.

The subsequent full 3,500-update QR/Fro qualification completed and passed
all gates. Its 214 CPU tests passed before launch, including lossless affinity-mask
deduplication. GPU3 is released and the private supervisor stopped. See the
latest full-horizon decision above; this earlier short-study next step is complete.
No historical experiment was rerun for context.

Earlier V-state continuation (historical next steps): [Fixed-update variance-state ablation](../optimizer_replay_results/soap_variance_round_20261006/README.md)
completed: twelve arms and 42,000 updates in 1031.7s. Fixed-update qualification
passes: all 18 historical anchors, model/full optimizer hashes and four repeated
trajectories/probes are exact. All 2100 refresh guards, 2016 targeted M probes
and 2016 independent matched-high V formula checks pass. Maximum physical M
drift is .11884%; warm full strict-reference checks pass, while 991 QR/reset
full-reference disagreements remain separately recorded.

Overlap worsens validation CE in every pair at 3500: QR +.061393/+.009562 and
Fro .5/cubic2 +.070220/+.054643 for covariance .99/.999. It does not close the
warm/QR gap. Retain carry/reindex; no overlap promotion or seed expansion. This
is one seed at 20 epochs of the original 200-epoch schedule. The study compares
the existing diagonal approximation, not exact full second-moment history.

Observed mean optimizer cost changes are +1.74%/+2.15% for QR/Fro at .99 and
+.26%/+1.99% at .999. Execution repeat spreads reach 8.18%, so these small mean
changes do not establish stable marginal costs. No wall endpoint or wall-quality
gate was collected. Prior wall failures remain unchanged. 206 CPU tests, 21
focused checks and five additional focused checks passed before launch. All 129
pre-existing Python files and six recent frozen evidence inventories are intact.
GPU1 claim released; private supervisor stopped; other projects remain active.

Next bounded task: audit retained timing evidence and choose one prospective
cause-led timing diagnostic/intervention. Require exact work and unchanged
<=3% wall / <=2% optimizer repeat limits before a new wall-quality campaign.
Do not rerun completed experiments merely for context or launch a broad sweep.

No production/default changes.

Earlier radius-learning continuation (historical next steps): [Radius .6 paired CIFAR learning](../optimizer_replay_results/soap_radius_learning_round_20261006/README.md)
completed: 16 arms, 65,175 updates. Fixed-update qualification passes; all 24
historical anchors and repeated model/full optimizer-state/metric/probe records
are exact. Radius-only CE deltas versus .5/cubic2 at 3500 are -.000095/.000338
(.99/.999); cubic3 deltas -.003876/+.005930 with about 1.86% extra mean optimizer
cost at .99. Neither produces a consistent learning gain; QR remains ahead.
No radius .6 promotion or seed expansion. All 3262 guards/2688 targeted probes
pass; 202 CPU tests passed before launch. New wall gate fails: QR range/mean
3.383% exceeds 3%, with optimizer variation 1.217% below 2%. Warm policies pass
repeat timing. All matched wall-quality endpoints remain exploratory; .999 has
one trajectory per policy. Earlier qualification and failures retain their scope.
Next bounded algorithm task: fixed-update variance-state ablation, current V
carry/reindex versus existing squared-overlap transport, with matched QR and
.5/cubic2 bases, strict M, both factors, .99/.999. This is a hypothesis about the
warm/QR learning gap; it is not an established mechanism. Keep radius/cadence
fixed and charge full costs. Separately diagnose timing from retained host/thread
records before a new wall campaign. GPU3 released; private supervisor stopped.
No production/default changes.

Earlier, the [timing round](../optimizer_replay_results/soap_timing_round_20261006/README.md)
reproduces unpinned wall-repeat failures, then establishes an 8.52% local
model/backward timing effect from main/autograd CPU placement with exact numerical
work. Both fixed placements pass the repeat limits at 1,000 QR updates. Next
qualify recorded placement on interleaved 3,500-update QR/Frobenius controls;
the shorter result does not qualify earlier wall endpoints. Host dispatch and
actual GPU kernels must remain separate. The supplemental replay and initial
trace-summary failures are preserved and corrected without editing frozen
evidence. No optimizer or permanent affinity default changed.

## Earlier priority: qualify timing and separate radius from bound conservatism

The [movement-bound round](../optimizer_replay_results/soap_movement_round_20261006/README.md)
implements a skew-Frobenius bound with two cubic corrections. Common-state
movement and residual reduction improve substantially; natural-cycle cost stays
close to row-cubic2. Paired CIFAR fixed-update CE improves only modestly and still
trails QR/NS6. The 90-second training-wall comparison failed its predeclared
repeat criterion: identical QR work varied by 11.85% in wall time while optimizer
time varied only .84%. Exact numerical repeats do not establish timing stability.

Keep the candidate experimental. First qualify repeated/interleaved identical
work for wall-quality interpretation. Separately compare allowed movement radius
and correction work on common states, using a small fixed shortlist and declared
moment/geometry tolerances. Select on learning and total cost; better instantaneous
covariance residual is insufficient. Replication and longer horizons remain due.

The [movement note](benchmarks/MOVEMENT_BOUND.md) records the bound and scalar
Gram-error maps. The [earlier design](benchmarks/WARM_RETRACTION_DESIGN.md) preserves
hypotheses and failures. BF16, natural-cycle compiled ownership and exact H200
provenance remain open. Production/defaults are unchanged; historical dated next
steps below are superseded by this priority and the shared roadmap.

## Research question

At a fixed total optimizer-time budget, which allocation between infrequent QR
refreshes, frequent warm basis updates, and hybrid schedules gives the best
training progress while preserving useful Adam state?

SOAP applies elementwise Adam in covariance-derived coordinates. Maintaining
those coordinates costs time, and changing them changes the interpretation of
historical moment estimates. Better instantaneous diagonalization need not imply
a better optimization update. Conversely, cheaper basis updates need not produce
a cheaper optimizer once projection, state transport and retraction are included.
Neither sparse hard refreshes nor frequent warm updates is presumed superior.

## Baselines and attribution

The existing fixed-10 LLM reference is already smaller-side TurboSOAP with warm
updates and occasional QR resets. It is a cadence control for that method, not
standard SOAP. The completed 4,000-step comparison establishes an observed
14.3% optimizer-time and 1.11% step-time saving for sparse cadence; it does not
establish a speed advantage over a matched QR-based SOAP implementation.

The original SOAP implementation initializes with eigh, then performs one
covariance-times-basis power iteration followed by QR. That QR refresh is not
an exact eigensolve. Pin original source for semantic reference and remove
avoidable redundant work in a separately verified runtime baseline. An every-
reset configuration of this repository is not automatically upstream-equivalent:
alignment, v handling, normalization and update ordering must be checked.

Primary attribution holds active factors, precision, covariance accumulation,
learning rates, decay, moment policy and model/data fixed. Compare:

| Arm | Basis policy | Cost to include |
|---|---|---|
| QR | QR every K updates; unchanged basis between | QR, projections and moment handling |
| Warm | Warm updates every k with rare QR safeguards | basis motion, retraction, m/v handling and safeguards |
| Hybrid | Both operations on declared schedules | the full combined cost |

Calibrate schedules from measured cost, not equal refresh counts. Report both
quality at equal updates and quality at equal wall time. A tuned practical
comparison is separate from the matched-setting attribution experiment.

## One-sided is a separate algorithmic choice

Full factors of an m-by-n tensor have nominal decomposition work O(m^3+n^3)
and storage O(m^2+n^2). Smaller-side SOAP removes a potentially dominant factor,
as well as associated covariance and projection work. For shape 4d-by-d, the
nominal cubic factor work changes from 65d^3 to d^3; this is not a wall-time ratio.

Run primary tracker comparisons with matched two-sided factors, then selected
one-sided comparisons. Neither regime should stand in for the other. The dense
warm tracker still uses cubic GEMMs. Possible gains are hardware efficiency,
continuity and frequency, not an established asymptotic reduction. Measure
covariance accumulation, projection, basis update, moment handling, retraction,
and complete optimizer steps. Cold initialization is reported separately.

Large factors with meaningful but trackable drift are a plausible favorable
regime. Nearly stationary bases may favor simply infrequent QR; rapidly changing
bases may defeat small warm updates. Small factors can be dominated by launches
and state movement. These are hypotheses to measure, not preselected winners.

## Preserve first moments; test second-moment handling

For orthonormal bases, define U=QL_old^T QL_new and V=QR_old^T QR_new. Exact
first-moment coordinate reexpression is M_new=U^T M_old V. Ordinary controls
must preserve the physical first moment. Keeping it in original coordinates and
projecting it for updates is another representation, with a different cost
schedule. Finite precision and imperfect orthogonality still require auditing.

Only diagonal second moments are retained. Squared-overlap transport is
V2_new approximately (U elementwise-squared)^T V2_old (V elementwise-squared).
It discards unrecorded cross second moments and is path-dependent when repeatedly
applied. Column continuity, carry/reindex, and approximate transport each have
tradeoffs. Compare their update transients and history error against a small full
second-moment oracle before adding repeated transport to training defaults.

The original dense SOAP class uses exact-form first-moment reexpression at every basis
change; v stays with continuous columns on warm steps and receives squared-
overlap mixing on hard resets. Optional warm v transport remains experimental.
The paper emphasizes preserved momentum; the inspected sources do not establish
a clean isolated ranking of m-transport versus v-transport importance.

## Memory settings

Only covariance beta and Adam beta2 values **.99 and .999** are in the current
study. .9999 is excluded by user direction. Keep beta1 fixed initially. Vary the
two second-moment memories independently; slower geometry with faster denominator
forgetting is one hypothesis, not the preferred answer. Half-lives are about 69
and 693 updates. At 175 CIFAR updates/epoch these are .39 and 3.96 epochs.

Use matched settings for attribution and fair LR calibration for practical
comparisons. Normalization, epsilon, startup behavior and schedule must remain
explicit. Slow covariance evolution does not guarantee stable eigenvectors near
small eigengaps. Stable coordinates may help preserve long-lived Adam statistics;
long memory can also preserve coordinate-history errors.

Covariance beta interacts with refresh cadence, reset cadence, warm rotation
step/cap/retraction, Adam memory and learning rate. For an EMA updated every
optimizer step, the total weight on the next k observations is `1 - beta**k`.
At k=10/20 this is about 9.56%/18.21% for beta .99, versus .996%/1.981% for .999.
This is an EMA identity, not a rule equating eigenspace movement or proving a
beta/cadence compensation. Eigenvalue gaps, gradient statistics and the tracker
matter. One-factor studies establish conditional effects; they cannot select a
universal beta independently of the remaining optimizer settings.

## Separate learning and runtime measurements

Keep three distinct sources of evidence:

1. Real CIFAR/OpenWebText learning: paired inputs, validated loss, learning-rate
   calibration, beta/cadence interactions and replication at actual workloads.
2. Synthetic runtime scaling: matrix dimensions/aspect ratios, factor sides,
   activation workload, cold initialization, ordinary updates and refresh/reset
   events using the actual optimizer, including m/v handling and retraction.
3. Target-model confirmation: complete natural cadence cycles in the intended
   trainer and precision/compile setup before claiming sustained throughput.

The new [synthetic protocol](../optimizer_replay_results/soap_synthetic_scaling_v3_20261002/PROTOCOL.md)
uses a short square-plus-4x-rectangular mock block. Explicit event sampling avoids
missing rare QR resets in a few arbitrary steps. Cadence-weighted event estimates
remain cost estimates, not measured long-run schedules or equal-quality speedups.
Report one-sided/two-sided panels separately: removing a large factor changes
both the method and potentially most of its cost. Report factor sizes, not just
model parameter count. Dense warm work remains cubic; constants, launch overhead,
solver efficiency, precision and memory determine the measured crossover.

CIFAR optimizer-cost rankings are scoped to its matrix shapes and execution path.
Neither small-model timings nor a synthetic crossover establish the ranking on
larger models. Do not combine synthetic time with CIFAR loss into a time-to-quality
curve for a workload that was never run. Current CIFAR wall-time variability is
a separate measurement issue, not an explanation of every dimensional effect.

The [first completed stress map](../optimizer_replay_results/soap_synthetic_scaling_v3_20261002/README.md)
retains all 72 cases, including 18 warm geometry rejections. A focused natural
200-update mock check also rejects warm at update 10 (two-sided, 3072 factor)
and 110 (smaller side, 768 factor); QR two-sided passes. On the failing natural
3072 input, the existing two-iteration retraction gives Gram error .76476;
six shadow iterations give .000663. Shadows do not change the executed update.
This identifies inadequate retraction convergence on these inputs, not a validated
fix or a failure claim about real-data training. Revisit rotation/scaling and
retraction safeguards, include their full cost, and repeat natural-cycle checks
before advancing large-factor speed claims. Event-cost advantages alone are
insufficient, including for cases that passed the short forced-event probe.

### Width, precision and compilation

The [large-factor protocol](benchmarks/LARGE_PRECISION.md) adds depth-one square
projections at widths 2048/4096. Depth and width serve different purposes: reducing
depth controls aggregate memory and runtime while preserving each large factor.
The earlier width-1536 4x MLP already had 6144 factors; a larger model width does
not automatically mean a larger eigensolve. Always report active factor shapes.

Separate BF16 model operations, covariance products, warm-kernel products and
FP32 state storage. Test strict FP32 against reduced internal precision, and
give QR controls the same applicable optimizations. Compile pure tensor work
first, account for first-call compilation separately, and retain full-optimizer
event costs including m/v handling. Optimizer-only inference mode needs explicit
state-lifecycle testing because inference tensors restrict later mutation;
training forward/backward must remain outside that context. Timing gains require
geometry and eventual learning gates at the same precision. NS6 is an explicit
experimental retraction setting, not a production default change or a general fix.

### Rectangular numerical gate completed

The [rectangular follow-up](../optimizer_replay_results/soap_rectangular_precision_20261004/README.md)
reproduces the original NS2 failures at updates10/110 and measures their momentum
damage. NS6, its compiled kernel, and BF16 covariance products pass the declared
1% momentum/update budgets over200 updates on both factor settings. BF16 gauge
passes the coarse Gram check but fails the stronger budgets; defer that precision
setting. Spectral-bound clipping passes but moves3072-factor bases about61x less
than NS6/average cap on their respective trajectories, so keep it a separate
tracking-policy choice. No optimizer defaults changed.

Current schedules accumulate covariance and update Adam each step, refresh warm
bases/transport m every10, and replace warm with scheduled QR every200. Research
checks run after refreshes outside measured optimizer time, stop failed cases,
and do not implement automatic QR fallback. Any eventual online checks must have
their overhead included. Next isolate cadence and covariance beta (.99/.999) with
the numerically surviving NS6 candidate and sparse QR controls before real-data
quality confirmation. Numerical budget acceptance alone is not convergence or
equal-quality speedup evidence.

### Cadence and covariance screen completed

The [cadence/beta screen](../optimizer_replay_results/soap_cadence_beta_20261004/README.md)
passes all24 numerical cases: covariance .99/.999, QR or NS6 warm every10/20/40,
both factors active on the original rectangular workload. All six repeated .999
controls exactly match prior loss and diagnostic records. Scheduled QR200 stays
fixed for warm; this does not introduce adaptive checks/fallbacks into the optimizer.

Warm is cheaper at matched intervals, but warm20 costs about20% more than QR40;
warm10 costs27–32% more than QR20. Sparse QR remains a serious alternative to more
frequent warm refreshes. These are short synthetic costs, not quality rankings.
Next: bounded real-data quality screening of NS6 warm20/40 and QR20/40 with both
covariance betas, verifying adapter settings first. No default changes.

### Real CIFAR cadence quality screen completed

The [CIFAR NS6 screen](../optimizer_replay_results/cifar_soap_cadence_20261004/README.md)
completes nine 20-epoch arms on the original 200-epoch schedule. At fixed LR .0005,
QR20/40 beats NS6 warm20/40 in validation CE and optimizer cost at both covariance
betas. At .99, QR20/QR40/warm20/warm40 CE is 1.74838/1.75978/1.80780/1.85002.
Covariance .99 improves all four early endpoints relative to .999. The earlier
100-epoch study showed horizon dependence, so do not treat this as convergence.

All 1,227 per-refresh/initialization probes pass (maximum Gram .0013854); QR20
historical anchors and the repeated control match exactly at all observations.
The control optimizer-time range is .38%, but training wall varies 20.7% and
model/backward 43.8%; no equal-wall claim. CIFAR matrix sizes remain separate
from the synthetic large-factor runtime map. No defaults change.

That follow-up is now complete; see the tracking and LR result below.

### CIFAR tracking and matrix-LR screen completed (2026-10-04 UTC)

[Report](../optimizer_replay_results/cifar_soap_tracking_lr_20261004/README.md): seven arms, 24,500 updates, 10.20 minutes. QR20 and NS6
warm20 each test matrix LR .00025/.0005/.001 with covariance .99, auxiliary
LR .0005 and Adam(.9,.999) fixed. Both choose .0005 at the 20-epoch endpoint:
CE 1.74838 versus 1.80780, a warm deficit of .05941. The same original 200-epoch
schedule and seed139 are used; this is early sensitivity, not convergence or
new-seed replication. The bounded LR check does not remove the quality gap.

On warm-trajectory incoming states, mean local covariance residuals for factors
192/576/768 are .3467/.2852/.3095 after one NS6 step, .3102/.2318/.2576 after two,
and .1047/.1038/.1108 after QR. Every second warm shadow improves its sampled
residual, but dimension means still trail QR. The cap binds in 58.9/41.7/44.4%
of those samples. QR-trajectory shadows show the same qualitative pattern.
These are identical-state local alternatives; they are never executed in
training, and lower residual alone does not establish better learning. QR
permutes v and warm carries v; update differences do not isolate transport.

All 1,225 refresh/initialization guards pass (maximum Gram .00143845). Six
historical observations, three repeated-control observations and all repeated
tracking records match exactly. 147 prelaunch CPU tests pass. Scalar probes
cover 21 events/504 matrices/1,008 factors including the repeat. A different
physical RTX5090 was used; numerical anchors are verified, timing is not pooled.
Control optimizer range/mean .575%, training-wall 1.659%. Selected-rate optimizer
cost is 9.773 ms QR versus 10.778 ms warm on these small factors. No equal-wall
or large-factor speedup claim. No defaults changed or weights saved; GPU released
and private supervisor stopped.

Next: a bounded tracking-efficiency gate, with two inner warm steps and a modest
cap change tested separately against unchanged warm and QR controls. Check
geometry, world-m/update fidelity and complete refresh cost before any expanded
training comparison. Distinguish one final m transport from two sequential
transports. The two-step shadow has no measured cost or learning result yet.
Keep v handling as a separate ablation, retain sparse QR budget controls, and
preserve .99/.999 and longer-horizon replication on the roadmap. No further
training grid was launched in this stage.

### Local refresh efficiency and precision gate (2026-10-04 UTC)

[Report](../optimizer_replay_results/cifar_soap_refresh_efficiency_20261004/README.md): three unchanged CIFAR trajectories, 10,500 updates,
6.58 minutes. QR20, NS6 warm20 and the QR repeat reproduce all nine historical
anchors exactly. The 504 actual-refresh Q/m/v controls and repeated numerical
probes match exactly. All 525 training guards pass; 156 CPU tests pass.

On 144 warm-trajectory matrix records excluding QR resets, one warm step at
cap .1 gives mean residual .32972 at 2.930 ms per complete local refresh. Cap .2
improves this to .31530 at 2.926 ms, about 27.1% more residual reduction per ms.
Two inner steps with final m transport reach .28729 at 5.764 ms: 1.97× cost,
8.9% lower local reduction/cost efficiency. QR reaches .10611 at 2.258 ms.
These are eager small-factor event measurements including m/v handling, not
whole-optimizer speedups or learning progress. No variant changes training.

All 2,520 candidate records pass Gram and both world-m fidelity checks, but no
variant passes every 1% strict-FP32 implied-update agreement check. Rejections
per 504 records: QR493, warm1 49, warm2-final72, warm2-sequential187, cap-.2 53.
Totals include the exact QR repeat and are not independent observations.
QR's maximum update difference is 73.3%, despite world-m preservation error
below .067%. This exposes full-refresh precision sensitivity in the baseline
as well; it does not prove broken transport or worse QR learning. Strict FP32
is a numerical reference, not an established learning optimum. Sequential versus
final-only m transport changes the two-step implied update by up to 3.63%, with
identical bases/v; sequential transport also costs slightly more.

Next: hold candidate Q and v fixed to isolate transport arithmetic, then isolate
basis construction and QR ordering/permutation effects. Preserve the full
strict-refresh comparison; do not relax its threshold after observing failures.
Resolve that attribution before promoting cap .2 into a training intervention.
Defer more inner steps unless their measured cost can be justified. Retain sparse
QR, .99/.999 and larger-factor follow-ups, with costs and learning kept separate.
No defaults changed, no weights/checkpoints/tensor snapshots saved. GPU released,
private supervisor stopped, prior evidence preserved. No further GPU run launched.

### Precision attribution completed (2026-10-04 UTC)

[Report](../optimizer_replay_results/cifar_soap_precision_attribution_20261004/README.md): 10,500 unchanged training updates, 4.61 minutes,
504 matrix records and 2,520 precision decompositions. All nine historical
anchors, actual-refresh Q/m/v controls, repeated probes and all previous
full-refresh update differences reproduce exactly. All 525 training guards
pass; 164 CPU tests pass. No candidate changes training.

Holding the high candidate basis/v fixed and using strict FP32 m transport
resolves every sampled one-step warm full-refresh agreement failure: cap .1
49→0 and cap .2 53→0 per 504 records, including the QR repeat. Worst remaining
difference is .844%. On the warm trajectory's 144 actual warm-event records,
both caps go 3→0, maximum .534%. This is a sufficient local intervention, not
proof that transport alone caused every combined discrepancy or that learning
improves. Two inner steps retain 7/13 failures with final/sequential transport.

QR remains different: failures 493→492, maximum remaining difference 73.098%.
The strict-basis stage with high ordering held fixed still has mean magnitude
12.086% of the strict update norm. Ordering and v differ in every sampled QR
record; their separate stage magnitudes are not additive. Vector closure is
at most 3.33e-8. Tiny world-m differences can become much larger after the Adam
denominator; geometry/world-m checks alone do not bound the implied update.
No claim of broken QR learning or a proven spectral cause follows from this.

Next: an opt-in strict m-transport path, full refresh cost measurement, then a
bounded paired learning check for one-step warm cap .1/.2 against unchanged
controls if its targeted checks pass. Retain both fixed-basis transport checks
and the full strict-refresh comparison. QR resets remain explicitly unresolved
under the old full-refresh gate; do not relax that threshold retrospectively
or claim the hybrid policy now passes it. Keep .99/.999 and large-factor work
separate. No defaults changed, weights/checkpoints/tensor snapshots saved, or
further GPU job launched. GPU released and private supervisor stopped.

## Strict transport, shape costs and learning completed (2026-10-04 UTC)

[Report](../optimizer_replay_results/cifar_soap_strict_transport_20261004/README.md):
implemented opt-in `ResearchSOAP(transport_precision="highest")`, then completed
18 paired shape checks and fourteen 3,500-update CIFAR arms (49,000 updates).
Strict precision covers first-moment transport at refreshes, including QR resets;
basis construction, v policy and nonrefresh work retain their prior behavior.
The setting restores on exceptions. Production `soap.py` and defaults are unchanged.

At covariance .99/.999, strict transport lowers QR CE by .010363/.004975.
Warm responds conditionally: strict transport worsens cap .1 at both betas and
improves cap .2 at both. Raising the cap at .999 hurts inherited transport but
helps strict transport. Best tested QR/warm CE is 1.738019/1.788615 at .99 and
1.774667/1.843668 at .999. QR also costs less on this small trainer. These are
one-seed, fixed-LR, 20-epoch endpoints on the original 200-epoch schedule;
neither a default recommendation nor a convergence ranking follows.

Synthetic complete refresh costs reverse rank with dimension: strict QR/warm
cap .1 cost 21.17/15.74 ms at 768×3072, but 76.82/98.99 ms at 1536×6144.
Selective strict-transport overhead is small in these short samples. These eager
NS6 fixture costs are separate from sustained model throughput or learning.

All 176 CPU tests and 2,450 refresh guards pass. All 1,176 strict-transport
matrix probes match the selected-basis update oracle exactly; actual warm events
also stay below 1% full strict-refresh disagreement (maximum .675%). QR/reset
full-reference failures remain explicit: the whole hybrid policy does not pass
that separate gate. Twelve historical observations and both repeated controls,
including numerical probes, reproduce exactly. No online controller or fallback
was added. The process-wide precision setting is qualified only for sequential
eager use; compiled/concurrent optimizer execution remains untested.

Next: stop expanding this precision/cap grid. Replicate selected QR and cap-.2
warm comparisons on paired seeds 271/811, retaining inherited precision and
both covariance controls; then extend promising settings on the original
schedule. Separately qualify matched-precision compiled refreshes and natural
cadence at the 3072/6144-factor shapes. Keep strict transport opt-in. No further
job is launched this round; no weights/checkpoints/tensor snapshots were saved.
GPU released and private supervisor stopped; compact evidence is retained.

## Foundation execution funnel (completed stages and general method)

1. Audit actual LLM compiled/eager BF16 evaluation at identical trained weights.
   SNRAdam found a CIFAR-specific discrepancy; LLM parity is not inferred from it.
2. Exact small-moment replay: abrupt versus incremental rotations, correct m,
   carry/reindex versus squared-overlap v, and .99/.999 memories. Include analytic
   sign/permutation controls and a full second-moment oracle.
3. Measure refresh and state-handling cost by factor size and aspect ratio, both
   sides versus smaller side. Benchmark the actual dense warm implementation and
   QR, with instrumentation overhead separated.
4. Build a semantically audited QR baseline and a frozen CIFAR-100 adapter using
   SNRAdam's corrected eager evaluator, cached split and explicit parameter groups.
   Preserve group-specific matrix/auxiliary LR schedules and decay rules.
5. Compare budget-calibrated strategies using common-state branches and an
   unchanged-control repeat, then replicate selected seeds. Confirm useful
   candidates on the LLM. CIFAR's width-192 factors are a mechanism laboratory,
   not evidence of large-factor speedup.

Use one claimed GPU, no automatic sweep expansion and no default checkpoints.
Keep compact configs, hashes, seeds, metrics and logs. If a diagnostic needs
weights or moment snapshots, declare purpose, size and retention before launch.

## First foundation results (2026-10-01)

The current LLM parity gate passes at three trained states (maximum compiled/
eager BF16 CE difference 2.9e-5). The 24-case moment oracle confirms exact-form
m reexpression and path-dependent diagonal v handling. Twelve eager refresh
cases show warm+m slower at 192-square two-sided factors (1.943 vs 1.455 ms),
but cheaper at 1536-square (2.317 vs 13.099 ms). These include state movement;
the later CIFAR calibration below adds full-optimizer measurements. Warm and QR yield
different residual/orthogonality errors, so equal cost is not equal quality.

[Complete results and reproduction](../optimizer_replay_results/soap_budget_20261001/README.md).
The QR reference and frozen CIFAR adapter are now complete. ResearchSOAP uses
upstream update/epsilon/decay ordering, sorts/reindexes v on QR, and carries v on
warm steps. It reexpresses m at basis changes, eliminating redundant round trips
on unchanged-basis steps. CPU semantics are tested against a pinned upstream
fixture; GPU differences are measured, not assumed zero.

The [CIFAR report](../optimizer_replay_results/cifar_soap_calibration_20261001/README.md)
finds QR10/warm10 at 12.17/12.07 ms per optimizer update, with QR20 at 9.80 ms.
QR20 saves 19.5% optimizer time and 9.4% steady step time. Warm5 costs more and
has worse early CE. An unchanged QR10 repeat has identical final weights/loss.
Warm Gram error is about 7e-4 versus 7e-7 for QR. The endpoints are only 640
updates, still in warmup, with untuned matrix LR and one seed.

The [20-epoch LR screen](../optimizer_replay_results/cifar_soap_lr_screen_20261001/README.md)
now covers three rates for all three policies, with an unchanged control repeat.
Every method selects the lower bracket boundary .0005. At that rate QR10 and
QR20 reach CE 1.77703 and 1.77964, while warm10 reaches 1.84933. QR20 costs about
19.5% less optimizer time. The control repeats bitwise, but training wall time
varies 13.7% despite only .2% optimizer-time variation, limiting wall-time claims.

The [lower bracket](../optimizer_replay_results/cifar_soap_lower_lr_20261001/README.md)
now resolves that boundary: exact .0005 anchors allow combining five rates, with
.0005 best for all three policies. This is one-seed early-learning calibration,
not a converged optimum. Warm10 has higher CE at every tested rate.

Three bitwise controls show 14.76% training-wall range/mean, failing the predeclared
3% interpretation gate. Model/backward elapsed time varies 31.51%, optimizer .53%.
This localizes the measured variation without isolating its cause; CUDA elapsed
includes dispatch gaps. No wall-quality winner is promoted.

The [independent memory screen](../optimizer_replay_results/cifar_soap_memory_20261002/README.md)
holds both LRs at .0005 and auxiliary AdamW beta2 at .999. Changing covariance beta
alone from .999 to .99 improves 20-epoch validation CE by .05121/.03126/.07148 for
QR10/QR20/warm10. Changing matrix Adam beta2 alone to .99 changes CE by
+.02317/+.02109/-.00779. Faster covariance adaptation is the stronger follow-up
candidate here; neither slower nor faster second-moment memory is universally
better. Warm's improvement does not establish that it beats matched QR: at
covariance .99, QR10/QR20/warm10 CE is 1.72581/1.74838/1.77785.

These are controlled fixed-LR, one-seed interventions, not separately tuned beta
rankings or convergence results. No joint .99/.99 arm was run. This motivated the
completed longer covariance comparison below; additional paired seeds are next.
Hold v/factor-side choices fixed, track warm orthogonality, and investigate
execution timing separately. No method defaults were changed.
See [adapter instructions](benchmarks/CIFAR100.md) for freezing a fresh run.

## Longer covariance learning (2026-10-02)

The [100-epoch extension](../optimizer_replay_results/cifar_soap_covariance_long_20261002/README.md)
changes the early readout: covariance .99 improves QR10 CE by .03014 and warm10
by .01433, while QR20's earlier gain shrinks to .00052 (accuracy -.04 percentage
points). At covariance .99, QR10/QR20/warm10 CE is 1.26909/1.30446/1.31237;
at .999 it is 1.29923/1.30498/1.32670. Treat QR20's endpoint effect as essentially
flat on this seed, not evidence of a meaningful benefit. Warm remains worse than
both QR schedules at matched covariance settings; QR20 costs about 19% less
optimizer time than QR10/warm10 on these small-factor matrices.

All 18 early anchors exactly reproduce the previous screen. The unchanged QR10
control repeats weights/metrics at all seven observation points through 100
epochs. This is the same seed extended on the original 200-epoch schedule, not
new-seed replication or convergence. Matrix/auxiliary beta2 .999 and both LRs
.0005 were fixed; no beta-specific LR retuning or default changes.

Next: paired seed replication before another memory or transport-policy sweep.
Predeclare seeds 271 and 811 on the same split, staging one seed at a time under
the one-GPU rule, with all six covariance/policy settings and the same 100-epoch
endpoint. Preserve the .999 controls: the covariance effect is schedule- and
horizon-dependent here, and its mechanism is unresolved. No seed job was added
to this campaign. Wider-factor confirmation remains necessary before dismissing
or promoting the warm approach; small CIFAR results have not shown its advantage.

## Evidence and references

- [SOAP paper, sections 3–4](https://arxiv.org/html/2409.11321v2#S3).
- [Original SOAP implementation](https://github.com/nikhilvyas/SOAP/blob/main/soap.py).
- [Subspace basis updates preprint](https://arxiv.org/abs/2605.26327): abstract reviewed; full comparison pending.
- [Shared current roadmap](../manifold_opt/optimizer_research_roadmap.md).
- [Research journal](../manifold_opt/optimizer_research_journal.md).
- [Receiving-device results](../optimizer_replay_results/rtx5090_cu128_20260930/README.md).
- [Current diagnostic bundle](../optimizer_replay_results/soap_budget_20261001/).
