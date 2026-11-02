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

> **Current anchor and status — 2026-10-27:** Stock betas(.95,.95), covariance.95 remain primary; finite moment/Gram errors are descriptive. The [seed204 replication](BF16_REPLICATION_20261027.md) is supervised independently. At the user's explicit request, a [bounded T3 automation](EXPERIMENT_AUTOMATION.md) now checks every10 minutes, reviews completion and can launch the remaining fixed QR20/QR50 controls and small QR profile. One GPU at a time; at most45 additional GPU execution minutes; no tuning/retries/new full runs. It reports here and self-disables when complete or requiring review, with an eight-hour expiry. The [ledger](automation/remaining_experiments_20261027.json) and live supervisor/progress state are authoritative. Earlier no-successor/ACTIVE labels describe historical scopes.

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

> **User-directed scope and representative benchmark decision — 2026-10-20:** Keep SOAP's parameter-update/second-moment rules fixed; defer the proposed update-mechanism audit/redesign. Practical objective: useful complete wall-time-to-loss improvement on a recognizable, source-backed SOAP training recipe. Current200 continues unchanged; review its completed matched-N10 result first. Next bounded preparation should select one public reference recipe and estimate model/data/token-batch/training-horizon/QR-cost and hardware feasibility, then compare native QR and warm at the same N with identical remaining math; occasional QR remains optional. Include a cheap sparse-QR contextual control and only the baseline calibration needed for a credible reference. Do not launch another cadence/precision grid or extrapolate width benefits without evidence. Official [SOAP example](https://github.com/nikhilvyas/SOAP) uses betas(.95,.95), covariance tied to beta2 by shampoo_beta=-1, and N10. [Original paper](https://arxiv.org/html/2409.11321v2) uses C4,seq1024,widths1024/1408 for main360m/660m non-embedding models,token batches~2m(N10) or256k(N80); its smaller-batch SOAP beta2=.99 result does not imply one universal beta. [Author's124M example](https://github.com/nikhilvyas/modded-nanogpt-SOAP) supplies a smaller public recipe on FineWeb. Current768x12/109M model is a useful engineering case, but betas(.9,.999),covariance beta.99,~8k-token batch and286M-token horizon differ materially. Step interval is not portable across batch sizes: ourN200 is~1.64M input tokens between refreshes, versus~20M in the original2m/N10 setting; these are not equivalent optimization dynamics. Larger token batches can also amortize QR and shrink warm's end-to-end saving, so do not assume the reference setup will favor warm. Judge complete time-to-quality,not refresh-only speed; choose practical success criteria before any future learning launch. If representative bounded cases show no useful tradeoff, narrow the claim to demonstrated basis-tracking/kernel results and stop expanding SOAP-specific experiments. Source/feasibility selection pending; no new recipe,hyperparameter,update rule,default,job or running protocol changed here.

> **Active matched-N warm comparison — 2026-10-20:** [Trial200](../optimizer_replay_results/soap_deep_matched_warm_trial200_20261020/ACTIVE_STATUS.md) launched ONCE on lifetime-claimed GPU7/private `manifold_deep_matched_warm_trial200`,trainerPID1459996. Same seed193,width768/depth12/109.397M model,full-OWT batches/panels,35000 primary updates/286.44M targets as completed/audited QR10 control199 (not rerun). Warm every10 ages; QR replaces every fifth refresh (repair50);699QR/2800warm per matrix. K5 is an initial optional repair policy, not a necessity/optimum; fixed/adaptive/no repair remain open. FastSOAP warm math/source unchanged from197/198,FP32 stored state,caller high,BF16 operands/directFP32 outputs,two corrections; native control retains highest momentum transport, so compare full implementations without isolated basis attribution. 302sources(299 unchanged+3 additive),311inputs,27working+27frozenCPU/fullmodel initial-reset/pins PASS;537oldPython/8956priorfiles preserved. GPU startup PASS: exact QR10 initial backward/panels,75disposable calls/96actual-M/192own reports/8compilegraphs,exact75-call full-state/scalar/RNG prefix,CPU snapshot released,and first260 read-only observation/restoration/workspaces/both banks/correct N10 metadata pass. Prospective full loss curve/endpoint/perplexity and complete cost/time; historical+.01 panel screen report-only, no automatic promotion. 75disposable+35000primary calls,480actual-M/192own factor reports/1760guards/7eval+observer/14banks planned. 140/141min,24GiB GPU/12GiB RSS,RAM-only reset/diagnostics,no weights/checkpoints/retry/queue. Monitor EXISTING200,never duplicate/restart. Earlier active199/next200 statements below historical.

> **Completed QR10 reference — 2026-10-20:** [Trial199 review](../optimizer_replay_results/soap_deep_qr10_trial199_20261019/REPORT.md) and independent audit PASS. 35000 primary/35075 total calls,133.003min,18.277GiB peak. Final CE3.435068 vs QR20/1933.437160 (−.002092 CE,~.209% lower perplexity); small one-seed difference, QR10 slightly worse at10k. Clean optimizer77.379 vs47.821ms (+61.81%),iteration217.652 vs188.184ms (+15.66%). All1758guards/432actual-M/exact75prefix/7eval+observer/14banks/releases pass;308inputs/299sources/534existingPython/8613priorfiles/11refs/80stats/dirtyHEAD-status/originaldocs preserved. Trainer1444357 dead/GPU7 free/private manager off. QR10 is now the control for matched-N warm10; historical+.01 QR20 screen report-only, not a universal quality limit. Prepare a separate warm10 candidate, initially repair50 (optional starting choice; fixed/adaptive/no repair remain open), then compare full loss curves/endpoints and complete cost/time. No warm10 result or automatic promotion. Earlier active199 statements below are historical.

> **Design clarification and next bounded task — 2026-10-19:** The user's N/K/J proposal is a loose research sketch, not a requirement for three configuration controls. Match QR and warm refresh opportunities; occasional QR is optional, not presumed necessary. Fixed schedules, error-triggered repairs or no repairs are all candidates. K5 is an initial test choice, not an optimum or established need. [Trial199 QR10 control](../optimizer_replay_results/soap_deep_qr10_trial199_20261019/ACTIVE_STATUS.md) is the first new bounded run; warm10 (initially considering repair50) follows after review and separate freezing. Keep comparisons interpretable without requiring a broad controller refactor or an N/K/J/threshold grid. The detailed choices below illustrate the design space rather than impose a mandatory API.

# SOAP refresh policy: matched intervals, repairs and monitoring

User-directed research framing, 2026-10-19. This governs future comparisons;
all completed protocols, measurements and screening labels remain unchanged.

## Primary question

At the same interval between basis updates, can a warm method deliver useful
SOAP training quality with less complete cost than QR? A warm method may use
occasional QR repairs. The repair policy and the cost of deciding to repair
are part of that method.

Interpret a matched refresh interval as the same number of training updates
between refresh opportunities. This does not mean equal wall-clock time per
refresh: that latency difference is precisely what we measure. Also evaluate
held-out progress against actual elapsed time and report matched-token loss.

QR200 hold is a secondary sparse-refresh ablation. Its cheaper schedule helps
measure the marginal benefit of intervening warm updates. Outperforming hold
is not a prerequisite for demonstrating useful performance against frequent
QR, and the hold result must not displace the primary matched-interval question.
Different histories also prevent treating measured costs as a mathematical
superset identity, although adding intervening warm work predictably costs more.

## Three independent policy choices

- `refresh_interval=N`: every N optimizer-age steps, update the basis using QR
  or the selected warm method. Covariance, Adam statistics and parameters
  continue updating every training step.
- `qr_every_k_refreshes=K`: for a warm method, use QR instead of the warm
  method at every Kth refresh opportunity. The repair interval in training
  steps is `N*K`. There is no discarded warm computation on a known scheduled
  QR repair. A separately qualified pure-warm policy could disable repairs.
- `check_every_j_refreshes=J`: evaluate a declared diagnostic at every Jth
  refresh opportunity, possibly triggering QR. Disabling monitoring is an
  explicit policy; unchecked intervals should incur no monitoring reductions,
  host scalar extraction, synchronization or precision switches.

For a fixed periodic policy, the existing API already expresses this as:

```python
precondition_frequency = N
hard_reset_interval = 0 if K is None else N * K
```

The basis refresh counter is one-based; initialization is handled separately:

```python
if optimizer_age % N == 0:
    refresh_index = optimizer_age // N
    if method == "qr" or (K is not None and refresh_index % K == 0):
        refresh_with_qr()
    else:
        refresh_with_warm_method()
```

These are policy sketches, not new implementations or optimizer defaults.
K counts refresh opportunities, not training steps. N is held fixed between
QR and warm arms within the primary comparison. K and J are separate choices.

## Where the completed comparison fits

Trial198 already had `N=20, K=10`: nine warm refreshes followed by one QR
repair per 200 optimizer-age steps. Native193 had `N=20` with QR at every
opportunity. Those two arms therefore already match the refresh interval.
Hold195 changed that interval to 200 and performed no intervening basis motion.

Trial198's speed/loss comparison against native193 remains directly relevant:
12.29% lower measured iteration cost with endpoint +0.005709 CE in one matched
trajectory. Its comparison against hold195 is additional evidence, not the
primary acceptance objective. Precision and implementation changes were
combined, so it does not isolate those effects or measure original J1's deeper
learning trajectory.

## Adaptive repair requires a specified diagnostic

An orthogonality error tests whether the basis remains numerically orthogonal.
It does not establish that it tracks useful covariance directions. An alignment
proxy, such as relative off-diagonal energy of the rotated covariance, addresses
a different question and still needs a connection to useful training behavior.
Neither small diagnostic error nor accurate momentum transport proves loss
equivalence.

Before implementing an adaptive policy, declare:

- Whether the check examines the old basis or a proposed warm basis, and
  whether existing warm intermediates can be reused accurately.
- Whether one factor, a whole matrix parameter or the whole optimizer repairs
  when a check fails, and whether the repair counter follows a fixed global
  schedule or counts refreshes since the last QR.
- The metric, threshold, J, response to nonfinite values, and any maximum
  number of intervals between repairs.

A check on the old basis can select QR before computing a warm candidate.
A check on a proposed warm basis pays for that candidate even when it then
falls back to QR. Reusing the last retraction intermediate is not automatically
the same as checking the final basis. A reference QR used to measure error is
also real work, not a free oracle.

Required production checks and fallbacks belong in complete optimizer and
iteration timing. Separate read-only research observations may be excluded
from clean intervals if disclosed, while remaining in all-in elapsed time.
Prefer checks inside an explicitly compiled monitored refresh over a Python
controller added to every ordinary step. No adaptive threshold is selected or
qualified by this document.

## Next bounded comparison

The proposed first pair uses `N=10` in both arms:

1. QR at every refresh opportunity.
2. The optimized J1-derived warm method, with QR replacing every fifth refresh
   (`K=5`, so QR repairs occur every 50 optimizer-age steps).

The warm schedule is warm at 10/20/30/40, QR at 50, then repeat. This is the
user's suggested outer-refresh structure; K=5 is a starting point rather than
a measured optimum. Keep the deeper model, data and
ordinary optimizer recipe controlled. A new QR10 learning control is needed;
the completed QR20 control cannot substitute for that matched schedule. Do not
rerun completed QR20/QR200 experiments to recover context.

First qualify and time the policy, then freeze the bounded learning recipe,
quality objective, seed/order, source, resource and artifact limits before a
GPU launch. Compare held-out loss versus updates and complete measured time;
record refresh, repair, transport, casts, compilation and checking costs. State
precision differences explicitly; a pure kernel attribution requires matched
precision as well. Retain the old +0.01-at-every-panel screen as historical
evidence rather than silently making it a universal future objective.

After the fixed K comparison, use a bounded read-only diagnostic to decide
whether a monitored policy with a declared J is worth a separate prospective
learning test. Avoid a simultaneous N/K/J/threshold/method sweep. This update
changes research framing only; no new GPU job, controller, checkpoint or
optimizer default is introduced.

[Trial198 readout](../optimizer_replay_results/soap_deep_fast_trial198_20261019/REPORT.md),
[FastSOAP implementation](fast_soap.py), [research roadmap](../manifold_opt/optimizer_research_roadmap.md).
