# TurboSOAP refresh panel (2026-11-20): plan and log

Requested by the main TurboSOAP session (on behalf of the user): a staged outside-model review panel to find
(A) ways to further accelerate the method and (B) ways to make the warm refresh match QR more closely (better
loss). Output: a ranked, testable candidate list whose specs fit the refresh-study harness
(`benchmarks/refresh_study_common.py`, `refresh_study_replay.py`, `refresh_study_microbench.py`).

Conventions follow the user's earlier panel, `/workspace/cautious-optimizers/research/reviews/PLAN.md`:
- stage briefs saved verbatim;
- answers saved verbatim as `<stage>_<model>.md`, with a short provenance header;
- raw event logs in `transcripts/`;
- tool calls checked and logged after every run;
- label randomization with a recorded seed when reviewers see each other's text;
- syntheses keep disagreements visible, mark literature claims "verify", and check algebra and cost claims numerically in `checks/`.

**Scope rules for this panel.** Files are created only under `research/reviews/panel_20261120/`. No code, report
or `paper/` file is edited, nothing is committed, no branch is switched. GPU checks run only via
`gpu-claim run --owner turbosoap-panel` and stay tiny (other projects' training jobs share the machine).
Candidates are not implemented in the harness here; that goes back through the main session.

## Reviewers

| Reviewer | How it runs | Status |
|---|---|---|
| Fable 5.2 | `Agent` subagent, `model: "fable"`. The brief goes behind a wrapper: "Your final message must be your complete answer … Do not use any tools". The only allowed tool use is a Read of its prompt file, when the prompt is long | see log |
| Gemini 4 Neon | `/root/.local/bin/agy -p --model gemini-4-neon-xhigh --output-format stream-json` (Antigravity CLI 1.5.13), run from an empty directory | first panel use; smoke-tested first |
| Gemini 4 Carbon | same CLI, `--model gemini-4-carbon-high` | used in the cautious-optimizers G stages |
| GPT-6.2 astra | `codex exec -m gpt-6.2-astra -s read-only --skip-git-repo-check -c web_search=disabled -c model_reasoning_effort=high --json` (codex-cli 0.167.2), brief on stdin, run from an empty directory | used in the cautious-optimizers G stages |

agy cannot disable `search_web`. In web-off stages the brief forbids tools, and the stream-json is checked for
`step_type: "tool"` events. A prompt too long for one argument goes as `prompt.md` in the empty working directory,
and the reviewer is told to read only that file.

Independence rules (P1): no reviewer sees another's text or the moderator priors, each runs in a fresh session,
no web, no files except the reviewer's own prompt file. Before anything leaves the machine the brief is checked
for personal details (author name, email, private links).

## Stages

### P1: blind brainstorm (web off)

- **Input:** `P1_BRIEF.md`, identical for all four reviewers. It contains:
  - the method and Algorithm 1;
  - the code of `warm_basis`, `qr_basis`, `transport_moment`, `covariance_ema` and `matrix_step`, plus the compile setup;
  - the settings and the measured costs and losses;
  - the known constraints;
  - two separated questions: (A) faster, (B) closer to QR, plus pitfalls.
- **Output:** `P1_<model>.md` ×4 (verbatim), event logs in `transcripts/`, then `P1_SYNTHESIS.md`:
  - a deduplicated candidate table;
  - agreements and disputes;
  - a comparison with `MODERATOR_PRIORS.md`;
  - numerical checks of claimed identities and costs in `checks/`.

### P2: reveal (web on)

Starts only after refresh-study Part 2 has finished: `replay.json`, `ablation.json` and the Part 2 section of
`reports/refresh_study_20261120/REPORT.md` must all exist. Each reviewer gets:
- its own P1 answer;
- the other three P1 answers, anonymized with seeded labels;
- the Part 2 results.

Web search is allowed, to check prior work (the two 2026 "FOAM" papers, Milligan et al. 2026). Each reviewer ranks
the pooled candidates and names its top three experiments. Then `P2_SYNTHESIS.md` and `CANDIDATES.md` follow: at
most 8 candidates, each with a harness-ready spec and the metric that decides it.

## Log

| Date | Stage | What | Where |
|---|---|---|---|
| 2026-11-20 | — | Folder and plan created. Moderator priors written before any brief (items 1–11 from the main session; M1–M3 from the moderator's code reading) | MODERATOR_PRIORS.md |
| 2026-11-20 | — | Smoke tests passed, all from empty scratch directories:<br>• `gemini-4-neon-xhigh` (first use; init event confirms the model id, 0 tool steps);<br>• `gemini-4-carbon-high` (0 tool steps);<br>• `gpt-6.2-astra` (codex 0.167.2; empty stderr);<br>• Fable subagent (0 tool uses).<br>Codex `memories` feature is off (`codex features list`), and its memories store has 0 entries | — |
| 2026-11-20 | P1 | Brief written: P1_BRIEF.md, 3,474 words, 22.7 KB.<br>• Contains: the method in plain maths, the verbatim code (`mm32`, `covariance_ema`, `warm_basis`, `qr_basis`, `transport_moment`, `matrix_step`, `group_step`, event selection, compile call), the settings, the event timings with units, the Part 1 cost table, the loss tables with caveats, r_C, the constraints, a description of the harness, and the A/B questions in a fixed candidate format.<br>• Kept out: file paths, repo and method names, the refresh-study ablation grid and Part 2, the moderator priors.<br>• Grep for author name, email, URLs and paths: clean.<br>• Size is under 128 KB, so agy gets the brief inline as `-p` | P1_BRIEF.md |
| 2026-11-20 | P1 | Sent to all four in parallel:<br>• codex: brief on stdin, `-o` final message, `--json` events;<br>• agy Neon and Carbon: brief inline as `-p`, stream-json;<br>• Fable: a fresh `general-purpose` subagent with `model: fable`. Its prompt (`prompts/P1_fable.md`: wrapper plus the verbatim brief) was copied alone into an empty scratch directory; it was allowed exactly one tool, a Read of that file, and asked to begin with a provenance note.<br>All reviewer working directories were empty | prompts/P1_fable.md |
| 2026-11-20 | P1 | All four answers saved verbatim, each with a provenance header:<br>• GPT-6.2-astra: 5,337 words, 7 min, no tool calls (one `agent_message` item);<br>• Fable: 4,121 words, 13.5 min, one Read of its own prompt file. It saw the instance CLAUDE.md and the one-line memory index, and reports that three index lines shaped its pitfalls;<br>• Gemini 4 Neon: 9,902 words, 20 min, 0 tool steps;<br>• Gemini 4 Carbon: 7,731 words, 23 min, 0 tool steps.<br>Fable's tool count was tallied from its transcript by script. Raw logs are in `transcripts/`; `checks/collect_answers.py` does the bookkeeping | P1_*.md |
| 2026-11-20 | P1 | Moderator checks, all CPU (`checks/*.py` + `.out`); no GPU was used, all GPUs being claimed by other projects' jobs:<br>• algebra: the QR first-order tangent, the NS and retraction identities, the Procrustes reflection, transport FLOP counts;<br>• the cap and half step;<br>• generator comparison, one-shot and chained, on the paper's Fig. 2 toy;<br>• Cholesky-QR numerics at n ≤ 4096 in FP32 vs emulated TF32;<br>• V rules after a refresh;<br>• the BF16 EMA dead-band;<br>• basis-frame EMA (prior 1).<br>`set_float32_matmul_precision('high')` was confirmed to set `allow_tf32=True` | checks/ |
| 2026-11-20 | P1 | Synthesis written (§9 is an addendum written after Part 2 appeared; §1–§8 are blind to it): 50 reviewer candidates pooled into 14 speed (PA1–PA14) and 12 tracking (PB1–PB12), with 6 disputes, the moderator checks, the comparison with the priors, 8 reviewer errors or overclaims, and a provisional ranking | P1_SYNTHESIS.md |
| 2026-11-20 | — | P2 gate met at 03:32–03:35: `replay.json`, `ablation.json` and a complete Part 2 section of REPORT.md now exist. (`microbench_ext.log` shows a further extension queued at 03:31; it is not part of the gate.) Parts of Part 2 bear directly on P1:<br>• unshifted Cholesky-QR10 is non-finite on trained factors, some of which are slightly indefinite in BF16;<br>• removing the Frobenius cap makes the current generator diverge;<br>• one NS correction gives the same quality at −22% time;<br>• two warm steps per refresh: −0.012 r_C, −0.019 r_C^w | — |
| 2026-11-20 | P1 report | P1 report sent to the main session. The main session relayed the user's choice, "P2 + early PB1 test": run P2 and **show reviewers the moderator CPU checks**, so they don't rank on gains the checks refuted. It also asked to add the Part 1 roofline facts, the note that Part 1b (n = 16384, shifted CholeskyQR3) is queued and not yet run, and the note that the refresh study is replay-testing PB1, a relative-gap damping variant and factor-1 Jacobi in parallel (reviewers must not assume an outcome) | — |
| 2026-11-20 | P2 | Reveal written as a template with reviewer placeholders: P2_REVEAL_TEMPLATE.md. It contains:<br>• R1 the pooled candidates, with proposers shown as labels;<br>• R2 disputes D1–D6;<br>• R3 the moderator checks M1–M7: what was checked, the numbers, which claims they contradict;<br>• R4 the Part 1 roofline, with warm and QR TFLOP/s recomputed from microbench.json;<br>• R5 the Part 2 REPORT section (paths, repo names and file names removed) plus the full ablation table from tables.md;<br>• R6 the parallel tests and the queued Part 1b.<br>Task: P2_BRIEF.md (what changed, prior work via web, ranking, top-3 experiments with exact specs, disputes plus the strongest disagreement, errors in the checks). | P2_REVEAL_TEMPLATE.md, P2_BRIEF.md |
| 2026-11-20 | P2 | Bundles built by `checks/build_p2.py`: `prompts/P2_<model>.md`, about 244 KB and 36.9k words each. Each holds Document 1 (the P1 brief), Document 2 (own P1 answer), Documents 3–5 (the others as Reviewers X, Y, Z), Document 6 (the reveal), then the task.<br>• Labels drawn with seed 20261120 (`prompts/P2_assignment.json`).<br>• Built from the raw responses in `transcripts/`, without provenance headers. Fable's own provenance section (which quoted memory-index lines) was stripped.<br>• Asserted free of model names, author name or email, `/workspace` paths and claude.ai links. | prompts/P2_*.md |
| 2026-11-20 | P2 | Web smoke tests:<br>• codex: `--search` must come before `exec` (`codex --search exec …`); 1 `web_search` item;<br>• agy Neon with `--sandbox`: `search_web` ×1 and `read_file` of `prompt.md` ×1 | — |
| 2026-11-20 | P2 | Sent to all four in parallel, **web on**, each from an empty directory:<br>• codex: `codex --search exec -m gpt-6.2-astra -s read-only --skip-git-repo-check -c model_reasoning_effort=high --json`, bundle on stdin;<br>• agy Neon and Carbon: `--sandbox`, bundle as `prompt.md` in the working dir (over the 128 KB argument limit), told to read only that file;<br>• Fable: a fresh `general-purpose` subagent (`model: fable`), allowed Read (its prompt file only), WebSearch and WebFetch | — |
| 2026-11-20 | P2 (moderator) | While the reviewers ran: CPU check of Cholesky-QR variants on the **trained** BF16 snapshots, read-only (`checks/trained_spectra_cholqr.py`; outputs `trained_spectra_cholqr_d256.out`, `trained_early_fp32.out`, `trained_early_tf32.out`). Widths d = 256 and 512, ages 10/20/50 (plus 500/1000).<br>• **Early factors are indefinite in BF16:** λ_min/λ_max ≈ −1e-4 to −4e-4, with 2–21% of eigenvalues negative. Late factors at d = 256 are PSD and well-conditioned.<br>• **Unshifted CholeskyQR2** fails 39/48 early factor-ages in FP32 (40/48 with emulated TF32). So the Part 2 failures come from indefiniteness and conditioning, not mainly TF32.<br>• **Gram-shifted CholeskyQR3 with s = 1e-6·‖Z‖₂²** (no ridge on C, so the target is unchanged) fails 1/48 in FP32 and 2/64 in TF32, and otherwise matches Householder to a column distance ≤ 2e-6.<br>• **Fukaya's theoretical shift** s = 11(n² + n(n+1))·u·‖Z‖² (as in the queued Part 1b script) fails 25–30/48: for square n×n it is about 0.8‖Z‖² at n = 768.<br>• **A ridge on C** changes the target (column distance up to 0.03 at 1e-6, 0.4 at 1e-4) and still fails 1–2/48 | checks/ |
| 2026-11-20 | P2 | All four answers saved verbatim, each with a provenance header and the labels it saw:<br>• GPT-6.2-astra: 5,002 words, 15 min, 52 web searches, no commands;<br>• Fable: 4,977 words, 12 min, 9 Reads of its own prompt, 14 WebSearch, 9 WebFetch, 1 ToolSearch, and **1 advisor call, outside the allowed tools** (disclosed by Fable itself; noted in P2_fable.md);<br>• Gemini 4 Neon: 7,032 words, 16 min, 9 `read_file` (prompt only), 49 `search_web`;<br>• Gemini 4 Carbon: 6,590 words, 20 min, 11 `read_file` (prompt only), 43 `search_web`.<br>Event logs are in `transcripts/` | P2_*.md |
| 2026-11-20 | P2 (moderator) | Citations spot-checked against arXiv/OpenAlex (research-desk reference lookup): 10 verified. Neon's arXiv:2604.14816 resolves to an unrelated paper; the intended SOAP-convergence paper is arXiv:2604.21616 | P2_SYNTHESIS.md §4 |
| 2026-11-20 | P2 (moderator) | Trained generator norms, CPU (`checks/trained_cap_norms.py`): the QR-tangent generator has ‖A‖₂ 1–116 (production 0.1–8.6), with a Frobenius/spectral ratio of only 1.1–3.5. So any bounded first-order step is scaled 2–200×.<br>Spec smoke test (`checks/candidate_specs.py`, `.out`) on trained snapshots with bases 50 ages stale:<br>• Gram-shift CholQR3 = Householder exactly;<br>• head panel (k = n/8) within 0.002 of QR's r_C;<br>• bounded tangents (Frobenius, Schatten-4, spectral, row+spectral) barely move | checks/ |
| 2026-11-20 | P2 | P2_SYNTHESIS.md written: what changed for the reviewers, ranking tables with mean ranks, the top-3 experiments, disputes plus each reviewer's strongest disagreement, prior work with the citation checks, the reviewers' critiques of M1–M7 (accepted), the new trained checks, errors caught, and the moderator ranking. CANDIDATES.md written: 8 harness-ready candidates C1–C8 with spec code (C1, C4, C5 smoke-tested), integration points and decision thresholds | P2_SYNTHESIS.md, CANDIDATES.md |
| 2026-11-20 | P2 (provenance) | **Method-name leak.** The sanitized Part 2 text in all four bundles kept one sentence naming the method ("closed loop TurboSOAP would see different C"); `build_p2.py`'s assert list did not include the name. None of the reviewers' web queries contains it (codex 52, Fable 14, Neon 49, Carbon 43 searches checked). Carbon repeats the name once in its answer | P2_SYNTHESIS.md |
| 2026-11-20 | P2 (addendum) | The refresh study's Part 2b (PB1, relative-gap and factor-1 Jacobi replay, 04:44) and Part 1b (n ≤ 16384, Fukaya-shift CholQR3, 04:40) were already written when the moderator's §6.2 prediction was drafted, but had not been read. So the prediction is labelled as not pre-registered.<br>• Part 2b: the best bound (spectral 0.5) closes 23–40% of the r_C^w gap; the reviewers' ≥ 50–60% gates are not met.<br>• Part 1b: CholQR3 prices C1 at ≈ 198 ms per 109M event (vs ≈ 586) and C3 at ≈ 20.3 ms/call (vs 15.94 today and ≈ 30.3 for QR20), so the C3 cost gate was corrected. The Fukaya shift fails at n ≥ 6144 on synthetic inputs too.<br>• Smoke test rerun with the whitened ρ_rms and a 10-age-stale basis: the head panel matches QR's first k columns and r_C but not the whitened tail.<br>Written up as P2_SYNTHESIS §9 and in CANDIDATES C1/C3/C4/C5 | P2_SYNTHESIS.md §9, CANDIDATES.md |
| 2026-11-20 | Correction | The main session asked for a reconciliation of C1/C3 with refresh-study's final Part 1b: IEEE-Gram CholQR3 ≈ 320 ms per 109M event; slower than Householder from n ≈ 4096 at batch 12; its later passes fail on 11/16 trained age-10 factors.<br>• Check `checks/reconcile_cholqr3.py` (CPU IEEE FP32, the same snapshots, Q and failure criterion). **The disagreement is the shift size.** Fukaya's 11(2n² + n)u‖X‖² fails 9/16 there and 34/96 overall; 1e-6·‖X‖² fails 0/16 and 4/96; 1e-7 fails 0/16 and 1/96. All small-shift failures are at ages ≤ 20, mostly on the n ≥ 2048 left factors at d = 1024 (last pivot of pass 2, numerically rank deficient), so the Householder fallback is required. Where it succeeds, accuracy is ≤ 1.5e-3 rad against FP64 Householder. "1 pass + NS" is rejected (0.82 rad).<br>• Pricing corrected to IEEE times: C1 saves ≈ 5.5 ms/call (not 7.8), only for n ≤ 3072. C3 costs ≈ 32.5 ms/call (not 20.3). C1 every 20 ages, at ≈ 16.3 ms/call, is the new cost-neutral option.<br>• The superseded "1.3–1.9× per ms vs QR20" was corrected to 2.9 / 2.3 / 0.92 / 0.35× (≈ 1.4× at 109M sizes). All four P2 reviewers had seen the old figure | CANDIDATES.md (dated blocks), P1_SYNTHESIS §9.4, P2_SYNTHESIS §10 |
