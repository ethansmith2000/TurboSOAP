## Stage 2 task

You answered the stage-1 brief (Document 1) independently; your answer is Document 2. You now also have the other
three stage-1 answers (Documents 3–5, anonymized as Reviewers X, Y and Z) and the stage-2 reveal (Document 6). The
reveal holds:
- the pooled candidate list;
- the open disputes;
- the moderator's numerical checks;
- the Part 1 roofline view;
- the Part 2 replay results on trained covariances;
- a note on tests still running.

**Web search is allowed and encouraged in this stage.** Use it to check prior work and implementation facts. Cite
each source with title, authors, year and a link, and say whether you **read** it (opened the paper or page) or saw it
**in search results only**. Mark anything you could not verify as [unverified]. Do not run code. Do not read any file
other than this prompt.

Please answer the following, in this order.

### 1. What changed for you (required)

Which of your own stage-1 claims do the moderator checks (R3) or the Part 2 results (R5) contradict, weaken or
confirm? Be specific: name your candidate IDs and numbers. Name the **single biggest change** to your view.

### 2. Prior work (web)

For each item below, say what the literature already establishes, and whether any pooled candidate is already known,
already shown to fail, or new:
- **Refresh frequency and cost in SOAP/Shampoo:**
  - the 2026 "FOAM" papers: one by Nam & Ahn on adaptive Shampoo refresh frequency, and one by Yu et al. on a cheaper
    SOAP;
  - Milligan et al. (2026), on storing statistics in basis coordinates;
  - any other work on warm-started or approximate eigenbasis tracking, or on adaptive refresh, for SOAP, Shampoo,
    KL-Shampoo or EShampoo.
- **CholeskyQR2 / shifted CholeskyQR3:**
  - shift choice and its effect on the computed Q;
  - behavior on rank-deficient or slightly indefinite inputs;
  - mixed-precision variants.
- **First-order (QR-tangent) and Jacobi-type tracking of eigenbases** under drift, including subspace or orthogonal
  iteration with Rayleigh–Ritz.
- **Batched or concurrent small QR on GPUs** (cuSOLVER, MAGMA): what is achievable at n = 768–3072 and batch 12.
- **Handling Adam's second moment when the basis changes** (PB8).

### 3. Ranking

Rank the pooled **A (speed)** candidates and, separately, the pooled **B (tracking)** candidates. Use every
candidate, or your top 8 per list with the rest marked "not ranked". Give one line of reasoning each. Take the Part 2
facts into account:
- trained factors are near-singular and some are slightly indefinite in BF16;
- unshifted Cholesky-QR fails on them;
- the current generator diverges without its cap, but the factor-ages that stay finite track better than QR10;
- the warm step barely moves weak directions;
- QR20 beats warm10/QR50 per refresh, but not per millisecond.

Where a candidate's rank depends on a test still running (R6), say what outcome would move it.

### 4. Your top 3 experiments

Name the three experiments, from either list, that you would run first. For each give:
1. **Exact spec.** A refresh function as batched PyTorch, at most about 40 lines. Its signature is
   `f(C_bf16[b,n,n], Q_bf16[b,n,n]) -> Q_new_bf16` (optionally also returning a column permutation), or a timing
   target for the microbenchmark. Include every constant (floors, caps, ridges, number of steps).
2. **Where it runs:**
   - the replay harness, which chains your function over the saved trained covariances (101 BF16 snapshots per width,
     d = 256–2048, refresh every 10 ages), with QR repairs at 50 if you say so;
   - the microbenchmark (compiled ms per factor at batch 1 and 12, n = 256–8192, plus the 109M event regions);
   - or, only if neither can test it, the smallest training run that could.
3. **The decision metric and threshold**, for example "late-window r_C^w interval within X of QR10 at all widths, at
   ≤ Y× the warm cost at n = 3072".
4. **Your probability** that it passes.

Use the harness's existing metrics where possible: r_C, r_C^w (diagonally normalized), top-1/8/n-8 overlap, e_Q, and
ms per factor. If you want a new metric, define it exactly.

### 5. Disputes (D1–D6) and your strongest disagreement

- Give your position on each of D1–D6 in one or two lines, updated with the evidence in R3 and R5.
- Then name your **single strongest disagreement with another reviewer** (required). Say which reviewer (X, Y or Z),
  the claim, why you disagree, and what measurement would settle it.

### 6. Errors in the moderator checks

If any check in R3 is wrong, mis-specified, or does not test what it claims, say so and why.

Write for an expert reader, in Markdown. Aim for 2,500–5,000 words. Prefer formulas and numbers to adjectives.
