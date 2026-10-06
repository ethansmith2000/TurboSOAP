# Same-state refresh efficiency gate

Predeclared 2026-10-04 UTC. Replay unchanged QR20 and NS6 warm20 at both LRs
.0005, covariance .99, Adam(.9,.999), seed139, first20 of200 scheduled epochs.
Finish with a QR20 repeat. All old training anchors must match exactly. This
recreates incoming states without retaining tensor snapshots; no candidate
changes the training trajectory. Covariance .999 and wider factors remain
separate follow-ups, not inferred from this test.

At matrix ages20/180/200/220/860/1740/3480 observe all24 matrices,48 factors.
Keep actual warm and scheduled QR-reset events separate. Five alternatives:
QR with permutation-only v, one NS6 warm step/cap.1, two NS6 steps/cap.1 with
one final m transport, the same two steps with sequential m transports, and
one NS6 step/cap.2. Warm carries v. All other basis settings remain unchanged.
Both inner steps use the same covariance. No combined cap/inner-step tuning.

Evaluate candidates using the training 'high' FP32 matmul setting, with strict
'highest' FP32 shadows of each SAME algorithm. Guard every candidate: finite
values, normalized Gram<.05, world-m preservation error<.01, world-m shadow
error<.01 and implied-update shadow error<.01. Retain rejected records; never
silently omit them from summaries. Lower covariance residual is a local tracking
diagnostic, not an exact optimizer oracle or a learning guarantee. Comparisons
between QR and warm also change v handling. Check that the local QR/warm control
exactly reproduces the actual refresh's Q/m/v.

Time complete per-matrix refresh kernels: covariance projection inside refresh,
basis construction/retraction, all first-moment transports, and QR v reindexing.
Exclude per-update covariance accumulation, Adam update, gradient projections,
guards, scores and shadow comparisons. This is refresh-event timing, not a full
optimizer-step or trainer throughput measurement. Use eager no_grad, training
'high' precision; no compile, BF16 gauge or inference_mode in this gate. Warm
each candidate once, then three rounds of three calls each with rotating order;
retain all CUDA-event round times and median. Compare residual reduction per
event-ms within identical incoming states and shape. No pooling across devices.

Advance no candidate automatically. Report numerical failures, residuals, event
costs, local reduction/cost ratios and final-versus-sequential update difference.
An improvement here only justifies a subsequent budget-matched training test.
No QR-free policy, online residual checks or adaptive fallback is introduced.

Budget:10,500 trainer updates plus compile smoke and offline probes;1200s
internal/1500s external, one lifetime GPU claim, private supervisor, no queue or
automatic restart. Require5GiB free. Save configs, frozen source, scalar metrics,
logs, plots and hashes within20MiB. No weights/checkpoints/tensor snapshots;
in-memory references are discarded immediately after their probe. Leave all
previous evidence and caches intact. Run CPU checks before freezing and launch.
