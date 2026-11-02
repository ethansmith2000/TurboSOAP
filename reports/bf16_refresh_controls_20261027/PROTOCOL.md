# Fixed full BF16 QR20 / QR50 controls

User-authorized bounded follow-up after the completed seed204 replication.
Two fresh controls, 2,000 calls each, sequentially on one shared GPU claim.
Only cadence changes from the frozen full BF16 method: QR20 uses frequency20
and repair20; QR50 uses frequency50 and repair50. Both have no warm events.
Production optimizer sources remain byte-identical to the pinned replication.

Same109,396,992-parameter OWT model, model seed204, exact cached train and
validation order, stock betas(.95,.95), covariance.95, full10,000-call LR horizon,
matrix peakLR.0015, auxiliary AdamW peakLR.0006/betas(.9,.999), decay.05,
clip1, epsilon1e-8, physical batch8/accumulation4. BF16 M/V/C/Q and matrix
arithmetic, FP32 master weights/accumulation/solver islands. No policy tuning.

Require exact initial weights, gradients, loss/norm and full1,024-row validation
CE against the verified seed204 run, with repeated initial backward. Full-panel
validation at0,50,100,200,500,1000,1500,2000; listed tensor preservation at
0/1000/2000. No model or optimizer tensor checkpoints.

The first call initializes matrix state and skips its matrix update. Through
call2000 QR20 therefore has1 cold,99QR,1900ordinary events; QR50 has1cold,
39QR,1960ordinary. Sample all96 factors/48 moments at seven actual QR refreshes
per control, four with coordinate residuals. Check eight stable ordinary/QR
graphs at101,201,501,1001,1501,2000; no warm graph is expected. Finite-state
and nonnegative-V checks in diagnostics, final model/matrix/auxiliary finite
check, no graph breaks and24GiB allocation cap. Finite M/Gram screen crossings
remain descriptive; no loss acceptance threshold is invented.

Compare to the cached fullBF16 seed204 warm10/QR50 first2,000 calls. Reuse
quality and timing records without rerunning warm training. Timing uses complete
shared100-call cycles, calls101..2000 (1,900 calls), including packing/casts/
auxiliary updates. Diagnostics/evaluation are outside timed steps but inside
execution budget. New control timings share one device; cached warm timings
are contextual across launches. Short-control loss cannot establish final10k
superiority. Report all validation curves and event counts.

Frozen runner benchmarks/bf16_refresh_controls.py; meaningful CPU checks cover
factory precision/recipe, actual optimizer cadence, unchanged warm behavior,
protocol and audit rejection of event/capture/source mismatches. Freeze code,
protocol, check receipt, input hashes and cached comparison before launch.
Managed soap-bf16-controls204, no autostart/restart/retry. Queue wait up to
7,200seconds is separate. A timeout inside gpu-claim bounds the entire claimed
execution to1,800seconds including interpreter imports, source/input verification,
parity checks, compilation, initialization, both controls and final auditing.
No successor is launched by the runner. Failure stops bounded automation.
