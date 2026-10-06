# Strict momentum transport: implementation, cost and paired learning

Predeclared 2026-10-04 UTC. Add an opt-in ResearchSOAP transport_precision=highest
that changes only first-moment transport at an actual basis refresh, including
QR resets. Restore the incoming matmul setting even on exceptions. Default is
inherit; SOAP production defaults, covariance updates, basis arithmetic, gradient
projection, Adam updates and v handling remain unchanged. No inference_mode or
optimizer compilation in this round. CPU checks precede freeze and launch.

Stage A: a synthetic implementation/cost gate on shapes192x192,576x192,768x192,
192x768,768x3072,1536x6144. Identity incoming bases and rank-limited PSD covariance
from a seeded rectangular gradient; random positive v. This is an implementation
fixture and dimension-specific event cost, not representative learning evidence.
Compare QR inherit/highest and warm inherit/highest at caps .1/.2. Warm uses NS6.
All bases/v must exactly match between a paired inherited/strict transport call.
Strict candidates must preserve world m within1%, Gram<.05 and agree with strict
m transport at the actually selected basis to1e-6 in implied update. Retain full
strict-refresh differences as a separate1% diagnostic, including QR failures.
Require all strict CIFAR-shape cases to pass before Stage B. Large-shape failures
are retained and limit large-factor conclusions; they do not change CIFAR scope.

Time real research refresh implementation, both factors, full m transport and
v handling, including precision-switch dispatch overhead in the CUDA interval.
One initial call per variant, then3 rotated rounds,3 calls per CIFAR round and1
per large-shape round. Preserve all round values and their median. Exclude model,
covariance accumulation, per-step Adam, diagnostics and shadows. No sustained
trainer-speed inference from these short eager event samples.

Stage B: at each covariance beta .99/.999, six settings: QR20 inherit/highest,
warm20 cap.1 inherit/highest, warm20 cap.2 inherit/highest. Add final repeats of
QR20/.99/inherit and warm20/.99/cap.2/highest:14 arms total. Fixed seed139, matrix
and auxiliary peak LR .0005, Adam(.9,.999), both factors, original200-epoch CIFAR
schedule truncated at20epochs/3500 updates. NS6 warm, QR reset200. Evaluate at
875/1750/3500. Do not retune LR, expand bracket or select a favorable checkpoint.
Compare paired precision, cap and beta effects at the3500 endpoint, including
their interaction. Historical unchanged QR/warm cap.1 anchors must match.

Every initialization/refresh gets the finite/Gram<.05 guard. At ages20/180/200/
220/860/1740/3480, all24 matrices get a read-only selected-basis strict-m oracle,
world-m/geometry check and full strict-refresh comparison. Abort an opt-in arm
if selected-basis update error>=1e-6, world-m error>=1% or Gram>=.05. Preserve
ordinary high controls and all full-strict failures; the targeted selected-basis
gate is a separate implementation check, not a relaxation or replacement of
the previous full-refresh gate. QR resets remain explicit. Count actual strict
transports and record effective group settings. Diagnostics are outside timing.

Budget:480s cost gate,2100s training,2700s external bound, one lifetime GPU claim
under private supervisor with no queue/restart. 49,000 trainer updates plus
compile smoke; expect roughly20minutes depending on shared workload. Check5GiB
free; compact evidence budget20MiB. No weights, checkpoints or tensor snapshots;
in-memory observer references discarded after the probe. Retain source, configs,
seeds, logs, scalar metrics, plots, manifests and checksums. Preserve old campaigns
and caches. One seed and an early endpoint cannot establish convergence; factor
scope and .99/.999 effects stay conditional on this recipe and cadence.
