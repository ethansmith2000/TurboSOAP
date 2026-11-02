# Full BF16 replication and two-seed evidence

Seed204 completed10,201 fresh calls (201 stock replay,10,000 fullBF16) in
57.11minutes; supervisor exit0. Repeated frozen-source, input and support hashes,
reference replay, full event/observation/evaluation/capture audits pass. The
seed205 frozen-source/input/support and both arm audits were also repeated.
See monitor_review.json for scalar evidence and per-seed provenance in manifests.

At10k calls seed204 stock/fullBF16 CE is3.411151532/3.422388958, delta+.011237426,
or1.1301% higher perplexity. Seed205 is3.409271536/3.419353740, delta+.010082204,
or1.0133% higher perplexity. Mean paired CE delta+.010659815. This is consistent
across two model-initialization seeds with a fixed data order; it establishes
neither exact matching nor statistical equivalence.

FullBF16 matrix state is1,245,708,320bytes (~1.160GiB), half stock's2.320GiB.
Seed204 matched calls101..200 take380.740ms stock versus314.638ms fullBF16,
17.36% less complete-step time. Candidate full steady mean314.058ms has a
different window; cached full stock time is contextual. Seed205's newly paired
9,900-call comparison saved18.66%. Do not average these unlike timing windows
as a single precise benchmark or attribute all speed to warm tracking alone.

All sampled state remains finite and compiled captures stay at12 graphs with
no graph breaks. Seed204 max actual-M error.9924%; seed2051.0356%. Normalized
Gram maxima~.002366/.002367. Preserve all raw historical flags: seed205 crossed
the old1% screen and both fail the original absolute Gram screen. Under the
user's latest steering these finite numerical observations are diagnostics,
not loss-quality acceptance thresholds or a reason to discard the method.

The approved short fullBF16 QR20/QR50 controls now test cadence context against
the cached seed204 warm prefix. No new full training run or method redesign.
