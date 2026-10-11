"""Freeze full-horizon QR/Fro qualification of the established co-location."""
import json
from pathlib import Path
import shutil
import sys
from prepare_cifar_benchmark import REPO,sha
import prepare_cifar_pinned_timing

def main():
    b=Path(sys.argv[1]).resolve();assert not b.exists()
    prior=Path('/workspace/optimizer_replay_results/cifar_soap_dispatch_20261006')
    s=json.loads((prior/'summary.json').read_text());assert s['local_repeat_qualified'] and s['material_placement_effect']
    prepare_cifar_pinned_timing.main()
    c=json.loads((b/'config.json').read_text());c['memory_screen']['timing_version']=4
    for arm in c['memory_screen']['arms']:arm['timing_affinity']=dict(main=116,autograd=116)
    for name in ['cifar_radius_learning_runner.py','dispatch_diagnostics.py','compact_dispatch_screen.py',
                 'prepare_colocated_timing.py','summarize_colocated_timing.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    protocol='''# Full3500 QR/Fro same-core timing qualification

Prospective2026-10-07. The six1000-update QR co-location diagnostic passed exact
numerical and unchanged repeat gates, showing35.69% lower model/backward events
and18.66% lower full training wall. This longer, two-policy qualification does
not claim a prior failed wall endpoint or uninstrumented throughput is restored.

Six3500-update arms QR/Fro/Fro/QR/QR/Fro, identical fresh model/RNG/optimizer/input,
seed139, covariance .99, Adam(.9,.999), both factors, strict highest M/high basis,
V carry/reindex, QR20 or Fro .5/cubic2 warm20/QR200, LR .0005 and original200-epoch
schedule, corrected cached CIFAR recipe/eager evaluation. Main and autograd both
CPU116 throughout primaries and replays. Record twelve builds and restore two
original masks. No other process, worker count, GPU clock, GC or policy changes.

Evaluate875/1750/3500, all arms stop3500, no wall-quality endpoints. All18
historical fixed observations INCLUDING full optimizer hashes match the V-study
carry QR/Fro controls; all prior probe fields match after stripping that study's
added variance fields. Within each method observations/probes/final optimizer
hashes exact. Initial model/data/RNG/split/base/trainer/core hashes match. Require
175 finite/geometry guards and168 targeted M samples per arm at prescribed
ages20/180/200/220/860/1740/3480: selected-basis error<1e-6, physical M<1%, Gram<.05.
Preserve full strict-reference QR/reset disagreements separately. Require stable
in-arm compiler counters and all140 chunk records/arm. Abort actual numerical
failures; preserve partial evidence, no tolerance adjustment or trimming.

All three repeats per method must meet unchanged<=3% full training-wall and
<=2% steady optimizer-event range/mean. First50 excluded only from event means.
Repeat timing is at this same-core instrumented scope. No cross-study/device
speed claim or synthetic timing/loss splice. Both QR/Fro gates mandatory.

Same observer as the successful1000 diagnostic: original host wall/thread/process
CPU/GC/context switches plus main user/system phase CPU, own-thread schedstat
runtime/runqueue-wait/ticks/lastCPU/masks and own-GPU samples every250 updates
(fourteen per arm). Affinity vectors deduplicated losslessly with a shared catalog;
all thread rows and counters retained, no dropped observations. Compaction is
offline after flush; included in stored observer overhead, outside primary wall.
Four rusage calls/update charged identically. Snapshot intervals include offline
guards/probes/hash/evaluation time; not phase-specific. Observations can perturb
subsequent execution. No online feedback or claim of exact causal microarchitecture.

After all primaries, last QR/Fro states held in RAM for four-update plain/profile1/
profile2 triplets. Require exact model/fulloptimizer/RNG/loss records within each
triplet. Four compressed traces; corrected actual kernel/gpu_memcpy/gpu_memset
accounting, all three CPU phases, positive work and complete launch attribution.
Profiler windows are perturbed and excluded from repeat gates. Allocate3504
inputs correctly; no historical replay. Host and GPU spans overlap; events can
include dispatch gaps and are not kernel duration.

If all gates pass, qualify this full3500 QR/Fro instrumented setup for a separately
declared new budget comparison. No current wall-quality winner or permanent
affinity/optimizer defaults. If timing fails retain it and withhold wall claims.

One freeRTX5090 recorded before launch, exclusive claim without queue; private
supervisor, internal1200/external1260 seconds. About323GiB shared disk free.
Expected<30MiB compact evidence, hard100MiB. No saved weights/checkpoints/tensor
files. Two RAM states only for stated profile diagnostic; discard afterward.
Retain compact source/config/protocol/provenance, metrics/logs/masks/records and
compressed traces until explicit cleanup. Preserve dirty source and old evidence.
'''
    (b/'PROTOCOL.md').write_text(protocol);(b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    m=json.loads((b/'manifest.json').read_text());v=Path('/workspace/optimizer_replay_results/cifar_soap_variance_20261006')
    m.update(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),
        prerequisite=str(prior),prerequisite_summary_sha256=sha(prior/'summary.json'),
        historical_bundle=str(v),historical_screen_sha256=sha(v/'screen.json'))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')

if __name__=='__main__':main()
