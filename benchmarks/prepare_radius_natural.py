"""Freeze the prospectively bounded actual-cycle radius screen."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    b = Path(sys.argv[1]).resolve()
    assert not (b/'source').exists()
    prior = Path('/workspace/optimizer_replay_results/cifar_soap_radius_attribution_20261006')
    r = json.loads((prior/'summary.json').read_text())
    assert r['status'] == 'completed'
    (b/'source/benchmarks').mkdir(parents=True)
    c = dict(widths=[768,1536], variants=['qr20','fro_cubic2_cap05','fro_cubic2_cap06','fro_cubic3_cap06'],
             betas=[.99,.999], updates=420, wall_budget_seconds=900, attribution=False,
             repeat_variants=['fro_cubic2_cap06','fro_cubic3_cap06'], seed=20261005,
             artifact_budget_bytes=20*2**20)
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (b/'PROTOCOL.md').write_text('''# Actual natural-cycle radius qualification

Prospective 2026-10-06. 16 matched cases plus two exact repeats at width1536,
covariance .999: QR20, skew-Frobenius .5/cubic2, .6/cubic2, .6/cubic3.
Same seeded MockBlock and four rotating 64-row activation/target batches as the
previous natural-cycle screen; seed20261005+width and data seed+64. Every case
initializes independently with identical model/data RNG. Three matrices, both
factors, widths768/1536 and largest factors3072/6144. BF16 model autocast is the
existing workload; optimizer covariance/state/bases FP32, high basis arithmetic,
highest first-moment transport, fixed V carry/reindex. Covariance accumulates at
every gradient-bearing call; cold init at age0 separately; 420 actual updates,
warm20 with QR200/400. No controller or extra online diagnostic.

Preserve Gram<.05 and per-refresh AND actual whole-history world-M error<.01.
Whole-history oracle independently accumulates physical gradient EMA every
update; evaluated at init and every20 updates for all three matrices. This is
sampled actual history, not an every-update bound or frozen-covariance chain.
Require finite complete420 cases, correct events/counts, two exact final model
and full optimizer-state hashes, every loss and every numerical probe. Matched
QR/.5 controls must exactly reproduce historical model/loss/diagnostic anchors;
new final optimizer hash is excluded only from historical comparison.
Finite/Gram guard aborts a case; M failures prevent qualification and remain saved.

Complete optimizer CUDA-event costs include covariance accumulation, Adam update,
both bases, strict M, V handling, casts, and observer dispatch. Timed step wall
includes model/backward and host dispatch. Cold init and offline diagnostics,
physical EMA oracle, model/state hashing, logging are excluded from update costs.
Save raw event costs and repeat variation; there is no equal-wall learning or
uninstrumented sustained-throughput claim, and no CIFAR/synthetic cost splice.
Own-trajectory residual/movement cannot establish causal common-state effects.

Prerequisite: completed radius common-state screen. Earlier failures and wall
endpoints stay preserved. Only qualified candidates may advance to actual paired
CIFAR learning with separately qualified CPU placement and interleaved controls.
Alternating3/2 remains an unlaunched fallback; no broad grid or default promotion.

One GPU3 exclusive claim, no queue, private manifold supervisor. Internal900s /
external960s. Initial324GiB free. Expected<10MiB, hard20MiB compact evidence.
No saved weights/checkpoints/tensor snapshots; state stays in RAM and is discarded.
Retain protocol/config/source hashes, metrics, logs, summaries and inventories
until explicit research cleanup. Existing source and historical evidence intact.
''')
    sources = ['soap.py','soap_reference.py'] + ['benchmarks/'+n for n in [
        'warm_retraction.py','movement_retraction.py','radius_natural.py',
        'retraction_diagnostic.py','benchmark_synthetic_scaling.py','cifar_tracking_probe.py',
        'prepare_radius_natural.py','summarize_radius_natural.py']]
    for rel in sources:
        dest=b/'source'/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/rel,dest)
    historical=Path('/workspace/optimizer_replay_results/soap_movement_natural_20261006/results.json')
    manifest=dict(source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),
        prerequisite=str(prior),prerequisite_summary_sha256=sha(prior/'summary.json'),
        historical_results=str(historical),historical_results_sha256=sha(historical),
        git_head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        evidence_budget_bytes=c['artifact_budget_bytes'])
    (b/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(b)

if __name__=='__main__': main()
