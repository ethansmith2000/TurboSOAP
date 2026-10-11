"""CPU-only preparation; a separate prospective device manifest is required to launch."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

try:
    from .legacy_cuda_probe import expected_config
except ImportError:
    from legacy_cuda_probe import expected_config


REPO=Path(__file__).resolve().parents[1]
FILES=['benchmarks/legacy_soap.py','benchmarks/audit_legacy_soap.py',
       'benchmarks/reference/soap_may_930a5a3.py','benchmarks/legacy_cuda_probe.py',
       'benchmarks/prepare_legacy_cuda_probe.py','tests/test_legacy_cuda_probe.py']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(bundle):
    bundle=Path(bundle)
    bundle.mkdir(parents=True,exist_ok=False)
    config=expected_config()
    (bundle/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for name in FILES:
        target=bundle/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/name,target)
    protected={str(p):sha(p) for p in REPO.rglob('*.py') if '.git' not in p.parts}
    results=REPO.parent/'optimizer_replay_results'
    for name in ['soap_historical_audit_20261008','soap_legacy_adapter_20261008',
                 'soap_fresh420_json_round_20261008','cifar_soap_fresh420_json_20261008']:
        for p in (results/name).rglob('*'):
            if p.is_file():protected[str(p)]=sha(p)
    for p in results.rglob('artifact_inventory.json'):protected[str(p)]=sha(p)
    (bundle/'protected_before.json').write_text(json.dumps(protected,indent=2)+'\n')
    (bundle/'TurboSOAP_preflight_git.txt').write_text(subprocess.check_output(
        ['git','status','--short'],cwd=REPO,text=True))
    manifest=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        config_sha256=sha(bundle/'config.json'),source_sha256={name:sha(bundle/'source'/name) for name in FILES},
        historical_commit='930a5a3a31a4bf74196eed358190c8f0bc71c454',
        phase='CPU_PREPARED_DEVICE_PENDING',gpu_selected=False,
        guide_sha256=sha(REPO.parent/'AGENTS.md'))
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (bundle/'STATUS.md').write_text('Prepared on CPU; device pending. No GPU run or queue.\n')
    (bundle/'PROTOCOL.md').write_text('''# Literal May CUDA support probe\n\nTwo shapes4x6 and8x8, seed172, 331 gradient-bearing calls each using literal constructor defaults. Each shape runs a hash-verified opt-in instance plus an independently imported original on identical prescribed CPU-generated gradients. No overrides/casts/normalizer fixes, dataset, model, compilation, checkpoint or model weights.\n\nQualify first-call skip, exact parameters/full optimizer state/stats after every observed call, 25 initial tensor QR refreshes through250, eight warm260–330, checked metrics260+310, variance sorting33 times and momentum round trips330 times. The call observer delegates unmodified and must restore. All buffers/parameters finite. BF16 trace is tested on the unchanged original path. Fail fast; retain partials and exceptions, no retry or second-shape execution after first failure. Gram diagnostics are unnormalized descriptive quantities; no physical-M/orthogonality threshold or learning/timing claim.\n\nScope: CUDA compatibility on one free claimed RTX5090 using this installed stack and tiny factors only. No H200 reproduction, large-factor feasibility, training, optimizer throughput, convergence, compiled-kernel or moment-transport qualification. Defaults are not proof of historical run config.\n\nBounds: internal120s from script entry (including imports), external150s from command launch, allocated CUDA memory2GiB, evidence5MiB including source/config/logs. No tensor/trace/checkpoint/weight artifacts. Temporary tensors only; verify owned allocated memory returns to initial baseline, no allocator cache clearing. No queue, automatic restart, retry or follow-on.\n\nPreparation does not select a GPU. Before launch, recheck claims/processes/disk and record index/UUID and frozen config/manifest hashes in a NEW launch_manifest.json. gpu-claim must set that exact visible UUID and claimed index. Use only the private research supervisor and a single new autostart=false/autorestart=false entry. Do not alter historical programs or other projects.\n''')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args();m=prepare(a.bundle.resolve());print(json.dumps(m))
