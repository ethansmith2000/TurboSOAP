"""Freeze a new library-only allocation control without initializing CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

try:
    from .cuda_library_ownership import config
except ImportError:
    from cuda_library_ownership import config


REPO=Path(__file__).resolve().parents[1]
FILES=['benchmarks/tensor_ownership.py','benchmarks/cuda_library_ownership.py',
       'benchmarks/prepare_cuda_library_ownership.py','tests/test_tensor_ownership.py']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(bundle):
    bundle.mkdir(parents=True,exist_ok=False)
    (bundle/'config.json').write_text(json.dumps(config(),indent=2)+'\n')
    for name in FILES:
        dest=bundle/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/name,dest)
    protected={str(p):sha(p) for p in REPO.rglob('*.py') if '.git' not in p.parts}
    results=REPO.parent/'optimizer_replay_results'
    for folder in ['soap_historical_audit_20261008','soap_legacy_adapter_20261008',
                   'soap_legacy_cuda_20261008','soap_fresh420_json_round_20261008',
                   'cifar_soap_fresh420_json_20261008']:
        for p in (results/folder).rglob('*'):
            if p.is_file():protected[str(p)]=sha(p)
    for p in results.rglob('artifact_inventory.json'):protected[str(p)]=sha(p)
    (bundle/'protected_before.json').write_text(json.dumps(protected,indent=2)+'\n')
    manifest=dict(config_sha256=sha(bundle/'config.json'),source_sha256={name:sha(bundle/'source'/name) for name in FILES},
                  guide_sha256=sha(REPO.parent/'AGENTS.md'),gpu_selected=False,
                  head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (bundle/'PROTOCOL.md').write_text('''# New library-only ownership control\n\nNo optimizer or trainer is run. FP32 then BF16 32x32 matmuls in one new process use seed173, native defaults, one exclusive nonqueued RTX5090 and unchanged library environment. Record cold/per-phase allocator totals and explicitly registered tensor/storage lifetimes. Require all registered tensors AND native storage weak handles to expire after each scope. Negative alias control must detect storage retained after the base tensor dies, then confirm release after alias deletion.\n\nPrimary hypothesis: first FP32 scope retains8519680 allocator bytes after all registered owners expire, matching installed non-sm90 default cuBLAS workspace. Different size is INCONCLUSIVE, not silently accepted; failures are retained. BF16 may initialize additional library state; its delta is measured separately without a zero-baseline ownership assertion. Existing failed optimizer probe remains unchanged; this cannot retrospectively identify an old block owner.\n\nInternal60/external90s, allocated2GiB/evidence5MiB. No tensors, weights, checkpoints, traces, cache/workspace clearing, environment/precision/controller changes, retries, queued waits or automatic next GPU campaign. CPU-tests validate aliases, retained tensors, exception/traceback references, observer references, nested cycles and fail-closed empty/closed handling. Observe only registered roots, no global tensor/GC scan. No forced GC or GC-setting changes.\n\nFreeze a NEW launch_manifest.json with index/UUID/base-manifest SHA before launch. Use the private research supervisor with one new autostart=false/autorestart=false entry. Check live claims/processes/disk; never displace other work. Capture failures and release only own resources.\n''')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.bundle.resolve())))
