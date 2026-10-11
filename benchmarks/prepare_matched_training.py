"""Freeze the runnable shared trainer after CPU checks; never select/launch a GPU."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

try:
    from .prepare_matched_training_design import verify as verify_design, sha, read, write
    from .launch_environment_contract import load_contract
except ImportError:
    from prepare_matched_training_design import verify as verify_design, sha, read, write
    from launch_environment_contract import load_contract


ROOT = Path(__file__).resolve().parents[1]
DESIGN = Path('/workspace/optimizer_replay_results/soap_matched_design_20261009/design')
NEW_SOURCES = ('benchmarks/matched_training_runtime.py', 'benchmarks/cifar_matched_training.py',
               'benchmarks/prepare_matched_training.py', 'tests/test_matched_training_runtime.py')
ADDITIONAL_DEPENDENCIES = ('benchmarks/tensor_ownership.py',)


def runnable_protocol(design_protocol):
    p = copy.deepcopy(design_protocol)
    p['status'] = 'PREPARED_NOT_LAUNCHED'
    p['remaining_before_launch'] = ['Bind one free GPU index/UUID and frozen private supervisor environment',
        'Verify actual supervisor inheritance before one nonqueued bounded execution']
    p['implementation'] = dict(entry='source/benchmarks/cifar_matched_training.py',
        optimizer_hash_chunk_bytes=2**20, main_and_autograd_cpu_after_smoke=116,
        all_original_masks_restored=True,
        optimizer_release_scope='All registered state/algorithm extras/gradients and optimizer objects; no allocator-zero claim',
        actual_first_capture_reuses_backward_and_gradients=True,
        endpoint_optimizer_hash_includes_state_and_algorithm_extras=True,
        exact_qr_repeat_full_loss_sequence_required=True,
        retained_preflight_probe_evidence_before_comparison=True)
    return p


def verify_runnable(bundle):
    b = Path(bundle); m = read(b/'manifest.json')
    for name, expected in m['source_sha256'].items():
        if sha(b/'source'/name) != expected:
            raise ValueError('Runnable source drift: '+name)
    if any(name not in m['source_sha256'] for name in ADDITIONAL_DEPENDENCIES):
        raise ValueError('Required runtime dependency absent from frozen manifest')
    for name, expected in m['artifacts'].items():
        if sha(b/name) != expected:
            raise ValueError('Runnable artifact drift: '+name)
    design = Path(m['design_bundle']); dm = verify_design(design)
    if sha(design/'manifest.json') != m['design_manifest_sha256']:
        raise ValueError('Design lineage drift')
    if (b/'config.json').read_bytes() != (design/'config.json').read_bytes():
        raise ValueError('Common recipe or study configuration changed')
    if read(b/'protocol.json') != runnable_protocol(read(design/'protocol.json')):
        raise ValueError('Declared shared design changed')
    for name, expected in dm['sources'].items():
        if m['source_sha256'].get(name) != expected:
            raise ValueError('Previously frozen source replaced: '+name)
    for name in ('split.json','data_prepared.json'):
        if (b/name).read_bytes() != (design/'reference'/name).read_bytes():
            raise ValueError('Frozen shared data/split changed')
    load_contract(b)
    for path, expected in read(b/'data_prepared.json')['data'].items():
        if sha(path) != expected:
            raise ValueError('Pinned training parquet changed')
    return m


def prepare(bundle, design=DESIGN):
    b = Path(bundle).resolve(); design = Path(design).resolve()
    if b.exists():
        raise ValueError('New unexecuted bundle required')
    if shutil.disk_usage(b.parent).free < 5*2**30:
        raise OSError('Shared disk capacity bound')
    dm = verify_design(design)
    b.mkdir(); shutil.copytree(design/'source', b/'source')
    for name in NEW_SOURCES:
        destination = b/'source'/name
        if destination.exists():
            raise ValueError('Cannot replace a frozen source')
        shutil.copy2(ROOT/name, destination)
    baseline = Path(read(design/'manifest.json')['baseline'])
    bm = read(baseline/'manifest.json')
    for name in ADDITIONAL_DEPENDENCIES:
        if sha(baseline/'source'/name) != bm['source_sha256'][name] or sha(ROOT/name) != bm['source_sha256'][name]:
            raise ValueError('Previously qualified ownership dependency changed')
        shutil.copy2(baseline/'source'/name, b/'source'/name)
    shutil.copy2(design/'config.json', b/'config.json')
    for name in ('split.json','data_prepared.json','environment_contract.json'):
        shutil.copy2(design/'reference'/name, b/name)
    write(b/'protocol.json', runnable_protocol(read(design/'protocol.json')))
    (b/'PROTOCOL.md').write_text('''# Matched shared-process comparison, prepared only

The linked CPU design and protocol.json bind QR20 / literalMay / Frocap.5 cubic2 /
QR20repeat,3500calls each,25discardedQR10smoke,4preflightbackwards/no preflightsteps.
Original1536/depth1/heads24/common5e-4/.99/(.9,.999) recipe,35000-call schedule,
875warmup, split/data/strongaugmentation/eagerBF16evaluation remain frozen.
LiteralMay defaults except explicit outer recipe overrides; no blog replication,
refresh-only ablation, tuning/EMA/default promotion or statistical winner claim.

One shared compiled model/batchplan and CPU initial snapshot, independent fresh
optimizer instances. All four exact full pre-update probes precede any comparison
step; each actual first pre-update must match before its step. Save probes before
comparing, abort on the first failure, no tolerance/family selection/retry/queue.
One populated optimizer at a time; registered state/extras/gradient tensor+storage
and weak optimizer-object release gates. Parameter storage stays persistent.
No whole-allocator-zero/GC/cache clearing or old failed-gate reclassification.
Full endpoint optimizer hashes stream1MiB chunks, include algorithm extras, match
the design logical-byte format. Initial109.33MiB/purpose226.62MiB<256MiB;
streaming CPU workspace <=3MiB, never a full host covariance-factor copy.

Observations0/100/400/875/1000/1750/2625/3500 on identical validation/trainpanel
inputs; finite all-state/cache/controller and FP32 Gram.05 guards at refresh and
observations. Exact QRrepeat sampled model/evaluation/RNG/batches/full loss stream
and full endpoint required for qualified single-seed exploratory quality deltas.
First50 only excluded from steady optimizer summaries; compile/smoke/preflight/
hash/evaluation/guards charged to continuous fullwall. Ordered instrumented cost
samples do not establish timing speedup.3600internal/3660watchdog,24GiBallocated/
256MiBpurposehost/100MiBevidence, no checkpoints/weights/tensorfiles/traces.

No device selected/claim/launch/program installed. New private supervisor
template binds nonqueued gpu-claim, compilerthreads4/OMP4/MKL4/no-bytecode1 and
historical environment contract. Later bind one index/UUID prospectively, render
only __GPU_INDEX__, and verify actual supervisor inheritance before starting.
JSON-wire backend before-smoke/after-measured gates preserve all recorded flags/
keys/libraries; only declaredUUID delta. Pin only our main/autograd toCPU116 after
smoke, restore original masks on all exits. Retain actual loaded wrappers and full
kernel metadata without changing graph outputs. Frozen-source/config/data checks
and environment gate precede CUDA initialization; one attempt, partials retained.

CPU preparation does not qualify compiled GPU numerical agreement,1536-dimMay
compatibility, full-horizon repeatability, convergence or timing. Prior failed and
inconclusive evidence remains unchanged. Execute only this frozen entry after
the separately bound managed launch; no automatic campaign follows.
''')
    env = read(b/'environment_contract.json')['phases']['backend_before_smoke']['environment']
    env = {k:v for k,v in env.items() if k!='CUDA_VISIBLE_DEVICES'}
    envtext = ','.join(k+'='+json.dumps(v) for k,v in sorted(env.items()))
    template = f'''[program:manifold_matched_learning]
command=/workspace/bin/gpu-claim run --owner optimizer-research --job matched-shared-20261010 --gpu __GPU_INDEX__ -- /usr/bin/timeout --kill-after=10s 3660 /venv/main/bin/python -u {b}/source/benchmarks/cifar_matched_training.py --bundle {b}
directory={b}
environment={envtext}
autostart=false
autorestart=false
startsecs=0
stopasgroup=true
killasgroup=true
stopwaitsecs=20
stdout_logfile={b}/run.log
stdout_logfile_maxbytes=0
redirect_stderr=true
'''
    (b/'supervisor_template.conf').write_text(template)
    artifacts = {str(p.relative_to(b)):sha(p) for p in b.rglob('*') if p.is_file() and b/'source' not in p.parents}
    write(b/'manifest.json', dict(status='PREPARED_NOT_LAUNCHED', gpu_selected=False, device=None,
        design_bundle=str(design), design_manifest_sha256=sha(design/'manifest.json'),
        source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        artifacts=artifacts, config_sha256=sha(b/'config.json'), protocol_sha256=sha(b/'PROTOCOL.md'),
        split_sha256=sha(b/'split.json'), base_prepared_sha256=sha(b/'data_prepared.json'),
        environment_contract_sha256=sha(b/'environment_contract.json'),
        supervisor_template_sha256=sha(b/'supervisor_template.conf')))
    verify_runnable(b)
    write(b/'PREPARATION_STATUS.json', dict(status='PREPARED_NOT_LAUNCHED',gpu_selected=False,
        launch_manifest_exists=False,comparison_updates_planned=14000,maximum_optimizer_calls=14025))
    (b/'source_worktree_status.txt').write_text(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True))
    return read(b/'manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--bundle',type=Path,required=True)
    m = prepare(parser.parse_args().bundle)
    print(json.dumps(dict(status=m['status'],source_files=len(m['source_sha256']))))
