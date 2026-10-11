"""Freeze a NEW cadence bundle using stdlib only; never select or initialize a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

BASELINE = Path('/workspace/optimizer_replay_results/cifar_may_refresh_frozen_20261010')
ROOT = Path(__file__).resolve().parents[1]
NEW_SOURCES = ('benchmarks/cadence_training_contract.py', 'benchmarks/cadence_training_runtime.py',
    'benchmarks/cifar_cadence_training.py', 'benchmarks/prepare_cadence_training.py',
    'tests/test_cadence_training.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify(bundle):
    b = Path(bundle)
    m = read(b / 'manifest.json')
    for rel, digest in m['artifacts'].items():
        if sha(b / rel) != digest:
            raise ValueError('Frozen artifact changed: ' + rel)
    for rel, digest in m['source_sha256'].items():
        if sha(b / 'source' / rel) != digest:
            raise ValueError('Frozen source changed: ' + rel)
    baseline = Path(m['baseline_bundle'])
    if sha(baseline / 'manifest.json') != m['baseline_manifest_sha256']:
        raise ValueError('Baseline provenance changed')
    old = read(baseline / 'manifest.json')
    for rel, digest in old['source_sha256'].items():
        if sha(baseline / 'source' / rel) != digest or m['source_sha256'].get(rel) != digest:
            raise ValueError('Previously frozen dependency changed: ' + rel)
    for rel in ('split.json', 'data_prepared.json', 'environment_contract.json'):
        if (b / rel).read_bytes() != (baseline / rel).read_bytes():
            raise ValueError('Shared data/environment changed: ' + rel)
    # Config changes are only the new prospective study declaration.
    c, old_c = read(b / 'config.json'), read(baseline / 'config.json')
    c.pop('matched_comparison'); old_c.pop('matched_comparison')
    if c != old_c:
        raise ValueError('Underlying training recipe changed')
    from launch_environment_contract import load_contract
    load_contract(b)
    for path, digest in read(b / 'data_prepared.json')['data'].items():
        if sha(path) != digest:
            raise ValueError('Cached training data changed')
    return m


def prepare(bundle, prepared_config, prepared_protocol, protocol_text):
    b = Path(bundle).resolve()
    if b.exists():
        raise ValueError('New unexecuted bundle required')
    if shutil.disk_usage(b.parent).free < 5 * 2**30:
        raise OSError('Shared disk bound')
    base = read(BASELINE / 'manifest.json')
    for rel, digest in base['source_sha256'].items():
        if sha(BASELINE / 'source' / rel) != digest:
            raise ValueError('Baseline source changed: ' + rel)
    b.mkdir(); shutil.copytree(BASELINE / 'source', b / 'source')
    for rel in NEW_SOURCES:
        if (b / 'source' / rel).exists():
            raise ValueError('Cannot replace an existing frozen source')
        (b / 'source' / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, b / 'source' / rel)
    write(b / 'config.json', read(prepared_config))
    write(b / 'protocol.json', read(prepared_protocol))
    shutil.copy2(protocol_text, b / 'PROTOCOL.md')
    for rel in ('split.json', 'data_prepared.json', 'environment_contract.json'):
        shutil.copy2(BASELINE / rel, b / rel)
    env = read(b / 'environment_contract.json')['phases']['backend_before_smoke']['environment']
    env = {k: v for k, v in env.items() if k != 'CUDA_VISIBLE_DEVICES'}
    envtext = ','.join(k + '=' + json.dumps(v) for k, v in sorted(env.items()))
    template = f'''[program:manifold_cadence_learning]
command=/workspace/bin/gpu-claim run --owner optimizer-research --job cadence-shared-20261010 --gpu __GPU_INDEX__ -- /usr/bin/timeout --kill-after=10s 3660 /venv/main/bin/python -u {b}/source/benchmarks/cifar_cadence_training.py --bundle {b}
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
    (b / 'supervisor_template.conf').write_text(template)
    artifacts = {str(p.relative_to(b)): sha(p) for p in b.iterdir() if p.is_file()}
    write(b / 'manifest.json', dict(status='PREPARED_NOT_LAUNCHED', gpu_selected=False, device=None,
        baseline_bundle=str(BASELINE), baseline_manifest_sha256=sha(BASELINE / 'manifest.json'),
        source_sha256={str(p.relative_to(b / 'source')): sha(p) for p in (b / 'source').rglob('*') if p.is_file()},
        artifacts=artifacts, config_sha256=sha(b / 'config.json'), protocol_sha256=sha(b / 'PROTOCOL.md'),
        split_sha256=sha(b / 'split.json'), base_prepared_sha256=sha(b / 'data_prepared.json'),
        environment_contract_sha256=sha(b / 'environment_contract.json'),
        supervisor_template_sha256=sha(b / 'supervisor_template.conf')))
    return verify(b)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--protocol-text', type=Path, required=True)
    a = p.parse_args()
    m = prepare(a.bundle, a.config, a.protocol, a.protocol_text)
    print(json.dumps(dict(status=m['status'], source_files=len(m['source_sha256']))))
