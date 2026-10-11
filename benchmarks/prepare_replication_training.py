"""New stdlib freeze with strict seed exception, completed-evidence lineage, no GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from replication_training_contract import BASELINE, validate_inputs

ROOT = Path(__file__).resolve().parents[1]
NEW_SOURCES = ('benchmarks/replication_training_contract.py',
    'benchmarks/prepare_replication_training.py', 'benchmarks/cifar_replication_training.py',
    'tests/test_replication_training.py')
REFERENCES = ('summary.json', 'execution_status.json', 'COMPLETED_STATUS.md')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_baseline():
    from prepare_cadence_training import verify as verify_completed_sources
    old = verify_completed_sources(BASELINE)
    if read(BASELINE / 'execution_status.json')['status'] != 'completed':
        raise ValueError('Require completed seed139 evidence')
    return old


def verify(bundle):
    b = Path(bundle)
    m = read(b / 'manifest.json')
    if m['baseline_bundle'] != str(BASELINE):
        raise ValueError('Undeclared seed139 lineage')
    for rel, digest in m['artifacts'].items():
        if sha(b / rel) != digest:
            raise ValueError('Frozen artifact changed: ' + rel)
    for rel, digest in m['source_sha256'].items():
        if sha(b / 'source' / rel) != digest:
            raise ValueError('Frozen source changed: ' + rel)
    if sha(BASELINE / 'manifest.json') != m['baseline_manifest_sha256']:
        raise ValueError('Baseline manifest changed')
    old = validate_baseline()
    for rel, digest in old['source_sha256'].items():
        if m['source_sha256'].get(rel) != digest:
            raise ValueError('Retained dependency changed: ' + rel)
    for rel, digest in m['reference_evidence_sha256'].items():
        if sha(BASELINE / rel) != digest:
            raise ValueError('Completed reference evidence changed: ' + rel)
    for rel in ('split.json', 'data_prepared.json', 'environment_contract.json'):
        if (b / rel).read_bytes() != (BASELINE / rel).read_bytes():
            raise ValueError('Shared data/environment changed: ' + rel)
    validate_inputs(read(b / 'config.json'), read(b / 'protocol.json'),
        read(BASELINE / 'config.json'), read(BASELINE / 'protocol.json'))
    from launch_environment_contract import load_contract
    load_contract(b)
    return m


def prepare(bundle, config, declared, protocol_text):
    b = Path(bundle).resolve()
    if b.exists():
        raise ValueError('Fresh unexecuted replication bundle required')
    validate_inputs(read(config), read(declared), read(BASELINE / 'config.json'), read(BASELINE / 'protocol.json'))
    old = validate_baseline()
    if shutil.disk_usage(b.parent).free < 5 * 2**30:
        raise OSError('Shared disk bound')
    b.mkdir()
    shutil.copytree(BASELINE / 'source', b / 'source')
    for rel in NEW_SOURCES:
        target = b / 'source' / rel
        if target.exists():
            raise ValueError('Cannot replace frozen source')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    write(b / 'config.json', read(config)); write(b / 'protocol.json', read(declared))
    shutil.copy2(protocol_text, b / 'PROTOCOL.md')
    for rel in ('split.json', 'data_prepared.json', 'environment_contract.json'):
        shutil.copy2(BASELINE / rel, b / rel)
    template = (BASELINE / 'supervisor_template.conf').read_text().replace(
        'manifold_cadence_learning', 'manifold_replication_learning').replace(
        'cadence-shared-20261010', 'replication-seed140-20261010').replace(
        str(BASELINE), str(b)).replace('cifar_cadence_training.py', 'cifar_replication_training.py')
    (b / 'supervisor_template.conf').write_text(template)
    artifacts = {str(p.relative_to(b)): sha(p) for p in b.iterdir() if p.is_file()}
    write(b / 'manifest.json', dict(status='PREPARED_NOT_LAUNCHED', gpu_selected=False, device=None,
        baseline_bundle=str(BASELINE), baseline_manifest_sha256=sha(BASELINE / 'manifest.json'),
        reference_evidence_sha256={rel: sha(BASELINE / rel) for rel in REFERENCES},
        allowed_config_changes=['seed', 'matched_comparison'],
        source_sha256={str(p.relative_to(b / 'source')): sha(p) for p in (b / 'source').rglob('*') if p.is_file()},
        artifacts=artifacts, config_sha256=sha(b / 'config.json'), protocol_sha256=sha(b / 'PROTOCOL.md'),
        environment_contract_sha256=sha(b / 'environment_contract.json'),
        supervisor_template_sha256=sha(b / 'supervisor_template.conf'),
        split_sha256=sha(b / 'split.json'), base_prepared_sha256=sha(b / 'data_prepared.json')))
    return verify(b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for flag in ('bundle', 'config', 'protocol', 'protocol-text'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    m = prepare(args.bundle, args.config, args.protocol, args.protocol_text)
    print(json.dumps(dict(status=m['status'], sources=len(m['source_sha256']), seed=140)))
