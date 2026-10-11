"""Fresh immutable profile bundle, with unchanged completed source dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from refresh_profile_contract import BASELINE, protocol, recipe, validate

ROOT = Path(__file__).resolve().parents[1]
NEW = ('benchmarks/refresh_profile_instrumentation.py', 'benchmarks/refresh_profile_contract.py',
       'benchmarks/prepare_refresh_profile.py', 'benchmarks/cifar_refresh_profile.py',
       'tests/test_refresh_profile.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')


def verify(b):
    b=Path(b); m=read(b/'manifest.json')
    if m['baseline_bundle'] != str(BASELINE) or sha(BASELINE/'manifest.json') != m['baseline_manifest_sha256']:
        raise ValueError('Baseline provenance changed')
    old=read(BASELINE/'manifest.json')
    for rel, digest in old['source_sha256'].items():
        if sha(BASELINE/'source'/rel) != digest or m['source_sha256'].get(rel) != digest:
            raise ValueError('Retained source changed: '+rel)
    for rel, digest in m['source_sha256'].items():
        if sha(b/'source'/rel) != digest:
            raise ValueError('Frozen source changed: '+rel)
    for rel, digest in m['artifacts'].items():
        if sha(b/rel) != digest:
            raise ValueError('Frozen input changed: '+rel)
    for rel, digest in m['reference_sha256'].items():
        if sha(BASELINE/rel) != digest:
            raise ValueError('Completed reference changed: '+rel)
    validate(read(b/'config.json'), read(b/'protocol.json'), read(BASELINE/'config.json'))
    for rel in ('split.json','data_prepared.json','environment_contract.json'):
        if (b/rel).read_bytes() != (BASELINE/rel).read_bytes():
            raise ValueError('Shared input changed: '+rel)
    return m


def prepare(b):
    b=Path(b)
    if b.exists():
        raise ValueError('Require a fresh bundle; no overwrite')
    if read(BASELINE/'execution_status.json')['status'] != 'completed':
        raise ValueError('Require completed baseline')
    if shutil.disk_usage(b.parent).free < 5*2**30:
        raise OSError('Insufficient shared disk')
    b.mkdir(); shutil.copytree(BASELINE/'source', b/'source')
    for rel in NEW:
        if (b/'source'/rel).exists():
            raise ValueError('Cannot replace retained source')
        shutil.copy2(ROOT/rel,b/'source'/rel)
    write(b/'config.json',recipe(read(BASELINE/'config.json')));write(b/'protocol.json',protocol())
    for rel in ('split.json','data_prepared.json','environment_contract.json'):
        shutil.copy2(BASELINE/rel,b/rel)
    template=(BASELINE/'supervisor_template.conf').read_text().replace(
        'manifold_longer_learning','manifold_refresh_profile').replace(
        'longer-seed140-20261010','refresh-profile-20261010').replace(
        str(BASELINE),str(b)).replace('cifar_longer_training.py','cifar_refresh_profile.py').replace(
        '--kill-after=10s 1260 ','--kill-after=10s 660 ')
    (b/'supervisor_template.conf').write_text(template)
    (b/'PROTOCOL.md').write_text('Width1536 plain/profile cost diagnostic\n\n'
        'The complete prospective contract is protocol.json. Four420-call fresh trajectories, '
        '25 disposable QR smoke calls, two empty first-backward probes; maximum1705 optimizer calls. '
        'Profile only calls281(warm/QR),282(ordinary),401(QRrepair/QR), after natural path warmup. '
        'Original35000 LR schedule/875warmup and optimizer math/precision/V/M are fixed. '
        'Plain/profile full model/optimizer/RNG endpoint and loss sequence must match exactly. '
        'No learning or wall speed ranking, no checkpoints/weights/tensors, no retry or queue. '
        'Six compressed traces deliberately retained for stage/dtype/kernel attribution. '
        '600s internal/660s external,24GiB GPU/256MiB CPU snapshot purpose/250MiB evidence.\n')
    artifacts={p.name:sha(p) for p in b.iterdir() if p.is_file()}
    write(b/'manifest.json',dict(status='PREPARED_NOT_LAUNCHED',baseline_bundle=str(BASELINE),
        baseline_manifest_sha256=sha(BASELINE/'manifest.json'),
        reference_sha256={r:sha(BASELINE/r) for r in ('summary.json','execution_status.json')},
        source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        artifacts=artifacts,config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),
        environment_contract_sha256=sha(b/'environment_contract.json'),
        supervisor_template_sha256=sha(b/'supervisor_template.conf')))
    return verify(b)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    args=p.parse_args();m=prepare(args.bundle)
    print(json.dumps(dict(status=m['status'],sources=len(m['source_sha256']))))
