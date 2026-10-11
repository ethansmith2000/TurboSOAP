"""Freeze a separately named JSON-wire diagnostic; no GPU launch or old repair."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

try:
    from .launch_environment_contract import load_contract, sha
    from .launch_environment_wire_contract import validate_child_change, validate_outer_change, validate_summary_change
    from .prepare_normalized_env import validate_frozen
except ImportError:
    from launch_environment_contract import load_contract, sha
    from launch_environment_wire_contract import validate_child_change, validate_outer_change, validate_summary_change
    from prepare_normalized_env import validate_frozen

ROOT = Path(__file__).resolve().parents[1]
FAILED = Path('/workspace/optimizer_replay_results/cifar_soap_normalized_env_20261008')
NEW = ['launch_environment_wire_contract.py', 'cifar_normalized_env_wire_child.py',
       'cifar_normalized_env_wire.py', 'summarize_normalized_env_wire.py', 'prepare_normalized_env_wire.py']


def prepare(bundle):
    bundle = Path(bundle).resolve()
    if bundle.exists():
        raise ValueError('Separate unexecuted bundle required')
    if shutil.disk_usage(bundle.parent).free < 5*2**30:
        raise OSError('Shared disk capacity guard')
    old = validate_frozen(FAILED); load_contract(FAILED)
    status = json.loads((FAILED/'execution_status.json').read_text())
    summary = json.loads((FAILED/'summary.json').read_text())
    if status['status'] != 'failed' or summary['completed'] is not False:
        raise ValueError('Preserve failed predecessor; no retrospective relabeling')
    for before, after, validator in [
        ('cifar_normalized_env_child.py', 'cifar_normalized_env_wire_child.py', validate_child_change),
        ('cifar_normalized_env.py', 'cifar_normalized_env_wire.py', validate_outer_change),
        ('summarize_normalized_env.py', 'summarize_normalized_env_wire.py', validate_summary_change)]:
        validator((FAILED/'source/benchmarks'/before).read_text(), (ROOT/'benchmarks'/after).read_text())
    bundle.mkdir()
    shutil.copytree(FAILED/'source', bundle/'source')
    shutil.copytree(FAILED/'upstream_reference', bundle/'upstream_reference')
    for name in NEW:
        shutil.copy2(ROOT/'benchmarks'/name, bundle/'source/benchmarks'/name)
    shutil.copy2(ROOT/'tests/test_launch_environment_wire.py', bundle/'source/tests/test_launch_environment_wire.py')
    for name in ('config.json', 'split.json', 'data_prepared.json', 'environment_contract.json'):
        shutil.copy2(FAILED/name, bundle/name)
    template = (FAILED/'supervisor_template.conf').read_text().replace(str(FAILED), str(bundle))
    template = template.replace('manifold_normalized_env]', 'manifold_normalized_env_wire]')
    template = template.replace('normalized-env-diagnostic', 'normalized-env-wire-diagnostic')
    template = template.replace('cifar_normalized_env.py', 'cifar_normalized_env_wire.py')
    (bundle/'supervisor_template.conf').write_text(template)
    (bundle/'PROTOCOL.md').write_text('''# Separate prospective JSON-wire normalized diagnostic; prepared only

Primary recipe, trajectory, all replay observers and all eight historical backend/
early-check gates unchanged. Four fresh QR20 controls in allocation420/3500/3500/420,
25QR10smoke+25measured each=200updates; original1536/depth1/heads24/seed139/batch256/
Adam(.9,.999)/covariance.99/currentV/strictM/bothfactors/flags/split/evaluation.
Internal900/external960/child210s,24GiBallocated/256MiBpurpose/100MiBevidence,
expectedcompact<25MiB. RAM-only diagnostic rollback/reference purpose and release
scopes unchanged. No tensor/weight/checkpoint/feature/trace files, retry, queue,
cache clearing, global.env/precision/library/driver/clock/GC/other-project change.

Only correction: compare live backend metadata in its persisted JSON wire form,
using JSON dumps/loads with allow_nan=False and no custom default serializer;
require string object keys. TorchVersion is a live string subclass whose JSON
value is identical to the retained plain string. Do not exempt version values,
coerce unknown objects, discard fields or ignore numerical/env/library deltas.
All field/key/value checks remain exact AFTER this prospective normalization.
The original checker/failed source/result/summary remains untouched.

New child differs only in backend-checker import; outer only child filename;
new summary only comparator imports. AST validates this limited scope. CPU tests
exercise actual torch.__version__ and actual backend_record(), not just decoded
historical JSON. CPU proof does not qualify CUDA/runtime/input localization.

Environment contract retains seven hash-bound historical inputs and all four
baseline arms/two phases. Explicit compiler threads4/OMP4/MKL4/no-bytecode1 and
all recorded before-smoke environment keys. No preseeded later Triton cache key.
Only prospectively declared single CUDA_VISIBLE_DEVICES UUID delta allowed; zero
numerical/backend deltas. No selected GPU/launch manifest at preparation. Before
future launch, verify all prepared hashes/live resources, bind one free index/UUID,
record exact baseline-to-selected binding and manifest/config/environment/template
hashes, render ONLY __GPU_INDEX__, and verify actual private-supervisor inherited
environment before starting the new managed nonqueued job. No old program launch.

Use summarize_normalized_env_wire.py after success/failure. Keep original observer,
repetition, ABI, code, parity, RNG, ownership, geometry and resource criteria; all
eight exact phase checks and fresh variation needed for replay localization. No
actual cold normalized tensor capture, named operator, validation-loss cause,
convergence/timing/learning/default promotion or automatic longer campaign.
The preceding11.647s campaign remains FAILED before first update, and earlier
200-update baseline-environment gate remains inconclusive. No old state reused.
''')
    manifest = dict(old)
    manifest.update(source_sha256={str(p.relative_to(bundle/'source')): sha(p) for p in (bundle/'source').rglob('*') if p.is_file()},
                    protocol_sha256=sha(bundle/'PROTOCOL.md'), supervisor_template_sha256=sha(bundle/'supervisor_template.conf'),
                    previous_bundle=str(FAILED), previous_manifest_sha256=sha(FAILED/'manifest.json'),
                    previous_summary_sha256=sha(FAILED/'summary.json'),
                    backend_comparison_representation='Exact persisted JSON; reject nonfinite/unknown objects/non-string keys',
                    device=None, gpu_selected=False, launch_pending=True)
    (bundle/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (bundle/'README.md').write_text('Prepared only; no GPU selected/launch/claim/queue. JSON-wire comparator correction is prospectively declared. Prior failed campaign is preserved. See NEXT_TASK.md.\n')
    (bundle/'PREPARATION_STATUS.json').write_text(json.dumps(dict(status='PREPARED_NOT_LAUNCHED', gpu_selected=False, updates_planned=200), indent=2)+'\n')
    (bundle/'source_worktree_status.txt').write_text(subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--bundle', type=Path, required=True)
    manifest = prepare(p.parse_args().bundle)
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED', source_files=len(manifest['source_sha256']))))
