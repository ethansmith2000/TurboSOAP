"""Freeze a new short diagnostic without selecting or initializing a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

try:
    from .normalized_diagnostic_contract import (BASE, NORMALIZED_POLICY, validate_config,
        validate_child_change, validate_outer_change, validate_reference_region)
except ImportError:
    from normalized_diagnostic_contract import (BASE, NORMALIZED_POLICY, validate_config,
        validate_child_change, validate_outer_change, validate_reference_region)


REPO = Path(__file__).resolve().parents[1]
NEW_FILES = ['normalized_input_replay.py', 'normalized_diagnostic_contract.py', 'normalized_input_abi.py',
             'tensor_ownership.py', 'cifar_normalized_child.py', 'cifar_normalized.py',
             'summarize_normalized_input.py', 'prepare_normalized_input.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(bundle):
    if bundle.exists():
        raise ValueError('Use a new unexecuted bundle')
    if shutil.disk_usage(bundle.parent).free < 5*2**30:
        raise OSError('Shared disk capacity guard')
    old = json.loads((BASE/'manifest.json').read_text())
    for name, digest in old['source_sha256'].items():
        assert sha(BASE/'source'/name) == digest, name
    for name, key in [('config.json', 'config_sha256'), ('split.json', 'split_sha256'),
                      ('data_prepared.json', 'base_prepared_sha256'), ('PROTOCOL.md', 'protocol_sha256')]:
        assert sha(BASE/name) == old[key], name
    selector = REPO.parent/'optimizer_replay_results/soap_normalized_selector_20261008/source_manifest.json'
    selector_manifest = json.loads(selector.read_text())
    for name in ['benchmarks/normalized_input_abi.py', 'benchmarks/tensor_ownership.py']:
        assert sha(REPO/name) == selector_manifest['source_sha256'][name]
    validate_child_change((BASE/'source/benchmarks/cifar_saved_operand_child.py').read_text(),
                          (REPO/'benchmarks/cifar_normalized_child.py').read_text())
    validate_outer_change((BASE/'source/benchmarks/cifar_saved_operand.py').read_text(),
                          (REPO/'benchmarks/cifar_normalized.py').read_text())
    validate_reference_region((REPO/'benchmarks/normalized_input_replay.py').read_text())
    c = json.loads((BASE/'config.json').read_text())
    c['max_wall_seconds'] = 900
    d = c['execution_diagnostic']; d['per_process_seconds'] = 210; d['normalized_input'] = NORMALIZED_POLICY
    for context in d['contexts']:
        context['label'] = context['label'].replace('saved_operand', 'normalized_input')
    validate_config(c)
    bundle.mkdir()
    shutil.copytree(BASE/'source', bundle/'source')
    shutil.copytree(BASE/'upstream_reference', bundle/'upstream_reference')
    for name in NEW_FILES:
        shutil.copy2(REPO/'benchmarks'/name, bundle/'source/benchmarks'/name)
    (bundle/'source/tests').mkdir(exist_ok=True)
    shutil.copy2(REPO/'tests/test_normalized_diagnostic.py', bundle/'source/tests/test_normalized_diagnostic.py')
    for name in ['split.json', 'data_prepared.json']:
        shutil.copy2(BASE/name, bundle/name)
    (bundle/'config.json').write_text(json.dumps(c, indent=2)+'\n')
    (bundle/'PROTOCOL.md').write_text('''# New normalized-input variation diagnostic, prepared only

Four fresh identical QR20 arms in balanced allocation order420/3500/3500/420,
25 disposable QR10 smoke +25 measured updates each,200total. Same frozen width1536/
depth1/heads24/seed139/batch256/original recipe/Adam(.9,.999)/covariance.99/currentV/
strictM/bothfactors/compiled graph/backend flags and common split/evaluation recipe.
Full primary and preceding model-only/initial/saved-operand/head replays are AST
unchanged. One new post-trajectory native normalized-input observer follows them.
Collect all four planned arms even if sampled primary fingerprints differ.
Exact repetition is independently reported; do not drop failures or tune flags.

Pin original full forward/backward sources; verify normalization producer,
affine/CLS consumer, classifier operand, saved21of30/backwardmul_7 use. Before
backward take metadata/references/native weak ownership handles only, with no
hooks/new graph outputs/retain_graph/clones/hashes/tensor math. After full backward
and synchronization clone/hash the FP32 normalized CLS slice only. Also retain
the first BF16 classifier operand for the unchanged HeldHead math. Three native
calls plain/plain/RAMprofile and three held-head calls plain/plain/RAMprofile;
compact kernel names/counts only, no trace files. Require actual-first/warm/earlier
capture/head output and all-gradient parity, complete parameter/optimizer/RNG
restoration, exact code/backend, finite/geometry checks, registered tensor AND
storage lifetime release, optimizer release and actual own affinity restoration.
Missing access/changed ABI is recorded as inconclusive; no guessed tensor/retry.

Purpose RAM allowance NEW256MiB. Existing replay reference118,675,856 bytes plus
transient full normalized reference102,236,160 is220,912,016. Account actual unique
storage extents and temporary CLS clone1,572,864 plus held-operand coexistence;
fail at the256MiB cap. Full normalized reference discarded after post-backward
CLS record; input/target/held operand discarded after state restoration. Snapshots
are RAM-only rollback owned by unchanged initial_state_scope, not saved weights.
Weak observer release scopes are explicit registered roots/aliases, not whole
allocator equality; separately record allocator totals without cache clearing.

Internal900/external960 seconds, child210 within global900, allocated24GiB,
evidence100MiB hard/expectedcompact<25MiB. No weights/checkpoints/tensor/features/
trace files, queued waits, retries, automatic long learning or unplanned stages.
No environment/precision/driver/library/cache/clocks/GC/other-project changes.
CPU verification alone uses empty CUDA_VISIBLE_DEVICES to prevent CPU Dynamo RNG
setup from initializing GPUs. Production flags and library environment unchanged.

No GPU selected or launch manifest/attempt at preparation. Before execution verify
every frozen source/config/protocol/data hash, live claims/processes/shared disk,
then bind one free RTX5090 index/UUID to this manifest in NEW launch_manifest.json.
Use exclusive nonqueued gpu-claim/private research supervisor/autostartfalse/
autorestartfalse. Preserve this prepared manifest; only new launch metadata binds
the device. Never launch historical supervisor entries or alter other projects.

Different native parity-valid FP32 normalized CLS localizes variation upstream
of final affine/cast. Equal FP32 but different BF16 operand localizes the final
affine/cast/lifetime path, not a named operator. No fresh gradient variation means
inconclusive localization. Parity/ABI/resource/code failures are inconclusive and
retained. No timing/convergence/quality/default promotion follows from25updates.
All earlier completed experiments and failed gates stay frozen.
''')
    m = dict(old)
    m.update(source_sha256={str(p.relative_to(bundle/'source')): sha(p) for p in (bundle/'source').rglob('*') if p.is_file()},
             config_sha256=sha(bundle/'config.json'), protocol_sha256=sha(bundle/'PROTOCOL.md'),
             baseline_bundle=str(BASE), baseline_manifest_sha256=sha(BASE/'manifest.json'),
             baseline_summary_sha256=sha(BASE/'summary.json'), device=None, gpu_selected=False,
             selector_manifest_sha256=sha(selector), declared_baseline_backend_deltas=[],
             base_repository_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
             evidence_budget_bytes=100*2**20, external_seconds=960, checkpoint_policy='No tensor/weight/checkpoint/trace files; RAM-only diagnostic references')
    # Prepared predecessor pointers are superseded explicitly, without editing old manifests.
    m.pop('next_task_sha256', None)
    (bundle/'manifest.json').write_text(json.dumps(m, indent=2)+'\n')
    (bundle/'source_worktree_status.txt').write_text(subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True))
    (bundle/'source_worktree.diff').write_text(subprocess.check_output(['git', 'diff'], cwd=REPO, text=True))
    return m


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args(); m = prepare(args.bundle.resolve())
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED', gpu_selected=False, source_files=len(m['source_sha256']))))
