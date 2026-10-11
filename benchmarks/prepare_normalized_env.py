"""Freeze a new environment-checked diagnostic without selecting/claiming a GPU."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

try:
    from .launch_environment_contract import (PHASES, POLICY, differences, sha, read,
        validate_contract, validate_child_change, validate_outer_change)
    from .normalized_diagnostic_contract import validate_config
except ImportError:
    from launch_environment_contract import (PHASES, POLICY, differences, sha, read,
        validate_contract, validate_child_change, validate_outer_change)
    from normalized_diagnostic_contract import validate_config

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = Path('/workspace/optimizer_replay_results/cifar_soap_normalized_input_20261008')
BASE = Path('/workspace/optimizer_replay_results/cifar_soap_saved_operand_20261008')
NEW = ['launch_environment_contract.py', 'cifar_normalized_env_child.py',
       'cifar_normalized_env.py', 'summarize_normalized_env.py', 'prepare_normalized_env.py']


def validate_frozen(base):
    m = read(base/'manifest.json')
    for name, key in [('config.json', 'config_sha256'), ('PROTOCOL.md', 'protocol_sha256'),
                      ('split.json', 'split_sha256'), ('data_prepared.json', 'base_prepared_sha256')]:
        if sha(base/name) != m[key]:
            raise ValueError('Frozen baseline '+name+' changed')
    for name, digest in m['source_sha256'].items():
        if sha(base/'source'/name) != digest:
            raise ValueError('Frozen baseline source changed: '+name)
    return m


def baseline_contract(base=BASE):
    validate_frozen(base)
    config = read(base/'config.json')
    paths = [base/'contexts'/c['label']/'result.json' for c in config['execution_diagnostic']['contexts']]
    runs = [read(p) for p in paths]
    phases = {p: runs[0][p] for p in PHASES}
    for run in runs:
        if run['status'] != 'completed' or any(differences(phases[p], run[p]) for p in PHASES):
            raise ValueError('Historical backend baseline not complete/exact in all four arms')
    inputs = [dict(path=str(p.resolve()), sha256=sha(p), kind='result') for p in paths]
    inputs += [dict(path=str((base/name).resolve()), sha256=sha(base/name), kind=name)
               for name in ('manifest.json', 'config.json', 'PROTOCOL.md')]
    return validate_contract(dict(policy=copy.deepcopy(POLICY), phases=phases, baseline_inputs=inputs))


def supervisor_template(bundle, contract):
    # Deliberately no device selection here. The claim supplies the single UUID.
    # Inherited extra tracked variables remain visible and are rejected at entry.
    env = contract['phases'][PHASES[0]]['environment']
    settings = ','.join(f'{k}="{v}"' for k, v in sorted(env.items()) if k != 'CUDA_VISIBLE_DEVICES')
    if any('"' in v or '%' in v or '\n' in v for v in env.values()):
        raise ValueError('Unsafe supervisor environment value')
    return f'''[program:manifold_normalized_env]
command=/workspace/bin/gpu-claim run --owner optimizer-research --job normalized-env-diagnostic --gpu __GPU_INDEX__ -- /usr/bin/timeout --kill-after=10s 960 /venv/main/bin/python -u {bundle}/source/benchmarks/cifar_normalized_env.py --bundle {bundle}
directory={bundle}
environment={settings}
autostart=false
autorestart=false
startsecs=0
stopasgroup=true
killasgroup=true
stopwaitsecs=20
stdout_logfile={bundle}/run.log
stdout_logfile_maxbytes=0
redirect_stderr=true
'''


def prepare(bundle):
    bundle = Path(bundle).resolve()
    if bundle.exists():
        raise ValueError('Use a separately named unexecuted bundle')
    if shutil.disk_usage(bundle.parent).free < 5*2**30:
        raise OSError('Shared disk capacity guard')
    old = validate_frozen(PREVIOUS); contract = baseline_contract()
    validate_child_change((PREVIOUS/'source/benchmarks/cifar_normalized_child.py').read_text(),
                          (ROOT/'benchmarks/cifar_normalized_env_child.py').read_text())
    validate_outer_change((PREVIOUS/'source/benchmarks/cifar_normalized.py').read_text(),
                          (ROOT/'benchmarks/cifar_normalized_env.py').read_text())
    config = read(PREVIOUS/'config.json'); validate_config(config)
    bundle.mkdir()
    shutil.copytree(PREVIOUS/'source', bundle/'source')
    shutil.copytree(PREVIOUS/'upstream_reference', bundle/'upstream_reference')
    for name in NEW:
        shutil.copy2(ROOT/'benchmarks'/name, bundle/'source/benchmarks'/name)
    shutil.copy2(ROOT/'tests/test_launch_environment.py', bundle/'source/tests/test_launch_environment.py')
    for name in ('config.json', 'split.json', 'data_prepared.json'):
        shutil.copy2(PREVIOUS/name, bundle/name)
    (bundle/'environment_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    (bundle/'supervisor_template.conf').write_text(supervisor_template(bundle, contract))
    (bundle/'PROTOCOL.md').write_text('''# New prospective normalized-input environment diagnostic; prepared only

Four fresh QR20 controls: allocation420/3500/3500/420,25disposable QR10 smoke and
25measured each,200total. Model1536/depth1/heads24/seed139/batch256, original recipe,
Adam(.9,.999)/covariance.99/currentV/strictM/bothfactors and all original graph,
earlier replays and normalized observer unchanged by AST. No historical states reused.
Internal900/external960/child210seconds, allocated24GiB, purpose RAM256MiB,
evidence100MiB expected<25MiB. RAM snapshots/native references exist only for the
already declared rollback/parity/ownership diagnostic, then expire. No tensor,
feature, model weight, checkpoint or trace files; no retry/queue/automatic longer job.

Correction: supervisor explicitly supplies all retained baseline-before-smoke
tracked environment variables except CUDA_VISIBLE_DEVICES. Compiler threads4,
OMP4/MKL4/no-bytecode1 are exact values; no unset/default equivalence. No global
.env/cache/library/driver/precision/clock/GC changes. Inherited extra tracked keys
are rejected. Baseline after-measured environment/library records are separately
retained, including the compiler-created Triton cache key. Do not preseed that key.

Prospective policy permits ONLY a single CUDA_VISIBLE_DEVICES UUID delta from the
historical baseline to the selected claimed device. No numerical/backend delta.
No device selected yet. Before starting a NEW private supervisor program, bind
physical index/single UUID in launch_manifest.json with exact baseline UUID,
manifest/config/environment-contract/supervisor-template hashes and concrete
rendered command/config. Verify inherited environment before a GPU workload;
the outer and child repeat read-only environment checks. The child compares full
retained backend metadata before the first smoke optimizer and after measurement.
Every missing/extra/value delta is reported; no first-key early return. Source and
historical baseline inputs are hash-checked. No dynamic device exemption.

Use summarize_normalized_env.py: preserve all prior observer/repetition/ABI/RNG/
storage/code/finite/resource gates and apply this prospectively frozen baseline
policy, requiring all eight exact phase/early-check records. Different normalized
CLS can localize replay variation before final affine/cast only if ALL gates pass
and fresh first-gradient variation exists. Otherwise inconclusive. The actual cold
first-step normalized tensor remains uncaptured; no named operator, regression
cause, learning/timing claim or default promotion. Historical inconclusive run,
May failed gate and all old source/config/summary remain frozen and unchanged.
''')
    manifest = dict(old)
    manifest.update(source_sha256={str(p.relative_to(bundle/'source')): sha(p)
                                  for p in (bundle/'source').rglob('*') if p.is_file()},
                    protocol_sha256=sha(bundle/'PROTOCOL.md'),
                    environment_contract_sha256=sha(bundle/'environment_contract.json'),
                    supervisor_template_sha256=sha(bundle/'supervisor_template.conf'),
                    previous_bundle=str(PREVIOUS), previous_manifest_sha256=sha(PREVIOUS/'manifest.json'),
                    previous_summary_sha256=sha(PREVIOUS/'summary.json'),
                    declared_baseline_backend_deltas=copy.deepcopy(POLICY), device=None,
                    gpu_selected=False, launch_pending=True,
                    environment_correction='Explicit historical compiler threads4; single UUID delta declared prospectively')
    (bundle/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (bundle/'source_worktree_status.txt').write_text(subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True))
    (bundle/'README.md').write_text('Prepared only. No GPU selected, launch manifest, claim, queue or attempt. Use the new environment contract and NEXT_TASK.md before launch. Historical diagnostic remains inconclusive.\n')
    (bundle/'PREPARATION_STATUS.json').write_text(json.dumps(dict(status='PREPARED_NOT_LAUNCHED', gpu_selected=False, updates_planned=200), indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--bundle', type=Path, required=True)
    m = prepare(p.parse_args().bundle)
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED', source_files=len(m['source_sha256']))))
