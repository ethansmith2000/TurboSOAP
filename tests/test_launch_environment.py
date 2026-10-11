import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from benchmarks.launch_environment_contract import (
    PHASES, POLICY, binding, check_backend, check_environment, differences,
    environment, expected_environment, load_contract, read_launch, sha,
    validate_child_change, validate_outer_change, verify_environment_launch,
)
from benchmarks.prepare_normalized_env import (
    BASE, PREVIOUS, ROOT, baseline_contract, prepare, supervisor_template,
)


GPU = {'index': 3, 'uuid': 'GPU-55fc0669-93bd-9223-674d-212086d4258b'}


@pytest.fixture(autouse=True)
def prohibit_cuda_initialization(monkeypatch):
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    def fail(*args, **kwargs):
        raise AssertionError('CPU contract tests must never initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', fail)
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def contract():
    return baseline_contract()


def test_all_four_retained_baselines_exact_and_phase_specific(contract):
    assert len(contract['baseline_inputs']) == 7
    assert contract['policy'] == POLICY
    assert 'TRITON_CACHE_DIR' not in contract['phases'][PHASES[0]]['environment']
    assert contract['phases'][PHASES[1]]['environment']['TRITON_CACHE_DIR'] == '/tmp/torchinductor_root/triton/0'


def test_actual_failed_environment_reports_both_differences(contract):
    p = PREVIOUS/'contexts/plan420_flush25_r0_normalized_input/result.json'
    run = json.loads(p.read_text())
    delta = differences(contract['phases'][PHASES[0]], run[PHASES[0]])
    assert [(r['path'], r['kind']) for r in delta] == [
        ('environment.CUDA_VISIBLE_DEVICES', 'value'),
        ('environment.TORCHINDUCTOR_COMPILE_THREADS', 'missing')]
    # Even a prospectively bound device cannot hide missing compiler settings.
    device = binding(contract, GPU)
    check = check_environment(contract, device, run[PHASES[0]]['environment'], '3')
    assert not check['passed']
    assert [r['path'] for r in check['differences']] == ['environment.TORCHINDUCTOR_COMPILE_THREADS']


@pytest.mark.parametrize('key', ['TORCHINDUCTOR_COMPILE_THREADS', 'OMP_NUM_THREADS',
    'MKL_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE', 'CUDA_HOME', 'CUDA_VERSION',
    'TORCHINDUCTOR_CACHE_DIR'])
def test_missing_or_changed_setting_never_defaulted(contract, key):
    device = binding(contract, GPU); env = expected_environment(contract, device)
    changed = dict(env); changed.pop(key)
    assert not check_environment(contract, device, changed, '3')['passed']
    env[key] = 'unexpected'
    assert not check_environment(contract, device, env, '3')['passed']


@pytest.mark.parametrize('key', ['TORCH_EXTRA', 'TRITON_CACHE_DIR', 'CUDA_EXTRA',
    'CUBLAS_WORKSPACE_CONFIG', 'CUDNN_EXTRA', 'NVIDIA_TF32_OVERRIDE', 'OMP_EXTRA', 'MKL_EXTRA'])
def test_extra_tracked_inheritance_rejected(contract, key):
    device = binding(contract, GPU); env = expected_environment(contract, device)
    env[key] = 'inherited'
    assert not check_environment(contract, device, env, '3')['passed']
    env.pop(key); env['UNRELATED_TEST_SETTING'] = 'ok'
    assert check_environment(contract, device, env, '3')['passed']


def test_wrong_or_multiple_claim_and_unbound_uuid(contract):
    device = binding(contract, GPU); env = expected_environment(contract, device)
    for claim in (None, '7', '3,7', '03'):
        assert not check_environment(contract, device, env, claim)['passed']
    env['CUDA_VISIBLE_DEVICES'] = contract['phases'][PHASES[0]]['environment']['CUDA_VISIBLE_DEVICES']
    assert not check_environment(contract, device, env, '3')['passed']
    for gpu in ({**GPU, 'uuid': GPU['uuid']+',GPU-second'}, {**GPU, 'index': True}, {**GPU, 'index': -1},
                {**GPU, 'uuid': 'GPU-fake'}):
        with pytest.raises(ValueError): binding(contract, gpu)
    altered = {**device, 'baseline_uuid': GPU['uuid']}
    with pytest.raises(ValueError): expected_environment(contract, altered)


@pytest.mark.parametrize('phase', PHASES)
def test_full_backend_phase_gate_only_uuid_delta(contract, phase):
    device = binding(contract, GPU); record = copy.deepcopy(contract['phases'][phase])
    record['environment']['CUDA_VISIBLE_DEVICES'] = GPU['uuid']
    assert check_backend(contract, device, record, phase)['passed']
    record['matmul_precision'] = 'highest'
    record['environment'].pop('TORCHINDUCTOR_COMPILE_THREADS')
    record['environment']['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    paths = {r['path'] for r in check_backend(contract, device, record, phase)['differences']}
    assert paths == {'matmul_precision', 'environment.TORCHINDUCTOR_COMPILE_THREADS',
                     'environment.CUBLAS_WORKSPACE_CONFIG'}


def test_source_gates_exact_readonly_regions_and_first_smoke_order():
    before = (PREVIOUS/'source/benchmarks/cifar_normalized_child.py').read_text()
    after = (ROOT/'benchmarks/cifar_normalized_env_child.py').read_text()
    assert validate_child_change(before, after)
    for change in (after.replace('cudnn.deterministic=True', 'cudnn.deterministic=False'),
                   after.replace("verify_environment_launch(b)", "verify_environment_launch(b);torch.cuda.empty_cache()"),
                   after.replace("'backend_before_smoke');save()", "'backend_after_measured');save()"),
                   after.replace("optimizer.step()", "optimizer.step();optimizer.step()")):
        with pytest.raises(ValueError): validate_child_change(before, change)
    outer = (ROOT/'benchmarks/cifar_normalized_env.py').read_text()
    old = (PREVIOUS/'source/benchmarks/cifar_normalized.py').read_text()
    assert validate_outer_change(old, outer)
    with pytest.raises(ValueError): validate_outer_change(old, outer.replace('if code:', 'if False:'))


def launch_fixture(bundle, contract):
    device = binding(contract, GPU)
    launch = dict(gpu=GPU, device_binding=device)
    for name, key in [('manifest.json', 'manifest_sha256'), ('config.json', 'config_sha256'),
                      ('environment_contract.json', 'environment_contract_sha256'),
                      ('supervisor_template.conf', 'supervisor_template_sha256')]:
        launch[key] = sha(bundle/name)
    (bundle/'launch_manifest.json').write_text(json.dumps(launch))
    return dict(expected_environment(contract, device), GPU_CLAIM_INDICES='3')


def test_new_preparation_and_bound_launch_remain_import_only_cpu(tmp_path, contract):
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    bundle = tmp_path/'new'; manifest = prepare(bundle)
    assert manifest['device'] is None and not manifest['gpu_selected']
    assert manifest['declared_baseline_backend_deltas'] == POLICY
    assert not (bundle/'launch_manifest.json').exists()
    assert load_contract(bundle) == contract
    with pytest.raises(ValueError): prepare(bundle)
    env = launch_fixture(bundle, contract)
    assert verify_environment_launch(bundle, env)['gpu'] == GPU
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('file', ['config.json', 'manifest.json', 'environment_contract.json',
    'supervisor_template.conf', 'source/benchmarks/cifar_normalized_env_child.py'])
def test_corrupted_bundle_rejected_before_gpu(tmp_path, contract, file):
    bundle = tmp_path/'new'; prepare(bundle); env = launch_fixture(bundle, contract)
    path = bundle/file; path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError): verify_environment_launch(bundle, env)


@pytest.mark.parametrize('kind', ['manifest.json', 'config.json'])
def test_corrupted_baseline_manifest_and_config_rejected(tmp_path, contract, kind):
    # Substitute disposable baseline fixtures; never mutate protected history.
    bundle = tmp_path/'new'; prepare(bundle)
    value = json.loads((bundle/'environment_contract.json').read_text())
    fake = tmp_path/kind; fake.write_text('{"config_sha256":"original"}')
    value['baseline_inputs'].append(dict(path=str(fake), sha256=sha(fake), kind=kind))
    (bundle/'environment_contract.json').write_text(json.dumps(value))
    m = json.loads((bundle/'manifest.json').read_text()); m['environment_contract_sha256'] = sha(bundle/'environment_contract.json')
    (bundle/'manifest.json').write_text(json.dumps(m))
    launch_fixture(bundle, contract); fake.write_text('{"config_sha256":"corrupted"}')
    with pytest.raises(ValueError, match='Historical baseline'): read_launch(bundle)


def test_contract_cannot_forge_historical_phase_even_with_rebound_hash(tmp_path, contract):
    bundle = tmp_path/'new'; prepare(bundle)
    value = json.loads((bundle/'environment_contract.json').read_text())
    value['phases'][PHASES[0]]['matmul_precision'] = 'highest'
    (bundle/'environment_contract.json').write_text(json.dumps(value))
    m = json.loads((bundle/'manifest.json').read_text()); m['environment_contract_sha256'] = sha(bundle/'environment_contract.json')
    (bundle/'manifest.json').write_text(json.dumps(m))
    launch_fixture(bundle, contract)
    with pytest.raises(ValueError, match='historical phase'): read_launch(bundle)


def test_new_qualifier_cannot_upgrade_failed_historical_environment(contract):
    from benchmarks.summarize_normalized_env import apply_backend_gate
    runs = [json.loads((PREVIOUS/'contexts'/c['label']/'result.json').read_text())
            for c in json.loads((PREVIOUS/'config.json').read_text())['execution_diagnostic']['contexts']]
    value = dict(localization_supported=True, pairs=[dict(classification='VARIATION_UPSTREAM_OF_FINAL_AFFINE_CAST')])
    result = apply_backend_gate(value, runs, contract, binding(contract, GPU))
    assert not result['localization_supported'] and not result['prospective_baseline_backend_match']
    assert result['pairs'][0]['classification'] == 'INCONCLUSIVE_PARITY_ABI_OR_RESOURCE_GATE'


def test_new_qualifier_requires_all_early_checks_and_keeps_other_gates(contract):
    from benchmarks.summarize_normalized_env import apply_backend_gate
    device = binding(contract, GPU); runs = []
    for i in range(4):
        run = dict(context=dict(label=str(i)))
        for phase in PHASES:
            record = copy.deepcopy(contract['phases'][phase]); record['environment']['CUDA_VISIBLE_DEVICES'] = GPU['uuid']
            run[phase] = record
            key = 'environment_'+('before_smoke' if phase == PHASES[0] else 'after_measured')+'_check'
            run[key] = check_backend(contract, device, record, phase)
        runs.append(run)
    assert apply_backend_gate(dict(localization_supported=True), runs, contract, device)['localization_supported']
    assert not apply_backend_gate(dict(localization_supported=False), runs, contract, device)['localization_supported']
    runs[0].pop('environment_before_smoke_check')
    assert not apply_backend_gate(dict(localization_supported=True), runs, contract, device)['localization_supported']


def test_actual_supervisor_inheritance_is_not_silently_filtered(tmp_path, contract):
    # Real supervisor, disposable stdlib-only CPU child; no research services.
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    target = tmp_path/'captured.json'; script = tmp_path/'capture.py'
    script.write_text('import os,json,pathlib\npathlib.Path('+repr(str(target))+').write_text(json.dumps(dict(os.environ)))\n')
    template = supervisor_template(tmp_path, contract)
    line = next(line for line in template.splitlines() if line.startswith('environment='))
    config = tmp_path/'supervisord.conf'
    config.write_text(f'''[supervisord]
nodaemon=true
logfile={tmp_path}/supervisor.log
pidfile={tmp_path}/supervisor.pid
childlogdir={tmp_path}
[program:cpu_probe]
command=/usr/bin/python3 {script}
{line}
autostart=true
autorestart=false
startsecs=0
stdout_logfile={tmp_path}/child.log
stderr_logfile={tmp_path}/child.err
''')
    inherited = dict(os.environ, TORCH_UNEXPECTED_INHERITED='must reject', TORCHINDUCTOR_COMPILE_THREADS='99')
    with (tmp_path/'supervisor.stdout').open('w') as log:
        proc = subprocess.Popen(['/usr/local/bin/supervisord', '-n', '-c', str(config)], env=inherited, stdout=log, stderr=log)
        try:
            deadline = time.monotonic()+10
            while not target.exists() and time.monotonic() < deadline:
                time.sleep(.05)
            assert target.exists()
        finally:
            proc.terminate(); proc.wait(timeout=10)
    env = json.loads(target.read_text())
    assert env['CUDA_VISIBLE_DEVICES'] == ''  # child never sees any GPU
    assert env['TORCHINDUCTOR_COMPILE_THREADS'] == '4'  # explicit override wins
    assert env['TORCH_UNEXPECTED_INHERITED'] == 'must reject'
    env['CUDA_VISIBLE_DEVICES'] = GPU['uuid']  # model the later claim, no CUDA calls
    result = check_environment(contract, binding(contract, GPU), env, '3')
    assert not result['passed']
    assert any(r['path'] == 'environment.TORCH_UNEXPECTED_INHERITED' for r in result['differences'])
