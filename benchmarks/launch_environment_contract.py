"""Read-only, stdlib launch gates for a NEW normalized-input campaign.

Historical records remain exact; only a prospective single-device binding can
change CUDA_VISIBLE_DEVICES. Never collapse missing keys into default values.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re

PREFIXES = ('TORCH', 'TRITON', 'CUDA', 'CUBLAS', 'CUDNN', 'NVIDIA_TF32',
            'OMP', 'MKL', 'PYTHONDONTWRITEBYTECODE')
PHASES = ('backend_before_smoke', 'backend_after_measured')
POLICY = {'allowed_historical_environment_delta': ['CUDA_VISIBLE_DEVICES'],
          'binding': 'single UUID and physical index declared before launch',
          'numerical_backend_deltas': [], 'compile_threads': '4'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def environment(env):
    return {k: v for k, v in env.items() if k.startswith(PREFIXES)}


def differences(left, right, path=''):
    """Enumerate ALL differences, including values shared by unequal key sets."""
    if type(left) is not type(right):
        return [{'path': path, 'kind': 'type', 'expected': left, 'actual': right}]
    if isinstance(left, dict):
        rows = []
        for key in sorted(left.keys() | right.keys()):
            name = (path + '.' + key).lstrip('.')
            if key not in right:
                rows.append(dict(path=name, kind='missing', expected=left[key]))
            elif key not in left:
                rows.append(dict(path=name, kind='extra', actual=right[key]))
            else:
                rows.extend(differences(left[key], right[key], name))
        return rows
    return [] if left == right else [dict(path=path, kind='value', expected=left, actual=right)]


def validate_contract(contract):
    if contract.get('policy') != POLICY or set(contract.get('phases', {})) != set(PHASES):
        raise ValueError('Undeclared environment/backend policy')
    before, after = (contract['phases'][p] for p in PHASES)
    for record in (before, after):
        env = record['environment']
        if environment(env) != env or not all(isinstance(v, str) for v in env.values()):
            raise ValueError('Invalid recorded environment')
        for key in ('TORCHINDUCTOR_COMPILE_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            if env.get(key) != '4':
                raise ValueError('Historical thread override must be explicit4')
        if env.get('PYTHONDONTWRITEBYTECODE') != '1':
            raise ValueError('No-bytecode setting changed')
    if before['environment']['CUDA_VISIBLE_DEVICES'] != after['environment']['CUDA_VISIBLE_DEVICES']:
        raise ValueError('Historical baseline device changed between phases')
    # Retain phase-specific cache/library initialization; never transplant it.
    return contract


def binding(contract, gpu):
    validate_contract(contract)
    if set(gpu) != {'index', 'uuid'} or type(gpu['index']) is not int or not 0 <= gpu['index'] < 8:
        raise ValueError('One physical GPU index required')
    uuid = gpu['uuid']
    if not isinstance(uuid, str) or not re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', uuid):
        raise ValueError('One UUID required')
    return dict(field='CUDA_VISIBLE_DEVICES',
                baseline_uuid=contract['phases'][PHASES[0]]['environment']['CUDA_VISIBLE_DEVICES'],
                selected_uuid=uuid, physical_index=gpu['index'])


def expected_environment(contract, device, phase=PHASES[0]):
    gpu = dict(index=device['physical_index'], uuid=device['selected_uuid'])
    if device != binding(contract, gpu) or phase not in PHASES:
        raise ValueError('Undeclared device or phase')
    env = dict(contract['phases'][phase]['environment'])
    env['CUDA_VISIBLE_DEVICES'] = device['selected_uuid']
    return env


def check_environment(contract, device, env, claim_index):
    delta = differences(expected_environment(contract, device), environment(env), 'environment')
    if claim_index != str(device['physical_index']):
        delta.append(dict(path='GPU_CLAIM_INDICES', kind='value',
                          expected=str(device['physical_index']), actual=claim_index))
    return dict(passed=not delta, differences=delta, device_binding=device)


def check_backend(contract, device, record, phase):
    expected = copy.deepcopy(contract['phases'][phase])
    expected['environment'] = expected_environment(contract, device, phase)
    delta = differences(expected, record)
    return dict(passed=not delta, phase=phase, differences=delta, device_binding=device)


def load_contract(bundle):
    bundle = Path(bundle)
    manifest = read(bundle/'manifest.json')
    for name, key in [('config.json', 'config_sha256'), ('PROTOCOL.md', 'protocol_sha256'),
                      ('environment_contract.json', 'environment_contract_sha256')]:
        if sha(bundle/name) != manifest[key]:
            raise ValueError('Frozen '+name+' changed')
    contract = validate_contract(read(bundle/'environment_contract.json'))
    for record in contract['baseline_inputs']:
        path = Path(record['path'])
        if sha(path) != record['sha256']:
            raise ValueError('Historical baseline input changed: '+str(path))
        if record.get('kind') == 'result':
            value = read(path)
            if value.get('status') != 'completed' or any(
                    differences(contract['phases'][phase], value.get(phase, {})) for phase in PHASES):
                raise ValueError('Contract does not reproduce historical phase records')
    if sum(r.get('kind') == 'result' for r in contract['baseline_inputs']) != 4:
        raise ValueError('Require four retained baseline records')
    for name, digest in manifest['source_sha256'].items():
        if sha(bundle/'source'/name) != digest:
            raise ValueError('Frozen source changed: '+name)
    return contract


def read_launch(bundle):
    bundle = Path(bundle)
    contract = load_contract(bundle); manifest = read(bundle/'manifest.json')
    launch = read(bundle/'launch_manifest.json')
    for name, key in [('manifest.json', 'manifest_sha256'), ('config.json', 'config_sha256'),
                      ('environment_contract.json', 'environment_contract_sha256'),
                      ('supervisor_template.conf', 'supervisor_template_sha256')]:
        if launch.get(key) != sha(bundle/name):
            raise ValueError('Launch not bound to '+name)
    if sha(bundle/'supervisor_template.conf') != manifest['supervisor_template_sha256']:
        raise ValueError('Supervisor template changed')
    if launch.get('device_binding') != binding(contract, launch['gpu']):
        raise ValueError('Prospective device binding changed')
    return contract, launch


def verify_environment_launch(bundle, env=None):
    env = os.environ if env is None else env
    contract, launch = read_launch(bundle)
    report = check_environment(contract, launch['device_binding'], env, env.get('GPU_CLAIM_INDICES'))
    if not report['passed']:
        raise ValueError('Launch environment mismatch: '+json.dumps(report, sort_keys=True))
    return launch


def verify_backend_launch(bundle, record, phase):
    contract, launch = read_launch(bundle)
    report = check_backend(contract, launch['device_binding'], record, phase)
    if not report['passed']:
        raise ValueError('Historical backend mismatch: '+json.dumps(report, sort_keys=True))
    return report


def validate_child_change(before, after):
    stripped = after
    expected_regions = {
        'ENVIRONMENT_PRELAUNCH': 'verify_environment_launch(b)',
        'ENVIRONMENT_BEFORE_SMOKE': "report['environment_before_smoke_check']=verify_backend_launch(b, report['backend_before_smoke'], 'backend_before_smoke');save()",
        'ENVIRONMENT_AFTER_MEASURED': "report['environment_after_measured_check']=verify_backend_launch(b, report['backend_after_measured'], 'backend_after_measured');save()",
    }
    for name in ('ENVIRONMENT_PRELAUNCH', 'ENVIRONMENT_BEFORE_SMOKE', 'ENVIRONMENT_AFTER_MEASURED'):
        start, end = '# '+name+'_START', '# '+name+'_END'
        if stripped.count(start) != 1 or stripped.count(end) != 1:
            raise ValueError('Missing declared region '+name)
        a, b = stripped.index(start), stripped.index(end)
        if a >= b:
            raise ValueError('Reversed region')
        region = '\n'.join(line.strip() for line in stripped[a+len(start):b].splitlines() if line.strip())
        if ast.dump(ast.parse(region)) != ast.dump(ast.parse(expected_regions[name])):
            raise ValueError('Read-only gate region changed: '+name)
        stripped = stripped[:a] + stripped[b+len(end):]
    line = 'from launch_environment_contract import verify_environment_launch, verify_backend_launch\n'
    if stripped.count(line) != 1:
        raise ValueError('One additive import required')
    stripped = stripped.replace(line, '')
    if ast.dump(ast.parse(stripped)) != ast.dump(ast.parse(before)):
        raise ValueError('Original primary/observer code changed')
    if not after.index('verify_environment_launch(b)') < after.index('torch.cuda.device_count()'):
        raise ValueError('Environment gate must precede CUDA initialization')
    if not after.index('verify_backend_launch(b, report[\'backend_before_smoke\']') < after.index('smoke=build_optimizer'):
        raise ValueError('Backend gate must precede first optimizer update')
    if not (after.index("report['backend_before_smoke']=backend_record();save()") <
            after.index('# ENVIRONMENT_BEFORE_SMOKE_START') < after.index('host=CompactDispatchHostTiming()')):
        raise ValueError('Before-smoke gate must immediately follow backend capture')
    if not (after.index("report['backend_after_measured']=backend_record()") <
            after.index('# ENVIRONMENT_AFTER_MEASURED_START') < after.index('def invariants()')):
        raise ValueError('Post-measurement gate must precede unchanged replays')
    return True


def validate_outer_change(before, after):
    line = 'from launch_environment_contract import verify_environment_launch\n'
    stripped = after.replace(line, '').replace('    verify_environment_launch(b)\n', '')
    expected = before.replace('cifar_normalized_child.py', 'cifar_normalized_env_child.py')
    if after.count(line) != 1 or after.count('    verify_environment_launch(b)\n') != 1:
        raise ValueError('One outer environment gate required')
    if ast.dump(ast.parse(stripped)) != ast.dump(ast.parse(expected)):
        raise ValueError('Outer changed beyond entry and read-only gate')
    return True
