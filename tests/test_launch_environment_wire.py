import copy
import json
import os

import pytest
import torch

from benchmarks.execution_fingerprints import backend_record
from benchmarks.launch_environment_contract import PHASES, binding, check_backend as original_check, differences
from benchmarks.launch_environment_wire_contract import canonical_backend, check_backend
from benchmarks.launch_environment_wire_contract import validate_child_change, validate_outer_change, validate_summary_change
from benchmarks.prepare_normalized_env import baseline_contract


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    def fail(*args, **kwargs):
        raise AssertionError('CPU metadata comparison must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', fail)
    yield
    assert not torch.cuda.is_initialized()


def fixture():
    contract = baseline_contract()
    device = binding(contract, dict(index=3, uuid='GPU-55fc0669-93bd-9223-674d-212086d4258b'))
    record = copy.deepcopy(contract['phases'][PHASES[0]])
    record['environment']['CUDA_VISIBLE_DEVICES'] = device['selected_uuid']
    record['torch'] = torch.__version__
    return contract, device, record


def test_real_torch_version_reproduces_old_failure_and_new_wire_pass():
    assert isinstance(torch.__version__, str) and type(torch.__version__) is not str
    contract, device, record = fixture()
    old = original_check(contract, device, record, PHASES[0])
    assert not old['passed'] and [(x['path'], x['kind']) for x in old['differences']] == [('torch', 'type')]
    assert check_backend(contract, device, record, PHASES[0])['passed']
    assert canonical_backend(record) == json.loads(json.dumps(record, allow_nan=False))


def test_actual_cpu_backend_record_compared_to_its_persisted_metadata():
    actual = backend_record()
    saved = json.loads(json.dumps(actual, allow_nan=False))
    assert differences(saved, actual) == [dict(path='torch', kind='type',
                                              expected=str(torch.__version__), actual=torch.__version__)]
    assert not differences(saved, canonical_backend(actual))
    assert canonical_backend(actual)['loaded_libraries'] == actual['loaded_libraries']
    assert canonical_backend(actual)['environment'] == actual['environment']


@pytest.mark.parametrize('change', ['version', 'precision', 'matmul', 'library', 'missing', 'extra', 'uuid', 'threads'])
def test_serialization_cannot_hide_real_backend_deltas(change):
    contract, device, record = fixture()
    if change == 'version': record['torch'] = type(torch.__version__)('different')
    elif change == 'precision': record['matmul_precision'] = 'highest'
    elif change == 'matmul': record['matmul']['allow_tf32'] = False
    elif change == 'library': record['loaded_libraries'].append('/unexpected/library.so')
    elif change == 'missing': record['environment'].pop('TORCHINDUCTOR_COMPILE_THREADS')
    elif change == 'extra': record['environment']['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    elif change == 'uuid': record['environment']['CUDA_VISIBLE_DEVICES'] = contract['phases'][PHASES[0]]['environment']['CUDA_VISIBLE_DEVICES']
    else: record['environment']['TORCHINDUCTOR_COMPILE_THREADS'] = '32'
    assert not check_backend(contract, device, record, PHASES[0])['passed']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf'), object(), {1: 'value'}])
def test_nonfinite_unknown_or_coerced_object_keys_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        canonical_backend(dict(value=value))


def test_wire_canonicalization_idempotent_and_preserves_typed_values():
    value = dict(flag=True, count=4, label='4', empty=None, floats=[.1, -0., 1e-20], shape=(256, 1536))
    wire = canonical_backend(value)
    assert canonical_backend(wire) == wire
    assert wire['flag'] is True and type(wire['count']) is int and wire['label'] == '4'
    assert wire['shape'] == [256, 1536]


def test_new_entry_sources_change_only_checker_import_and_filename():
    from benchmarks.prepare_normalized_env_wire import FAILED, ROOT
    for before, after, validator in [
        ('cifar_normalized_env_child.py', 'cifar_normalized_env_wire_child.py', validate_child_change),
        ('cifar_normalized_env.py', 'cifar_normalized_env_wire.py', validate_outer_change),
        ('summarize_normalized_env.py', 'summarize_normalized_env_wire.py', validate_summary_change)]:
        old = (FAILED/'source/benchmarks'/before).read_text(); new = (ROOT/'benchmarks'/after).read_text()
        assert validator(old, new)
        with pytest.raises(ValueError): validator(old, new.replace('def main(', 'def changed_main('))


def test_new_wire_bundle_frozen_config_and_environment_unchanged(tmp_path):
    from benchmarks.prepare_normalized_env_wire import FAILED, prepare
    from benchmarks.launch_environment_contract import load_contract
    bundle = tmp_path/'wire'; manifest = prepare(bundle)
    assert not manifest['gpu_selected'] and manifest['device'] is None
    assert not (bundle/'launch_manifest.json').exists()
    assert (bundle/'config.json').read_bytes() == (FAILED/'config.json').read_bytes()
    assert (bundle/'environment_contract.json').read_bytes() == (FAILED/'environment_contract.json').read_bytes()
    assert load_contract(bundle) == load_contract(FAILED)
    with pytest.raises(ValueError): prepare(bundle)
