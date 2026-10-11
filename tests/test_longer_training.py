"""Horizon/resource contract, retained paired gates, and failure release checks."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from longer_training_contract import BASELINE, HORIZON, OBSERVATIONS, protocol, recipe, run_longer, validate_inputs
from matched_training_contract import digest

spec = importlib.util.spec_from_file_location('_longer_cpu_components', Path(__file__).with_name('test_replication_training.py'))
components = importlib.util.module_from_spec(spec); spec.loader.exec_module(components)


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda *a, **k: pytest.fail('CPU test attempted CUDA'))
    yield
    assert not torch.cuda.is_initialized()


def inputs():
    return json.loads((BASELINE / 'config.json').read_text()), json.loads((BASELINE / 'protocol.json').read_text())


def test_longer_contract_keeps_learning_recipe_and_final_counters():
    c, p = inputs(); new = recipe(c); declared = protocol(p)
    assert {k: v for k, v in new.items() if k not in ('matched_comparison', 'max_wall_seconds')} == {
        k: v for k, v in c.items() if k not in ('matched_comparison', 'max_wall_seconds')}
    validate_inputs(new, declared, c, p)
    assert declared['steps_per_arm'] == HORIZON == 10000
    assert declared['maximum_optimizer_updates'] == 20025 and declared['all_arm_preflight_backward_calls'] == 2
    assert declared['observation_steps'] == list(OBSERVATIONS)
    assert declared['schedule'] == p['schedule']
    assert declared['bounds']['internal_wall_seconds'] == new['max_wall_seconds'] == 1200
    assert declared['bounds']['external_watchdog_seconds'] == 1260
    assert declared['bounds']['maximum_evidence_bytes'] == 200 * 2**20
    from cadence_training_contract import expected_counts
    for a, method, reset in [('qr20', 'qr', 0), ('warm20', 'warm', 200)]:
        assert declared['cadence']['final_counts'][a] == expected_counts(HORIZON, method, 20, reset)


@pytest.mark.parametrize('kind', ['lr', 'seed', 'horizon', 'observations', 'bound', 'order'])
def test_reject_even_resealed_undeclared_changes(kind):
    c, p = inputs(); new = recipe(c); declared = protocol(p)
    if kind in ('lr', 'seed'):
        new[kind] = .0006 if kind == 'lr' else 141
    elif kind == 'horizon':
        declared['steps_per_arm'] = 11000
    elif kind == 'observations':
        declared['observation_steps'].remove(3500)
    elif kind == 'bound':
        declared['bounds']['internal_wall_seconds'] = 1400
    else:
        declared['execution_order'].reverse()
    with pytest.raises(ValueError):
        validate_inputs(new, declared, c, p)


@pytest.mark.parametrize('fault', [None, 'preflight', 'training'])
def test_private_longer_dispatch_exact_gates_and_first_failure_release(fault):
    import cadence_training_contract as cc
    import cadence_training_runtime as cr
    before = (cc.ARMS, cr.execution_order())
    runtime = components.CPU(fault)
    declared = protocol(); declared.update(disposable_qr10_smoke_updates=1,
        steps_per_arm=3, observation_steps=[0, 1, 2, 3])
    report = run_longer(runtime, declared)
    assert not report['cleanup_errors'] and report['snapshot_release']['all_registered_released']
    assert runtime.closed and runtime.snapshot.closed
    assert (cc.ARMS, cr.execution_order()) == before
    assert not report['quality_qualified'] and not report['timing_qualified']
    if fault is None:
        assert report['status'] == 'completed' and len(runtime.updates) == 6
        probes = [r for e, r in runtime.events if e == 'preflight_probe']
        assert probes[0]['observation'] == probes[1]['observation']
        assert all(r['release']['passed'] for r in report['runs'])
    else:
        assert report['status'] == 'failed' and report['abort_audit'] is not None
        assert digest(runtime.model.state_dict()) == runtime.initial_model
        assert digest(runtime.streams.capture()) == runtime.initial_rng
        assert not runtime.updates if fault == 'preflight' else ('warm20', 1) not in runtime.updates
