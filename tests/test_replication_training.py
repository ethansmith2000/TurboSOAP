"""Only new seed/order/strict-lineage/pair reset and failure semantics."""
import copy
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from replication_training_contract import ARMS, BASELINE, SEED, protocol, recipe, run_replication, validate_inputs
from matched_training_contract import RandomStreams, SharedInitialState, digest

spec = importlib.util.spec_from_file_location('_replication_cpu_components', Path(__file__).with_name('test_cadence_training.py'))
components = importlib.util.module_from_spec(spec)
spec.loader.exec_module(components)


class CPU(components.CPU):
    def __init__(self, fault=None, seed=SEED):
        super().__init__(fault)
        self.snapshot.close()
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.model = components.Tiny(); self.active = self.model; self.config['seed'] = seed
        self.streams = RandomStreams(torch.Generator().manual_seed(seed + 1001),
            SimpleNamespace(rng=np.random.default_rng(seed + 2001), counts={'mixup': 0}))
        self.snapshot = SharedInitialState(self.model, self.streams)
        self.initial_model = self.snapshot.initial_digest; self.initial_rng = digest(self.streams.capture())

    def make_batch(self, index):
        indices, x, target = super().make_batch(index)
        if self.fault == 'preflight' and self.phase == 'probe' and self.arm == 'warm20':
            x += .01
        return indices, x, target

    def before_optimizer_step(self, arm, index):
        if self.fault == 'training' and arm == 'warm20' and index == 1:
            raise RuntimeError('Injected replication failure')
        super().before_optimizer_step(arm, index)


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail(*a, **k):
        raise AssertionError('Replication CPU test attempted CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', fail)
    yield
    assert not torch.cuda.is_initialized()


def tiny_protocol():
    result = protocol()
    result.update(disposable_qr10_smoke_updates=1, steps_per_arm=3, observation_steps=[0, 1, 2, 3])
    return result


def test_only_declared_seed_and_pair_recipe_change():
    old = json.loads((BASELINE / 'config.json').read_text())
    new = recipe(old)
    assert new['seed'] == 140 and old['seed'] == 139
    assert {k: v for k, v in new.items() if k not in ('seed', 'matched_comparison')} == {
        k: v for k, v in old.items() if k not in ('seed', 'matched_comparison')}
    validate_inputs(new, protocol(), old, json.loads((BASELINE / 'protocol.json').read_text()))
    assert protocol()['maximum_optimizer_updates'] == 7025
    assert tuple(protocol()['execution_order']) == ARMS == ('qr20', 'warm20')


@pytest.mark.parametrize('field,value', [('seed', 141), ('lr', .0006), ('shampoo_beta', .999)])
def test_strict_recipe_rejects_undeclared_changes(field, value):
    old = json.loads((BASELINE / 'config.json').read_text()); new = recipe(old); new[field] = value
    with pytest.raises(ValueError, match='Only declared'):
        validate_inputs(new, protocol(), old, json.loads((BASELINE / 'protocol.json').read_text()))


def test_pair_private_order_exact_reset_preflight_and_release():
    import cadence_training_contract as original_contract
    import cadence_training_runtime as original_runtime
    old_order = original_runtime.execution_order(); old_arms = original_contract.ARMS
    runtime = CPU(); report = run_replication(runtime, tiny_protocol())
    assert report['status'] == 'completed' and report['paired_recipe_preflight_qualified']
    assert [r['arm'] for r in report['runs']] == ['qr20', 'warm20']
    assert len(runtime.updates) == 6
    probes = [r for event, r in runtime.events if event == 'preflight_probe']
    assert len(probes) == 2 and probes[0]['observation'] == probes[1]['observation']
    assert original_contract.ARMS == old_arms and original_runtime.execution_order() == old_order
    assert report['snapshot_release']['all_registered_released'] and runtime.closed and runtime.snapshot.closed
    assert all(r['release']['passed'] for r in report['runs'])
    assert not report['quality_qualified'] and not report['timing_qualified']


@pytest.mark.parametrize('fault', ['preflight', 'training'])
def test_pair_first_failure_stops_restores_and_releases(fault):
    runtime = CPU(fault); report = run_replication(runtime, tiny_protocol())
    assert report['status'] == 'failed' and not report['paired_recipe_preflight_qualified']
    assert not report['cleanup_errors'] and report['abort_audit'] is not None
    assert runtime.snapshot.closed and runtime.closed and report['snapshot_release']['all_registered_released']
    assert digest(runtime.model.state_dict()) == runtime.initial_model
    assert digest(runtime.streams.capture()) == runtime.initial_rng
    if fault == 'preflight':
        assert not runtime.updates
    else:
        assert ('warm20', 1) not in runtime.updates


def test_independent_seed_changes_initial_state_reproducibly():
    first = CPU(seed=139); second = CPU(seed=140); repeated = CPU(seed=140)
    assert first.initial_model != second.initial_model == repeated.initial_model
    assert first.initial_rng != second.initial_rng == repeated.initial_rng
    for runtime in (first, second, repeated):
        runtime.snapshot.close()
