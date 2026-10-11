"""New cadence/reset/failure checks; no old experiment or broad suite rerun."""
from contextlib import nullcontext
import copy
import json
import random
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from cadence_training_contract import (ARMS, build_arm, expected_counts, execution_order,
    protocol, validate_counters)
from cadence_training_runtime import run_cadence
from matched_training_contract import COMMON, RandomStreams, SharedInitialState, digest

CONFIG = dict(COMMON, precondition_mode='all', covariance_compute_dtype='float32',
    clip_norm=1., min_lr_ratio=.01)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.2))
        self.head = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.head(self.blocks(x))


class Event:
    def record(self):
        self.time = time.monotonic()


class CPU:
    def __init__(self, fault=None):
        torch.manual_seed(139); np.random.seed(139); random.seed(139)
        self.model = Tiny(); self.active = self.model; self.config = CONFIG.copy()
        self.streams = RandomStreams(torch.Generator().manual_seed(1140),
            SimpleNamespace(rng=np.random.default_rng(3140), counts={'mixup': 0}))
        self.snapshot = SharedInitialState(self.model, self.streams)
        self.initial_model = self.snapshot.initial_digest
        self.initial_rng = digest(self.streams.capture())
        self.schedule_steps = 100; self.warmup_steps = 5
        self.events = []; self.updates = []; self.phase = None; self.arm = None
        self.fault = fault; self.closed = False

    def make_batch(self, index):
        x = torch.randn(5, 4, generator=self.streams.crop)
        x += float(self.streams.augmentation.rng.random()) + random.random() + float(np.random.random())
        self.streams.augmentation.counts['mixup'] += 1
        if self.fault == 'preflight' and self.phase == 'probe' and self.arm == 'warm10':
            x += .01
        return torch.arange(5) + index * 5, x, torch.tensor([0, 1, 2, 0, 1])

    def evaluate(self):
        with torch.no_grad():
            logits = self.model(torch.arange(20).reshape(5, 4).float() / 20)
            y = torch.tensor([0, 1, 2, 0, 1])
            return dict(ce=float(F.cross_entropy(logits, y)), correct=int((logits.argmax(1) == y).sum()))

    def emit(self, event, payload):
        self.events.append((event, json.loads(json.dumps(payload, allow_nan=False))))

    def before_smoke(self):
        self.phase = 'smoke'

    def after_smoke(self):
        assert digest(self.model.state_dict()) == self.initial_model
        assert digest(self.streams.capture()) == self.initial_rng

    def before_probe(self, arm):
        self.phase = 'probe'; self.arm = arm

    def before_training_step(self, arm, index):
        self.phase = 'train'; self.arm = arm

    def before_optimizer_step(self, arm, index):
        assert any(event == 'preflight_passed' for event, _ in self.events)
        if self.fault == 'training' and arm == 'warm10' and index == 1:
            raise RuntimeError('Injected cadence training failure')
        self.updates.append((arm, index))

    def after_measured(self):
        pass

    def smoke_parity(self, x):
        return dict(cpu_unit_test_only=True)

    def budget(self, resources=False):
        pass

    def autocast(self):
        return nullcontext()

    def marks(self):
        return [Event() for _ in range(4)]

    def elapsed(self, marks):
        return [(marks[i + 1].time - marks[i].time) * 1000 for i in range(3)]

    def synchronize(self):
        pass

    def close(self):
        self.closed = True
        return dict(passed=True, cpu_unit_test_only=True)

    def capture_abort_resources(self):
        return dict(cpu_unit_test_only=True)

    capture_abort_backend = capture_abort_resources
    capture_abort_code = capture_abort_resources


@pytest.fixture(autouse=True)
def block_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail(*a, **k):
        raise AssertionError('CPU preparation attempted CUDA initialization')
    monkeypatch.setattr(torch.cuda, '_lazy_init', fail)
    yield
    assert not torch.cuda.is_initialized()


def test_same_warm_policy_and_auxiliary_except_cadence():
    model = Tiny()
    a, b = build_arm(model, CONFIG, 'warm20'), build_arm(model, CONFIG, 'warm10')
    left = {k: v for k, v in a.matrix.param_groups[0].items() if k != 'params'}
    right = {k: v for k, v in b.matrix.param_groups[0].items() if k != 'params'}
    assert left.pop('precondition_frequency') == 20
    assert right.pop('precondition_frequency') == 10
    assert left == right and left['hard_reset_interval'] == 200
    assert left['variance_policy'] == 'permutation' and left['transport_precision'] == 'highest'
    assert left['research_movement_policy'] == 'fro_cubic2_cap05'
    assert [{k: v for k, v in g.items() if k != 'params'} for g in a.auxiliary.param_groups] == [
        {k: v for k, v in g.items() if k != 'params'} for g in b.auxiliary.param_groups]


@pytest.mark.parametrize('arm,qr,warm', [('qr20', 20, 0), ('warm20', 2, 18), ('warm10', 2, 38)])
def test_natural_supplied_gradient_counters_two_full_repair_cycles(arm, qr, warm):
    model = Tiny(); opt = build_arm(model, CONFIG, arm)
    for call in range(401):
        for p in model.parameters():
            p.grad = torch.full_like(p, .01)
        opt.step()
    rows = validate_counters(opt, 401)['parameters']
    assert all(row['actual']['research_qr_refreshes'] == qr for row in rows)
    assert all(row['actual']['research_warm_refreshes'] == warm for row in rows)
    p = opt.matrix.param_groups[0]['params'][0]
    opt.matrix.state[p]['research_warm_refreshes'] += 1
    with pytest.raises(RuntimeError, match='counter mismatch'):
        validate_counters(opt, 401)


def small_protocol():
    p = protocol()
    p.update(disposable_qr10_smoke_updates=1, steps_per_arm=3, observation_steps=[0, 1, 2, 3])
    return p


def test_three_arm_private_orchestration_reset_preflight_and_release():
    import matched_training_diagnostic_runtime as original
    original_arms = original.ARMS
    runtime = CPU(); report = run_cadence(runtime, small_protocol())
    assert report['status'] == 'completed' and report['paired_recipe_preflight_qualified']
    assert not report['quality_qualified'] and not report['timing_qualified'] and not report['qr_repeat_completed']
    assert [r['arm'] for r in report['runs']] == list(execution_order())
    assert len(runtime.updates) == 9
    assert original.ARMS == original_arms  # Parameterization never touches old module globals.
    assert runtime.closed and runtime.snapshot.closed and report['snapshot_release']['all_registered_released']
    assert all(r['release']['passed'] for r in report['runs'])
    probes = [row for event, row in runtime.events if event == 'preflight_probe']
    assert len(probes) == 3 and all(r['observation'] == probes[0]['observation'] for r in probes)
    final = [row for event, row in runtime.events if event == 'finished'][0]
    assert not final['quality_qualified'] and final['paired_recipe_preflight_qualified']


@pytest.mark.parametrize('fault', ['preflight', 'training'])
def test_new_arm_failure_aborts_and_releases(fault):
    runtime = CPU(fault); report = run_cadence(runtime, small_protocol())
    assert report['status'] == 'failed' and not report['paired_recipe_preflight_qualified']
    assert runtime.closed and runtime.snapshot.closed and report['snapshot_release']['all_registered_released']
    assert report['abort_audit'] is not None and not report['cleanup_errors']
    assert digest(runtime.model.state_dict()) == runtime.initial_model
    assert digest(runtime.streams.capture()) == runtime.initial_rng
    if fault == 'preflight':
        assert not runtime.updates
    else:
        assert ('warm10', 1) not in runtime.updates


def test_final_counts_include_cold_call_without_training():
    p = protocol()
    assert p['cadence']['final_counts']['qr20']['research_qr_refreshes'] == 174
    assert p['cadence']['final_counts']['warm20']['research_warm_refreshes'] == 157
    assert p['cadence']['final_counts']['warm10']['research_warm_refreshes'] == 332
    for arm in ['warm20', 'warm10']:
        assert p['cadence']['final_counts'][arm]['research_qr_refreshes'] == 17
    assert expected_counts(1, 'warm', 10, 200)['research_warm_refreshes'] == 0
    assert set(execution_order()) == set(ARMS)
