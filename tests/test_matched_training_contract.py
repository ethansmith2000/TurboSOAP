import copy
from pathlib import Path
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from benchmarks.matched_training_contract import (
    ARMS, COMMON, MAY_OVERRIDES, HORIZON, OBSERVATIONS, RandomStreams,
    SharedInitialState, all_arm_preupdate_gate, assert_fresh_independent,
    build_arm, digest, optimizer_metadata, preupdate_observation, tensor_fingerprint,
)


CONFIG = {**COMMON, 'precondition_mode': 'all', 'covariance_compute_dtype': 'float32'}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.2))
        self.head = torch.nn.Linear(4, 3)
        self.register_buffer('scale', torch.tensor(1.))

    def forward(self, x):
        return self.head(self.blocks(x)) * self.scale


@pytest.fixture(autouse=True)
def never_initialize_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail(*args, **kwargs):
        raise AssertionError('CPU contract test attempted CUDA initialization')
    monkeypatch.setattr(torch.cuda, '_lazy_init', fail)
    yield
    assert not torch.cuda.is_initialized()


def setup():
    torch.manual_seed(139); random.seed(139); np.random.seed(139)
    model = TinyModel()
    augmentation = SimpleNamespace(rng=np.random.default_rng(3140), counts={'mixup': 0, 'cutmix': 0})
    streams = RandomStreams(torch.Generator().manual_seed(1140), augmentation)
    snapshot = SharedInitialState(model, streams)
    arms = [(arm, build_arm(model, CONFIG, arm)) for arm in ARMS]
    def batch():
        x = torch.randn(5, 4, generator=streams.crop)
        x += float(augmentation.rng.random()) + random.random() + float(np.random.random())
        augmentation.counts['mixup'] += 1
        return torch.arange(5), x, torch.tensor([0, 1, 2, 0, 1])
    def observe(label, opt):
        return preupdate_observation(model, model, batch, streams, clip_norm=1.)
    return model, streams, snapshot, arms, observe


def test_full_observations_exact_before_any_update():
    model, streams, snapshot, arms, observe = setup()
    before = digest(streams.capture())
    result = all_arm_preupdate_gate(snapshot, arms, observe)
    assert result['status'] == 'PASS_PREUPDATE_ONLY' and result['optimizer_steps'] == 0
    assert len(result['arms']) == 4
    assert all(r['observation'] == result['arms'][0]['observation'] for r in result['arms'])
    assert digest(streams.capture()) == before
    assert digest(model.state_dict()) == snapshot.initial_digest
    assert all(p.grad is None for p in model.parameters())
    assert_fresh_independent([o for _, o in arms])


@pytest.mark.parametrize('field', ['input', 'target', 'output', 'loss', 'rng_before',
    'rng_after_augmentation', 'rng_after_backward', 'gradients_before_clip', 'gradients_after_clip'])
def test_any_full_record_mismatch_aborts_with_no_steps(field):
    model, streams, snapshot, arms, observe = setup()
    initial_rng = digest(streams.capture())
    calls = []
    def changed(label, opt):
        calls.append(label)
        record = observe(label, opt)
        if label == 'may_literal':
            if field.startswith('gradients'):
                record[field]['head.weight']['sha256'] = 'corruption'
            elif isinstance(record[field], dict):
                record[field]['sha256'] = 'corruption'
            else:
                record[field] = 'corruption'
        return record
    with pytest.raises(RuntimeError, match='Exact all-arm'):
        all_arm_preupdate_gate(snapshot, arms, changed)
    assert calls == ['qr20', 'may_literal']
    assert_fresh_independent([o for _, o in arms])
    assert digest(model.state_dict()) == snapshot.initial_digest
    assert digest(streams.capture()) == initial_rng
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('component', ['split', 'matrix', 'auxiliary'])
def test_steps_blocked_and_original_methods_restored(component):
    model, streams, snapshot, arms, observe = setup()
    original = [o.step for _, o in arms]
    def bad(label, opt):
        (opt if component == 'split' else getattr(opt, component)).step()
    with pytest.raises(RuntimeError, match='steps forbidden'):
        all_arm_preupdate_gate(snapshot, arms, bad)
    assert [o.step for _, o in arms] == original
    assert_fresh_independent([o for _, o in arms])
    assert digest(model.state_dict()) == snapshot.initial_digest


def test_snapshot_restores_buffers_flags_requirements_rng_and_parameter_identity():
    model, streams, snapshot, arms, observe = setup()
    model.blocks[1].eval()
    snapshot.close()
    snapshot = SharedInitialState(model, streams)
    initial_rng = digest(streams.capture())
    identities = snapshot.identities
    with torch.no_grad():
        for p in model.parameters():
            p.add_(3.)
            p.grad = torch.ones_like(p)
        model.scale.fill_(9.)
    model.eval(); model.head.weight.requires_grad_(False)
    observe('unused', arms[0][1])
    snapshot.reset()
    assert snapshot._identities() == identities
    assert model.training and not model.blocks[1].training and model.head.training
    assert model.head.weight.requires_grad
    assert float(model.scale) == 1.
    assert all(p.grad is None for p in model.parameters())
    assert digest(streams.capture()) == initial_rng


@pytest.mark.parametrize('mutation', ['weights', 'policy', 'state'])
def test_probe_mutations_detected_without_optimizer_steps(mutation):
    model, streams, snapshot, arms, observe = setup()
    def bad(label, opt):
        record = observe(label, opt)
        if mutation == 'weights':
            model.scale.add_(1.)
        elif mutation == 'policy':
            opt.matrix.param_groups[0]['lr'] = .1
        else:
            opt.matrix.state[model.blocks[0].weight]['unexpected'] = 1
        return record
    with pytest.raises(RuntimeError, match={'weights': 'mutated model',
        'policy': 'mutated optimizer policy', 'state': 'start empty'}[mutation]):
        all_arm_preupdate_gate(snapshot, arms, bad)
    assert digest(model.state_dict()) == snapshot.initial_digest
    assert all(p.grad is None for p in model.parameters())


def test_actual_optimizer_states_are_independent_after_one_arm_updates():
    model, streams, snapshot, arms, observe = setup()
    all_arm_preupdate_gate(snapshot, arms, observe)
    observe(*arms[0]); arms[0][1].step()
    assert arms[0][1].matrix.state and arms[0][1].auxiliary.state
    assert all(not o.matrix.state and not o.auxiliary.state for _, o in arms[1:])
    snapshot.reset()
    assert digest(model.state_dict()) == snapshot.initial_digest


def test_duplicate_optimizer_and_nonempty_state_rejected():
    model, streams, snapshot, arms, observe = setup()
    with pytest.raises(RuntimeError, match='alias'):
        assert_fresh_independent([arms[0][1], arms[0][1]])
    arms[0][1].matrix.state[model.blocks[0].weight]['x'] = 0
    with pytest.raises(RuntimeError, match='start empty'):
        all_arm_preupdate_gate(snapshot, arms, observe)


def test_literal_may_defaults_and_all_common_overrides_are_visible():
    model, streams, snapshot, arms, observe = setup()
    row = optimizer_metadata(arms[1][1], 'may_literal')['matrix']
    assert row['source_sha256'] == 'fff531c00b321531d07121f60cd25d4c7b46b9033d8617c5d03e11b28ac93b5d'
    for group in row['groups']:
        g = group['settings']
        for k, v in MAY_OVERRIDES.items():
            assert g[k] == (list(v) if isinstance(v, tuple) else v)
        for k in ('precondition_frequency', 'basis_compute_dtype', 'basis_storage_dtype',
                  'covariance_compute_dtype', 'basis_direction_normalizer',
                  'basis_direction_normalizer_beta', 'basis_warmup_steps', 'basis_compile'):
            assert g[k] == row['constructor_defaults'][k]
        assert g['precondition_frequency'] == 10
        assert g['covariance_compute_dtype'] == 'bfloat16'
    coverage = []
    for opt in [o for _, o in arms]:
        names = [n for g in opt.param_groups for n in g['param_names']]
        assert sorted(names) == sorted(n for n, p in model.named_parameters())
        coverage.append([(g['role'], g['param_names'], g['weight_decay'], g['peak_lr'])
                         for g in opt.param_groups])
    assert all(v == coverage[0] for v in coverage)
    for g in arms[2][1].matrix.param_groups:
        assert g['research_movement_policy'] == 'fro_cubic2_cap05'
        assert g['basis_rotation_cap'] == .5 and g['basis_ns_iterations'] == 2
        assert g['hard_reset_interval'] == 200 and g['precondition_frequency'] == 20


@pytest.mark.parametrize('key,value', [('lr', .0006), ('betas', [.95, .999]),
    ('shampoo_beta', .999), ('matrix_betas', [.9, .99]),
    ('precondition_mode', 'left'), ('covariance_compute_dtype', 'bfloat16')])
def test_recipe_drift_rejected(key, value):
    with pytest.raises(ValueError):
        build_arm(TinyModel(), {**CONFIG, key: value}, 'qr20')


def test_bfloat16_full_byte_fingerprint_and_nonfinite_guard():
    x = torch.arange(16).to(torch.bfloat16).reshape(4, 4)
    assert tensor_fingerprint(x) == tensor_fingerprint(x.clone())
    x[3, 2] += 1
    assert tensor_fingerprint(x) != tensor_fingerprint(torch.arange(16).to(torch.bfloat16).reshape(4, 4))
    assert not tensor_fingerprint(torch.tensor(float('nan')))['finite']
    model, streams, snapshot, arms, observe = setup()
    def bad(label, opt):
        return preupdate_observation(model, model,
            lambda: (torch.arange(2), torch.full((2, 4), float('nan')), torch.tensor([0, 1])),
            streams, clip_norm=1.)
    with pytest.raises(FloatingPointError):
        all_arm_preupdate_gate(snapshot, arms, bad)
    assert digest(model.state_dict()) == snapshot.initial_digest


def test_snapshot_bounds_release_and_nonpersistent_buffer_rejection():
    model, streams, snapshot, arms, observe = setup()
    with pytest.raises(MemoryError):
        SharedInitialState(model, streams, ram_cap=1)
    snapshot.close()
    assert snapshot.weights is None and snapshot.rng is None
    with pytest.raises(RuntimeError, match='released'):
        snapshot.reset()
    model.register_buffer('hidden', torch.ones(1), persistent=False)
    with pytest.raises(ValueError, match='Nonpersistent'):
        SharedInitialState(model, streams)


@pytest.mark.parametrize('kind', ['missing_field', 'missing_gradient', 'missing_arm'])
def test_partial_gate_cannot_pass(kind):
    model, streams, snapshot, arms, observe = setup()
    def partial(label, opt):
        row = observe(label, opt)
        if kind == 'missing_field':
            del row['target']
        else:
            del row['gradients_before_clip']['head.weight']
        return row
    with pytest.raises(ValueError):
        all_arm_preupdate_gate(snapshot, arms[:-1] if kind == 'missing_arm' else arms, partial)


def test_original_schedule_retained_beyond_warmup():
    from benchmarks.cifar_support.cifar100_qualification import lr_at
    assert HORIZON == 3500 > 175 * 5
    assert 875 in OBSERVATIONS and OBSERVATIONS[-1] == HORIZON
    assert lr_at(1, 35000, 875, .0005, .01) == .0005 / 875
    assert lr_at(875, 35000, 875, .0005, .01) == .0005
    assert .00049 < lr_at(3500, 35000, 875, .0005, .01) < .0005


def test_reused_cpu_model_qr_repeat_full_model_optimizer_rng_exact():
    model, streams, snapshot, arms, observe = setup()
    endpoints = []
    for name in ('qr20', 'qr20_repeat'):
        snapshot.reset()
        opt = build_arm(model, CONFIG, name)
        for _ in range(3):
            observe(name, opt); opt.step()
        endpoints.append((digest(model.state_dict()), digest(opt.state_dict()), digest(streams.capture())))
        del opt
    assert endpoints[0] == endpoints[1]
    # Typed optimizer index keys are distinct from text keys and preserve BF16 bytes.
    assert digest({1: torch.tensor([1.], dtype=torch.bfloat16)}) != digest({'1': torch.tensor([1.], dtype=torch.bfloat16)})


@pytest.mark.parametrize('field,value', [('lr', .0006), ('epochs', 20),
    ('checkpoint', True), ('official_test', True), ('width', 2048),
    ('warmup_epochs', 5.0), ('evaluation_policy', 'compiled_bf16')])
def test_design_cannot_silently_change_recipe_or_compress_schedule(field, value):
    from benchmarks.prepare_matched_training_design import validate_config, protocol
    baseline = dict(CONFIG, epochs=200, width=1536, warmup_epochs=5,
        checkpoint=False, official_test=False, evaluation_policy='eager_bf16',
        prefix_steps=128, branch_steps=3500, execution_diagnostic={}, max_wall_seconds=900)
    expected = {k:v for k,v in baseline.items() if k not in {'prefix_steps', 'branch_steps', 'execution_diagnostic'}}
    expected.update(max_wall_seconds=3600, matched_comparison=dict(arms=protocol()['arms'], steps_per_arm=3500))
    assert validate_config(expected, baseline)
    with pytest.raises(ValueError):
        validate_config({**expected, field:value}, baseline)


def test_prospective_limits_and_unknown_historical_recipe_are_explicit():
    from benchmarks.prepare_matched_training_design import protocol
    p = protocol()
    assert p['status'] == 'FROZEN_DESIGN_NOT_LAUNCHABLE'
    assert p['maximum_optimizer_updates'] == 4*3500+25
    assert p['process']['selected_device'] is None
    assert p['bounds']['maximum_initial_snapshot_purpose_cpu_bytes'] == 256*2**20
    assert not p['artifact_policy']['checkpoints'] and not p['artifact_policy']['final_weights']
    assert p['historical_recipe']['recalled_shampoo_beta'] == .999
    assert p['historical_recipe']['tentative_lr'] == .0006
    assert p['historical_recipe']['beta1'].startswith('Unknown')
    assert not p['gate']['matched_hash_family_selection'] and p['gate']['retries'] == 0
    assert 'Instrumented' in p['timing']['interpretation']
