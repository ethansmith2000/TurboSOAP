"""Fast candidate learning contract, actual diagnostics and source safety."""
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from cadence_training_contract import validate_counters
from deep_matched_warm_kernel import build_optimizer, qualify_actual_m, event_kind, LayerObserver
from deep_matched_warm_learning import recipe, verify_control, decision, compact_guard, BASELINE
from matched_training_runtime import streaming_digest
from test_deep_llm_learning import Tiny

CONFIG = dict(matrix_lr=.0005, lr=.0005, weight_decay=.05, betas=(.9, .999),
              shampoo_beta=.99, precondition_mode='all', covariance_compute_dtype='float32')


@pytest.fixture(autouse=True)
def high_precision():
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(old)


def test_full_75_prefix_covers_warm_repair_postrepair_and_real_m_diagnostics():
    torch.manual_seed(200)
    model = Tiny()
    opt = build_optimizer(model, CONFIG, compile_warm=False)
    events = []
    router = SimpleNamespace(rng_digest=lambda: streaming_digest(torch.get_rng_state()),
                             moments=[], parities=[], emit=lambda k, v: events.append((k, v)))
    qualify_actual_m(opt.matrix, router, 'fast10', 'disposable')
    for call in range(1, 76):
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        opt.step()
    validate_counters(opt, 75)
    guard = compact_guard(opt, model, 75)
    assert guard['qr_refreshes_per_matrix'] == 1 and guard['warm_refreshes_per_matrix'] == 6
    assert len(router.moments) == 96 and {r['age'] for r in router.moments} == {10, 50}
    assert len(router.parities) == 192 and {r['age'] for r in router.parities} == {10, 60}
    assert all(row['passed'] and row['rng_unchanged'] for row in router.moments)
    assert all(row['all_registered_released'] for event, row in events if event == 'actual_m_release')
    assert all(group['transport_precision'] == 'inherit' for group in opt.matrix.param_groups)
    ref = weakref.ref(opt.matrix)
    del opt
    gc.collect()
    assert ref() is None
    assert sum(event_kind(call) == 'qr' for call in range(1, 35001)) == 699
    assert sum(event_kind(call) == 'warm' for call in range(1, 35001)) == 2800


def test_fast_observer_restores_source_and_records_last_warm_refresh_age():
    torch.manual_seed(200)
    model = Tiny()
    opt = build_optimizer(model, CONFIG, compile_warm=False)
    observer = LayerObserver(model, lambda *args: None, lambda: streaming_digest(torch.get_rng_state()),
                             stages=(16,), lags=(1, 2))
    for call in range(1, 17):
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        opt.step()
        observer.tick(call, opt)
    row = observer.observations[0]
    assert row['source_policy'] == 'fast10' and row['current_last_refresh_call'] == 11
    assert all(f['lags'][str(lag)]['capture_last_refresh_call'] == 11 for f in row['factors'] for lag in (1, 2))
    assert row['source_unchanged'] and row['all_registered_released']
    assert observer.close()['released_history_banks'] == 2


def test_decision_charges_both_kinds_and_retains_quality_screen_as_report_only():
    kinds = dict(ordinary=dict(optimizer_host_ms=20., iteration_host_ms=160.),
                 warm=dict(optimizer_host_ms=60., iteration_host_ms=200.),
                 qr=dict(optimizer_host_ms=600., iteration_host_ms=740.))
    data = dict(phases=[{}, dict(steady_means=kinds)])
    quality = [dict(call=c, postwarmup=True, passes_panel=c != 3500)
               for c in (1000, 3500, 10000, 20000, 30000, 35000)]
    result = decision(data, data, quality)
    assert result['candidate_optimizer_ms'] == pytest.approx(34.8)
    assert result['native_optimizer_ms'] == pytest.approx(78.)
    assert result['optimizer_lower'] and result['iteration_lower']
    assert not result['all_postwarmup_panels_pass'] and 'provisional_screen_pass' not in result
    assert 'None' in result['promotion']


def test_matched_recipe_only_changes_explicit_execution_policy_and_pins_native_evidence(tmp_path):
    from deep_llm_data import sha
    control = json.loads((BASELINE/'config.json').read_text())
    candidate = recipe(control)
    changed = {key for key in control if control[key] != candidate[key]}
    assert BASELINE.name == 'llm_soap_deep_qr10_trial199_frozen_20261019'
    assert changed == {'matched_comparison', 'model_execution', 'transport_precision', 'max_wall_seconds'}
    assert candidate['matched_comparison']['trial'] == 200
    assert candidate['matched_comparison']['native_control_trial'] == 199
    assert candidate['max_wall_seconds'] == 8400
    assert candidate['matched_comparison']['disposable_calls'] == 75
    (tmp_path/'config.json').write_text(json.dumps(candidate))
    for name in ('data_audit.json', 'data_stat.json', 'initial_reference.json', 'backend_contract.json'):
        (tmp_path/name).write_bytes((BASELINE/name).read_bytes())
    comparison = dict(trial=200, matched_seed=193, quality_margin_ce=.01,
                      quality_calls=[1000, 3500, 10000, 20000, 30000, 35000],
                      native_files={'summary.json': sha(BASELINE/'summary.json')},
                      qualification_files={'summary.json': sha(Path('/workspace/optimizer_replay_results/soap_fast_qualification197_frozen_20261019/summary.json'))},
                      hold_files={'summary.json': sha(Path('/workspace/optimizer_replay_results/llm_soap_deep_sparse_trial195_frozen_20261019/summary.json'))})
    (tmp_path/'comparison.json').write_text(json.dumps(comparison))
    verify_control(tmp_path)
    comparison['native_files']['summary.json'] = 'changed'
    (tmp_path/'comparison.json').write_text(json.dumps(comparison))
    with pytest.raises(RuntimeError, match='Native comparison evidence changed'):
        verify_control(tmp_path)
