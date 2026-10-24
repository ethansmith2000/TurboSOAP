"""Native QR10 arithmetic, held coordinates, actual transport and observation safety."""
import copy
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'benchmarks'), str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import pytest
import torch
from cadence_training_contract import validate_counters
from cifar_optimizers import build_optimizer as native_builder
from deep_qr10_kernel import build_optimizer, LayerObserver, qualify_actual_m, event_kind
from deep_qr10_learning import recipe, verify_control, decision, compact_guard
from matched_training_runtime import streaming_digest
from soap_reference import ResearchSOAP
from test_deep_llm_learning import Tiny


CONFIG = dict(matrix_lr=.0005, lr=.0005, weight_decay=.05, betas=(.9,.999), shampoo_beta=.99,
              precondition_mode='all', covariance_compute_dtype='float32', transport_precision='highest')


def test_hold_preserves_basis_coordinates_while_covariance_and_moments_update():
    torch.manual_seed(199)
    model=Tiny(); opt=build_optimizer(model,CONFIG,'qr10')
    for p in model.parameters():p.grad=torch.randn_like(p)
    opt.step()
    p=next(iter(opt.matrix.state)); state=opt.matrix.state[p]
    q=streaming_digest(state['Q']); cov=streaming_digest(state['GG']); moments=streaming_digest([state['exp_avg'],state['exp_avg_sq']])
    for call in range(2,11):
        for param in model.parameters():param.grad=torch.randn_like(param)
        opt.step()
        assert streaming_digest(state['Q'])==q
        assert state.get('research_strict_transports',0)==0
    assert streaming_digest(state['GG'])!=cov and streaming_digest([state['exp_avg'],state['exp_avg_sq']])!=moments
    for param in model.parameters():param.grad=torch.randn_like(param)
    opt.step(); validate_counters(opt,11)
    assert state['research_qr_refreshes']==state['research_strict_transports']==1
    assert streaming_digest(state['Q'])!=q
    assert compact_guard(opt,model,11)['qr_refreshes_per_matrix']==1
    state['research_strict_transports']=2
    with pytest.raises(RuntimeError,match='Cadence counter mismatch'):compact_guard(opt,model,11)


def test_all_75_steps_match_independent_native_qr10_with_same_gradient_stream():
    torch.manual_seed(199); a=Tiny(); b=copy.deepcopy(a)
    candidate=build_optimizer(a,CONFIG,'qr10'); control=native_builder(b,CONFIG,'qr20')
    for group in control.matrix.param_groups:group['precondition_frequency']=10
    for call in range(1,76):
        for pa,pb in zip(a.parameters(),b.parameters()):
            grad=torch.randn_like(pa);pa.grad=grad.clone();pb.grad=grad.clone()
        candidate.step();control.step()
    assert streaming_digest(a.state_dict())==streaming_digest(b.state_dict())
    assert streaming_digest(candidate.state_dict())==streaming_digest(control.state_dict())
    validate_counters(candidate,75)
    assert sum(event_kind(c)=='qr' for c in range(1,35001))==3499


def test_actual_qr10_transport_checks_selected_ages_and_releases_weak_owner():
    torch.manual_seed(199);p=torch.nn.Parameter(torch.randn(5,3))
    opt=ResearchSOAP([p],precondition_frequency=10,transport_precision='highest')
    events=[]; router=SimpleNamespace(rng_digest=lambda:streaming_digest(torch.get_rng_state()),moments=[],emit=lambda k,v:events.append((k,v)))
    qualify_actual_m(opt,router,'qr20','primary')
    for call in range(1,52):p.grad=torch.randn_like(p);opt.step()
    assert [r['age'] for r in router.moments]==[10,50]
    assert all(r['passed'] and r['rng_unchanged'] and r['arm']=='qr10' for r in router.moments)
    assert all(v['all_registered_released'] for k,v in events if k=='actual_m_release')
    ref=weakref.ref(opt);del opt;gc.collect();assert ref() is None


def test_qr10_observer_records_actual_hold_age_and_restores_and_releases():
    torch.manual_seed(199);model=Tiny();opt=build_optimizer(model,CONFIG,'qr10');events=[]
    observer=LayerObserver(model,lambda k,v:events.append((k,v)),lambda:streaming_digest(torch.get_rng_state()),stages=(26,),lags=(1,2))
    for call in range(1,27):
        for p in model.parameters():p.grad=torch.randn_like(p)
        opt.step();observer.tick(call,opt)
    row=observer.observations[0]
    assert row['source_policy']=='qr10' and row['current_last_refresh_call']==21
    assert all(f['lags'][str(lag)]['capture_last_refresh_call']==21 for f in row['factors'] for lag in (1,2))
    assert row['source_unchanged'] and row['all_registered_released']
    assert observer.close()['released_history_banks']==2


def test_disposable_first_transport_is_qualified_before_75_prefix():
    p=torch.nn.Parameter(torch.randn(4,3));opt=ResearchSOAP([p],precondition_frequency=10,transport_precision='highest')
    router=SimpleNamespace(rng_digest=lambda:streaming_digest(torch.get_rng_state()),moments=[],emit=lambda *args:None)
    qualify_actual_m(opt,router,'qr20','disposable')
    for _ in range(75):p.grad=torch.randn_like(p);opt.step()
    assert len(router.moments)==1 and router.moments[0]['age']==10 and router.moments[0]['passed']


def test_report_uses_qr10_cadence_and_does_not_promote_from_contextual_qr20_screen():
    kinds={'ordinary':{'optimizer_host_ms':20.,'iteration_host_ms':160.},'qr':{'optimizer_host_ms':600.,'iteration_host_ms':740.}}
    data={'phases':[{}, {'steady_means':kinds}]}
    quality=[dict(call=c,postwarmup=True,passes_panel=c!=3500) for c in (1000,3500,10000,20000,30000,35000)]
    d=decision(data,data,quality)
    assert d['candidate_optimizer_ms']==pytest.approx(78.)
    assert d['native_optimizer_ms']==pytest.approx(49.)
    assert not d['optimizer_lower'] and not d['iteration_lower']
    assert not d['all_postwarmup_panels_pass'] and 'provisional_screen_pass' not in d
    assert 'Pending' in d['matched_warm_comparison']
    with pytest.raises(RuntimeError,match='Missing quality panels'):decision(data,data,quality[:-1])


def test_qr10_recipe_changes_metadata_only_and_control_hashes_are_enforced(tmp_path):
    from deep_qr10_learning import BASELINE
    from deep_llm_data import sha
    old=json.loads((BASELINE/'config.json').read_text());new=recipe(old)
    assert {k:v for k,v in old.items() if k not in ('matched_comparison','max_wall_seconds')}=={k:v for k,v in new.items() if k not in ('matched_comparison','max_wall_seconds')}
    assert new['max_wall_seconds']==10200
    assert new['matched_comparison']['disposable_calls']==75 and new['matched_comparison']['trial']==199
    (tmp_path/'config.json').write_text(json.dumps(new))
    for name in ('data_audit.json','data_stat.json','initial_reference.json','backend_contract.json'):
        (tmp_path/name).write_bytes((BASELINE/name).read_bytes())
    comparison=dict(trial=199,matched_seed=193,quality_margin_ce=.01,quality_calls=[1000,3500,10000,20000,30000,35000],native_files={'summary.json':sha(BASELINE/'summary.json')})
    (tmp_path/'comparison.json').write_text(json.dumps(comparison));verify_control(tmp_path)
    comparison['native_files']['summary.json']='bad';(tmp_path/'comparison.json').write_text(json.dumps(comparison))
    with pytest.raises(RuntimeError,match='Native comparison evidence changed'):verify_control(tmp_path)
