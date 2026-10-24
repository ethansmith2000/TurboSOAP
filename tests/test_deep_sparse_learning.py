"""Native QR200 arithmetic, held coordinates, actual transport and observation safety."""
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
from deep_sparse_kernel import build_optimizer, LayerObserver, qualify_actual_m, event_kind
from deep_sparse_learning import recipe, verify_control, decision, compact_guard
from matched_training_runtime import streaming_digest
from soap_reference import ResearchSOAP
from test_deep_llm_learning import Tiny


CONFIG = dict(matrix_lr=.0005, lr=.0005, weight_decay=.05, betas=(.9,.999), shampoo_beta=.99,
              precondition_mode='all', covariance_compute_dtype='float32', transport_precision='highest')


def test_hold_preserves_basis_coordinates_while_covariance_and_moments_update():
    torch.manual_seed(195)
    model=Tiny(); opt=build_optimizer(model,CONFIG,'qr20')
    for p in model.parameters():p.grad=torch.randn_like(p)
    opt.step()
    p=next(iter(opt.matrix.state)); state=opt.matrix.state[p]
    q=streaming_digest(state['Q']); cov=streaming_digest(state['GG']); moments=streaming_digest([state['exp_avg'],state['exp_avg_sq']])
    for call in range(2,201):
        for param in model.parameters():param.grad=torch.randn_like(param)
        opt.step()
        assert streaming_digest(state['Q'])==q
        assert state.get('research_strict_transports',0)==0
    assert streaming_digest(state['GG'])!=cov and streaming_digest([state['exp_avg'],state['exp_avg_sq']])!=moments
    for param in model.parameters():param.grad=torch.randn_like(param)
    opt.step(); validate_counters(opt,201)
    assert state['research_qr_refreshes']==state['research_strict_transports']==1
    assert streaming_digest(state['Q'])!=q
    assert compact_guard(opt,model,201)['qr_refreshes_per_matrix']==1
    state['research_strict_transports']=2
    with pytest.raises(RuntimeError,match='Cadence counter mismatch'):compact_guard(opt,model,201)


def test_all_225_steps_match_independent_native_qr200_with_same_gradient_stream():
    torch.manual_seed(195); a=Tiny(); b=copy.deepcopy(a)
    candidate=build_optimizer(a,CONFIG,'qr20'); control=native_builder(b,CONFIG,'qr20')
    for group in control.matrix.param_groups:group['precondition_frequency']=200
    for call in range(1,226):
        for pa,pb in zip(a.parameters(),b.parameters()):
            grad=torch.randn_like(pa);pa.grad=grad.clone();pb.grad=grad.clone()
        candidate.step();control.step()
    assert streaming_digest(a.state_dict())==streaming_digest(b.state_dict())
    assert streaming_digest(candidate.state_dict())==streaming_digest(control.state_dict())
    validate_counters(candidate,225)
    assert sum(event_kind(c)=='qr' for c in range(1,35001))==174


def test_actual_qr200_transport_checks_selected_ages_and_releases_weak_owner():
    torch.manual_seed(195);p=torch.nn.Parameter(torch.randn(5,3))
    opt=ResearchSOAP([p],precondition_frequency=200,transport_precision='highest')
    events=[]; router=SimpleNamespace(rng_digest=lambda:streaming_digest(torch.get_rng_state()),moments=[],emit=lambda k,v:events.append((k,v)))
    qualify_actual_m(opt,router,'qr20','primary')
    for call in range(1,402):p.grad=torch.randn_like(p);opt.step()
    assert [r['age'] for r in router.moments]==[200,400]
    assert all(r['passed'] and r['rng_unchanged'] and r['arm']=='hold200' for r in router.moments)
    assert all(v['all_registered_released'] for k,v in events if k=='actual_m_release')
    ref=weakref.ref(opt);del opt;gc.collect();assert ref() is None


def test_sparse_observer_records_actual_hold_age_and_restores_and_releases():
    torch.manual_seed(195);model=Tiny();opt=build_optimizer(model,CONFIG,'qr20');events=[]
    observer=LayerObserver(model,lambda k,v:events.append((k,v)),lambda:streaming_digest(torch.get_rng_state()),stages=(26,),lags=(1,2))
    for call in range(1,27):
        for p in model.parameters():p.grad=torch.randn_like(p)
        opt.step();observer.tick(call,opt)
    row=observer.observations[0]
    assert row['source_policy']=='hold200' and row['current_last_refresh_call']==1
    assert all(f['lags'][str(lag)]['capture_last_refresh_call']==1 for f in row['factors'] for lag in (1,2))
    assert row['source_unchanged'] and row['all_registered_released']
    assert observer.close()['released_history_banks']==2


def test_disposable_first_transport_is_qualified_before_225_prefix():
    p=torch.nn.Parameter(torch.randn(4,3));opt=ResearchSOAP([p],precondition_frequency=200,transport_precision='highest')
    router=SimpleNamespace(rng_digest=lambda:streaming_digest(torch.get_rng_state()),moments=[],emit=lambda *args:None)
    qualify_actual_m(opt,router,'qr20','disposable')
    for _ in range(225):p.grad=torch.randn_like(p);opt.step()
    assert len(router.moments)==1 and router.moments[0]['age']==200 and router.moments[0]['passed']


def test_decision_uses_own_cadence_and_cannot_average_away_quality_miss():
    kinds={'ordinary':{'optimizer_host_ms':20.,'iteration_host_ms':160.},'qr':{'optimizer_host_ms':600.,'iteration_host_ms':740.}}
    data={'phases':[{}, {'steady_means':kinds}]}
    quality=[dict(call=c,postwarmup=True,passes_panel=c!=3500) for c in (1000,3500,10000,20000,30000,35000)]
    d=decision(data,data,quality)
    assert d['candidate_optimizer_ms']==pytest.approx(22.9)
    assert d['native_optimizer_ms']==pytest.approx(49.)
    assert d['optimizer_lower'] and d['iteration_lower'] and not d['provisional_screen_pass']
    with pytest.raises(RuntimeError,match='Missing quality panels'):decision(data,data,quality[:-1])


def test_sparse_recipe_changes_metadata_only_and_control_hashes_are_enforced(tmp_path):
    from deep_sparse_learning import BASELINE
    from deep_llm_data import sha
    old=json.loads((BASELINE/'config.json').read_text());new=recipe(old)
    assert {k:v for k,v in old.items() if k!='matched_comparison'}=={k:v for k,v in new.items() if k!='matched_comparison'}
    assert new['matched_comparison']['disposable_calls']==225 and new['matched_comparison']['trial']==195
    (tmp_path/'config.json').write_text(json.dumps(new))
    for name in ('data_audit.json','data_stat.json','initial_reference.json','backend_contract.json'):
        (tmp_path/name).write_bytes((BASELINE/name).read_bytes())
    comparison=dict(trial=195,matched_seed=193,quality_margin_ce=.01,quality_calls=[1000,3500,10000,20000,30000,35000],native_files={'summary.json':sha(BASELINE/'summary.json')})
    (tmp_path/'comparison.json').write_text(json.dumps(comparison));verify_control(tmp_path)
    comparison['native_files']['summary.json']='bad';(tmp_path/'comparison.json').write_text(json.dumps(comparison))
    with pytest.raises(RuntimeError,match='Native comparison evidence changed'):verify_control(tmp_path)
