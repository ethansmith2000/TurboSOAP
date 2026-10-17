"""Own-history BF16projection/reset/M/lease failure controls and immutable learning criteria."""
from pathlib import Path
import sys,copy,json
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import pytest
import torch
from test_llm_sparse_screen import tiny_runtime, Tiny
from llm_sparse_screen_kernel import Router
from matched_training_runtime import OptimizerLease, streaming_digest
from projection_learning_contract import recipe,protocol,validate,ARMS,PARENT,counts,event_kind
from projection_learning_kernel import build_arm,qualify_actual_m
from projection_learning import preflight,trajectory

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda *a,**k:(_ for _ in ()).throw(AssertionError('CUDA in CPU check')))
    torch.set_num_threads(2);previous=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(previous)
    assert not torch.cuda.is_initialized()

def setup(arm):
    torch.manual_seed(72);model=Tiny();router=Router(lambda *a:None,lambda:'rng',compiler=lambda f,**kw:f)
    return model,router,build_arm(model,recipe(),arm,router)

def runtime(tmp_path):
    r,e=tiny_runtime(tmp_path);r.config=recipe();p=protocol();p['coverage']=r.protocol['coverage'];p['evaluation_calls']=[];r.protocol=p
    return r,e

@pytest.mark.parametrize('arm',ARMS)
def test_own_history_runner_exact_prefix_and_sampled_physical_m(tmp_path,arm):
    r,e=runtime(tmp_path)
    try:
        expected=preflight(r);d=trajectory(r,arm,'disposable',245,expected);p=trajectory(r,arm,'primary',245,expected)
        assert d['prefixes']['245']==p['prefixes']['245']
        assert [(v['loss'],v['gradient_norm']) for v in d['timings']]==[(v['loss'],v['gradient_norm']) for v in p['timings']]
        assert d['release']['passed'] and p['release']['passed']
        assert [v['age'] for v in r.kernel.moments if v['phase']=='primary']==[20,200]
        assert all(v['passed'] and v['rng_unchanged'] for v in r.kernel.moments)
        assert [v['call'] for v in p['timings'] if v['actual_m_diagnostic']]==[21,201]
        assert p['final_counters']['parameters'][0]['actual']==counts(245,arm)
    finally:r.snapshot.close()

def test_native_untouched_bf16_adapter_used_but_refresh_transport_stays_highest(monkeypatch):
    import projection_profile_kernel as k
    original=k.project;seen=[]
    def observed(*a,**kw):
        assert torch.get_float32_matmul_precision()=='high'
        seen.append(kw.get('back',a[2] if len(a)>2 else False));return original(*a,**kw)
    monkeypatch.setattr(k,'project',observed)
    for arm in ARMS:
        seen.clear();model,router,opt=setup(arm)
        assert '_research_refresh' not in opt.matrix.__dict__
        assert ('_project_with_basis' in opt.matrix.__dict__)==(arm=='bf1620')
        p=opt.matrix.param_groups[0]['params'][0]
        for call in range(1,42):p.grad=torch.tensor([[.3,.2,-.1],[.1,-.2,.4]])+.001*call;opt.step()
        assert len(seen)==(0 if arm=='qr20' else 80)
        assert seen.count(True)==(0 if arm=='qr20' else 40)
        for state in opt.matrix.state.values():
            assert all(v.dtype==torch.float32 for v in [state['exp_avg'],state['exp_avg_sq'],*state['GG'],*state['Q']])

@pytest.mark.parametrize('arm',ARMS)
def test_real_m_corruption_fails_and_releases_buffers(arm):
    model,router,opt=setup(arm);p=opt.matrix.param_groups[0]['params'][0];p.grad=torch.ones_like(p);opt.step()
    state=opt.matrix.state[p];state['step']=20;state['exp_avg'].fill_(1.)
    original=opt.matrix._transport_first_moment
    def corrupt(state,old,new,group):original(state,old,new,group);state['exp_avg'].mul_(2.)
    opt.matrix._transport_first_moment=corrupt;events=[];router.emit=lambda e,p:events.append((e,p))
    qualify_actual_m(opt.matrix,router,arm,'primary')
    with pytest.raises(FloatingPointError,match='momentum'):opt.matrix._research_refresh(state,opt.matrix.param_groups[0])
    assert events[-1][0]=='actual_m_release' and events[-1][1]['all_registered_released']
    del state,p
    lease=OptimizerLease(model,opt);del opt
    # The injected test closure retains a bound method; remove it before testing the actual hook ownership.
    del original,corrupt
    assert lease.release()['passed']

@pytest.mark.parametrize('arm',ARMS)
def test_instance_hook_has_no_owner_cycle_and_does_not_touch_other_instances(arm):
    model,router,opt=setup(arm);qualify_actual_m(opt.matrix,router,arm,'primary')
    other=build_arm(model,recipe(),'qr20',router);assert '_research_refresh' not in other.matrix.__dict__
    p=opt.matrix.param_groups[0]['params'][0];p.grad=torch.ones_like(p);opt.step();del p,other
    lease=OptimizerLease(model,opt);del opt;assert lease.release()['passed']

def test_failure_records_completed_work_and_releases_candidate_state(tmp_path):
    r,e=runtime(tmp_path)
    try:
        expected=preflight(r);calls=[]
        def budget(**kw):
            calls.append(1)
            if len(calls)==3:raise RuntimeError('injected budget')
        r.budget=budget
        with pytest.raises(RuntimeError,match='Trajectory failed'):trajectory(r,'bf1620','primary',245,expected)
        row=json.loads((tmp_path/'bf1620_primary.json').read_text());assert row['completed_optimizer_step_calls']==2 and row['release']['passed']
        assert streaming_digest(r.model.state_dict())==r.snapshot.initial_digest
    finally:r.snapshot.close()

def test_declared_learning_margin_counts_cadence_and_legacy_record_are_immutable():
    c,p=recipe(),protocol();validate(c,p)
    assert p['maximum_optimizer_updates']==25+2*(245+3500)==7515
    assert p['expected']['actual_m_transactions']==40
    assert sum(v for k,v in p['memory'].items() if type(v) is int and k!='maximum_tensor_purpose_bytes')<p['memory']['maximum_tensor_purpose_bytes']
    assert p['artifacts']['probe']['expected_max_extra_bytes']==p['memory']['rectangular_m_workspace_bytes']+p['memory']['projection_cast_and_outputs_bytes']
    assert p['decision']['quality_margin_ce']==.01 and p['decision']['quality_calls']==[1000,1500,2000,2500,3000,3500]
    for arm in ARMS:assert sum(event_kind(call,arm)=='qr' for call in range(1,3501))==174
    for path,value in [('seed',184),('steps_per_arm',4000)]:
        changed=copy.deepcopy(p);changed[path]=value
        with pytest.raises(ValueError,match='Undeclared'):validate(c,changed)
    changed=copy.deepcopy(p);changed['decision']['quality_margin_ce']=.02
    with pytest.raises(ValueError,match='Undeclared'):validate(c,changed)

def test_parent255sources_preserved_and_forged_criteria_rejected(tmp_path):
    import prepare_projection_learning as prep
    ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=188,parameters=144875648,cuda_initialized=False,snapshot_purpose_bytes=2230649344,model_sha256='5'*64))
    checks=tmp_path/'checks.json';prep.write(checks,dict(status='PASS_CPU_PROJECTION_LEARNING',cuda_initialized=False,tests_passed=True))
    b=tmp_path/'bundle';m=prep.prepare(b,ref,188,PARENT/'token_cache_audit.json',checks);old=prep.read(PARENT/'manifest.json')['source_sha256']
    assert len(old)==255 and len(m['source_sha256'])==261 and all(m['source_sha256'][n]==h for n,h in old.items())
    p=prep.read(b/'protocol.json');p['decision']['quality_margin_ce']=.02;prep.write(b/'protocol.json',p)
    m=prep.read(b/'manifest.json');m['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',m)
    with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
    with pytest.raises(ValueError,match='Fresh bundle'):prep.prepare(b,ref,188,PARENT/'token_cache_audit.json',checks)
