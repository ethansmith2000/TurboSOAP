"""Independent cadence/update controls and preparation/runner failure checks."""
from pathlib import Path
import copy
import json
import math
import sys
from types import SimpleNamespace
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[1]/'benchmarks'),
    str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import pytest
import torch
from llm_sparse_screen_contract import ARMS, SEEDS, recipe, protocol, event_kind, counts, validate, PARENT
from llm_sparse_screen_kernel import Router, build_arm, qualify_actual_m, common_geometry_guard
from matched_training_runtime import OptimizerLease, streaming_digest
from llm_sparse_screen import evaluate_transaction, endpoint, coverage_guard
from cadence_training_contract import validate_counters


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError('CPU qualification attempted CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    torch.set_num_threads(2)
    yield
    assert not torch.cuda.is_initialized()


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(3,2,bias=False))])
        self.embedding = torch.nn.Embedding(4,3)

    def forward(self, x):
        return self.blocks[0](self.embedding(x))


def setup(arm):
    torch.manual_seed(901)
    model = Tiny()
    router = Router(lambda *a: None, lambda: 'fixedrng', compiler=lambda f, **kw: f)
    return model, router, build_arm(model, recipe(180), arm, router)


@pytest.mark.parametrize('arm,qr,warm', [('qr20',174,0),('highest_residual20',17,157),('hold200',17,0),('qr40',87,0)])
def test_cold_event_precedence_and_full_budget(arm, qr, warm):
    kinds = [event_kind(c,arm) for c in range(1,3501)]
    assert kinds[0]=='cold' and kinds.count('qr')==qr and kinds.count('warm')==warm
    assert kinds.count('ordinary')==3499-qr-warm
    assert counts(3500,arm)==dict(step=3499,research_qr_refreshes=qr,research_warm_refreshes=warm,research_strict_transports=qr+warm)
    assert event_kind(200,arm)=='ordinary' and event_kind(201,arm)=='qr'


@pytest.mark.parametrize('arm', ARMS)
def test_native_245_call_trajectory_and_actual_transport_qualification(arm):
    model, router, opt = setup(arm)
    qualify_actual_m(opt.matrix, router, arm)
    p = next(iter(opt.matrix.param_groups[0]['params']))
    events = []
    for call in range(1,246):
        p.grad = torch.tensor([[.3,.2,-.1],[.1,-.2,.4]]) + .01*math.sin(call)*torch.tensor([[1.,-.3,.2],[-.2,.4,-1.]])
        state = opt.matrix.state.get(p)
        old_ids = [id(q) for q in state['Q']] if state else []
        opt.step()
        state = opt.matrix.state[p]
        actual = validate_counters(opt, call)
        assert actual['parameters'][0]['actual']==counts(call,arm)
        if call>1 and event_kind(call,arm)=='ordinary':
            assert old_ids==[id(q) for q in state['Q']]
        elif call>1:
            events.append(call)
    assert events==[c for c in range(2,246) if event_kind(c,arm)!='ordinary']
    assert len(router.moments)==(2 if arm=='highest_residual20' else 1)
    assert all(r['passed'] and r['rng_unchanged'] for r in router.moments)
    assert len(common_geometry_guard(opt,model))==2


def test_hold_updates_adam_covariance_without_coordinate_transport():
    model, router, opt = setup('hold200')
    group=opt.matrix.param_groups[0];p=group['params'][0]
    p.grad=torch.tensor([[.3,.2,-.1],[.1,-.2,.4]]);opt.step()
    state=opt.matrix.state[p]
    for call in range(2,7):
        old=copy.deepcopy(state);parameter=p.detach().clone()
        g=torch.tensor([[.2,.7,-.3],[-.1,.5,.6]])/call;p.grad=g.clone()
        ql,qr=old['Q'];projected=ql.T@g@qr
        m=.9*old['exp_avg']+.1*projected
        v=.999*old['exp_avg_sq']+.001*projected.square()
        update=ql@(m/(v.sqrt()+group['eps']))@qr.T
        age=call-1;rate=group['lr']*math.sqrt(1-.999**age)/(1-.9**age)
        expected=(parameter-rate*update)*(1-group['lr']*group['weight_decay'])
        expected_c=[.99*old['GG'][0]+.01*(g@g.T),.99*old['GG'][1]+.01*(g.T@g)]
        old_q_ids=[id(q) for q in state['Q']]
        opt.step()
        torch.testing.assert_close(p,expected,rtol=2e-5,atol=2e-7)
        torch.testing.assert_close(state['exp_avg'],m)
        torch.testing.assert_close(state['exp_avg_sq'],v)
        for actual,target in zip(state['GG'],expected_c):torch.testing.assert_close(actual,target)
        assert old_q_ids==[id(q) for q in state['Q']]
        assert not state.get('research_strict_transports',0)


def test_first_hold_qr_permutation_and_physical_m_match_direct_formula():
    model,router,opt=setup('hold200');group=opt.matrix.param_groups[0];p=group['params'][0]
    ql=torch.eye(2);qr=torch.eye(3)
    cov=[torch.tensor([[1.,.2],[.2,3.]]),torch.tensor([[1.,.1,.1],[.1,4.,.2],[.1,.2,2.]])]
    m=torch.arange(6,dtype=torch.float32).reshape(2,3)+1;v=m.square()+.5
    state=dict(Q=[ql,qr],GG=cov,exp_avg=m.clone(),exp_avg_sq=v.clone(),step=200)
    opt.matrix.state[p]=state
    orders=[torch.argsort(c.diagonal(),descending=True) for c in cov]
    bases=[torch.linalg.qr(c[:,order])[0] for c,order in zip(cov,orders)]
    opt.matrix._research_refresh(state,group)
    torch.testing.assert_close(state['exp_avg'],bases[0].T@m@bases[1])
    torch.testing.assert_close(state['exp_avg_sq'],v[orders[0]][:,orders[1]])
    for actual,target in zip(state['Q'],bases):torch.testing.assert_close(actual,target)


def test_nonfinite_auxiliary_and_absolute_gram_are_not_hidden():
    model,router,opt=setup('hold200');p=opt.matrix.param_groups[0]['params'][0]
    p.grad=torch.ones_like(p);opt.step()
    with torch.no_grad():model.embedding.weight[0,0]=float('nan')
    with pytest.raises(FloatingPointError,match='Nonfinite'):common_geometry_guard(opt,model)
    with torch.no_grad():model.embedding.weight[0,0]=0
    opt.matrix.state[p]['Q'][0][0,0]*=1.1
    with pytest.raises(FloatingPointError,match='geometry'):common_geometry_guard(opt,model)


def test_m_gate_failure_and_qualification_buffers_release_on_exception():
    model,router,opt=setup('hold200');p=opt.matrix.param_groups[0]['params'][0]
    state=dict(Q=[torch.eye(2),torch.eye(3)],GG=[torch.eye(2),torch.eye(3)],
        exp_avg=torch.ones(2,3),exp_avg_sq=torch.ones(2,3),step=200)
    opt.matrix.state[p]=state;events=[];router.emit=lambda e,p:events.append((e,p))
    original=opt.matrix._transport_first_moment
    def corrupt(state,old,new,group):
        original(state,old,new,group);state['exp_avg'].mul_(2)
    opt.matrix._transport_first_moment=corrupt
    qualify_actual_m(opt.matrix,router,'hold200')
    with pytest.raises(FloatingPointError,match='momentum'):opt.matrix._research_refresh(state,opt.matrix.param_groups[0])
    assert events[-1][0]=='disposable_m_release' and events[-1][1]['all_registered_released']


def test_original_evaluator_failure_restores_flags_rng_and_rejects_mutation(tmp_path):
    model,router,opt=setup('qr20')
    from matched_training_contract import RandomStreams
    import numpy as np
    streams=RandomStreams(torch.Generator().manual_seed(10),SimpleNamespace(rng=np.random.default_rng(3),counts={}))
    runtime=SimpleNamespace(model=model,streams=streams,budget=lambda **kw:None)
    before=streaming_digest(streams.capture())
    def fail():
        streams.crop.manual_seed(999)
        raise ValueError('evaluation interruption')
    runtime.evaluate=fail
    with pytest.raises(ValueError,match='interruption'):evaluate_transaction(runtime,opt)
    assert model.training and streaming_digest(streams.capture())==before
    def mutate():
        with torch.no_grad():model.embedding.weight.add_(1)
        raise ValueError('evaluation interruption')
    runtime.evaluate=mutate
    with pytest.raises(RuntimeError,match='mutated'):evaluate_transaction(runtime,opt)
    assert model.training and streaming_digest(streams.capture())==before


def test_lease_releases_qualified_instance_hook_without_owner_cycle():
    model,router,opt=setup('hold200');qualify_actual_m(opt.matrix,router,'hold200')
    p=opt.matrix.param_groups[0]['params'][0];p.grad=torch.ones_like(p);opt.step()
    lease=OptimizerLease(model,opt);del opt,p
    assert lease.release()['passed']


def test_two_seed_semantics_and_utility_are_frozen():
    for seed in SEEDS:
        c,p=recipe(seed),protocol(seed);validate(c,p)
        assert p['maximum_optimizer_updates']==25+4*(245+3500)
        assert p['qualification']['no_primary_diagnostics'] and not p['probe_calls']
        assert p['guard_calls']==protocol(SEEDS[0])['guard_calls']
        assert p['llm_evaluation']['panels']['validation']['valid_tokens']==261632
        changed=copy.deepcopy(p);changed['cadence']['hold200']['frequency']=100
        with pytest.raises(ValueError,match='Undeclared'):validate(c,changed)
        changed=copy.deepcopy(p);changed['decision']['practical_screen']='loosen after seeing results'
        with pytest.raises(ValueError,match='Undeclared'):validate(c,changed)
    assert protocol(181)['execution_order']==list(reversed(protocol(180)['execution_order']))


def test_frozen_preparation_rejects_semantic_tampering_even_with_updated_hash(tmp_path):
    import prepare_llm_sparse_screen as prep
    initial=tmp_path/'initial.json';prep.write(initial,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=180,
        parameters=144875648,cuda_initialized=False,snapshot_purpose_bytes=2230649344,model_sha256='1'*64))
    calibration=tmp_path/'checks.json';prep.write(calibration,dict(status='PASS_CPU_SPARSE_SCREEN',cuda_initialized=False,tests_passed=True))
    bundle=tmp_path/'bundle'
    prep.prepare(bundle,initial,180,PARENT/'token_cache_audit.json',calibration)
    manifest=prep.read(bundle/'manifest.json');declared=prep.read(bundle/'protocol.json');declared['cadence']['hold200']['frequency']=100
    prep.write(bundle/'protocol.json',declared);manifest['artifacts']['protocol.json']=prep.sha(bundle/'protocol.json');prep.write(bundle/'manifest.json',manifest)
    with pytest.raises(ValueError,match='Undeclared'):prep.verify(bundle)
    with pytest.raises(ValueError,match='Fresh bundle'):prep.prepare(bundle,initial,180,PARENT/'token_cache_audit.json',calibration)


def tiny_runtime(tmp_path):
    from contextlib import nullcontext
    import numpy as np
    from matched_training_contract import RandomStreams, SharedInitialState
    from llm_feasibility_runtime import FlatOutput
    model, router, opt=setup('qr20');del opt
    # Three output classes and changing examples avoid an exactly stationary
    # rank-one fixture: the retained movement-cosine gate is intentionally strict.
    model.blocks[0][0]=torch.nn.Linear(3,3,bias=False)
    streams=RandomStreams(torch.Generator().manual_seed(44),SimpleNamespace(rng=np.random.default_rng(22),counts={}))
    snapshot=SharedInitialState(model,streams,extra_workspace_bytes=1024,ram_cap=8*2**20)
    p=protocol(180);p['coverage']=dict(matrix_parameters=1,two_sided_factors=2,dimensions_per_refresh={'3':2})
    p['evaluation_calls']=[]
    events=[]
    class Event:
        def record(self):pass
        def elapsed_time(self, other):return .25
    runtime=SimpleNamespace(bundle=tmp_path,model=model,active=FlatOutput(model),streams=streams,
        snapshot=snapshot,config=recipe(180),protocol=p,kernel=router,current_arm=None,
        schedule_steps=35000,warmup_steps=875,autocast=nullcontext,
        budget=lambda **kw:None,synchronize=lambda:None,optimizer_events=lambda:[Event(),Event()],
        emit=lambda e,p:events.append((e,p)))
    runtime.make_batch=lambda index:(torch.tensor([index]),
        torch.tensor([[index%4,(index//2+1)%4,(index//3+2)%4,(index//5+3)%4]]),
        torch.tensor([(index+j)%3 for j in range(4)]))
    return runtime,events


def test_whole_four_arm_runner_exact_disposable_state_prefix_and_registered_release(tmp_path):
    from llm_sparse_screen import preflight, trajectory
    runtime,events=tiny_runtime(tmp_path)
    try:
        expected=preflight(runtime)
        for arm in ARMS:
            disposable=trajectory(runtime,arm,'disposable',245,expected)
            primary=trajectory(runtime,arm,'primary',245,expected)
            assert disposable['prefixes']['245']==primary['prefixes']['245']
            assert [(r['loss'],r['gradient_norm']) for r in disposable['timings']]==[(r['loss'],r['gradient_norm']) for r in primary['timings']]
            assert disposable['release']['passed'] and primary['release']['passed']
            assert primary['completed_optimizer_step_calls']==primary['attempted_optimizer_step_calls']==245
            assert [g['call'] for g in primary['guards']]==[1,21,41,201,245]
            assert primary['training_seconds']>0 and primary['observation_seconds']>0
        assert len(runtime.kernel.probes)==6 and len(runtime.kernel.moments)==5
    finally:runtime.snapshot.close()


def test_runner_failure_records_completed_work_and_releases_state(tmp_path):
    from llm_sparse_screen import preflight, trajectory
    runtime,events=tiny_runtime(tmp_path)
    try:
        expected=preflight(runtime);calls=[]
        def budget(**kw):
            calls.append(1)
            if len(calls)==3:raise RuntimeError('injected all-in budget failure')
        runtime.budget=budget
        with pytest.raises(RuntimeError,match='Trajectory failed'):trajectory(runtime,'hold200','primary',245,expected)
        row=json.loads((tmp_path/'hold200_primary.json').read_text())
        assert row['release']['passed'] and row['completed_optimizer_step_calls']==2 and row['attempted_optimizer_step_calls']==2
        assert streaming_digest(runtime.model.state_dict())==runtime.snapshot.initial_digest
    finally:runtime.snapshot.close()


def test_original_token_evaluator_weights_uneven_batches_independently():
    from contextlib import nullcontext
    from llm_math_learning_runtime import Runtime
    from llm_feasibility_runtime import FlatOutput
    model,_,_=setup('qr20')
    with torch.no_grad():model.blocks[0][0].weight.zero_()
    dataset=SimpleNamespace(tokens=torch.arange(13,dtype=torch.int64)%2)
    runtime=SimpleNamespace(config={'sequence_length':2},model=model,active=FlatOutput(model),
        ids={'validation':torch.arange(5),'train_probe':torch.arange(3)},
        validation_dataset=dataset,train_dataset=dataset,autocast=nullcontext,budget=lambda **kw:None,
        protocol={'llm_evaluation':{'batch_size':2}})
    result=Runtime.evaluate(runtime)
    assert result['validation']['examples']==5 and result['validation']['valid_tokens']==10
    assert result['train_probe']['examples']==3 and result['train_probe']['valid_tokens']==6
    for row in result.values():
        assert row['ce']==pytest.approx(math.log(2),abs=1e-7) and row['accuracy']==.5


def test_actual_default_compiler_limits_are_enforced(monkeypatch):
    from llm_sparse_screen import compilation_guard
    runtime=SimpleNamespace(protocol=protocol(180))
    compilation_guard(runtime)
    monkeypatch.setattr(torch._dynamo.config,'recompile_limit',16)
    with pytest.raises(RuntimeError,match='compiler limits'):compilation_guard(runtime)
