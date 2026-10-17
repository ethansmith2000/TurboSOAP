"""Seed/order-only confirmation, immutable provenance and unchanged runner wiring."""
from pathlib import Path
import sys,copy,json
from types import SimpleNamespace
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import pytest
import torch
import reuse_confirmation as entry
import reuse_learning as runner
from reuse_confirmation_contract import recipe,protocol,validate,PARENT,ARMS
from reuse_learning_contract import recipe as parent_recipe,protocol as parent_protocol
from test_reuse_learning import runtime as tiny_runtime

@pytest.fixture(autouse=True)
def cpu_and_scoped_binding(monkeypatch):
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda *a,**k:(_ for _ in ()).throw(AssertionError('CUDA in CPU check')))
    torch.set_num_threads(2)
    for name in ['ARMS','Runtime','verify','verify_launch','verify_backend']:monkeypatch.setattr(runner,name,getattr(runner,name))
    yield
    assert not torch.cuda.is_initialized()

def test_only_declared_seed_order_evaluation_and_entry_metadata_change():
    c=recipe();old=parent_recipe();c['seed']=old['seed'];c['matched_comparison']['execution_order']=old['matched_comparison']['execution_order'];assert c==old
    p=protocol();old=parent_protocol()
    for name in ['seed','execution_order','implementation']:p[name]=old[name]
    p['llm_evaluation']['panels']=old['llm_evaluation']['panels'];p.pop('confirmation');assert p==old
    assert protocol()['execution_order']==['reuse20','qr20'] and protocol()['maximum_optimizer_updates']==7515
    assert protocol()['decision']==parent_protocol()['decision'] and protocol()['bounds']==parent_protocol()['bounds']
    validate(recipe(),protocol())
    with pytest.raises(ValueError,match='fresh186'):recipe(185)
    for key,val in [('seed',185),('execution_order',['qr20','reuse20'])]:
        changed=protocol();changed[key]=val
        with pytest.raises(ValueError,match='Undeclared'):validate(recipe(),changed)

def test_binding_reuses_exact_trajectory_kernel_main_and_only_scopes_launch_functions():
    original={n:getattr(runner,n) for n in ['trajectory','run_pilot','preflight','build_arm','qualify_actual_m','main']}
    bound=entry.bind_runner()
    assert bound is runner and bound.ARMS==ARMS and bound.Runtime is entry.Runtime
    assert bound.verify is entry.verify and bound.verify_launch is entry.verify_launch
    assert all(getattr(bound,k) is v for k,v in original.items())

def test_reversed_full_two_arm_disposable_prefix_and_release(tmp_path):
    core=entry.bind_runner();r,e=tiny_runtime(tmp_path);r.config=recipe();p=protocol();p['coverage']=r.protocol['coverage'];p['evaluation_calls']=[];r.protocol=p
    try:
        expected=core.preflight(r);seen=[]
        for arm in p['execution_order']:
            d=core.trajectory(r,arm,'disposable',245,expected);v=core.trajectory(r,arm,'primary',245,expected);seen.append(arm)
            assert d['prefixes']['245']==v['prefixes']['245']
            assert [(x['loss'],x['gradient_norm']) for x in d['timings']]==[(x['loss'],x['gradient_norm']) for x in v['timings']]
            assert d['release']['passed'] and v['release']['passed']
        assert seen==['reuse20','qr20']
        assert all(v['passed'] and v['rng_unchanged'] for v in r.kernel.moments)
    finally:r.snapshot.close()

@pytest.mark.parametrize('phase',['before_smoke','after_measured'])
def test_new_runtime_uses_confirmation_backend_verifier(monkeypatch,phase):
    record={'backend':'fixture'};calls=[];emits=[]
    monkeypatch.setattr(entry,'backend_record',lambda:record)
    monkeypatch.setattr(entry,'verify_backend',lambda b,r:calls.append((b,r)))
    fake=SimpleNamespace(bundle='new186',emit=lambda e,p:emits.append((e,p)),execution_contract=lambda:{'passed':True},retain_code=lambda p:None,budget=lambda **kw:None)
    getattr(entry.Runtime,phase)(fake)
    assert calls==[('new186',record)] and emits[0][1]['passed']

def test_confirmation_prepare_preserves243sources_and_rejects_criterion_tampering(tmp_path):
    import prepare_reuse_confirmation as prep
    ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=186,parameters=144875648,cuda_initialized=False,snapshot_purpose_bytes=2230649344,model_sha256='6'*64))
    cal=tmp_path/'cal.json';prep.write(cal,dict(status='PASS_CPU_REUSE_CONFIRMATION',cuda_initialized=False,tests_passed=True))
    b=tmp_path/'bundle';m=prep.prepare(b,ref,186,PARENT/'token_cache_audit.json',cal)
    old=prep.read(PARENT/'manifest.json')['source_sha256'];assert len(old)==243 and len(m['source_sha256'])==248 and all(m['source_sha256'][n]==h for n,h in old.items())
    p=prep.read(b/'protocol.json');p['decision']['quality_margin_ce']=.02;prep.write(b/'protocol.json',p)
    m=prep.read(b/'manifest.json');m['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',m)
    with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
    with pytest.raises(ValueError,match='Fresh bundle'):prep.prepare(b,ref,186,PARENT/'token_cache_audit.json',cal)
