"""CPU-only new integration/reset/precision/provenance checks."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch
from cadence_training_contract import build_arm as old_arm, expected_counts as old_counts
from hybrid_training_kernel import HybridKernel, build_arm, eager_hybrid, precision
from hybrid_training_contract import ARMS, protocol, recipe, expected_counts, validate
from matched_training_contract import COMMON, RandomStreams, SharedInitialState, digest
from matched_training_runtime import OptimizerLease, streaming_digest
from soap import SOAP, _damped_jacobi_generator
from movement_retraction import clip_skew_frobenius

CONFIG=dict(COMMON,precondition_mode='all',covariance_compute_dtype='float32',clip_norm=1.,min_lr_ratio=.01)

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=torch.nn.Sequential(*[torch.nn.Linear(4,4) for _ in range(4)],torch.nn.Dropout(.2))
        self.head=torch.nn.Linear(4,3)
    def forward(self,x):return self.head(self.blocks(x))

def identity_compiler(fn,**kwargs):return fn

def manual(cov,q,mixed):
    with precision('high'):
        b=SOAP._basis_covariance(cov,q)
        a=clip_skew_frobenius(.5*_damped_jacobi_generator(b,.01),.5)
        delta=(q.bfloat16()@a.bfloat16()).float() if mixed else q@a
        y=q+delta
    with precision('highest'):
        for _ in range(2):y=1.5*y-.5*(y@(y.T@y))
    return y

@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail(*a,**k):raise AssertionError('CPU preparation attempted CUDA initialization')
    monkeypatch.setattr(torch.cuda,'_lazy_init',fail)
    yield
    assert not torch.cuda.is_initialized()

@pytest.mark.parametrize('mixed',[False,True])
def test_eager_exact_independent_formula_and_nonmutation(mixed):
    torch.manual_seed(411)
    q=torch.linalg.qr(torch.randn(16,16)).Q;g=torch.randn(16,16);cov=g@g.T+torch.eye(16)
    before=[v.clone() for v in (cov,q)]
    torch.set_float32_matmul_precision('high')
    assert torch.equal(eager_hybrid(cov,q,mixed),manual(cov,q,mixed))
    assert torch.get_float32_matmul_precision()=='high'
    assert all(torch.equal(a,b) for a,b in zip(before,(cov,q)))

def test_compile_cached_boundaries_and_failure_restoration():
    compiled=[]
    def compiler(fn,**kw):compiled.append(kw);return fn
    kernel=HybridKernel(compiler);opt=build_arm(Tiny(),CONFIG,'hybrid20',kernel);g=opt.matrix.param_groups[0]
    q=torch.eye(4);cov=torch.eye(4)
    for _ in range(3):kernel(cov,q,g)
    assert compiled==[dict(mode='default',fullgraph=True,dynamic=False)]*2
    assert kernel.calls=={'4':3}
    def broken(y):raise RuntimeError('injected retraction error')
    kernel.retraction=broken;torch.set_float32_matmul_precision('high')
    with pytest.raises(RuntimeError,match='injected'):kernel(cov,q,g)
    assert torch.get_float32_matmul_precision()=='high' and torch.backends.cuda.matmul.allow_tf32
    g['basis_lr']=.6
    with pytest.raises(ValueError,match='policy'):kernel(cov,q,g)

def test_actual_optimizer_ordinary_qr_m_v_paths_and_two_repairs():
    torch.manual_seed(42);a=Tiny();b=copy.deepcopy(a);c=copy.deepcopy(a)
    kernel=HybridKernel(identity_compiler)
    original=old_arm(a,CONFIG,'warm20');hybrid=build_arm(b,CONFIG,'hybrid20',kernel)
    oracle=old_arm(c,CONFIG,'warm20');oracle.matrix._gauge_step_one=lambda cov,q,g:manual(cov,q,True)
    assert type(original.matrix) is type(hybrid.matrix)
    assert original.matrix.step.__func__ is hybrid.matrix.step.__func__
    assert original.matrix._research_refresh.__func__ is hybrid.matrix._research_refresh.__func__
    assert original.matrix._transport_first_moment.__func__ is hybrid.matrix._transport_first_moment.__func__
    assert [{k:v for k,v in g.items() if k!='params'} for g in original.auxiliary.param_groups]==[
        {k:v for k,v in g.items() if k!='params'} for g in hybrid.auxiliary.param_groups]
    generator=torch.Generator().manual_seed(97)
    for call in range(1,422):
        for pa,pb,pc in zip(a.parameters(),b.parameters(),c.parameters()):
            grad=torch.randn(pa.shape,generator=generator)*.01
            pa.grad=grad.clone();pb.grad=grad.clone();pc.grad=grad.clone()
        original.step();hybrid.step();oracle.step()
        if call<=20:
            assert streaming_digest(original.state_dict())==streaming_digest(hybrid.state_dict())
        if call in (21,41,201,221,401,421):
            assert streaming_digest(hybrid.state_dict())==streaming_digest(oracle.state_dict())
            assert streaming_digest(b.state_dict())==streaming_digest(c.state_dict())
    assert kernel.calls=={'4':19*8} # 21refresh events minus2QR repairs.
    expected=old_counts(421,'warm',20,200)
    for state in hybrid.matrix.state.values():
        assert all(state[k]==v for k,v in expected.items())

class Event:
    def record(self):self.t=time.perf_counter()
    def elapsed_time(self,other):return (other.t-self.t)*1000

class CPU:
    def __init__(self,bundle,fault=None):
        torch.manual_seed(141);np.random.seed(141);random.seed(141)
        self.bundle=bundle;self.model=Tiny();self.active=self.model;self.config=CONFIG.copy()
        self.streams=RandomStreams(torch.Generator().manual_seed(1142),
            SimpleNamespace(rng=np.random.default_rng(3142),counts={'mixup':0}))
        self.snapshot=SharedInitialState(self.model,self.streams)
        self.kernel=HybridKernel(identity_compiler);self.schedule_steps=35000;self.warmup_steps=875
        self.protocol=protocol();self.protocol.update(disposable_smoke_calls=1,disposable_calls_per_arm=45,steps_per_arm=221,
            prefix_hashes=[1,21,201,221],sampled_batch_calls=[21,201])
        self.protocol['coverage']=dict(matrix_parameters=4,two_sided_factors=8,dimensions_per_refresh={'4':8})
        self.protocol['expected']=dict(hybrid_calls_by_dimension={'4':96},diagnostic_reference_calls=8)
        self.events=[];self.current_arm=None;self.fault=fault
        self.initial_rng=streaming_digest(self.streams.capture());self.initial_model=self.snapshot.initial_digest
    def make_batch(self,index):
        x=torch.randn(5,4,generator=self.streams.crop)
        x+=float(self.streams.augmentation.rng.random())+random.random()+float(np.random.random())
        self.streams.augmentation.counts['mixup']+=1
        if self.fault=='first' and self.current_arm=='hybrid20_primary':x+=.01
        return torch.arange(5)+index*5,x,torch.tensor([0,1,2,0,1])
    def emit(self,event,payload):self.events.append((event,payload))
    def budget(self,resources=False):pass
    def synchronize(self):pass
    def autocast(self):return nullcontext()
    def optimizer_events(self):return [Event(),Event()]
    def smoke_parity(self,x):return dict(cpu_only=True)
    def before_smoke(self):pass
    def after_smoke(self):pass
    def after_measured(self):pass
    def close(self):return dict(passed=True,cpu_only=True)

def test_disposable_warmup_fresh_reset_preflight_and_release(tmp_path,monkeypatch):
    from cifar_hybrid_training import run_pilot
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    runtime=CPU(tmp_path);result=run_pilot(runtime)
    assert result['status']=='completed' and not result['qualified_learning'] and not result['qualified_training_timing']
    assert result['optimizer_calls']==799 and result['hybrid_calls']=={'4':96}
    assert result['snapshot_release']['all_registered_released'] and runtime.snapshot.closed
    assert all(r['release']['passed'] for r in result['primary']+result['disposable'])
    assert len({r['endpoint']['rng_sha256'] for r in result['primary']})==1
    # Primary trajectories start from the exact same captured backward, not
    # from disposable trained weights/moments. This gate runs six times.
    first=[row['observation'] for event,row in runtime.events if event=='actual_first_passed']
    assert len(first)==6 and all(r==first[0] for r in first)
    assert len(result['actual_state_probes'])==8

def test_failure_releases_disposable_primary_and_snapshot(tmp_path,monkeypatch):
    from cifar_hybrid_training import run_pilot
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    runtime=CPU(tmp_path,fault='first')
    with pytest.raises(RuntimeError,match='Trajectory failed'):run_pilot(runtime)
    assert runtime.snapshot.closed
    cleanup=json.loads((tmp_path/'final_cleanup.json').read_text())
    assert cleanup['snapshot_release']['all_registered_released']
    failed=json.loads((tmp_path/'hybrid20_primary.json').read_text())
    assert failed['release']['passed'] and failed['calls']==0

def test_prospective_counts_recipe_and_semantic_rejection():
    p=protocol()
    assert p['maximum_optimizer_updates']==2020 and p['warmup']['covers_post_repair_warm_calls']==[221,241]
    assert p['expected']['hybrid_kernel_calls']==232
    assert sum(p['expected']['hybrid_calls_by_dimension'].values())==232
    for calls in (1,245,420):
        assert expected_counts(calls,'warm',20,200)==old_counts(calls,'warm',20,200)
    base=dict(CONFIG);c=recipe(base);validate(c,p,base)
    c['matrix_lr']=.0006
    with pytest.raises(ValueError,match='Undeclared'):validate(c,p,base)


def test_sealed_preparation_tamper_rejection_and_stdlib_only_verify(tmp_path):
    import prepare_hybrid_training as prep
    reference=tmp_path/'initial.json'
    prep.write(reference,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=141,parameters=28660324,
        cuda_initialized=False,snapshot_purpose_bytes=237631888,model_sha256='0'*64))
    b=tmp_path/'fresh';m=prep.prepare(b,reference)
    assert not (b/'launch_manifest.json').exists()
    code="import sys;sys.path.insert(0,sys.argv[1]);import prepare_hybrid_training as p;p.verify(sys.argv[2]);assert 'torch' not in sys.modules;print('STDLIB_ONLY_VERIFIED')"
    result=subprocess.check_output(['/usr/bin/python3','-I','-B','-c',code,str(Path(prep.__file__).parent),str(b)],text=True)
    assert result.strip()=='STDLIB_ONLY_VERIFIED'
    source=b/'source/benchmarks/hybrid_training_kernel.py';original=source.read_bytes()
    source.write_bytes(original+b'\n# corruption\n')
    with pytest.raises(ValueError,match='Frozen source'):prep.verify(b)
    source.write_bytes(original)
    for name,field,value in [('config.json','seed',142),('protocol.json','steps_per_arm',421)]:
        original=(b/name).read_bytes();obj=json.loads(original);obj[field]=value;prep.write(b/name,obj)
        resealed=copy.deepcopy(m);resealed['artifacts'][name]=prep.sha(b/name);prep.write(b/'manifest.json',resealed)
        with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
        (b/name).write_bytes(original);prep.write(b/'manifest.json',m)
    prep.verify(b)
    gpu=dict(index=2,uuid='GPU-f9397a2d-9940-b5ab-bd06-54c68b83117b')
    prep.write(b/'launch_manifest.json',dict(gpu=gpu,sha256={name:prep.sha(b/name) for name in
        ('manifest.json','config.json','protocol.json','environment_contract.json','supervisor_template.conf')}))
    env=prep.expected_environment(b);env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],GPU_CLAIM_INDICES='2')
    prep.verify_launch(b,env)
    env['TORCH_UNDECLARED_FLAG']='1'
    with pytest.raises(ValueError,match='environment'):prep.verify_launch(b,env)
