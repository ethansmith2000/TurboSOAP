from contextlib import nullcontext
from types import SimpleNamespace
import ast
import json
import random
import os
import subprocess
import sys
import time
import weakref

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from benchmarks.matched_training_contract import (COMMON, ARMS, RandomStreams, SharedInitialState,
    build_arm, digest, tensor_fingerprint, preupdate_observation)
from benchmarks.matched_training_runtime import (OptimizerLease, captured_backward, check_state,
    evaluation_scope, refresh_due, run_shared, streaming_digest, streaming_tensor_record)


CONFIG = dict(COMMON, precondition_mode='all', covariance_compute_dtype='float32', clip_norm=1., min_lr_ratio=.01)
PROTOCOL = dict(disposable_qr10_smoke_updates=1, steps_per_arm=3, observation_steps=[0,1,2,3])


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.Sequential(torch.nn.Linear(4,4), torch.nn.Dropout(.2))
        self.head = torch.nn.Linear(4,3)
        self.register_buffer('scale', torch.tensor(1.))

    def forward(self, x):
        return self.head(self.blocks(x))*self.scale


class Event:
    def record(self):
        self.t = time.monotonic()


class CPU:
    def __init__(self, fault=None):
        torch.manual_seed(139); np.random.seed(139); random.seed(139)
        self.model = Tiny(); self.active = self.model; self.config = CONFIG.copy()
        self.streams = RandomStreams(torch.Generator().manual_seed(1140),
            SimpleNamespace(rng=np.random.default_rng(3140), counts={'mixup':0,'cutmix':0}))
        self.snapshot = SharedInitialState(self.model, self.streams)
        self.initial_rng = digest(self.streams.capture()); self.initial_model = self.snapshot.initial_digest
        self.schedule_steps = 100; self.warmup_steps = 5
        self.events = []; self.phase = None; self.arm = None; self.index = None
        self.updates = []; self.fault = fault; self.closed = False

    def make_batch(self, index):
        x = torch.randn(5,4,generator=self.streams.crop)
        x += float(self.streams.augmentation.rng.random()) + random.random() + float(np.random.random())
        self.streams.augmentation.counts['mixup'] += 1
        if self.fault in ('preflight_input','actual_first_input') and self.arm=='may_literal':
            if self.phase == ('probe' if self.fault=='preflight_input' else 'train'):
                x += .01
        return torch.arange(5)+index*5, x, torch.tensor([0,1,2,0,1])

    def evaluate(self):
        if self.fault=='evaluation' and self.phase=='train' and self.arm=='may_literal' and self.index==1:
            raise RuntimeError('Injected evaluation failure')
        with torch.no_grad():
            logits = self.model(torch.arange(20).reshape(5,4).float()/20)
            y = torch.tensor([0,1,2,0,1])
            return dict(ce=float(F.cross_entropy(logits,y)), correct=int((logits.argmax(1)==y).sum()))

    def emit(self, event, payload):
        self.events.append((event,json.loads(json.dumps(payload,allow_nan=False))))

    def before_smoke(self):
        self.phase='smoke'

    def after_smoke(self):
        assert self.snapshot.initial_digest == digest(self.model.state_dict())
        assert digest(self.streams.capture()) == self.initial_rng

    def before_probe(self, label):
        self.phase='probe'; self.arm=label

    def before_training_step(self, arm, index):
        self.phase='train'; self.arm=arm; self.index=index
        if self.fault=='repeat_numerics' and arm=='qr20_repeat' and index==1:
            with torch.no_grad(): self.model.head.weight.add_(.01)

    def before_optimizer_step(self, arm, index):
        assert any(event=='preflight_passed' for event,_ in self.events)
        if self.fault=='optimizer' and arm=='may_literal' and index==1:
            raise RuntimeError('Injected optimizer failure')
        self.updates.append((arm,index))

    def after_measured(self):
        if self.fault=='backend':
            raise RuntimeError('Injected backend check failure')

    def smoke_parity(self, x):
        return {'cpu_test_only':True}

    def budget(self, resources=False):
        if self.fault=='budget' and self.phase=='train' and self.arm=='may_literal' and self.index==1:
            raise RuntimeError('Injected continuous wall limit')

    def autocast(self):
        return nullcontext()

    def marks(self):
        return [Event() for _ in range(4)]

    def elapsed(self, marks):
        return [(marks[i+1].t-marks[i].t)*1000 for i in range(3)]

    def synchronize(self):
        pass

    def close(self):
        self.closed=True
        return dict(passed=self.fault!='placement',cpu_test_only=True)


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail(*a,**k):
        raise AssertionError('CPU runner test attempted CUDA initialization')
    monkeypatch.setattr(torch.cuda,'_lazy_init',fail)
    yield
    assert not torch.cuda.is_initialized()


def test_same_orchestration_all_four_arms_exact_qr_repeat_and_ownership():
    runtime=CPU(); result=run_shared(runtime,PROTOCOL)
    assert result['status']=='completed', result['failure']
    assert result['quality_qualified'] and not result['timing_qualified']
    assert [r['arm'] for r in result['runs']]==list(ARMS)
    assert len(runtime.updates)==12
    assert result['snapshot_release']['all_registered_released']
    assert all(r['release']['passed'] for r in result['runs'])
    assert runtime.snapshot.closed and runtime.closed
    assert all(p.grad is None for p in runtime.model.parameters())
    preflight_rows=[v for event,v in runtime.events if event=='preflight_probe']
    assert len(preflight_rows)==4
    assert all(r['observation']==preflight_rows[0]['observation'] for r in preflight_rows)
    assert result['runs'][0]['endpoint']==result['runs'][3]['endpoint']
    assert all(r['steps']==3 for r in result['runs'])


@pytest.mark.parametrize('fault', ['preflight_input','actual_first_input','optimizer','evaluation',
                                  'budget','backend','placement','repeat_numerics'])
def test_abort_stops_retains_failure_and_releases_snapshots_and_optimizers(fault):
    runtime=CPU(fault); result=run_shared(runtime,PROTOCOL)
    assert result['status']=='failed' and not result['quality_qualified']
    assert runtime.snapshot.closed and runtime.closed
    assert result['snapshot_release']['all_registered_released']
    releases=[v for event,v in runtime.events if event.endswith('_release')]
    releases += [v['release'] for event,v in runtime.events if event=='arm_result']
    assert all(r['passed'] for group in releases for r in (group if isinstance(group,list) else [group]))
    if fault=='preflight_input':
        assert not runtime.updates
        assert len([1 for event,_ in runtime.events if event=='preflight_probe'])==2
    if fault=='actual_first_input':
        assert runtime.updates==[('qr20',0),('qr20',1),('qr20',2)]
    if fault!='placement':
        assert digest(runtime.model.state_dict())==runtime.initial_model
        assert digest(runtime.streams.capture())==runtime.initial_rng
    assert any(event=='finished' for event,_ in runtime.events)


def test_capture_same_full_observation_and_clipped_gradients_as_frozen_helper():
    runtime=CPU()
    original=preupdate_observation(runtime.model,runtime.active,lambda:runtime.make_batch(0),
        runtime.streams,clip_norm=1.)
    gradients={n:p.grad.clone() for n,p in runtime.model.named_parameters()}
    runtime.snapshot.reset()
    row,loss,norm=captured_backward(runtime.model,runtime.active,lambda:runtime.make_batch(0),
        runtime.streams,clip_norm=1.)
    assert row==original
    assert tensor_fingerprint(loss)==original['loss'] and tensor_fingerprint(norm)==original['clip_norm']
    assert all(torch.equal(p.grad,gradients[n]) for n,p in runtime.model.named_parameters())


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.int64,torch.bool])
@pytest.mark.parametrize('shape',[(0,),(),(19,),(17,31)])
def test_streamed_full_bytes_match_original_digest(dtype,shape):
    x=torch.arange(max(1,int(np.prod(shape)))).reshape(shape) if shape!=(0,) else torch.empty(0)
    x=x.to(dtype)
    if x.ndim==2: x=x.T
    chunk=4096
    assert streaming_tensor_record(x,chunk)==tensor_fingerprint(x)
    assert streaming_digest({'state':{3:[x,None,[]]},'value':np.array([1,2])})==digest({'state':{3:[x,None,[]]},'value':np.array([1,2])})


def test_streaming_large_covariance_and_row_workspace_guard():
    x=torch.randn(4096,1024) # 16MiB tensor, still streamed in 1MiB chunks.
    assert streaming_tensor_record(x)==tensor_fingerprint(x)
    with pytest.raises(MemoryError,match='row exceeds'):
        streaming_tensor_record(x,32)


def test_external_storage_alias_fails_release_even_after_registered_tensor_dies():
    runtime=CPU(); opt=build_arm(runtime.model,CONFIG,'qr20')
    captured_backward(runtime.model,runtime.active,lambda:runtime.make_batch(0),runtime.streams,clip_norm=1.)
    opt.step(); lease=OptimizerLease(runtime.model,opt)
    alias=next(iter(opt.matrix.state.values()))['exp_avg'].view(-1)
    del opt
    result=lease.release()
    assert result['optimizer_objects_released']
    assert not result['all_registered_released'] and result['live_unique_storages']>=1
    assert not result['passed']
    del alias


def test_external_optimizer_object_reference_fails_release():
    runtime=CPU(); opt=build_arm(runtime.model,CONFIG,'qr20'); lease=OptimizerLease(runtime.model,opt)
    result=lease.release()
    assert not result['optimizer_objects_released'] and not result['passed']


def test_may_skipped_sentinels_and_every_state_tensor_finite_checked():
    runtime=CPU(); opt=build_arm(runtime.model,CONFIG,'may_literal')
    p=next(iter(opt.matrix.param_groups[0]['params']))
    opt.matrix.state[p].update(Q=[torch.eye(4,dtype=torch.bfloat16),[],None],
                              controller={'ema':torch.ones(1)},exp_avg=torch.zeros_like(p))
    row=check_state(runtime.model,opt)
    assert row['skipped_factors']==2 and row['factor_count']==1
    prior=torch.get_float32_matmul_precision()
    opt.matrix.state[p]['controller']['ema'].fill_(float('nan'))
    with pytest.raises(FloatingPointError):check_state(runtime.model,opt)
    assert torch.get_float32_matmul_precision()==prior
    opt.matrix.state[p]['controller']['ema'].fill_(1.)
    opt.matrix.state[p]['scalar']=float('inf')
    with pytest.raises(FloatingPointError,match='scalar'):check_state(runtime.model,opt)


def test_refresh_off_by_one_includes_cold_call_and_internal_step20():
    runtime=CPU(); opt=build_arm(runtime.model,CONFIG,'qr20')
    assert refresh_due(opt)
    for g in opt.matrix.param_groups:
        for p in g['params']: opt.matrix.state[p].update(step=18,Q=[torch.eye(4),torch.eye(4)])
    assert not refresh_due(opt)
    for state in opt.matrix.state.values():state['step']=19
    assert refresh_due(opt)


def test_evaluation_failure_restores_rng_and_mixed_training_flags():
    runtime=CPU(); runtime.model.blocks[1].eval(); before=digest(runtime.streams.capture())
    with pytest.raises(RuntimeError):
        with evaluation_scope(runtime):
            torch.rand(3); runtime.streams.crop.manual_seed(2)
            runtime.streams.augmentation.counts['mixup']=91
            raise RuntimeError('Injected evaluator')
    assert runtime.model.training and not runtime.model.blocks[1].training
    assert digest(runtime.streams.capture())==before


def test_gpu_entry_import_is_cpu_safe_and_runtime_requires_explicit_initialization():
    from benchmarks.cifar_matched_training import GPURuntime
    assert hasattr(GPURuntime,'initialize')
    assert not torch.cuda.is_initialized()


def test_cached_diagnostic_tensor_registered_and_nonfinite_cache_rejected():
    runtime=CPU(); opt=build_arm(runtime.model,CONFIG,'may_literal')
    opt.matrix._basis_weight_cache[('diagnostic',)] = torch.ones(4)
    alias=opt.matrix._basis_weight_cache[('diagnostic',)].view(-1)
    check_state(runtime.model,opt)
    opt.matrix.latest_basis_stats={'bad':float('nan')}
    with pytest.raises(FloatingPointError,match='scalar'):check_state(runtime.model,opt)
    opt.matrix.latest_basis_stats={}
    lease=OptimizerLease(runtime.model,opt); del opt
    row=lease.release()
    assert not row['passed'] and row['live_unique_storages']==1
    del alias


@pytest.mark.parametrize('corruption',['source','config','protocol','supervisor'])
def test_new_runnable_bundle_preserves_design_and_rejects_corruption(tmp_path,corruption):
    from benchmarks.prepare_matched_training import prepare,verify_runnable,DESIGN,sha
    before=sha(DESIGN/'manifest.json')
    b=tmp_path/'runnable'
    m=prepare(b)
    assert sha(DESIGN/'manifest.json')==before
    assert m['status']=='PREPARED_NOT_LAUNCHED' and not m['gpu_selected']
    assert len(m['source_sha256'])==22
    assert (b/'config.json').read_bytes()==(DESIGN/'config.json').read_bytes()
    assert not (b/'launch_manifest.json').exists() and not (b/'execution_status.json').exists()
    verify_runnable(b)
    with pytest.raises(ValueError,match='New unexecuted'):prepare(b)
    names={'source':'source/benchmarks/matched_training_runtime.py','config':'config.json',
           'protocol':'protocol.json','supervisor':'supervisor_template.conf'}
    p=b/names[corruption];p.write_bytes(p.read_bytes()+b'\n ')
    with pytest.raises(ValueError,match='drift|changed'):verify_runnable(b)


def test_prospective_runnable_operational_changes_preserve_all_design_gates():
    from benchmarks.prepare_matched_training import read,DESIGN,runnable_protocol
    design=read(DESIGN/'protocol.json');new=runnable_protocol(design)
    for key in ('arms','steps_per_arm','maximum_optimizer_updates','bounds','schedule','gate',
                'observation_steps','historical_recipe','artifact_policy','timing','guards'):
        assert design[key]==new[key]
    assert new['implementation']['main_and_autograd_cpu_after_smoke']==116
    assert new['implementation']['optimizer_hash_chunk_bytes']==2**20
    assert new['status']=='PREPARED_NOT_LAUNCHED'


def test_captured_training_math_nodes_match_frozen_observation_source():
    import inspect
    old=ast.parse(inspect.getsource(preupdate_observation));new=ast.parse(inspect.getsource(captured_backward))
    def critical(tree):
        rows=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and ast.unparse(node.func) in (
                'F.cross_entropy','loss.backward','torch.nn.utils.clip_grad_norm_',
                'active','model.zero_grad'):
                rows.append(ast.dump(node))
        return sorted(rows)
    assert critical(old)==critical(new)


def test_gpu_environment_failure_precedes_device_query_and_retains_one_attempt(tmp_path,monkeypatch):
    from benchmarks.prepare_matched_training import prepare
    from benchmarks import cifar_matched_training as cli
    b=tmp_path/'runnable';prepare(b)
    monkeypatch.setattr(cli,'HERE',b/'source/benchmarks')
    monkeypatch.setattr('sys.argv',['runner','--bundle',str(b)])
    def env_fail(bundle):raise RuntimeError('Injected inherited environment mismatch')
    def device_fail():raise AssertionError('CUDA query preceded environment gate')
    monkeypatch.setattr(cli,'verify_environment_launch',env_fail)
    monkeypatch.setattr(torch.cuda,'device_count',device_fail)
    with pytest.raises(RuntimeError,match='environment mismatch'):cli.main()
    status=(b/'execution_status.json').read_bytes()
    assert json.loads(status)['status']=='failed'
    with pytest.raises(RuntimeError,match='One attempt'):cli.main()
    assert (b/'execution_status.json').read_bytes()==status


def test_continuous_wall_clock_covers_initial_verification_time(tmp_path):
    from benchmarks.prepare_matched_training import prepare
    from benchmarks.cifar_matched_training import GPURuntime
    b=tmp_path/'runnable';prepare(b)
    runtime=GPURuntime(b,started=time.monotonic()-3601)
    with pytest.raises(RuntimeError,match='Continuous all-in wall'):runtime.budget()


def test_steady_cuda_optimizer_interval_ends_before_batch_hashes():
    import inspect
    from benchmarks.matched_training_runtime import train_arm
    source=inspect.getsource(train_arm)
    assert source.index('lease.opt.step(); marks[3].record()') < source.index("result['sampled_batches'][str(index+1)]")


def test_frozen_runtime_imports_in_clean_subprocess_without_worktree_fallback(tmp_path):
    from benchmarks.prepare_matched_training import prepare
    b=tmp_path/'runnable';prepare(b)
    script='''import pathlib,sys,torch
root=pathlib.Path(sys.argv[1])/'source'
sys.path[:0]=[str(root),str(root/'benchmarks'),str(root/'benchmarks/cifar_support')]
assert not torch.cuda.is_initialized()
def fail(*a,**k):raise AssertionError('Isolated frozen import attempted CUDA')
torch.cuda._lazy_init=fail
import benchmarks.matched_training_runtime as runtime
import benchmarks.cifar_matched_training as cli
import benchmarks.prepare_matched_training as prep
import tensor_ownership
for module in [runtime,cli,prep,tensor_ownership]:
 assert root in pathlib.Path(module.__file__).resolve().parents,module.__file__
prep.verify_runnable(root.parent)
assert not torch.cuda.is_initialized()
print('PASS_ISOLATED_FROZEN_IMPORT')
'''
    env={**os.environ,'PYTHONPATH':'','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
    result=subprocess.run([sys.executable,'-c',script,str(b)],cwd=tmp_path,env=env,
                          text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'PASS_ISOLATED_FROZEN_IMPORT' in result.stdout
