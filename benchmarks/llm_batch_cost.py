"""Trial201: batch128 cost feasibility on the established deep OpenWebText model.

Only the public reference's token batch is adopted here. This is not a paper
reproduction, convergence comparison, new optimizer or update-mechanism audit.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE),str(HERE/'cifar_support')]
import numpy as np
import torch
from llm_accumulation import backward_accumulated, weighted_cost
from cifar_optimizers import apply_lr_multiplier
from cifar100_qualification import lr_at
from deep_qr10_kernel import build_optimizer as qr_builder, qualify_actual_m as qr_qualify
from deep_qr10_learning import compact_guard as qr_guard
from deep_matched_warm_kernel import build_optimizer as warm_builder, qualify_actual_m as warm_qualify, event_kind as warm_kind
from deep_matched_warm_learning import compact_guard as warm_guard
from deep_llm_data import CACHE,tokens_from_rows,shifted_loss,sha
from deep_llm_feasibility import verify,write,event_record,forward_context
from deep_llm_learning_data import data_plan,input_sha,verify_stats
from execution_fingerprints import backend_record
from llm_feasibility_runtime import build_model
from matched_training_runtime import streaming_digest
from tensor_ownership import TensorOwnership

PARENT=Path('/workspace/optimizer_replay_results/llm_soap_deep_matched_warm_trial200_frozen_20261020')
ARMS=('qr10','fast10')
PREFIX=75
MEASURED=201
ACCUMULATION=16


def recipe(parent):
    return {**parent,'trial':201,'batch_size':128,'microbatch_size':8,
        'gradient_accumulation_steps':16,'primary_calls':201,'disposable_calls':75,
        'max_wall_seconds':1800,'transport_precision':'per-arm: native highest / FastSOAP inherit',
        'matched_comparison':dict(arms=list(ARMS),seed=193,phase_order=['qr10 disposable','qr10 measured','fast10 disposable','fast10 measured'],
            scope='Batch-amortization feasibility only; fixed existing model/data/math, not full independent reference recipe or convergence evidence')}


def verify_protocol(bundle):
    c=json.loads((bundle/'config.json').read_text())
    if c!=recipe(json.loads((PARENT/'config.json').read_text())):raise RuntimeError('Undeclared recipe change')
    for name,digest in json.loads((bundle/'parent_pins.json').read_text()).items():
        if sha(PARENT/name)!=digest:raise RuntimeError('Completed parent changed: '+name)
    for name in ('data_audit.json','data_stat.json','initial_reference.json','backend_contract.json'):
        if (bundle/name).read_bytes()!=(PARENT/name).read_bytes():raise RuntimeError('Shared reference differs: '+name)
    return c


class Run:
    def __init__(self,bundle):
        self.bundle=bundle;self.started=time.monotonic();self.model=None;self.opt=None
        self.snapshot={};self.snapshot_owner=TensorOwnership();self.snapshot_release=None
        self.phases=[];self.releases=[];self.prefixes={};self.calls=0;self.microcalls=0
        self.current_arm='setup';self.current_phase='setup';self.current_call=0

    def emit(self,event,row):
        with (self.bundle/'events.jsonl').open('a') as f:f.write(json.dumps(event_record(event,time.monotonic()-self.started,row),allow_nan=False)+'\n')
        if event in ('started','preflight','prefix_passed','phase_complete','finished'):
            print(json.dumps(dict(event=event,**row),allow_nan=False),flush=True)

    def status(self,status='running',**extra):
        write(self.bundle/'execution_status.json',dict(status=status,trial=201,pid=os.getpid(),arm=self.current_arm,phase=self.current_phase,
            phase_call=self.current_call,optimizer_calls=self.calls,microbatch_backwards=self.microcalls,elapsed_seconds=time.monotonic()-self.started,**extra))

    def budget(self):
        import psutil
        if time.monotonic()-self.started>self.c['max_wall_seconds']:raise TimeoutError('Declared wall limit')
        if torch.cuda.max_memory_allocated()>self.c['maximum_allocated_bytes']:raise MemoryError('Declared GPU limit')
        if psutil.Process().memory_info().rss>12*2**30:raise MemoryError('Declared RSS limit')
        if shutil.disk_usage(self.bundle).free<5*2**30:raise OSError('Shared free disk')
        if self.calls%50==0:
            verify_stats(self.audit['cache_file_stat'])
            compact=sum(p.stat().st_size for p in self.bundle.rglob('*') if p.is_file() and 'compile_cache' not in p.parts)
            cache=sum(p.stat().st_size for p in (self.bundle/'compile_cache').rglob('*') if p.is_file())
            if compact>100*2**20 or cache>512*2**20:raise OSError('Declared compact/cache limits')

    def rng(self):return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())
    def rng_digest(self):return streaming_digest(self.rng())

    def reset(self):
        self.model.load_state_dict(self.snapshot);self.model.zero_grad(set_to_none=True);self.model.train()
        r=self.initial_rng;torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state(r['cuda']);random.setstate(r['python']);np.random.set_state(r['numpy'])
        if streaming_digest(self.model.state_dict())!=self.initial:raise RuntimeError('Exact initial reset failed')

    def endpoint(self):return streaming_digest(dict(model=self.model.state_dict(),optimizer=self.opt.state_dict(),gradients=[p.grad for p in self.model.parameters()],rng=self.rng_digest()))

    def setup(self):
        self.c=verify_protocol(self.bundle);self.manifest,self.audit=verify(self.bundle)
        launch=json.loads((self.bundle/'launch.json').read_text())
        if os.environ.get('GPU_CLAIM_INDICES')!='7' or os.environ.get('CUDA_VISIBLE_DEVICES')!=launch['uuid'] or torch.cuda.device_count()!=1:raise RuntimeError('GPU7 lifetime claim required')
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.cuda.set_per_process_memory_fraction(self.c['maximum_allocated_bytes']/torch.cuda.get_device_properties(0).total_memory)
        random.seed(193);np.random.seed(193);torch.manual_seed(193);torch.cuda.manual_seed_all(193)
        self.model=build_model(self.c);self.initial=streaming_digest(self.model.state_dict())
        ref=json.loads((self.bundle/'initial_reference.json').read_text())
        if self.initial!=ref['model_sha256'] or sum(p.numel() for p in self.model.parameters())!=109396992:raise RuntimeError('Existing initial model reference differs')
        self.snapshot={k:t.detach().clone() for k,t in self.model.state_dict().items()};self.snapshot_owner.watch_tree(self.snapshot,'initial_cpu_reset')
        self.model=self.model.cuda();self.initial_rng=self.rng()
        import datasets
        self.data=datasets.load_from_disk(str(CACHE),keep_in_memory=False)
        for split,ds in self.data.items():
            if len(ds)!=self.audit['parent_structure'][split]['rows'] or ds._fingerprint!=self.audit['parent_structure'][split]['fingerprint']:raise RuntimeError('Shared cache structure differs')
        self.order,_=data_plan(len(self.data['train']),len(self.data['validation']),193)
        if hashlib.sha256(self.order.astype('<i8').tobytes()).hexdigest()!=self.audit['training_plan_sha256']:raise RuntimeError('Existing frozen microbatch plan differs')
        if len(np.unique(self.order[:MEASURED*ACCUMULATION]))!=MEASURED*128:raise RuntimeError('Repeated data within a phase')
        backend=backend_record();write(self.bundle/'backend_before.json',backend)
        if {k:v for k,v in backend.items() if k not in ('environment','loaded_libraries')}!=json.loads((self.bundle/'backend_contract.json').read_text()):raise RuntimeError('Numerical backend differs')
        self.emit('started',dict(pid=os.getpid(),trial=201,parameters=109396992,global_batch=128,microbatch=8,accumulation=16,sequence=1024,
            target_tokens_per_update=130944,total_optimizer_calls=552,total_microbatch_backwards=8864,phase_microbatch_backwards=8832,measured_calls_per_arm=201,disposable_calls_per_arm=75,
            scope='Batch-cost feasibility on existing model, not a full reference-recipe reproduction or learning conclusion'))
        self.budget()

    def backward(self,call):
        batches=[]
        for index in range((call-1)*ACCUMULATION,call*ACCUMULATION):
            tokens=tokens_from_rows(self.data['train'],self.order[index])
            if input_sha(tokens)!=self.audit['train_batches_sha256'][index]:raise RuntimeError('Actual microbatch input differs')
            batches.append(tokens)
        def loss_fn(model,cpu):
            tokens=cpu.cuda()
            with forward_context():return shifted_loss(model(tokens),tokens)
        loss,norm=backward_accumulated(self.model,batches,loss_fn,count=ACCUMULATION,clip_norm=self.c['clip_norm'])
        self.microcalls+=ACCUMULATION
        return loss,norm

    def preflight(self):
        observations=[]
        for _ in range(2):
            self.reset();loss,norm=self.backward(1)
            observations.append(dict(loss=float(loss),norm=float(norm),gradients=streaming_digest([p.grad for p in self.model.parameters()])))
            del loss,norm;self.budget()
        if observations[0]!=observations[1]:raise RuntimeError('Repeated accumulated backward differs')
        self.preflight_record=observations[0];self.model.zero_grad(set_to_none=True)
        self.emit('preflight',dict(exact_repeat=True,microbatch_backwards=32,observation=observations[0]))

    def phase(self,arm,phase,calls):
        self.current_arm=arm;self.current_phase=phase;self.current_call=0;self.reset()
        c={**self.c,'transport_precision':'highest' if arm=='qr10' else 'inherit'}
        self.opt=qr_builder(self.model,c,'qr10') if arm=='qr10' else warm_builder(self.model,c,'fast10')
        router=SimpleNamespace(rng_digest=self.rng_digest,emit=self.emit,moments=[],parities=[])
        (qr_qualify if arm=='qr10' else warm_qualify)(self.opt.matrix,router,arm,'disposable' if phase=='disposable' else 'primary')
        graph_start=int(torch._dynamo.utils.counters['stats']['unique_graphs']);rows=[];owner=TensorOwnership()
        try:
            for call in range(1,calls+1):
                self.budget();torch.cuda.synchronize();started=time.perf_counter()
                apply_lr_multiplier(self.opt,lr_at(call,35000,875,1.,self.c['min_lr_ratio']))
                loss,norm=self.backward(call)
                if call==1 and dict(loss=float(loss),norm=float(norm),gradients=streaming_digest([p.grad for p in self.model.parameters()]))!=self.preflight_record:raise RuntimeError('Actual first accumulated gradient differs')
                torch.cuda.synchronize();backward_ms=1000*(time.perf_counter()-started)
                marks=[torch.cuda.Event(enable_timing=True) for _ in range(2)];host=time.perf_counter();marks[0].record();self.opt.step();marks[1].record();torch.cuda.synchronize()
                opt_ms=1000*(time.perf_counter()-host);iteration_ms=1000*(time.perf_counter()-started)
                kind=('cold' if call==1 else 'qr' if (call-1)%10==0 else 'ordinary') if arm=='qr10' else warm_kind(call)
                diagnostic=call in ((11,) if arm=='qr10' and phase=='disposable' else (11,51,61) if phase=='disposable' else (11,51))
                row=dict(arm=arm,phase=phase,call=call,kind=kind,diagnostic=diagnostic,loss=float(loss),norm=float(norm),backward_host_ms=backward_ms,
                    optimizer_host_ms=opt_ms,optimizer_event_ms=marks[0].elapsed_time(marks[1]),iteration_host_ms=iteration_ms,accumulation=16,target_tokens=130944)
                if not all(math.isfinite(v) for v in row.values() if isinstance(v,float)):raise FloatingPointError('Nonfinite update')
                del loss,norm,marks;rows.append(row);self.calls+=1;self.current_call=call;self.emit('update',row)
                if call in (1,11,51) or (call-1)%20==0 or call==calls:
                    guard=(qr_guard if arm=='qr10' else warm_guard)(self.opt,self.model,call);self.emit('guard',dict(arm=arm,phase=phase,**guard))
                if call==PREFIX:
                    prefix=dict(scalars=[(r['loss'],r['norm']) for r in rows],state=self.endpoint())
                    if phase=='disposable':self.prefixes[arm]=prefix
                    else:
                        if prefix!=self.prefixes[arm]:raise RuntimeError('Exact accumulated75-call prefix differs')
                        self.emit('prefix_passed',dict(arm=arm,call=75,all_model_optimizer_gradient_rng_scalars=True))
                        if arm==ARMS[-1]:self.release_snapshot()
                if call%10==0:self.status()
            expected=48 if arm=='qr10' and phase=='disposable' else 96
            if len(router.moments)!=expected or not all(r['passed'] and r['rng_unchanged'] for r in router.moments):raise RuntimeError('Actual-M qualification differs')
            if len(router.parities)!=(192 if arm=='fast10' and phase=='disposable' else 0):raise RuntimeError('Own warm reports differ')
            clean=[r for r in rows if r['call']>50 and not r['diagnostic']]
            means={kind:dict(samples=len(chosen),**{key:statistics.mean(r[key] for r in chosen) for key in ('backward_host_ms','optimizer_host_ms','optimizer_event_ms','iteration_host_ms')})
                for kind in ('ordinary','warm','qr') if (chosen:=[r for r in clean if r['kind']==kind])}
            graphs=int(torch._dynamo.utils.counters['stats']['unique_graphs'])-graph_start
            if phase=='measured' and graphs:raise RuntimeError('Compilation during measured phase')
            result=dict(arm=arm,phase=phase,calls=calls,steady_means=means,actual_m_checks=len(router.moments),own_reports=len(router.parities),new_compile_graphs=graphs)
            self.phases.append(result);self.emit('phase_complete',result)
        finally:
            owner.watch_tree(self.opt.state_dict(),'optimizer');owner.watch_tree([p.grad for p in self.model.parameters()],'gradients')
            self.opt.matrix.state.clear();self.opt.auxiliary.state.clear();self.model.zero_grad(set_to_none=True);self.opt=None;del router;gc.collect();torch.cuda.synchronize()
            release=owner.report();owner.close();self.releases.append(release);self.emit('optimizer_release',dict(arm=arm,phase=phase,**release))
            if not release['all_registered_released']:raise RuntimeError('Optimizer state retained')

    def release_snapshot(self):
        if self.snapshot_release is None:
            self.snapshot.clear();gc.collect()
            self.snapshot_release=self.snapshot_owner.report() if self.snapshot_owner._tensors else dict(all_registered_released=True)
            self.snapshot_owner.close();self.emit('snapshot_release',self.snapshot_release)
            if not self.snapshot_release['all_registered_released']:raise RuntimeError('Snapshot retained')

    def run(self):
        try:
            self.setup();self.preflight()
            for arm in ARMS:self.phase(arm,'disposable',PREFIX);self.phase(arm,'measured',MEASURED)
            verify_protocol(self.bundle);verify(self.bundle)
            backend=backend_record();write(self.bundle/'backend_after.json',backend)
            if {k:v for k,v in backend.items() if k not in ('environment','loaded_libraries')}!=json.loads((self.bundle/'backend_contract.json').read_text()):raise RuntimeError('Backend changed')
            if self.calls!=552 or self.microcalls!=8864:raise RuntimeError('Global or microbatch call count differs')
        finally:
            self.release_snapshot()
            if self.model is not None:
                owner=TensorOwnership();owner.watch_tree(self.model.state_dict(),'model');self.model=None;gc.collect();torch.cuda.synchronize()
                self.model_release=owner.report();owner.close();self.emit('model_release',self.model_release)
                if not self.model_release['all_registered_released']:raise RuntimeError('Model retained')
        measured={r['arm']:r['steady_means'] for r in self.phases if r['phase']=='measured'}
        costs={arm:{key:weighted_cost(means,arm,key) for key in ('backward_host_ms','optimizer_host_ms','optimizer_event_ms','iteration_host_ms')} for arm,means in measured.items()}
        savings={key:100*(1-costs['fast10'][key]/costs['qr10'][key]) for key in ('optimizer_host_ms','iteration_host_ms')}
        result=dict(status='completed',trial=201,optimizer_calls=self.calls,microbatch_backwards=self.microcalls,phases=self.phases,costs=costs,saving_percent=savings,
            all_in_seconds=time.monotonic()-self.started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),snapshot_release=self.snapshot_release,releases=self.releases,model_release=self.model_release,
            scope='One sequential batch-cost pilot on existing109M model; only batch128/seq1024 is taken from independent reference. Not paper reproduction,quality evaluation,full-reference speedup or width extrapolation.')
        write(self.bundle/'summary.json',result);self.emit('finished',result);self.status('completed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);args=p.parse_args()
    if HERE.parent!=args.bundle.resolve()/'source':raise RuntimeError('Frozen entry only')
    if (args.bundle/'execution_status.json').exists() or (args.bundle/'events.jsonl').exists():raise RuntimeError('One attempt only')
    runtime=Run(args.bundle);runtime.status()
    try:runtime.run()
    except BaseException as exc:
        import traceback
        runtime.status('failed',error=repr(exc),traceback=traceback.format_exc());raise
