"""One frozen deeper native baseline with no-wrap data and read-only layer probes."""
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
from cifar_optimizers import build_optimizer,apply_lr_multiplier
from cifar100_qualification import lr_at
from cadence_training_contract import validate_counters
from deep_llm_data import CACHE,tokens_from_rows,shifted_loss
from deep_llm_feasibility import verify,write,event_record,forward_context
from deep_llm_learning_data import data_plan,input_sha,verify_stats,HORIZON
from deep_llm_learning_observer import LayerObserver,STAGES,MAX_BANK_BYTES
from execution_fingerprints import backend_record
from llm_feasibility_runtime import build_model
from llm_sparse_screen_kernel import common_geometry_guard
from matched_training_runtime import streaming_digest
from stage_stability_kernel import qualify_actual_m
from stage_stability_contract import PRIMARY_M_AGES
from tensor_ownership import TensorOwnership

EVALUATIONS=(0,1000,3500,10000,20000,30000,35000)
M_CALLS=tuple(age+1 for age in PRIMARY_M_AGES)


def compact_guard(opt,model,call):
    rows=common_geometry_guard(opt,model);counts=validate_counters(opt,call)
    if len(rows)!=96 or counts['parameter_count']!=48:raise RuntimeError('SOAP coverage differs')
    layers={}
    for row in rows:
        layer=str(int(row['parameter'].split('.')[1]))
        group=layers.setdefault(layer,dict(factors=0,absolute_gram_max=0.,normalized_gram_max=0.))
        group['factors']+=1
        group['absolute_gram_max']=max(group['absolute_gram_max'],row['absolute_gram_fro'])
        group['normalized_gram_max']=max(group['normalized_gram_max'],row['normalized_gram'])
    if len(layers)!=12 or any(row['factors']!=8 for row in layers.values()):raise RuntimeError('Layer coverage differs')
    return dict(call=call,all_96_factor_guards_pass=True,all_48_counters_pass=True,
        matrix_age=call-1,qr_refreshes_per_matrix=(call-1)//20,layers=layers)


class Run:
    def __init__(self,bundle):
        self.bundle=bundle;self.started=time.monotonic();self.model=None;self.opt=None;self.observer=None
        self.snapshot={};self.snapshot_owner=TensorOwnership();self.snapshot_release=None
        self.phases=[];self.evaluations=[];self.releases=[];self.budget_calls=0;self.calls=0
        self.current_phase='setup';self.primary_call=0

    def emit(self,event,row):
        with (self.bundle/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event_record(event,time.monotonic()-self.started,row),allow_nan=False)+'\n')
        if event in ('started','preflight','prefix_passed','evaluation','phase_complete','layer_observation','finished'):
            display={k:v for k,v in row.items() if k not in ('factors','gradients','adaptive','guards')}
            print(json.dumps(dict(event=event,**display),allow_nan=False),flush=True)

    def status(self,status='running',**extra):
        write(self.bundle/'execution_status.json',dict(status=status,pid=os.getpid(),phase=self.current_phase,
            primary_calls=self.primary_call,optimizer_calls=self.calls,elapsed_seconds=time.monotonic()-self.started,**extra))

    def budget(self):
        self.budget_calls+=1
        if time.monotonic()-self.started>self.c['max_wall_seconds']:raise TimeoutError('Internal wall limit')
        if torch.cuda.max_memory_allocated()>self.c['maximum_allocated_bytes']:raise MemoryError('GPU allocation limit')
        if self.budget_calls%50==0:
            import psutil
            if psutil.Process().memory_info().rss>12*2**30:raise MemoryError('Process RSS limit')
        if self.budget_calls%1000==0:
            if shutil.disk_usage(self.bundle).free<5*2**30:raise OSError('Shared free-disk limit')
            size=self.source_bytes+sum(p.stat().st_size for p in self.bundle.iterdir() if p.is_file())
            if size>100*2**20:raise OSError('Compact evidence limit')
            verify_stats(self.audit['cache_file_stat'])

    def rng_state(self):
        return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())

    def rng_digest(self):return streaming_digest(self.rng_state())

    def endpoint(self,opt):
        return streaming_digest(dict(model=self.model.state_dict(),optimizer=opt.state_dict(),
            gradients=[p.grad for p in self.model.parameters()],rng=self.rng_digest()))

    def setup(self):
        self.manifest,self.audit=verify(self.bundle)
        self.c=json.loads((self.bundle/'config.json').read_text());c=self.c
        if c['primary_calls']!=HORIZON or c['seed']!=193:raise ValueError('Only frozen seed193/horizon declared')
        launch=json.loads((self.bundle/'launch.json').read_text())
        if os.environ.get('GPU_CLAIM_INDICES')!='7' or os.environ.get('CUDA_VISIBLE_DEVICES')!=launch['uuid'] or torch.cuda.device_count()!=1:
            raise RuntimeError('Single physical GPU7 lifetime claim required')
        if shutil.disk_usage(self.bundle).free<5*2**30:raise OSError('Shared free-disk limit')
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.cuda.set_per_process_memory_fraction(c['maximum_allocated_bytes']/torch.cuda.get_device_properties(0).total_memory)
        random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        self.model=build_model(c);reference=json.loads((self.bundle/'initial_reference.json').read_text())
        self.initial=streaming_digest(self.model.state_dict())
        if self.initial!=reference['model_sha256'] or sum(p.numel() for p in self.model.parameters())!=109396992:
            raise RuntimeError('Full CPU initial reference differs')
        self.snapshot={name:t.detach().clone() for name,t in self.model.state_dict().items()}
        self.snapshot_owner.watch_tree(self.snapshot,'initial_cpu_snapshot')
        self.model=self.model.cuda();self.initial_rng=self.rng_state()
        import datasets
        self.data=datasets.load_from_disk(str(CACHE),keep_in_memory=False)
        for name,dataset in self.data.items():
            if len(dataset)!=self.audit['parent_structure'][name]['rows'] or dataset._fingerprint!=self.audit['parent_structure'][name]['fingerprint']:
                raise RuntimeError('Loaded cached dataset differs')
        self.order,self.panels=data_plan(len(self.data['train']),len(self.data['validation']),c['seed'])
        if hashlib.sha256(self.order.astype('<i8').tobytes()).hexdigest()!=self.audit['training_plan_sha256']:
            raise RuntimeError('Full nonrepeating batch plan differs')
        for name,(split,ids) in self.panels.items():
            if split!=self.audit['panels'][name]['split'] or ids.tolist()!=self.audit['panels'][name]['ids']:
                raise RuntimeError('Fixed evaluation IDs differ')
        self.source_bytes=sum(p.stat().st_size for p in (self.bundle/'source').rglob('*') if p.is_file())
        backend=backend_record();write(self.bundle/'backend_before.json',backend)
        expected=json.loads((self.bundle/'backend_contract.json').read_text())
        if {k:v for k,v in backend.items() if k not in ('environment','loaded_libraries')}!=expected:
            raise RuntimeError('Declared native numerical/software backend differs')
        self.emit('started',dict(pid=os.getpid(),parameters=109396992,primary_calls=HORIZON,
            source='Native QR20, eager MATH BF16 model; no candidate learning',data_plan_sha256=self.audit['training_plan_sha256'],
            primary_targets=286440000,layers=[0,5,11],stages=list(STAGES),maximum_bank_bytes=MAX_BANK_BYTES))

    def reset(self):
        self.model.load_state_dict(self.snapshot);self.model.zero_grad(set_to_none=True);self.model.train()
        state=self.initial_rng
        torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda']);random.setstate(state['python']);np.random.set_state(state['numpy'])
        if streaming_digest(self.model.state_dict())!=self.initial:raise RuntimeError('Full initial reset differs')

    def backward(self,call):
        cpu_tokens=tokens_from_rows(self.data['train'],self.order[call-1])
        if input_sha(cpu_tokens)!=self.audit['train_batches_sha256'][call-1]:raise RuntimeError('Actual training input differs from frozen batch hash')
        tokens=cpu_tokens.cuda();self.model.zero_grad(set_to_none=True)
        with forward_context():logits=self.model(tokens);loss=shifted_loss(logits,tokens)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.c['clip_norm'],foreach=True)
        return loss.detach(),norm.detach()

    def initial_observation(self,loss,norm):
        return dict(loss=float(loss),gradient_norm=float(norm),gradients_sha256=streaming_digest([p.grad for p in self.model.parameters()]),model_sha256=streaming_digest(self.model.state_dict()))

    def preflight(self):
        observations=[]
        for _ in range(2):
            self.reset();loss,norm=self.backward(1);row=self.initial_observation(loss,norm);del loss,norm
            if row['model_sha256']!=self.initial or (observations and row!=observations[0]):raise RuntimeError('Exact initial repeated backward differs')
            observations.append(row);self.budget()
        self.preflight_record=observations[0];self.model.zero_grad(set_to_none=True)
        self.emit('preflight',dict(exact_repeated_backward=True,optimizer_calls=0,observation=self.preflight_record))

    @torch.no_grad()
    def evaluate(self,call):
        from qr_reuse_profile_core import flags
        before=self.endpoint(self.opt);old_flags=[(module,module.training) for module in self.model.modules()]
        outside=flags();rng=self.rng_state();metrics={};started=time.monotonic()
        try:
            self.model.eval()
            for name,(split,ids) in self.panels.items():
                total=0.;count=0
                for start in range(0,len(ids),8):
                    cpu=tokens_from_rows(self.data[split],ids[start:start+8])
                    if input_sha(cpu)!=self.audit['panels'][name]['batch_sha256'][start//8]:raise RuntimeError('Actual evaluation tokens differ')
                    tokens=cpu.cuda()
                    with forward_context():logits=self.model(tokens);loss=shifted_loss(logits,tokens)
                    n=len(tokens)*1023;total+=float(loss)*n;count+=n;del tokens,logits,loss,cpu
                metrics[name]=dict(ce=total/count,valid_tokens=count,blocks=len(ids))
                if not math.isfinite(metrics[name]['ce']):raise FloatingPointError('Nonfinite evaluation')
        finally:
            for module,flag in old_flags:module.training=flag
            torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda']);random.setstate(rng['python']);np.random.set_state(rng['numpy'])
            if before!=self.endpoint(self.opt) or outside!=flags() or [m.training for m,_ in old_flags]!=[f for _,f in old_flags]:
                raise RuntimeError('Evaluation source/gradient/RNG/flag restoration failed')
        row=dict(call=call,metrics=metrics,restoration_passed=True,seen_train_rows_all_presented=call>=8,seconds=time.monotonic()-started)
        self.evaluations.append(row);self.emit('evaluation',row);self.budget()

    def release_snapshot(self):
        if self.snapshot_release is None:
            self.snapshot.clear();gc.collect()
            self.snapshot_release=self.snapshot_owner.report() if self.snapshot_owner._tensors else dict(all_registered_released=True,tracked_tensors=0)
            self.snapshot_owner.close();self.emit('snapshot_release',self.snapshot_release)
            if not self.snapshot_release['all_registered_released']:raise RuntimeError('CPU initial snapshot storage retained')

    def trajectory(self,phase,calls):
        self.current_phase=phase;self.reset();self.opt=build_optimizer(self.model,self.c,'qr20')
        owner=TensorOwnership();samples=[];guard_count=0;guard_seconds=0.;layer_release=None
        router=SimpleNamespace(rng_digest=self.rng_digest,emit=self.emit,moments=[])
        qualify_actual_m(self.opt.matrix,router,'qr20',phase)
        if phase=='primary':self.observer=LayerObserver(self.model,self.emit,self.rng_digest)
        try:
            if phase=='primary':self.evaluate(0)
            for call in range(1,calls+1):
                self.budget();torch.cuda.synchronize();iteration_started=time.perf_counter()
                apply_lr_multiplier(self.opt,lr_at(call,HORIZON,875,1.,self.c['min_lr_ratio']))
                loss,norm=self.backward(call)
                if call==1 and self.initial_observation(loss,norm)!=self.preflight_record:raise RuntimeError('First actual backward differs from preflight')
                torch.cuda.synchronize();marks=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                host=time.perf_counter();marks[0].record();self.opt.step();marks[1].record();torch.cuda.synchronize()
                optimizer_ms=1000*(time.perf_counter()-host);iteration_ms=1000*(time.perf_counter()-iteration_started)
                row=dict(phase=phase,call=call,kind='cold' if call==1 else ('qr' if (call-1)%20==0 else 'ordinary'),
                    diagnostic=phase=='disposable' and call==21 or phase=='primary' and call in M_CALLS,
                    loss=float(loss),gradient_norm=float(norm),lr=self.opt.matrix.param_groups[0]['lr'],
                    optimizer_host_ms=optimizer_ms,optimizer_event_ms=marks[0].elapsed_time(marks[1]),iteration_host_ms=iteration_ms)
                if not all(math.isfinite(v) for v in row.values() if isinstance(v,float)):raise FloatingPointError('Nonfinite update')
                del loss,norm,marks;samples.append(row);self.calls+=1
                if phase=='primary':self.primary_call=call
                self.emit('update',row)
                if call==1 or (call-1)%20==0 or call==calls:
                    start=time.monotonic();guard=compact_guard(self.opt,self.model,call)
                    elapsed=time.monotonic()-start;guard_seconds+=elapsed;guard_count+=1
                    self.emit('guard',dict(phase=phase,seconds=elapsed,**guard))
                if call==25:
                    prefix=dict(loss_norm=[(r['loss'],r['gradient_norm']) for r in samples],state_sha256=self.endpoint(self.opt))
                    if phase=='disposable':self.disposable_prefix=prefix
                    else:
                        if prefix!=self.disposable_prefix:raise RuntimeError('Exact25-call full-state/scalar prefix differs')
                        self.emit('prefix_passed',dict(call=25,exact_model_optimizer_gradients_rng_and_scalars=True))
                        self.release_snapshot()
                if phase=='primary':
                    self.observer.tick(call,self.opt)
                    if call in EVALUATIONS:self.evaluate(call)
                if call%100==0:self.status()
            expected=48 if phase=='disposable' else 384
            if len(router.moments)!=expected or not all(r['passed'] and r['rng_unchanged'] for r in router.moments):raise RuntimeError('Actual-M qualification count or gate differs')
            clean=[r for r in samples if r['call']>50 and not r['diagnostic']]
            means={kind:dict(samples=len(rows),**{k:statistics.mean(r[k] for r in rows) for k in ['optimizer_host_ms','optimizer_event_ms','iteration_host_ms']})
                for kind in ('ordinary','qr') if (rows:=[r for r in clean if r['kind']==kind])}
            result=dict(phase=phase,calls=calls,steady_means=means,guard_count=guard_count,guard_seconds=guard_seconds,
                actual_m_checks=len(router.moments),worst_actual_m_relative=max(r['physical_m_relative'] for r in router.moments),first_sample=samples[0],last_sample=samples[-1])
            self.phases.append(result);self.emit('phase_complete',result)
        finally:
            if self.observer is not None:
                layer_release=self.observer.close();self.emit('layer_history_release',layer_release)
                if phase=='primary' and calls==HORIZON and self.primary_call==HORIZON:
                    if len(self.observer.observations)!=7 or layer_release['released_history_banks']!=14:raise RuntimeError('Layer observation count differs')
                    write(self.bundle/'layer_observations.json',self.observer.observations)
                self.observer=None
            owner.watch_tree(self.opt.state_dict(),'optimizer_state');owner.watch_tree([p.grad for p in self.model.parameters()],'gradients')
            self.opt.matrix.state.clear();self.opt.auxiliary.state.clear();self.model.zero_grad(set_to_none=True)
            self.opt=None;del router;gc.collect();torch.cuda.synchronize()
            release=owner.report() if owner._tensors else dict(all_registered_released=True,tracked_tensors=0)
            owner.close();self.releases.append(release);self.emit('optimizer_release',dict(phase=phase,**release))
            if not release['all_registered_released']:raise RuntimeError('Optimizer/gradient storage retained')

    def run(self):
        try:
            self.setup();self.preflight();self.trajectory('disposable',25);self.trajectory('primary',HORIZON)
            verify(self.bundle);backend=backend_record();write(self.bundle/'backend_after.json',backend)
            expected=json.loads((self.bundle/'backend_contract.json').read_text())
            if {k:v for k,v in backend.items() if k not in ('environment','loaded_libraries')}!=expected:raise RuntimeError('Backend changed during primary')
            if self.calls!=35025 or len(self.evaluations)!=7:raise RuntimeError('Final call/evaluation count differs')
        finally:
            self.release_snapshot()
            if self.model is not None:
                owner=TensorOwnership();owner.watch_tree(self.model.state_dict(),'model')
                self.model=None;gc.collect();torch.cuda.synchronize();self.model_release=owner.report();owner.close()
                self.emit('model_release',self.model_release)
                if not self.model_release['all_registered_released']:raise RuntimeError('Model storage retained')
        result=dict(status='completed',optimizer_calls=self.calls,primary_calls=self.primary_call,
            all_in_seconds=time.monotonic()-self.started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            phases=self.phases,evaluations=self.evaluations,releases=self.releases,
            snapshot_release=self.snapshot_release,model_release=self.model_release,exact_25_prefix=True,
            read_only_layer_observations=7,history_banks_released=14,
            scope='Native deeper-model no-wrap learning baseline; no candidate learning, convergence or production-speed qualification')
        write(self.bundle/'summary.json',result);self.emit('finished',{k:result[k] for k in ['status','optimizer_calls','all_in_seconds','peak_allocated_bytes']})
        self.status('completed');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);args=parser.parse_args()
    runtime=Run(args.bundle);runtime.status()
    try:runtime.run()
    except BaseException as exc:
        import traceback
        runtime.status('failed',error=repr(exc),traceback=traceback.format_exc());raise
