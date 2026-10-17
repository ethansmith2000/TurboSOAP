"""One frozen matched-learning attempt; fixed observations and exact evaluation restoration."""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE),str(HERE/'cifar_support')]
import torch
from torch.nn import functional as F
from cifar_matched_training import GPURuntime, write
from cifar_optimizers import apply_lr_multiplier
from cifar100_qualification import lr_at
from cadence_training_contract import validate_counters
from execution_fingerprints import backend_record
from native_damped_kernel import PolicyRouter
from native_damped_diagnostic import build_arm
from native_damped_contract import ARMS, POLICIES
from native_damped_contract import HORIZON, WARMUP
from cifar_hybrid_width import larger_snapshot_cap
from matched_training_runtime import (OptimizerLease, captured_backward, check_state,
    disposable_smoke, failure_record, optimizer_extras, refresh_due, streaming_digest)
from prepare_native_damped import verify, verify_launch, verify_backend
from tensor_ownership import TensorOwnership


def compiler_stats():
    from torch._dynamo.utils import counters
    return {k:dict(v) for k,v in counters.items() if v}


class Runtime(GPURuntime):
    def initialize(self):
        with larger_snapshot_cap():super().initialize()
        if self.snapshot.budget_bytes!=414257552 or self.snapshot.snapshot_bytes!=203186576:raise RuntimeError("Native2048 snapshot resource delta")
        if self.snapshot.initial_digest!=json.loads((self.bundle/'initial_reference.json').read_text())['model_sha256']:
            raise RuntimeError('Actual initial model differs from frozen CPU learning seed reference')
        # Common prospective backend delta for EVERY arm, not a BF16-only confound.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        self.kernel=PolicyRouter(self.emit,lambda:streaming_digest(self.streams.capture()))

    def before_smoke(self):
        record=backend_record();verify_backend(self.bundle,record)
        self.emit('backend_before_smoke',dict(record=record,passed=True))

    def after_measured(self):
        record=backend_record();verify_backend(self.bundle,record)
        self.emit('backend_after_measured',dict(record=record,passed=True))
        self.retain_code('after_measured');self.budget(resources=True)

    def budget(self, resources=False):
        super().budget(resources=False)
        if resources:
            cache=Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])
            size=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file()) if cache.exists() else 0
            if size>=self.protocol['bounds']['maximum_compiler_cache_bytes']:
                raise OSError('Declared isolated compiler cache bound exceeded')
            if torch.cuda.max_memory_allocated()>=self.protocol['bounds']['maximum_peak_allocated_bytes']:
                raise MemoryError('Declared allocated GPU memory bound exceeded')
            evidence=sum(p.stat().st_size for p in self.bundle.rglob('*') if p.is_file() and cache not in p.parents)
            if evidence>=self.protocol['bounds']['maximum_evidence_bytes']:
                raise OSError('Declared compact evidence bound exceeded')

    def optimizer_events(self):
        return [torch.cuda.Event(enable_timing=True) for _ in range(2)]


def endpoint(runtime,opt):
    return dict(model_sha256=streaming_digest(runtime.model.state_dict()),
        optimizer_sha256=streaming_digest(dict(state=opt.state_dict(),extras=dict(
            matrix=optimizer_extras(opt.matrix),auxiliary=optimizer_extras(opt.auxiliary)))),
        rng_sha256=streaming_digest(runtime.streams.capture()))


def evaluate_transaction(runtime,opt):
    """Evaluate eagerly under eval flags and restore all branch state, including failures."""
    started=time.monotonic()
    modules=list(runtime.model.modules());flags=[m.training for m in modules]
    rng=runtime.streams.capture();before=endpoint(runtime,opt)
    gradient_before=streaming_digest([p.grad for p in runtime.model.parameters()])
    stats=compiler_stats()
    try:
        runtime.model.eval()
        metrics=runtime.evaluate()
    finally:
        runtime.streams.restore(rng)
        for module,flag in zip(modules,flags):module.training=flag
    after=endpoint(runtime,opt)
    if after!=before or streaming_digest([p.grad for p in runtime.model.parameters()])!=gradient_before:
        raise RuntimeError('Evaluation mutated model/optimizer/gradient/RNG')
    if [m.training for m in modules]!=flags or compiler_stats()!=stats:
        raise RuntimeError('Evaluation flags/compiler restoration failed')
    for name,row in metrics.items():
        if set(row)!={'ce','accuracy','examples'} or not math.isfinite(row['ce']) or row['ce']<0 or not 0<=row['accuracy']<=1 or row['examples']!=len(runtime.ids[name]):
            raise RuntimeError('Invalid fixed evaluation metrics: '+name)
    runtime.budget(resources=True)
    return dict(metrics=metrics,restoration_passed=True,before=before,after=after,
        gradients_sha256=gradient_before,seconds=time.monotonic()-started)


def preflight(runtime):
    records=[]
    for arm in ARMS:
        lease=OptimizerLease(runtime.model,build_arm(runtime.model,runtime.config,arm,runtime.kernel))
        try:
            with runtime.snapshot.probe_scope():
                row,loss,norm=captured_backward(runtime.model,runtime.active,
                    lambda:runtime.make_batch(0),runtime.streams,
                    clip_norm=runtime.config['clip_norm'],autocast=runtime.autocast)
                del loss,norm
                if streaming_digest(runtime.model.state_dict())!=runtime.snapshot.initial_digest:
                    raise RuntimeError('Initial preflight mutated model')
                if records and row!=records[0]:
                    raise RuntimeError('Exact initial all-arm backward mismatch: '+arm)
                records.append(row)
        finally:
            release=lease.release()
            runtime.emit('preflight_release',dict(arm=arm,**release))
        if not release['passed']:
            raise RuntimeError('Preflight registered release failed')
    runtime.emit('preflight_passed',dict(status='running',observations=records))
    return records[0]


def coverage_guard(runtime,opt):
    guard=check_state(runtime.model,opt,maximum_gram_error=runtime.protocol['gates']['state_gram_limit'])
    sizes=Counter(str(q.shape[0]) for state in opt.matrix.state.values() for q in state.get('Q',[]) if torch.is_tensor(q))
    c=runtime.protocol['coverage']
    if len(opt.matrix.state)!=c['matrix_parameters'] or guard['factor_count']!=c['two_sided_factors'] or guard['skipped_factors'] or dict(sizes)!=c['dimensions_per_refresh']:
        raise RuntimeError('Actual two-sided factor coverage differs: '+str(sizes))
    guard['dimensions']=dict(sizes)
    return guard


def trajectory(runtime,arm,phase,steps,expected):
    measured=phase=='primary';runtime.current_arm=arm+'_'+phase;runtime.snapshot.reset()
    if streaming_digest(runtime.streams.capture())!=streaming_digest(runtime.snapshot.rng):
        raise RuntimeError('Initial RNG not restored')
    lease=OptimizerLease(runtime.model,build_arm(runtime.model,runtime.config,arm,runtime.kernel))
    result=dict(arm=arm,phase=phase,status='running',calls=0,timings=[],guards=[],prefixes={},sampled_batches={},evaluations=[])
    started=time.monotonic();error=None;initial_stats=compiler_stats();initial_kernel_calls=sum(runtime.kernel.calls.values())
    try:
        for call in range(1,steps+1):
            runtime.budget();opt=lease.opt
            apply_lr_multiplier(opt,lr_at(call,runtime.schedule_steps,runtime.warmup_steps,1.,runtime.config['min_lr_ratio']))
            is_refresh=refresh_due(opt)
            if call==1:
                row,loss,norm=captured_backward(runtime.model,runtime.active,
                    lambda:runtime.make_batch(0),runtime.streams,
                    clip_norm=runtime.config['clip_norm'],autocast=runtime.autocast)
                if row!=expected:raise RuntimeError('Actual first pre-update mismatch: '+runtime.current_arm)
                result['sampled_batches']['1']={k:row[k] for k in ('indices','input','target','rng_after_augmentation')}
                runtime.emit('actual_first_passed',dict(arm=arm,phase=phase,observation=row))
            else:
                indices,x,target=runtime.make_batch(call-1);opt.zero_grad(set_to_none=True)
                if measured and call in runtime.protocol['sampled_batch_calls']:
                    result['sampled_batches'][str(call)]=dict(indices=streaming_digest(indices),
                        input=streaming_digest(x),target=streaming_digest(target),
                        rng_after_augmentation=streaming_digest(runtime.streams.capture()))
                with runtime.autocast():
                    logits=runtime.active(x);loss=F.cross_entropy(logits.float(),target)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(runtime.model.parameters(),runtime.config['clip_norm'],foreach=True)
                loss=loss.detach();norm=norm.detach();del indices,x,target,logits
            diagnostic=measured and call in runtime.protocol['probe_calls']
            runtime.kernel.probe=diagnostic;probe_before=len(runtime.kernel.probes)
            runtime.synchronize();marks=runtime.optimizer_events();host_started=time.perf_counter()
            marks[0].record();opt.step();marks[1].record();runtime.synchronize()
            host_ms=1000*(time.perf_counter()-host_started);device_ms=marks[0].elapsed_time(marks[1]);del marks
            runtime.kernel.probe=False
            if diagnostic:
                rows=runtime.kernel.probes[probe_before:]
                if len(rows)!=(runtime.protocol['coverage']['two_sided_factors'] if arm in POLICIES else 0) or not all(r['passed'] for r in rows):
                    raise RuntimeError('Diagnostic factor coverage failure')
                runtime.emit('actual_state_probes',dict(call=call,rows=rows))
            scalar_loss=float(loss);scalar_norm=float(norm);del loss,norm
            if not all(math.isfinite(v) for v in (scalar_loss,scalar_norm)):
                raise FloatingPointError('Nonfinite actual loss/gradient norm')
            kind='cold' if call==1 else ('qr' if arm=='qr20' or (call-1)%200==0 else 'warm') if is_refresh else 'ordinary'
            result['timings'].append(dict(call=call,kind=kind,diagnostic=diagnostic,
                optimizer_host_ms=host_ms,optimizer_event_ms=device_ms,loss=scalar_loss,gradient_norm=scalar_norm))
            result['calls']=call
            if is_refresh:
                guard=coverage_guard(runtime,opt);guard.update(call=call,counters=validate_counters(opt,call))
                result['guards'].append(guard)
                if compiler_stats().get('graph_break'):
                    raise RuntimeError('Compiled model/optimizer graph break')
                if measured and compiler_stats()!=initial_stats:
                    raise RuntimeError('Steady compilation counters changed after disposable warmup')
            if measured and call in runtime.protocol['prefix_hashes']:
                result['prefixes'][str(call)]=endpoint(runtime,opt)
            if measured and call in runtime.protocol['evaluation_calls']:
                observation=evaluate_transaction(runtime,opt);observation['call']=call
                result['evaluations'].append(observation)
                runtime.emit('validation',dict(arm=arm,**observation))
            if call%250==0 or call==steps:
                runtime.emit('progress',dict(arm=arm,phase=phase,call=call,loss=scalar_loss))
                runtime.budget(resources=True)
            del opt
        result['final_counters']=validate_counters(lease.opt,steps)
        if measured:result['endpoint']=endpoint(runtime,lease.opt)
        result.update(status='completed',phase_seconds=time.monotonic()-started,
            compiler_before=initial_stats,compiler_after=compiler_stats(),
            hybrid_factor_calls=sum(runtime.kernel.calls.values())-initial_kernel_calls)
    except BaseException as exc:
        # Exception formatting inspects frame locals; drop the borrowed optimizer
        # before it can become a lingering alias during the registered-release check.
        opt=None
        error=failure_record(exc);result.update(status='failed',failure=error)
    finally:
        runtime.kernel.probe=False
        opt=None
        result['release']=lease.release();runtime.emit('trajectory_result',result)
        write(runtime.bundle/(arm+'_'+phase+'.json'),result)
        runtime.snapshot.reset()
    if error is not None or not result['release']['passed']:
        raise RuntimeError('Trajectory failed: '+runtime.current_arm)
    return result


def run_pilot(runtime):
    owner=TensorOwnership();owner.watch_tree(runtime.snapshot.weights,'initial.weights');owner.watch_tree(runtime.snapshot.rng,'initial.rng')
    try:
        runtime.before_smoke();disposable_smoke(runtime,runtime.protocol['disposable_smoke_calls']);runtime.after_smoke()
        expected=preflight(runtime)
        disposable=[trajectory(runtime,a,'disposable',runtime.protocol['disposable_calls_per_arm'],expected) for a in runtime.protocol['execution_order']]
        steady_before=compiler_stats()
        primary=[]
        for arm in runtime.protocol['execution_order']:
            row=trajectory(runtime,arm,'primary',runtime.protocol['steps_per_arm'],expected)
            if primary and (row['sampled_batches']!=primary[0]['sampled_batches'] or
                row['endpoint']['rng_sha256']!=primary[0]['endpoint']['rng_sha256']):
                raise RuntimeError('Paired batch/augmentation/global RNG mismatch')
            prior=next(r for r in disposable if r['arm']==arm)
            if [(r['loss'],r['gradient_norm']) for r in row['timings'][:prior['calls']]] != [(r['loss'],r['gradient_norm']) for r in prior['timings']]:
                raise RuntimeError('Disposable/primary scalar prefix mismatch: '+arm)
            row['exact_disposable_scalar_prefix']=True
            primary.append(row)
        if compiler_stats()!=steady_before:raise RuntimeError('Steady compiler counts changed')
        if dict(runtime.kernel.calls)!=runtime.protocol['expected']['hybrid_calls_by_dimension']:
            raise RuntimeError('Unexpected hybrid factor calls')
        if len(runtime.kernel.probes)!=runtime.protocol['expected']['diagnostic_reference_calls']:
            raise RuntimeError('Unexpected actual-state reference calls')
        if len(runtime.kernel.moment_probes)!=runtime.protocol['expected']['full_moment_transactions']:raise RuntimeError('Full moment transaction count differs')
        if {a:dict(k.calls) for a,k in runtime.kernel.kernels.items()}!=runtime.protocol['expected']['policy_calls_by_dimension']:raise RuntimeError('Per-policy kernel count differs')
        if runtime.kernel.cf_calls!=runtime.protocol['expected']['counterfactual_basis_calls']:raise RuntimeError('Native counterfactual basis call count differs')
        runtime.after_measured()
        result=dict(status='completed',disposable=disposable,primary=primary,
            optimizer_calls=runtime.protocol['disposable_smoke_calls']+sum(r['calls'] for r in disposable+primary),
            hybrid_calls=dict(runtime.kernel.calls),actual_state_probes=runtime.kernel.probes,
            counterfactual_basis_calls=runtime.kernel.cf_calls,full_moment_probes=runtime.kernel.moment_probes,policy_calls={a:dict(k.calls) for a,k in runtime.kernel.kernels.items()},
            compiler_final=compiler_stats(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            snapshot_purpose_bytes=runtime.snapshot.budget_bytes,
            qualified_learning=False,paired_fixed_observations=True,qualified_training_timing=False,no_candidate_promoted=True)
    finally:
        runtime.model.zero_grad(set_to_none=True);runtime.snapshot.close()
        release=owner.report();owner.close();masks=runtime.close()
        write(runtime.bundle/'final_cleanup.json',dict(snapshot_release=release,masks=masks))
    if not release['all_registered_released'] or not masks['passed']:
        raise RuntimeError('Initial snapshot/placement release failed')
    result.update(snapshot_release=release,masks=masks)
    return result


def main():
    started=time.monotonic();p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    b=p.parse_args().bundle.resolve();runtime=None
    if HERE.parent!=b/'source':raise RuntimeError('Frozen entry only')
    if (b/'execution_status.json').exists() or (b/'events.jsonl').exists():raise RuntimeError('One attempt only')
    try:
        verify(b);verify_launch(b)
        write(b/'execution_status.json',dict(status='initializing',attempt=1,pid=os.getpid()))
        if torch.cuda.device_count()!=1:raise RuntimeError('One lifetime-claimed GPU required')
        runtime=Runtime(b,started=started);runtime.initialize()
        result=run_pilot(runtime);result['all_in_seconds']=time.monotonic()-started
        verify(b);write(b/'summary.json',result);runtime.emit('finished',dict(status='completed'))
    except BaseException as exc:
        write(b/'execution_status.json',dict(status='failed',error=repr(exc),all_in_seconds=time.monotonic()-started))
        if runtime is not None:
            if runtime.snapshot is not None and not runtime.snapshot.closed:
                runtime.model.zero_grad(set_to_none=True);runtime.snapshot.close()
            write(b/'failure_cleanup.json',runtime.close())
        raise


if __name__=='__main__':main()
