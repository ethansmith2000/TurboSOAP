"""One managed actual-trainer profile attempt; no precision/math changes."""
import argparse
from contextlib import nullcontext
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE),str(HERE/'cifar_support')]
import numpy as np
import torch
from torch.nn import functional as F
from cifar_matched_training import GPURuntime, write
from cadence_training_contract import build_arm, validate_counters
from cifar_optimizers import apply_lr_multiplier
from cifar100_qualification import lr_at
from execution_fingerprints import backend_record
from launch_environment_contract import verify_environment_launch
from launch_environment_wire_contract import verify_backend_launch
from matched_training_runtime import (OptimizerLease, captured_backward, check_state,
    disposable_smoke, failure_record, optimizer_extras, refresh_due, streaming_digest)
from prepare_refresh_profile import verify
from refresh_profile_contract import HORIZON, PROFILE_CALLS, RUNS
from refresh_profile_instrumentation import profiled_step


def endpoint(runtime,opt):
    return dict(model_sha256=streaming_digest(runtime.model.state_dict()),
        optimizer_sha256=streaming_digest(dict(state=opt.state_dict(),extras=dict(
            matrix=optimizer_extras(opt.matrix),auxiliary=optimizer_extras(opt.auxiliary)))),
        rng_sha256=streaming_digest(runtime.streams.capture()))


def export(prof,bundle,name):
    raw=bundle/(name+'.json');target=bundle/(name+'.json.gz')
    prof.export_chrome_trace(str(raw))
    if raw.stat().st_size >= 200*2**20:
        raise OSError('Single raw trace bound exceeded')
    with raw.open('rb') as source,gzip.open(target,'wb') as out:
        shutil.copyfileobj(source,out)
    raw.unlink() # Own compression temporary only; retained evidence is target.
    return dict(path=target.name,bytes=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest())


def preflight(runtime):
    records=[]
    for arm in ('qr20','warm20'):
        lease=OptimizerLease(runtime.model,build_arm(runtime.model,runtime.config,arm))
        try:
            with runtime.snapshot.probe_scope():
                row,loss,norm=captured_backward(runtime.model,runtime.active,
                    lambda:runtime.make_batch(0),runtime.streams,
                    clip_norm=runtime.config['clip_norm'],autocast=runtime.autocast)
                del loss,norm
                records.append(row)
                if streaming_digest(runtime.model.state_dict()) != runtime.snapshot.initial_digest:
                    raise RuntimeError('Preflight mutated initial model')
        finally:
            release=lease.release()
        if not release['passed']:
            raise RuntimeError('Preflight release failed')
    if records[0] != records[1]:
        raise RuntimeError('Exact initial QR/warm backward differs')
    runtime.emit('preflight_passed',dict(status='running',observations=records))
    return records[0]


def run(runtime,label,expected):
    arm=label.split('_')[0];profiling=label.endswith('_profile')
    runtime.current_arm=label;runtime.snapshot.reset()
    lease=OptimizerLease(runtime.model,build_arm(runtime.model,runtime.config,arm))
    result=dict(label=label,arm=arm,status='running',calls=0,timings=[],prefixes={},guards=[],traces=[])
    pending=[];loss_hash=hashlib.sha256();error=None
    runtime.emit('arm_started',dict(label=label))
    try:
        for call in range(1,HORIZON+1):
            runtime.budget();opt=lease.opt
            apply_lr_multiplier(opt,lr_at(call,runtime.schedule_steps,runtime.warmup_steps,
                1.,runtime.config['min_lr_ratio']))
            is_refresh=refresh_due(opt)
            if call==1:
                row,loss,norm=captured_backward(runtime.model,runtime.active,
                    lambda:runtime.make_batch(call-1),runtime.streams,
                    clip_norm=runtime.config['clip_norm'],autocast=runtime.autocast)
                if row!=expected:raise RuntimeError('Actual first pre-update mismatch: '+label)
                runtime.emit('actual_first_passed',dict(label=label,observation=row))
            else:
                indices,x,target=runtime.make_batch(call-1);opt.zero_grad(set_to_none=True)
                with runtime.autocast():
                    logits=runtime.active(x);loss=F.cross_entropy(logits.float(),target)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(runtime.model.parameters(),runtime.config['clip_norm'],foreach=True)
                loss=loss.detach();norm=norm.detach();del indices,x,target,logits
            # Serialized optimizer diagnostic. Host/event interval excludes model work,
            # includes optimizer launch gaps; not a throughput/wall-learning measurement.
            runtime.synchronize()
            selected=profiling and call in PROFILE_CALLS
            prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) if selected else None
            events=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
            started=time.perf_counter()
            with prof if selected else nullcontext():
                events[0].record()
                if selected:profiled_step(opt)
                else:opt.step()
                events[1].record();runtime.synchronize()
            host_ms=1000*(time.perf_counter()-started)
            event_ms=events[0].elapsed_time(events[1])
            result['timings'].append(dict(call=call,age=call-1,profiled=selected,
                kind='cold' if call==1 else ('qr' if arm=='qr20' or (call-1)%200==0 else 'warm')
                    if (call-1)%20==0 else 'ordinary',
                optimizer_event_ms=event_ms,optimizer_host_ms=host_ms))
            del events
            if selected:
                result['traces'].append(dict(call=call,**export(prof,runtime.bundle,label+'_call'+str(call))))
                del prof
                runtime.emit('trace_saved',result['traces'][-1]);runtime.budget(resources=True)
            pending.append((loss,norm));del loss,norm
            result['calls']=call
            if is_refresh:
                guard=check_state(runtime.model,opt)
                if guard['factor_count']!=8 or guard['skipped_factors']!=0:
                    raise RuntimeError('Two-sided factor coverage changed')
                guard.update(call=call,counters=validate_counters(opt,call))
                result['guards'].append(guard)
            if call%50==0 or call==HORIZON:
                losses=torch.stack([p[0] for p in pending]).cpu().numpy()
                norms=torch.stack([p[1] for p in pending]).cpu().numpy()
                if not np.isfinite(losses).all() or not np.isfinite(norms).all():
                    raise FloatingPointError('Nonfinite loss or gradient norm')
                loss_hash.update(losses.astype(np.float32).tobytes());pending.clear()
                runtime.emit('history',dict(label=label,call=call,mean_loss=float(losses.mean())))
                del losses,norms;runtime.budget(resources=True)
            if call in runtime.protocol['gates']['prefix_hashes']:
                result['prefixes'][str(call)]=endpoint(runtime,opt)
                runtime.emit('prefix',dict(label=label,call=call,hashes=result['prefixes'][str(call)]))
            del opt
        result['endpoint']=result['prefixes'][str(HORIZON)]
        result['loss_sequence_sha256']=loss_hash.hexdigest()
        result['final_counters']=validate_counters(lease.opt,HORIZON)
        result['status']='completed'
    except BaseException as exc:
        error=failure_record(exc);result['status']='failed';result['failure']=error
    finally:
        pending.clear()
        # No persistent bound-method profiler wrappers survive a call.
        if 'opt' in locals():del opt
        result['release']=lease.release()
        runtime.emit('arm_result',result)
    if error is not None or not result['release']['passed']:
        raise RuntimeError('Diagnostic arm failed: '+label)
    return result


def main():
    started=time.monotonic();p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    b=p.parse_args().bundle.resolve();runtime=None
    if HERE.parent!=b/'source':raise RuntimeError('Frozen execution only')
    if (b/'execution_status.json').exists() or (b/'events.jsonl').exists():raise RuntimeError('One attempt only')
    try:
        verify(b);verify_environment_launch(b)
        write(b/'execution_status.json',dict(status='initializing',attempt=1))
        if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one lifetime-claimed device')
        runtime=GPURuntime(b,started=started);runtime.initialize();runtime.before_smoke()
        disposable_smoke(runtime,25);runtime.after_smoke();expected=preflight(runtime)
        arms=[]
        for label in RUNS:
            arms.append(run(runtime,label,expected))
            if label.endswith('_profile'):
                plain=arms[-2];profile=arms[-1]
                for key in ('endpoint','prefixes','loss_sequence_sha256','final_counters'):
                    if plain[key]!=profile[key]:raise RuntimeError('Plain/profile exact replay mismatch: '+key)
                runtime.emit('observer_invariance_passed',dict(arm=profile['arm'],full_endpoint=True,
                    full_loss_sequence=True,prefix_calls=runtime.protocol['gates']['prefix_hashes']))
        runtime.after_measured();peak=torch.cuda.max_memory_allocated()
        purpose=runtime.snapshot.budget_bytes;runtime.model.zero_grad(set_to_none=True)
        runtime.snapshot.close();masks=runtime.close()
        if not masks['passed']:raise RuntimeError('Own thread masks not restored')
        verify(b)
        summary=dict(status='completed',scope='Profile cost attribution; no learning/precision/speed qualification',
            arms=arms,optimizer_calls=sum(a['calls'] for a in arms)+25,
            all_in_seconds=time.monotonic()-started,peak_allocated_bytes=peak,
            snapshot_purpose_bytes=purpose,masks=masks,observer_invariance=True,
            precision_changed=False,new_learning_comparison=False)
        write(b/'summary.json',summary);runtime.emit('finished',dict(status='completed'))
    except BaseException as exc:
        write(b/'execution_status.json',dict(status='failed',error=repr(exc),all_in_seconds=time.monotonic()-started))
        if runtime is not None:
            runtime.retain_code('failure')
            if runtime.snapshot is not None and not runtime.snapshot.closed:
                runtime.model.zero_grad(set_to_none=True);runtime.snapshot.close()
            write(b/'failure_cleanup.json',runtime.close())
        raise


if __name__=='__main__':main()
