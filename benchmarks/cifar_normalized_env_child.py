"""One fresh-process width1536 execution context, no retained tensors or graph hooks."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import weakref

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE),str(HERE/'cifar_support')]
import numpy as np
import torch
from torch.nn import functional as F
from cifar_optimizer_benchmark import (support,StrongAugmentation,json_write,rng_state,
                                      restore_rng,state_digest)
from cifar_optimizers import build_optimizer,apply_lr_multiplier,counters,check_basis_state
from compact_dispatch_screen import CompactDispatchHostTiming
from saved_operand_replay import run_saved_operand_replay
from normalized_input_replay import run_normalized_replay
from normalized_diagnostic_contract import verify_launch
from launch_environment_contract import verify_environment_launch, verify_backend_launch
from initial_state_replay import run_initial_replay
from owned_matrix_hooks import OwnedMatrixHooks
from execution_fingerprints import tensor_record,gradient_records,basis_records,loaded_code,backend_record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--context',required=True);a=parser.parse_args();b=a.bundle.resolve()
    assert HERE.parent==b/'source' and os.environ.get('GPU_CLAIM_INDICES')
    # ENVIRONMENT_PRELAUNCH_START
    verify_environment_launch(b)
    # ENVIRONMENT_PRELAUNCH_END
    # NORMALIZED_LAUNCH_START
    verify_launch(b)
    # NORMALIZED_LAUNCH_END
    assert torch.cuda.device_count()==1
    m=json.loads((b/'manifest.json').read_text())
    for name,digest in m['source_sha256'].items():assert support.digest(b/'source'/name)==digest,name
    for name,key in [('config.json','config_sha256'),('split.json','split_sha256'),
                     ('data_prepared.json','base_prepared_sha256'),('PROTOCOL.md','protocol_sha256')]:
        assert support.digest(b/name)==m[key],name
    c=json.loads((b/'config.json').read_text());d=c['execution_diagnostic']
    context=next(x for x in d['contexts'] if x['label']==a.context)
    out=b/'contexts'/a.context;out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();original_affinities={};opt=lease=None
    report=dict(status='running',context=context,pid=os.getpid(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
                measured=[],smoke_fingerprints=[],timing_chunks=[])
    def save():json_write(out/'result.json',report)
    def budget():
        assert time.monotonic()-started<d['per_process_seconds'],'Child wall bound exceeded'
        assert torch.cuda.max_memory_allocated()<d['maximum_peak_allocated_bytes'],'Allocated memory bound exceeded'
        assert sum(p.stat().st_size for p in b.rglob('*') if p.is_file())<m['evidence_budget_bytes']
    try:
        torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        random.seed(c['seed']);torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        raw,labels,_=support.load_training_data(c)
        images=torch.from_numpy(raw).permute(0,3,1,2).contiguous().cuda()
        labels=torch.tensor(labels,device='cuda');del raw
        split=json.loads((b/'split.json').read_text());prepared=json.loads((b/'data_prepared.json').read_text())
        # Preserve the common split-index allocations from the prior trainer.
        ids={key:torch.tensor(value,device='cuda') for key,value in split.items()}
        mean=torch.tensor(prepared['mean'],device='cuda')[None,:,None,None]
        std=torch.tensor(prepared['std'],device='cuda')[None,:,None,None]
        model=support.build_model(c).cuda();initial=copy.deepcopy(model.state_dict())
        initial_hash=state_digest(initial);report['initial_model_sha256']=initial_hash
        active=torch.compile(model,mode='default',fullgraph=True,dynamic=False)
        strong=StrongAugmentation(c,mean,std);crop=torch.Generator(device='cuda').manual_seed(c['seed']+1001)
        initial_rng=rng_state(crop,strong)
        per_epoch=len(split['train'])//c['batch_size'];schedule_steps=per_epoch*c['epochs']
        order_rng=np.random.default_rng(c['seed']+2001);batches=[]
        for _ in range(math.ceil(context['allocation_horizon']/per_epoch)):
            order=order_rng.permutation(np.asarray(split['train']))
            batches.extend(order[:per_epoch*c['batch_size']].reshape(per_epoch,c['batch_size']))
        device_batches=[torch.tensor(x,device='cuda') for x in batches]
        report['batch_plan']=dict(count=len(batches),bytes=sum(x.numel()*x.element_size() for x in device_batches),
                                 first25_ids_sha256=hashlib.sha256(np.asarray(batches[:25]).tobytes()).hexdigest())
        report['backend_before_smoke']=backend_record();save()
        # ENVIRONMENT_BEFORE_SMOKE_START
        report['environment_before_smoke_check']=verify_backend_launch(b, report['backend_before_smoke'], 'backend_before_smoke');save()
        # ENVIRONMENT_BEFORE_SMOKE_END
        host=CompactDispatchHostTiming()
        def step(index,optimizer,capture):
            ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)];ev[0].record();host.begin()
            with host.scope('augmentation'):
                batch=device_batches[index]
                offsets=torch.randint(0,9,(c['batch_size'],2),device='cuda',generator=crop)
                flips=torch.rand(c['batch_size'],device='cuda',generator=crop)<.5
                y=labels[batch];x,target,_=strong(images[batch],y,offsets,flips)
            ev[1].record();host.mark()
            with host.scope('model_backward'):
                multiplier=support.lr_at(index+1,schedule_steps,per_epoch*c['warmup_epochs'],1.,c['min_lr_ratio'])
                apply_lr_multiplier(optimizer,multiplier);optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=active(x);loss=F.cross_entropy(logits.float(),target)
                loss.backward()
                row=None
                if capture:
                    # RAM-only caller diagnostics, no compiled-model hooks or math changes.
                    row=dict(step=index+1,input=tensor_record(x),target=tensor_record(target),
                        ids_sha256=hashlib.sha256(batches[index].tobytes()).hexdigest(),
                        rng_sha256=state_digest(rng_state(crop,strong)),output=tensor_record(logits),
                        loss=float(loss.detach()),grad_before_clip=gradient_records(model))
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_norm'],foreach=True)
                if capture:row.update(grad_after_clip=gradient_records(model),gradient_norm=float(norm))
            ev[2].record();host.mark()
            with host.scope('optimizer'):optimizer.step()
            ev[3].record();host.finish()
            if capture:
                row.update(model_sha256=state_digest(model.state_dict()),
                    optimizer_sha256=state_digest(optimizer.state_dict()),basis=basis_records(optimizer),
                    counters=counters(optimizer))
            return row,loss.detach(),norm.detach(),x,y,target,ev
        model.train();smoke=build_optimizer(model,c,'qr10')
        for index in range(d['smoke_steps']):
            row,loss,norm,x,y,target,events=step(index,smoke,index+1 in [1,25])
            if row:report['smoke_fingerprints'].append(row);save()
            budget()
        torch.cuda.synchronize();model.eval()
        report['smoke']=dict(steps=d['smoke_steps'],training_parity=support.parity_check(model,active,x,y),
            evaluation_parity=support.evaluation_parity_check(model,x,y))
        report['code_after_smoke']=loaded_code()
        ref=weakref.ref(smoke.matrix);del smoke
        assert ref() is None
        model.load_state_dict(initial);model.train();restore_rng(initial_rng,crop,strong)
        assert state_digest(model.state_dict())==initial_hash
        workers=[int(p.name) for p in Path('/proc/self/task').iterdir()
                 if (p/'comm').read_text().strip()=='pt_autograd_0']
        assert len(workers)==1,workers
        for tid in [os.getpid(),workers[0]]:
            original_affinities[tid]=os.sched_getaffinity(tid)
            assert 116 in original_affinities[tid]
            os.sched_setaffinity(tid,{116})
        report['placement']=dict(main_cpu=116,autograd_cpu=116)
        cfg={**c,'transport_precision':'highest'};opt=build_optimizer(model,cfg,'qr20')
        lease=OwnedMatrixHooks(opt.matrix)
        report['allocated_before_measured_bytes']=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats();host.reset();save()
        events=[];losses=[];norms=[]
        for index in range(d['measured_steps']):
            row,loss,norm,x,y,target,ev=step(index,opt,index+1 in d['fingerprint_steps'])
            events.append(ev);losses.append(loss);norms.append(norm)
            if row:
                report['measured'].append(row)
                if index+1 in [1,21]:
                    report.setdefault('geometry',[]).append(dict(step=index+1,**check_basis_state(opt,.05)))
                save();print('FINGERPRINT '+a.context+' '+str(index+1),flush=True)
            if (index+1)%context['timing_chunk']==0 or index+1==d['measured_steps']:
                values=torch.stack([torch.stack(losses).mean(),torch.stack(norms).max()]).cpu().tolist()
                torch.cuda.synchronize()
                assert all(math.isfinite(v) for v in values),'Nonfinite training'
                samples=[[v[0].elapsed_time(v[1]),v[1].elapsed_time(v[2]),v[2].elapsed_time(v[3])]
                         for v in events]
                report['timing_chunks'].append(dict(step=index+1,samples=samples,loss=values[0],
                    gradient_norm=values[1],host=host.flush()))
                events=[];losses=[];norms=[];save()
            budget()
        report['code_after_measured']=loaded_code();report['backend_after_measured']=backend_record()
        # ENVIRONMENT_AFTER_MEASURED_START
        report['environment_after_measured_check']=verify_backend_launch(b, report['backend_after_measured'], 'backend_after_measured');save()
        # ENVIRONMENT_AFTER_MEASURED_END
        # Two model-only calls at fixed final weights and held input. No optimizer step.
        # Profile is separately charged and leaves parameter/optimizer/RNG state exact.
        def invariants():return dict(model=state_digest(model.state_dict()),optimizer=state_digest(opt.state_dict()),
                                     rng=state_digest(rng_state(crop,strong)))
        before=invariants();profile_started=time.monotonic()
        def backward_only():
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=active(x);value=F.cross_entropy(logits.float(),target)
            value.backward();torch.cuda.synchronize()
            return dict(output=tensor_record(logits),loss=float(value.detach()),grad=gradient_records(model))
        plain=backward_only()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            # Keep hashes/CPU diagnostics outside the profiler window.
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=active(x);value=F.cross_entropy(logits.float(),target)
            value.backward();torch.cuda.synchronize()
        profiled=dict(output=tensor_record(logits),loss=float(value.detach()),grad=gradient_records(model))
        assert profiled==plain,'Plain/profile model execution differs'
        after=invariants();assert before==after,'Model-only profile mutated parameter/optimizer/RNG'
        kernels={}
        for event in prof.events():
            if str(event.device_type)=='DeviceType.CUDA':
                kernels[event.name]=kernels.get(event.name,0)+1
        assert kernels,'No runtime GPU kernel provenance'
        report['runtime_profile']=dict(scope='final-state model-only forward/backward; no optimizer step',
            kernel_names_counts=kernels,plain_profile_exact=True,invariants_before=before,invariants_after=after,
            seconds=time.monotonic()-profile_started)
        report['code_after_profile']=loaded_code();del prof,plain,profiled
        # INITIAL_STATE_REPLAY_START
        def first_batch():
            batch=device_batches[0]
            offsets=torch.randint(0,9,(c['batch_size'],2),device='cuda',generator=crop)
            flips=torch.rand(c['batch_size'],device='cuda',generator=crop)<.5
            first_x,first_target,_=strong(images[batch],labels[batch],offsets,flips)
            return first_x,first_target
        report['initial_state_replay']=run_initial_replay(model=model,active=active,
            initial=initial,initial_rng=initial_rng,get_rng=lambda:rng_state(crop,strong),
            restore_rng=lambda saved:restore_rng(saved,crop,strong),make_batch=first_batch,
            first_record=report['measured'][0],invariants=invariants)
        report['code_after_initial_replay']=loaded_code();save();budget()
        # INITIAL_STATE_REPLAY_END
        # SAVED_OPERAND_REPLAY_START
        def saved_progress(value):
            report['saved_operand_replay']=value;save();budget()
        report['saved_operand_replay']=run_saved_operand_replay(model=model,active=active,
            initial=initial,initial_rng=initial_rng,get_rng=lambda:rng_state(crop,strong),
            restore_rng=lambda saved:restore_rng(saved,crop,strong),make_batch=first_batch,
            first_record=report['measured'][0],warm_record=report['initial_state_replay']['plain'],
            invariants=invariants,progress=saved_progress)
        report['code_after_saved_operand_replay']=loaded_code();save();budget()
        # SAVED_OPERAND_REPLAY_END
        # NORMALIZED_REPLAY_START
        def normalized_progress(value):
            report['normalized_input_replay']=value;save();budget()
        report['normalized_input_replay']=run_normalized_replay(model=model,active=active,
            initial=initial,initial_rng=initial_rng,get_rng=lambda:rng_state(crop,strong),
            restore_rng=lambda saved:restore_rng(saved,crop,strong),make_batch=first_batch,
            first_record=report['measured'][0],warm_record=report['initial_state_replay']['plain'],
            invariants=invariants,progress=normalized_progress,previous=report['saved_operand_replay'],
            ram_cap_bytes=d['normalized_input']['ram_cap_bytes'])
        report['code_after_normalized_replay']=loaded_code();save();budget()
        # NORMALIZED_REPLAY_END
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        matrix_ref=weakref.ref(opt.matrix);opt_ref=weakref.ref(opt)
        lease.close();del lease,opt;lease=opt=None
        report['owned_optimizer_released']=matrix_ref() is None and opt_ref() is None
        assert report['owned_optimizer_released'];budget()
        report.update(status='completed',all_in_seconds=time.monotonic()-started);save()
    except BaseException as exc:
        report.update(status='failed',failure=repr(exc),all_in_seconds=time.monotonic()-started);save();raise
    finally:
        if lease is not None:lease.close()
        for tid,mask in original_affinities.items():
            try:os.sched_setaffinity(tid,mask)
            except ProcessLookupError:pass
        # NORMALIZED_FINALIZE_START
        report['own_affinity_restored']=all(os.sched_getaffinity(tid)==mask for tid,mask in original_affinities.items())
        # NORMALIZED_FINALIZE_END
        report['affinity_restored']={str(k):sorted(v) for k,v in original_affinities.items()};save()


if __name__=='__main__':main()
