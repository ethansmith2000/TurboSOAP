"""Fresh420: original primary math, owned wide probes, exact QR stream and later replays."""
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
from movement_retraction import configure_matrix
from wide_probe import WideTransportProbe
from fresh420_contract import canonical,check_record
from cifar_optimizers import screen_arm_config
from saved_operand_replay import run_saved_operand_replay
from initial_state_replay import run_initial_replay
from owned_matrix_hooks import OwnedMatrixHooks
from execution_fingerprints import tensor_record,gradient_records,basis_records,loaded_code,backend_record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--context',required=True);a=parser.parse_args();b=a.bundle.resolve()
    assert HERE.parent==b/'source' and os.environ.get('GPU_CLAIM_INDICES')
    assert torch.cuda.device_count()==1
    m=json.loads((b/'manifest.json').read_text());assert os.environ['CUDA_VISIBLE_DEVICES']==m['device']['uuid']
    for name,digest in m['source_sha256'].items():assert support.digest(b/'source'/name)==digest,name
    for name,key in [('config.json','config_sha256'),('split.json','split_sha256'),
                     ('data_prepared.json','base_prepared_sha256'),('PROTOCOL.md','protocol_sha256')]:
        assert support.digest(b/name)==m[key],name
    c=json.loads((b/'config.json').read_text());d=c['fresh420'];study=c['memory_screen']
    context=next(x for x in study['arms'] if x['label']==a.context)
    out=b/'contexts'/a.context;out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();original_affinities={};opt=lease=probe=host=None
    report=dict(status='running',context=context,pid=os.getpid(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
                measured=[],smoke_fingerprints=[],timing_chunks=[],observations=[],refresh_guards=[],tracking_probes=[],comparison_records=[])
    reference=None
    if a.context=='qr_2':
        reference=json.loads((b/'contexts/qr_0/result.json').read_text())
        assert reference['status']=='completed'
        reference=reference['comparison_records']
    def emit(kind,value):
        row=canonical(kind,value);report['comparison_records'].append(row);save()
        if reference is not None:check_record(reference,len(report['comparison_records'])-1,row)
    def save():json_write(out/'result.json',report)
    def budget():
        assert time.monotonic()-started<d['per_process_seconds'],'Child wall bound exceeded'
        assert torch.cuda.max_memory_allocated()<study['maximum_peak_allocated_bytes'],'Allocated memory bound exceeded'
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
        report['parameter_count']=sum(p.numel() for p in model.parameters())
        active=torch.compile(model,mode='default',fullgraph=True,dynamic=False)
        strong=StrongAugmentation(c,mean,std);crop=torch.Generator(device='cuda').manual_seed(c['seed']+1001)
        initial_rng=rng_state(crop,strong)
        report['initial_rng_sha256']=state_digest(initial_rng)
        per_epoch=len(split['train'])//c['batch_size'];schedule_steps=per_epoch*c['epochs']
        order_rng=np.random.default_rng(c['seed']+2001);batches=[]
        for _ in range(math.ceil(420/per_epoch)):
            order=order_rng.permutation(np.asarray(split['train']))
            batches.extend(order[:per_epoch*c['batch_size']].reshape(per_epoch,c['batch_size']))
        device_batches=[torch.tensor(x,device='cuda') for x in batches]
        report['batch_plan']=dict(count=len(batches),bytes=sum(x.numel()*x.element_size() for x in device_batches),
                                 first25_ids_sha256=hashlib.sha256(np.asarray(batches[:25]).tobytes()).hexdigest())
        report['backend_before_smoke']=backend_record();save()
        emit('initial',dict(model=initial_hash,rng=state_digest(initial_rng),batch_plan=report['batch_plan']))
        emit('backend',dict(phase='before_smoke',record=report['backend_before_smoke']))
        host=CompactDispatchHostTiming()
        def step(index,optimizer,capture):
            hash_seconds=0.
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
                    torch.cuda.synchronize()
                    hash_start=time.monotonic()
                    # RAM-only caller diagnostics, no compiled-model hooks or math changes.
                    row=dict(step=index+1,input=tensor_record(x),target=tensor_record(target),
                        ids_sha256=hashlib.sha256(batches[index].tobytes()).hexdigest(),
                        rng_sha256=state_digest(rng_state(crop,strong)),output=tensor_record(logits),
                        loss=float(loss.detach()),grad_before_clip=gradient_records(model))
                    hash_seconds+=time.monotonic()-hash_start
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_norm'],foreach=True)
                if capture:
                    torch.cuda.synchronize()
                    hash_start=time.monotonic()
                    row.update(grad_after_clip=gradient_records(model),gradient_norm=float(norm))
                    hash_seconds+=time.monotonic()-hash_start
            ev[2].record();host.mark()
            with host.scope('optimizer'):optimizer.step()
            ev[3].record();host.finish()
            if capture:
                torch.cuda.synchronize()
                hash_start=time.monotonic()
                row.update(model_sha256=state_digest(model.state_dict()),
                    optimizer_sha256=state_digest(optimizer.state_dict()),basis=basis_records(optimizer),
                    counters=counters(optimizer))
                hash_seconds+=time.monotonic()-hash_start
            return row,loss.detach(),norm.detach(),x,y,target,ev,hash_seconds
        model.train();smoke=build_optimizer(model,c,'qr10')
        for index in range(d['smoke_steps']):
            row,loss,norm,x,y,target,events,hash_seconds=step(index,smoke,index+1 in [1,25])
            if row:
                report['smoke_fingerprints'].append(row);emit('fingerprint',dict(phase='smoke',record=row))
            budget()
        torch.cuda.synchronize();model.eval()
        report['smoke']=dict(steps=d['smoke_steps'],training_parity=support.parity_check(model,active,x,y),
            evaluation_parity=support.evaluation_parity_check(model,x,y))
        report['code_after_smoke']=loaded_code()
        emit('smoke',report['smoke']);emit('code',dict(phase='after_smoke',record=report['code_after_smoke']))
        report['matrix_inventory']=[dict(name=name,shape=list(p.shape),factor_dimensions=list(p.shape)) for g in smoke.matrix.param_groups for p,name in zip(g['params'],g['param_names'])]
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
        cfg=screen_arm_config(c,context);opt=build_optimizer(model,cfg,context['policy'])
        lease=OwnedMatrixHooks(opt.matrix)
        if context.get('movement_policy'):configure_matrix(opt.matrix,context['movement_policy'])
        probe=WideTransportProbe(opt,study['tracking_ages'])
        assert not opt.matrix.state and not opt.auxiliary.state
        report['effective_groups']=[{k:v for k,v in g.items() if k in ['role','betas','shampoo_beta','peak_lr','weight_decay','eps','basis_method','precondition_frequency','hard_reset_interval','variance_policy','basis_ns_iterations','transport_precision','basis_rotation_cap']} for g in opt.param_groups]
        emit('initial',dict(effective_groups=report['effective_groups']))
        report['allocated_before_measured_bytes']=torch.cuda.memory_allocated()
        report['peak_before_measured_bytes']=torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats();host.reset();save()
        training_seconds=0.;inside_hash_seconds=0.;segment=time.monotonic()
        all_events=[];events=[];losses=[];norms=[];consumed=hashlib.sha256();aug_hashes={}
        def pause():
            nonlocal training_seconds,inside_hash_seconds
            torch.cuda.synchronize()
            training_seconds+=time.monotonic()-segment-inside_hash_seconds
            inside_hash_seconds=0.
        def resume():
            nonlocal segment
            segment=time.monotonic()
        def observation(index):
            saved_rng=rng_state(crop,strong);before=state_digest(model.state_dict())
            model.eval()
            val=support.evaluate(active,images,labels,ids['validation'],mean,std,c['eval_batch_size'])
            panel=support.evaluate(active,images,labels,ids['train_panel'],mean,std,c['eval_batch_size'])
            parity=support.evaluation_parity_check(model,x,y)
            geometry=check_basis_state(opt,study['maximum_gram_error'])
            assert state_digest(model.state_dict())==before
            model.train();restore_rng(saved_rng,crop,strong)
            assert state_digest(rng_state(crop,strong))==state_digest(saved_rng)
            return dict(step=index,training_seconds=training_seconds,validation=val,train_panel=panel,
                evaluation_parity=parity,geometry=geometry,model_sha256=before,
                optimizer_sha256=state_digest(opt.state_dict()),rng_sha256=state_digest(saved_rng),
                consumed_ids_sha256=consumed.hexdigest(),augmented_batches=dict(aug_hashes),
                counters=counters(opt))
        for index in range(d['measured_steps']):
            step_number=index+1;consumed.update(batches[index].tobytes())
            row,loss,norm,x,y,target,ev,hash_seconds=step(index,opt,step_number in d['fingerprint_steps'])
            inside_hash_seconds+=hash_seconds
            events.append(ev);all_events.append(ev);losses.append(loss);norms.append(norm)
            if row:
                pause();report['measured'].append(row)
                emit('fingerprint',dict(phase='measured',record=row));resume()
            if step_number==1 or opt.matrix.last_basis_refreshes:
                # Online finite/Gram guards count in full training wall.
                guard_started=time.monotonic();diagnostic=check_basis_state(opt,study['maximum_gram_error'])
                guard=dict(step=step_number,**diagnostic,diagnostic_seconds=time.monotonic()-guard_started)
                pause();report['refresh_guards'].append(guard);emit('guard',guard);resume()
            if probe.pending:
                pause();probe_started=time.monotonic();matrices=probe.flush()
                value=dict(step=step_number,diagnostic_seconds=time.monotonic()-probe_started,matrices=matrices)
                report['tracking_probes'].append(value);emit('probe',value);resume()
            if step_number in [1,2,*study['evaluation_updates']]:
                pause();aug_hashes[str(step_number)]=state_digest([x,target]);resume()
            if step_number%study['timing_chunk']==0:
                pause()
                values=torch.stack([torch.stack(losses).mean(),torch.stack(norms).max()]).cpu().tolist()
                assert all(math.isfinite(v) for v in values),'Nonfinite training'
                samples=[[v[0].elapsed_time(v[1]),v[1].elapsed_time(v[2]),v[2].elapsed_time(v[3])] for v in events]
                value=dict(step=step_number,samples=samples,loss=values[0],gradient_norm=values[1],host=host.flush())
                report['timing_chunks'].append(value);emit('chunk',value)
                events=[];losses=[];norms=[]
                if step_number in study['evaluation_updates']:
                    value=observation(step_number);report['observations'].append(value);emit('observation',value)
                    print('OBSERVATION '+a.context+' '+str(step_number)+' CE '+str(value['validation']['ce']),flush=True)
                budget();save();resume()
        pause()
        report['training_seconds']=training_seconds
        report['optimizer_ms']=sum(e[2].elapsed_time(e[3]) for e in all_events[50:])/len(all_events[50:])
        report['model_backward_ms']=sum(e[1].elapsed_time(e[2]) for e in all_events[50:])/len(all_events[50:])
        report['final_state']=dict(model=state_digest(model.state_dict()),optimizer=state_digest(opt.state_dict()),rng=state_digest(rng_state(crop,strong)))
        emit('final',report['final_state'])
        report['code_after_measured']=loaded_code();report['backend_after_measured']=backend_record()
        emit('code',dict(phase='after_measured',record=report['code_after_measured']))
        emit('backend',dict(phase='after_measured',record=report['backend_after_measured']))
        def invariants():return dict(model=state_digest(model.state_dict()),optimizer=state_digest(opt.state_dict()),
                                     rng=state_digest(rng_state(crop,strong)))
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
        emit('code',dict(phase='after_saved_operand_replay',record=report['code_after_saved_operand_replay']))
        if reference is not None:assert len(report['comparison_records'])==len(reference)
        # SAVED_OPERAND_REPLAY_END
        report['peak_allocated_bytes']=max(report['peak_before_measured_bytes'],torch.cuda.max_memory_allocated())
        matrix_ref=weakref.ref(opt.matrix);opt_ref=weakref.ref(opt)
        probe.close();lease.close();del probe,lease,opt;probe=lease=opt=None
        report['owned_optimizer_released']=matrix_ref() is None and opt_ref() is None
        assert report['owned_optimizer_released'];budget()
        report.update(status='completed',all_in_seconds=time.monotonic()-started);save()
    except BaseException as exc:
        report.update(status='failed',failure=repr(exc),all_in_seconds=time.monotonic()-started);save();raise
    finally:
        matrix_ref=weakref.ref(opt.matrix) if opt is not None else None
        opt_ref=weakref.ref(opt) if opt is not None else None
        if probe is not None:probe.close()
        if lease is not None:lease.close()
        probe=lease=opt=None
        if matrix_ref is not None:
            report['owned_optimizer_released']=matrix_ref() is None and opt_ref() is None
        if host is not None:host.close()
        for tid,mask in original_affinities.items():
            try:os.sched_setaffinity(tid,mask)
            except ProcessLookupError:pass
        report['affinity_restored']={str(k):sorted(v) for k,v in original_affinities.items()}
        report['own_affinity_restored']=all(os.sched_getaffinity(k)==v for k,v in original_affinities.items())
        save()


if __name__=='__main__':main()
