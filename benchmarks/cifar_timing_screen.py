"""Isolated timing study derived from cifar_lr_screen; shared trainer is unchanged."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE),str(HERE/'cifar_support')]
import numpy as np
import torch
from torch.nn import functional as F
from cifar_optimizer_benchmark import (json_write,rng_state,restore_rng,state_digest,
                                      support,StrongAugmentation)
from timing_diagnostics import HostTiming, telemetry, profile_summary
from movement_retraction import configure_matrix
from cifar_optimizers import apply_lr_multiplier,build_optimizer,counters,screen_arm_config,check_basis_state


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    bundle=parser.parse_args().bundle.resolve()
    assert HERE.parent==bundle/'source' and os.environ.get('GPU_CLAIM_INDICES')
    assert torch.cuda.device_count()==1,'Exactly one claimed GPU'
    manifest=json.loads((bundle/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert support.digest(bundle/'source'/name)==digest,name
    for name,key in [('config.json','config_sha256'),('split.json','split_sha256'),
                     ('data_prepared.json','base_prepared_sha256'),('PROTOCOL.md','protocol_sha256')]:
        assert support.digest(bundle/name)==manifest[key],name
    c=json.loads((bundle/'config.json').read_text())
    assert not c['checkpoint'] and not c['official_test']
    study=c.get('memory_screen',c.get('lr_screen'));started=time.monotonic()
    assert study['equal_updates']>875 and study['maximum_updates']>=study['equal_updates']
    wall_target=study['wall_seconds']
    assert wall_target is None or wall_target>0
    assert c['betas'][1] in (.99,.999) and c['shampoo_beta'] in (.99,.999)
    torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    random.seed(c['seed']);torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
    raw,labels,_=support.load_training_data(c)
    images=torch.from_numpy(raw).permute(0,3,1,2).contiguous().cuda()
    labels=torch.tensor(labels,device='cuda');del raw
    split=json.loads((bundle/'split.json').read_text())
    ids={k:torch.tensor(v,device='cuda') for k,v in split.items()}
    prepared=json.loads((bundle/'data_prepared.json').read_text())
    mean=torch.tensor(prepared['mean'],device='cuda')[None,:,None,None]
    std=torch.tensor(prepared['std'],device='cuda')[None,:,None,None]
    model=support.build_model(c).cuda();initial=copy.deepcopy(model.state_dict())
    initial_hash=state_digest(initial)
    active=torch.compile(model,mode='default',fullgraph=True,dynamic=False)
    strong=StrongAugmentation(c,mean,std)
    crop=torch.Generator(device='cuda').manual_seed(c['seed']+1001)
    initial_rng=rng_state(crop,strong)
    per_epoch=len(split['train'])//c['batch_size'];schedule_steps=per_epoch*c['epochs']
    assert study['maximum_updates']<=schedule_steps
    order_rng=np.random.default_rng(c['seed']+2001)
    batches=[]
    for _ in range(math.ceil(study['maximum_updates']/per_epoch)):
        order=order_rng.permutation(np.asarray(split['train']))
        batches.extend(order[:per_epoch*c['batch_size']].reshape(per_epoch,c['batch_size']))
    device_batches=[torch.tensor(x,device='cuda') for x in batches]
    report=dict(status='running',config=c,torch=torch.__version__,cuda=torch.version.cuda,
                gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],initial_model_sha256=initial_hash,
                parameter_count=sum(p.numel() for p in model.parameters()),
                cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_threads=torch.get_num_threads(),
                timing_version=2,runs=[])
    json_write(bundle/'screen.json',report)

    def check_budget():
        if time.monotonic()-started>c['max_wall_seconds']:
            raise RuntimeError('Predeclared study wall budget exceeded')

    host = HostTiming()
    replay_states = {}

    def train_step(index,opt):
        ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[0].record();host.begin()
        with host.scope('augmentation'):
            batch=device_batches[index]
            offsets=torch.randint(0,9,(c['batch_size'],2),device='cuda',generator=crop)
            flips=torch.rand(c['batch_size'],device='cuda',generator=crop)<.5
            y=labels[batch];x,target,_=strong(images[batch],y,offsets,flips)
        ev[1].record();host.mark()
        with host.scope('model_backward'):
            multiplier=support.lr_at(index+1,schedule_steps,per_epoch*c['warmup_epochs'],1.,c['min_lr_ratio'])
            apply_lr_multiplier(opt,multiplier);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(active(x).float(),target)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_norm'],foreach=True)
        ev[2].record();host.mark()
        with host.scope('optimizer'):
            opt.step()
        ev[3].record();host.finish()
        return loss.detach(),norm.detach(),x,y,target,ev

    # Compile/cold-path qualification is separate, then discard every trained state.
    model.train();smoke=build_optimizer(model,c,'qr10')
    for index in range(c['smoke_steps']):
        loss,norm,x,y,target,events=train_step(index,smoke)
        if (index+1)%25==0:check_budget()
    torch.cuda.synchronize();model.eval()
    report['smoke']=dict(steps=c['smoke_steps'],training_parity=support.parity_check(model,active,x,y),
                        evaluation_parity=support.evaluation_parity_check(model,x,y))
    del smoke
    json_write(bundle/'screen.json',report)
    print('SMOKE '+json.dumps(report['smoke']),flush=True)

    def observation(opt,step,training_seconds,id_hash,aug_hashes,x,y):
        saved_rng=rng_state(crop,strong);before=state_digest(model.state_dict())
        model.eval()
        val=support.evaluate(active,images,labels,ids['validation'],mean,std,c['eval_batch_size'])
        panel=support.evaluate(active,images,labels,ids['train_panel'],mean,std,c['eval_batch_size'])
        parity=support.evaluation_parity_check(model,x,y)
        prior=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
        errors=[]
        for state in opt.matrix.state.values():
            if not all(bool(torch.isfinite(state[k]).all()) for k in ['exp_avg','exp_avg_sq']):
                raise FloatingPointError('Nonfinite optimizer moments')
            for q in state['Q']:
                if q is not None:errors.append((q.T@q-torch.eye(q.shape[0],device=q.device)).norm()/q.shape[0]**.5)
        errors=torch.stack(errors).cpu().tolist();torch.set_float32_matmul_precision(prior)
        if not all(math.isfinite(v) for v in errors) or max(errors)>study['maximum_gram_error']:
            raise FloatingPointError('Predeclared basis stability guard exceeded')
        assert state_digest(model.state_dict())==before,'Evaluation mutated weights'
        model.train();restore_rng(saved_rng,crop,strong)
        return dict(step=step,epoch=step/per_epoch,training_seconds=training_seconds,
                    validation=val,train_panel=panel,evaluation_parity=parity,
                    maximum_gram_error=max(errors),mean_gram_error=statistics.mean(errors),
                    model_sha256=before,rng_sha256=state_digest(saved_rng),
                    consumed_ids_sha256=id_hash.hexdigest(),augmented_batches=dict(aug_hashes),
                    optimizer_counters=counters(opt))

    paired={}
    phase_names=['augmentation_cuda_ms','model_backward_cuda_ms','optimizer_ms','iteration_cuda_ms']
    for arm in study['arms']:
        check_budget();model.load_state_dict(initial);model.train()
        restore_rng(initial_rng,crop,strong)
        assert state_digest(model.state_dict())==initial_hash
        arm_config=screen_arm_config(c,arm)
        opt=build_optimizer(model,arm_config,arm['policy'])
        if arm.get('movement_policy'):configure_matrix(opt.matrix,arm['movement_policy'])
        probe=None
        if arm.get('tracking_probe',False):
            if study.get('tracking_probe_kind')=='strict_transport':
                from cifar_strict_transport import StrictTransportProbe as TrackingProbe
            elif study.get('tracking_probe_kind')=='precision_attribution':
                from cifar_precision_attribution import PrecisionProbe as TrackingProbe
            elif study.get('tracking_probe_kind')=='efficiency':
                from cifar_refresh_efficiency import EfficiencyProbe as TrackingProbe
            else:
                from cifar_tracking_probe import TrackingProbe
            probe=TrackingProbe(opt,study['tracking_ages'])
        assert not opt.matrix.state and not opt.auxiliary.state
        result=dict(**arm,telemetry_start=telemetry(),status='running',observations=[],history=[],initial_model_sha256=initial_hash,
                    effective_groups=[{k:v for k,v in g.items() if k in
                        ['role','betas','shampoo_beta','peak_lr','weight_decay','eps','basis_method',
                         'precondition_frequency','hard_reset_interval','variance_policy','basis_ns_iterations',
                         'transport_precision','basis_rotation_cap']} for g in opt.param_groups])
        if study.get('guard_every_refresh',False):result['refresh_guards']=[]
        if probe is not None:result['tracking_probes']=[]
        report['runs'].append(result);json_write(bundle/'screen.json',report)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        host.reset()
        id_hash=hashlib.sha256();aug_hashes={};events=[];losses=[];norms=[]
        phase_samples=[];process_cpu_seconds=0.;chunk_cpu=time.process_time()
        training_seconds=0.;wall_observed=wall_target is None;equal_observed=False
        chunk_start=time.monotonic();arm_start=chunk_start
        for index in range(study['maximum_updates']):
            step=index+1;id_hash.update(batches[index].tobytes())
            loss,norm,x,y,target,ev=train_step(index,opt)
            events.append(ev)
            losses.append(loss);norms.append(norm)
            if study.get('guard_every_refresh',False) and (step==1 or opt.matrix.last_basis_refreshes):
                torch.cuda.synchronize();training_seconds+=time.monotonic()-chunk_start
                process_cpu_seconds+=time.process_time()-chunk_cpu
                guard_start=time.monotonic()
                diagnostic=check_basis_state(opt,study['maximum_gram_error'])
                result['refresh_guards'].append(dict(step=step,**diagnostic,diagnostic_seconds=time.monotonic()-guard_start))
                chunk_start=time.monotonic();chunk_cpu=time.process_time()
            if probe is not None and probe.pending:
                torch.cuda.synchronize();training_seconds+=time.monotonic()-chunk_start
                process_cpu_seconds+=time.process_time()-chunk_cpu
                probe_start=time.monotonic();measurements=probe.flush()
                result['tracking_probes'].append(dict(step=step,diagnostic_seconds=time.monotonic()-probe_start,matrices=measurements))
                chunk_start=time.monotonic();chunk_cpu=time.process_time()
            # Hash probes are excluded from training-time accounting.
            if step in [1,2,*study['evaluation_updates']]:
                torch.cuda.synchronize();training_seconds+=time.monotonic()-chunk_start
                process_cpu_seconds+=time.process_time()-chunk_cpu
                aug_hashes[str(step)]=state_digest([x,target])
                chunk_start=time.monotonic();chunk_cpu=time.process_time()
            if step%study['timing_chunk']==0 or step==study['maximum_updates']:
                values=torch.stack([torch.stack(losses).mean(),torch.stack(norms).max()]).cpu().tolist()
                torch.cuda.synchronize();training_seconds+=time.monotonic()-chunk_start
                process_cpu_seconds+=time.process_time()-chunk_cpu
                if not all(math.isfinite(v) for v in values):raise FloatingPointError('Nonfinite training')
                samples=[(a.elapsed_time(b),b.elapsed_time(d),d.elapsed_time(e),a.elapsed_time(e))
                         for a,b,d,e in events]
                phase_samples.extend(samples);events=[]
                row=dict(step=step,training_seconds=training_seconds,process_cpu_seconds=process_cpu_seconds,
                         loss=values[0],maximum_gradient_norm=values[1],
                         **{name:statistics.mean(x[i] for x in samples) for i,name in enumerate(phase_names)})
                row['host']=host.flush()
                result['history'].append(row);losses=[];norms=[]
                reasons=[]
                if step in study['evaluation_updates']:reasons.append('fixed_updates')
                if not wall_observed and training_seconds>=wall_target:
                    reasons.append('equal_wall');wall_observed=True
                if reasons:
                    obs=observation(opt,step,training_seconds,id_hash,aug_hashes,x,y)
                    obs['reasons']=reasons
                    obs['timing']=dict(process_cpu_seconds=process_cpu_seconds,
                        all_update_cuda_seconds=sum(x[3] for x in phase_samples)/1000,
                        steady_samples=len(phase_samples[50:]),
                        **{name:statistics.mean(x[i] for x in phase_samples[50:]) for i,name in enumerate(phase_names)})
                    result['observations'].append(obs)
                    if step in study['evaluation_updates']:
                        keys=['rng_sha256','consumed_ids_sha256','augmented_batches']
                        comparable={k:obs[k] for k in keys}
                        if step in paired:assert comparable==paired[step],('Unpaired input stream',arm['label'],step)
                        else:paired[step]=comparable
                    print(arm['label']+' '+json.dumps(obs),flush=True)
                    json_write(bundle/'screen.json',report)
                if step==study['equal_updates']:equal_observed=True
                if step%250==0:print(arm['label']+' PROGRESS '+json.dumps(row),flush=True)
                check_budget();chunk_start=time.monotonic();chunk_cpu=time.process_time()
                if equal_observed and wall_observed:break
        torch.cuda.synchronize()
        result.update(telemetry_end=telemetry(),optimizer_sha256=state_digest(opt.state_dict()),status='completed',steps=step,wall_target_reached=wall_observed if wall_target is not None else None,
                      **{name:statistics.mean(x[i] for x in phase_samples[50:]) for i,name in enumerate(phase_names)},
                      process_cpu_seconds=process_cpu_seconds,
                      training_seconds=training_seconds,all_in_seconds=time.monotonic()-arm_start,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        if not wall_observed:result['wall_limit_note']='Maximum-update cap reached; no equal-wall claim'
        assert equal_observed
        json_write(bundle/'screen.json',report)
        if probe is not None:probe.close()
        if arm.get('capture_replay'):
            replay_states[arm['kind']]=dict(model=copy.deepcopy(model.state_dict()),
                optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state(crop,strong),arm=arm)
        del probe,opt
    def fixed(run):return next(x for x in run['observations'] if x['step']==study['equal_updates'])
    pairs=study.get('control_pairs',[['qr10_lr0.001','qr10_lr0.001_repeat']])
    by_label={x['label']:x for x in report['runs']}
    comparisons=[]
    for original,repeated in pairs:
        first=fixed(by_label[original]);repeat=fixed(by_label[repeated])
        comparisons.append(dict(original=original,repeated=repeated,
            model_hash_equal=first['model_sha256']==repeat['model_sha256'],
            ce_difference=repeat['validation']['ce']-first['validation']['ce']))
    report['control_repeats']=comparisons
    report['control_repeat']=dict(model_hash_equal=all(x['model_hash_equal'] for x in comparisons),
                                 ce_difference=comparisons[-1]['ce_difference'])
    report['paired_fixed_update_checks_passed']=True
    report['status']='completed' if report['control_repeat']['model_hash_equal'] else 'repeat_mismatch'
    # Primary controls finish before any profiler work. Saved states stay in RAM.
    report['status']='profiling'
    report['profile_replays']=[]
    json_write(bundle/'screen.json',report)
    for kind,saved in replay_states.items():
        arm=saved['arm'];reference=None
        for mode in ['plain','profile1','profile2']:
            check_budget();model.load_state_dict(saved['model']);model.train()
            opt=build_optimizer(model,screen_arm_config(c,arm),arm['policy'])
            if arm.get('movement_policy'):configure_matrix(opt.matrix,arm['movement_policy'])
            opt.load_state_dict(copy.deepcopy(saved['optimizer']))
            restore_rng(saved['rng'],crop,strong);host.reset();host.profile=mode!='plain'
            prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if host.profile else None
            from contextlib import nullcontext
            torch.cuda.synchronize();t=time.monotonic()
            losses=[];event_rows=[]
            with prof if prof is not None else nullcontext():
                for index in range(study['equal_updates'],study['equal_updates']+study['profile_steps']):
                    loss,norm,x,y,target,ev=train_step(index,opt)
                    losses.append(loss);event_rows.append(ev)
                torch.cuda.synchronize()
            wall=time.monotonic()-t
            control=dict(model=state_digest(model.state_dict()),optimizer=state_digest(opt.state_dict()),
                         rng=state_digest(rng_state(crop,strong)),losses=torch.stack(losses).cpu().tolist())
            if reference is None:reference=control
            row=dict(kind=kind,mode=mode,numerical_control=control,exact=control==reference,
                     wall_seconds=wall,host=host.flush(),cuda_ms={name:sum(e[i].elapsed_time(e[i+1]) for e in event_rows)
                     for i,name in enumerate(['augmentation','model_backward','optimizer'])})
            if prof is not None:row['profiler']=profile_summary(prof,bundle,kind+'_'+mode)
            report['profile_replays'].append(row);json_write(bundle/'screen.json',report)
            assert sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file()) < manifest['evidence_budget_bytes'],'Evidence budget exceeded'
            assert row['exact'],'Profiler replay changed numerical result'
            del opt,prof
    host.close()
    report['status']='completed' if report['control_repeat']['model_hash_equal'] else 'repeat_mismatch'
    report['all_in_seconds']=time.monotonic()-started
    json_write(bundle/'screen.json',report)
    print('COMPLETE '+json.dumps(report['control_repeat']),flush=True)
    if report['status']!='completed':raise RuntimeError('Unchanged-control repeat mismatch; investigate before ranking')


if __name__=='__main__':main()
