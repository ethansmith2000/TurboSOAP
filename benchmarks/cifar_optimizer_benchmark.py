"""Frozen CIFAR-100 adapter and common-state optimizer cost calibration.

This bounded calibration is not a convergence or tuned optimizer comparison.
No model/optimizer files are saved. The shared prefix and branch states live in
memory only. Dataset/split/support code are copied from corrected SNRAdam inputs.
"""
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

import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE),str(HERE/'cifar_support')]
import cifar100_qualification as support
from cifar100_strong_augmentation import StrongAugmentation
from cifar_optimizers import (apply_lr_multiplier,build_optimizer,counters,set_policy)


def json_write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def state_digest(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            y=x.detach().cpu().contiguous()
            h.update(str((y.dtype,tuple(y.shape))).encode());h.update(y.numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=str):h.update(str(key).encode());visit(x[key])
        elif isinstance(x,(list,tuple)):
            for item in x:visit(item)
        else:h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def rng_state(gen,strong):
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),
                python=random.getstate(),crop=gen.get_state(),
                mix=copy.deepcopy(strong.rng.bit_generator.state),counts=copy.deepcopy(strong.counts))


def restore_rng(s,gen,strong):
    torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state_all(s['cuda'])
    random.setstate(s['python']);gen.set_state(s['crop'])
    strong.rng.bit_generator.state=copy.deepcopy(s['mix']);strong.counts=copy.deepcopy(s['counts'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args();bundle=args.bundle.resolve()
    assert os.environ.get('GPU_CLAIM_INDICES'),'Use lifetime gpu-claim'
    assert HERE.parent==bundle/'source','Execute frozen source'
    manifest=json.loads((bundle/'manifest.json').read_text())
    for name,sha in manifest['source_sha256'].items():
        assert support.digest(bundle/'source'/name)==sha,name
    c=json.loads((bundle/'config.json').read_text())
    assert support.digest(bundle/'config.json')==manifest['config_sha256']
    assert support.digest(bundle/'split.json')==manifest['split_sha256']
    assert support.digest(bundle/'data_prepared.json')==manifest['base_prepared_sha256']
    assert not c['checkpoint'] and not c['official_test']
    if c['prefix_steps']<=50 or c['branch_steps']<=50:
        raise ValueError('Need more than 50 updates for steady timing')
    overall_started=time.monotonic()
    torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    random.seed(c['seed']);torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
    raw,labels,_=support.load_training_data(c)
    images=torch.from_numpy(raw).permute(0,3,1,2).contiguous().cuda()
    labels=torch.tensor(labels,device='cuda')
    split=json.loads((bundle/'split.json').read_text())
    ids={k:torch.tensor(v,device='cuda') for k,v in split.items()}
    prepared=json.loads((bundle/'data_prepared.json').read_text())
    mean=torch.tensor(prepared['mean'],device='cuda')[None,:,None,None]
    std=torch.tensor(prepared['std'],device='cuda')[None,:,None,None]
    model=support.build_model(c).cuda()
    initial=copy.deepcopy(model.state_dict());initial_hash=state_digest(initial)
    active=torch.compile(model,mode='default',fullgraph=True,dynamic=False)
    strong=StrongAugmentation(c,mean,std)
    crop=torch.Generator(device='cuda').manual_seed(c['seed']+1001)
    initial_rng=rng_state(crop,strong)
    per_epoch=len(split['train'])//c['batch_size']
    schedule_steps=per_epoch*c['epochs']
    needed=c['prefix_steps']+c['branch_steps']
    order_rng=np.random.default_rng(c['seed']+2001)
    batches=[]
    for _ in range(math.ceil(needed/per_epoch)):
        order=order_rng.permutation(np.asarray(split['train']))
        batches.extend(order[:per_epoch*c['batch_size']].reshape(per_epoch,c['batch_size']))
    device_batches=[torch.tensor(v,device='cuda') for v in batches]
    report=dict(status='running',config=c,initial_model_sha256=initial_hash,
                parameter_count=sum(p.numel() for p in model.parameters()),
                torch=torch.__version__,cuda=torch.version.cuda,
                gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],runs=[],
                limitations=['Calibration only; matrix LR not tuned; endpoints are early learning, not rankings.',
                             'Historical SNRAdam scores are workload context, not paired controls.',
                             'Two-sided matrix factors; auxiliary parameters use fused AdamW.',
                             'Permutations of v follow QR sorting; warm preserves column labels; general v rotation not applied.',
                             'No compiled optimizer kernels; timings include eager optimizer overhead.'])

    def evaluate():
        model.eval()
        val=support.evaluate(active,images,labels,ids['validation'],mean,std,c['eval_batch_size'])
        panel=support.evaluate(active,images,labels,ids['train_panel'],mean,std,c['eval_batch_size'])
        model.train()
        return dict(validation=val,train_panel=panel)

    def run(label,opt,start,count,check_parity=False):
        started=time.monotonic();model.train();before=counters(opt)
        event_pairs=[];event_kinds=[];losses=[];max_norms=[];history=[];hashes={}
        id_hash=hashlib.sha256();steady_started=None;last_x=last_y=None
        for index in range(count):
            absolute=start+index
            batch=device_batches[absolute]
            id_hash.update(batches[absolute].tobytes())
            ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
            ev[0].record()
            offsets=torch.randint(0,9,(c['batch_size'],2),device='cuda',generator=crop)
            flips=torch.rand(c['batch_size'],device='cuda',generator=crop)<.5
            y=labels[batch];x,target,_=strong(images[batch],y,offsets,flips)
            multiplier=support.lr_at(absolute+1,schedule_steps,per_epoch*c['warmup_epochs'],1.,c['min_lr_ratio'])
            apply_lr_multiplier(opt,multiplier);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(active(x).float(),target)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_norm'],foreach=True)
            ev[1].record();opt.step();ev[2].record();ev[3].record()
            if index>=50:
                event_pairs.append(ev)
                matrix=getattr(opt,'matrix',None)
                event_kinds.append('qr' if matrix and matrix.last_hard_reset_events else
                                   'warm' if matrix and matrix.last_basis_refreshes else 'ordinary')
            losses.append(loss.detach());max_norms.append(norm.detach())
            # Identical actual augmented tensors/targets, not only sample IDs.
            # Checks are outside timed events and the steady-state wall window.
            if index<2:hashes[str(index)]=state_digest([x,target])
            if (index+1)%50==0 or index+1==count:
                values=torch.stack([torch.stack(losses).mean(),torch.stack(max_norms).max()]).cpu().tolist()
                if not all(math.isfinite(v) for v in values):raise RuntimeError('Nonfinite training')
                row=dict(step=absolute+1,branch_step=index+1,loss=values[0],maximum_gradient_norm=values[1])
                history.append(row);print(label+' '+json.dumps(row),flush=True)
                losses=[];max_norms=[]
                if time.monotonic()-overall_started>c['max_wall_seconds']:
                    raise RuntimeError('Calibration wall budget exceeded')
            if index==49:
                torch.cuda.synchronize();steady_started=time.monotonic()
            last_x,last_y=x,y
        torch.cuda.synchronize();steady_seconds=time.monotonic()-steady_started
        sample_times=[(a.elapsed_time(d),b.elapsed_time(e)) for a,b,e,d in event_pairs]
        ending_rng=rng_state(crop,strong)
        result=dict(label=label,start_step=start,steps=count,consumed_ids_sha256=id_hash.hexdigest(),
                    first_augmented_batches=hashes,ending_rng_sha256=state_digest(ending_rng),
                    history=history,final_model_sha256=support.model_digest(model),
                    optimizer_counters={k:v-before.get(k,0) for k,v in counters(opt).items()},
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    iteration_cuda_ms=statistics.mean(x[0] for x in sample_times),
                    optimizer_cuda_ms=statistics.mean(x[1] for x in sample_times),
                    steady_wall_ms=steady_seconds*1000/(count-50),
                    optimizer_p95_ms=float(np.percentile([x[1] for x in sample_times],95)),
                    timed_steps=len(sample_times),group_lrs=[dict(role=g['role'],lr=g['lr'],peak_lr=g['peak_lr'],weight_decay=g['weight_decay']) for g in opt.param_groups])
        result['optimizer_timing_by_kind']={kind:dict(
            samples=event_kinds.count(kind),
            mean_ms=statistics.mean(t[1] for t,k in zip(sample_times,event_kinds) if k==kind))
            for kind in sorted(set(event_kinds))}
        matrix=getattr(opt,'matrix',None)
        if matrix is not None:
            prior=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
            gram_errors=[]
            for state in matrix.state.values():
                assert all(bool(torch.isfinite(state[k]).all()) for k in ['exp_avg','exp_avg_sq'])
                for q in state['Q']:
                    if q is not None:
                        gram_errors.append((q.T@q-torch.eye(q.shape[0],device=q.device)).norm()/q.shape[0]**.5)
            errors=torch.stack(gram_errors).cpu().tolist()
            result['basis_geometry']=dict(factors=len(errors),maximum_normalized_gram_error=max(errors),
                                          mean_normalized_gram_error=statistics.mean(errors))
            torch.set_float32_matmul_precision(prior)
        result.update(evaluate())
        if check_parity:
            model.eval()
            result['training_parity']=support.parity_check(model,active,last_x,last_y)
            result['evaluation_parity']=support.evaluation_parity_check(model,last_x,last_y)
            model.train();restore_rng(ending_rng,crop,strong)
        result['all_in_seconds']=time.monotonic()-started
        report['runs'].append(result);json_write(bundle/'calibration.json',report)
        return result

    adam=build_optimizer(model,c,'adamw')
    run('adamw_adapter_smoke',adam,0,c['prefix_steps'],True)
    del adam
    model.load_state_dict(initial);restore_rng(initial_rng,crop,strong)
    optimizer=build_optimizer(model,c,'qr10')
    report['parameter_groups']=[{k:v for k,v in g.items() if k in ['param_names','role','peak_lr','weight_decay']} for g in optimizer.param_groups]
    prefix=run('qr_shared_prefix',optimizer,0,c['prefix_steps'],True)
    shared_model=copy.deepcopy(model.state_dict());shared_optimizer=copy.deepcopy(optimizer.state_dict())
    shared_rng=rng_state(crop,strong)
    report['shared_model_sha256']=state_digest(shared_model)
    report['shared_optimizer_sha256']=state_digest(shared_optimizer)
    reports=[]
    for label in ['qr10','qr10_repeat','qr20','warm5','warm10']:
        model.load_state_dict(shared_model);optimizer.load_state_dict(copy.deepcopy(shared_optimizer))
        assert state_digest(model.state_dict())==report['shared_model_sha256']
        assert state_digest(optimizer.state_dict())==report['shared_optimizer_sha256']
        set_policy(optimizer,'qr10' if label=='qr10_repeat' else label)
        restore_rng(shared_rng,crop,strong)
        result=run(label,optimizer,c['prefix_steps'],c['branch_steps'])
        if reports:
            for key in ['consumed_ids_sha256','first_augmented_batches','ending_rng_sha256']:
                assert result[key]==reports[0][key],(label,key)
        reports.append(result)
    report['control_repeat']=dict(validation_ce_difference=reports[1]['validation']['ce']-reports[0]['validation']['ce'],
        final_model_hash_equal=reports[0]['final_model_sha256']==reports[1]['final_model_sha256'])
    report['status']='completed';json_write(bundle/'calibration.json',report)
    print('COMPLETE '+json.dumps(report['control_repeat']),flush=True)


if __name__=='__main__':main()
