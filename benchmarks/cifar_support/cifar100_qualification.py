"""Single-GPU CIFAR-100 ViT qualification, separate from frozen GPT trainers.

No candidate optimizer, official test evaluation, checkpoint, or implicit resume.
GPU execution requires the workspace lifetime-claim launcher.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from vit import ViT

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def stratified_split(labels, seed, classes=100, validation_per_class=50):
    labels = np.asarray(labels)
    if labels.ndim != 1 or set(labels.tolist()) != set(range(classes)):
        raise ValueError('Unexpected label coverage')
    rng = np.random.default_rng(seed)
    train, val, panel = [], [], []
    for c in range(classes):
        ids = np.flatnonzero(labels == c)
        if len(ids) <= validation_per_class:
            raise ValueError('Insufficient class examples')
        ids = rng.permutation(ids)
        val.extend(ids[:validation_per_class])
        train.extend(ids[validation_per_class:])
        panel.extend(ids[validation_per_class:validation_per_class + 20])
    return {k: np.asarray(v, dtype=np.int64).tolist()
            for k, v in [('train', train), ('validation', val), ('train_panel', panel)]}


def validate_config(c):
    fixed = dict(dataset='cifar100',image_size=32,patch_size=4,classes=100,
                 optimizer='adamw',checkpoint=False,official_test=False)
    if any(c.get(k) != v for k,v in fixed.items()):
        raise ValueError('Outside qualification scope')
    for k in ['batch_size','eval_batch_size','epochs','warmup_epochs','width','depth','heads','log_every']:
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError(k)
    if c['width'] % c['heads'] or c['warmup_epochs'] >= c['epochs']:
        raise ValueError('Architecture or schedule')
    if not (0 < c['lr'] < 1 and c['weight_decay'] >= 0 and
            0 <= c['label_smoothing'] < 1 and 0 <= c['min_lr_ratio'] <= 1):
        raise ValueError('Invalid scalar configuration')
    if len(c['betas']) != 2 or not all(0 <= b < 1 for b in c['betas']):
        raise ValueError('Invalid betas')
    if c.get('evaluation_policy','eager_bf16')!='eager_bf16':
        raise ValueError('Only verified eager evaluation is permitted')
    if c.get('recipe','minimal') not in ['minimal','mixup_cutmix_randaugment']:
        raise ValueError('Unknown recipe')
    if c.get('recipe')=='mixup_cutmix_randaugment':
        if not (0<c['mixup_alpha']<=2 and 0<c['cutmix_alpha']<=2 and
                type(c['randaugment_magnitude']) is int and 0<=c['randaugment_magnitude']<=10):
            raise ValueError('Strong augmentation configuration')


def lr_at(step, total, warmup, peak, minimum):
    if not 1 <= step <= total or not 0 < warmup < total:
        raise ValueError('Schedule bounds')
    if step <= warmup:
        return peak * step / warmup
    fraction = (step-warmup)/(total-warmup)
    return peak*(minimum+(1-minimum)*.5*(1+math.cos(math.pi*fraction)))


def horizon_report(steps_per_epoch):
    return {str(b): dict(half_life_steps=math.log(.5)/math.log(b),
                        half_life_epochs=math.log(.5)/math.log(b)/steps_per_epoch,
                        mean_age_steps=b/(1-b)) for b in [.9,.95,.99,.999,.9999]}


def build_model(c):
    return ViT(img_size=32, patch_size=c['patch_size'], num_classes=100,
               embed_dim=c['width'], depth=c['depth'], heads=c['heads'],
               qk_norm=False, drop_rate=0., gradient_checkpointing=False)


def parameter_groups(model, decay):
    groups = {True: [], False: []}
    names = {True: [], False: []}
    for name, p in model.named_parameters():
        parent, _, leaf = name.rpartition('.')
        module = model.get_submodule(parent)
        use_decay = leaf == 'weight' and isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))
        groups[use_decay].append(p)
        names[use_decay].append(name)
    return [dict(params=groups[k], weight_decay=decay if k else 0., param_names=names[k])
            for k in (True,False)]


def model_digest(model):
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def augment(images, offsets, flips, mean, std):
    """Independent reflection-padded crops and flips for NCHW uint8 images."""
    x = F.pad(images.float()/255., (4,4,4,4), mode='reflect')
    b = torch.arange(len(images), device=images.device)[:,None,None,None]
    c = torch.arange(3, device=images.device)[None,:,None,None]
    rows = offsets[:,0,None] + torch.arange(32, device=images.device)[None,:]
    cols = offsets[:,1,None] + torch.arange(32, device=images.device)[None,:]
    x = x[b,c,rows[:,None,:,None],cols[:,None,None,:]]
    x = torch.where(flips[:,None,None,None], x.flip(-1), x)
    return (x-mean)/std


def load_training_data(c):
    """Decode only the pinned training parquet, never the official test split."""
    import pyarrow.parquet as pq
    from PIL import Image
    path=Path(c['data_root'])/'cifar100/train-00000-of-00001.parquet'
    if digest(path)!=c['data_sha256']:
        raise ValueError('Pinned CIFAR training parquet checksum mismatch')
    table=pq.read_table(path,columns=['img','fine_label'])
    images=np.stack([np.asarray(Image.open(io.BytesIO(row['bytes'])).convert('RGB'))
                     for row in table['img'].to_pylist()])
    labels=np.asarray(table['fine_label'].to_pylist(),dtype=np.int64)
    if images.shape!=(50000,32,32,3) or images.dtype!=np.uint8:
        raise ValueError('Unexpected CIFAR training images')
    if not np.array_equal(np.bincount(labels,minlength=100),np.full(100,500)):
        raise ValueError('Unexpected class counts')
    return images,labels,path


def prepare(bundle, config_path):
    bundle = Path(bundle).resolve()
    if bundle.exists():
        raise ValueError('Fresh bundle required')
    if shutil.disk_usage(ROOT).free < 5*2**30:
        raise RuntimeError('Require 5 GiB shared disk headroom')
    c = json.loads(Path(config_path).read_text()); validate_config(c)
    images,labels,data_path = load_training_data(c)
    split = stratified_split(labels, c['split_seed'])
    # Training-only normalization; no validation or test contribution.
    pixels = images[np.asarray(split['train'])]
    mean = pixels.mean(axis=(0,1,2), dtype=np.float64)/255.
    std = pixels.std(axis=(0,1,2), dtype=np.float64)/255.
    bundle.mkdir(parents=True)
    snapshot = bundle/'source'; snapshot.mkdir()
    sources = {}
    for name in ['cifar100_qualification.py','cifar100_strong_augmentation.py','vit.py',
                 'CIFAR100_QUALIFICATION_PLAN_2026-09-30.md','CIFAR100_RECIPE_REFINEMENT_2026-09-30.md']:
        shutil.copyfile(ROOT/name, snapshot/name)
        sources[name] = digest(snapshot/name)
    write_json(bundle/'config.json',c); write_json(bundle/'split.json',split)
    manifest = dict(sources=sources, config_sha256=digest(bundle/'config.json'),
        split_sha256=digest(bundle/'split.json'),
        data={str(data_path):digest(data_path)}, mean=mean.tolist(), std=std.tolist(),
        counts={k:len(v) for k,v in split.items()},
        steps_per_epoch=len(split['train'])//c['batch_size'],
        test_policy='Pinned training-only mirror; no test file downloaded or evaluated',
        retention='No weights/checkpoints; training parquet under 120 MiB, compact evidence under 50 MiB')
    manifest['horizons'] = horizon_report(manifest['steps_per_epoch'])
    write_json(bundle/'prepared.json',manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def verify(bundle):
    bundle=Path(bundle)
    m=json.loads((bundle/'prepared.json').read_text())
    for name,h in m['sources'].items():
        if digest(bundle/'source'/name)!=h:
            raise ValueError('Frozen source drift: '+name)
    for name,h in m['data'].items():
        if digest(name)!=h: raise ValueError('Data drift: '+name)
    for name in ['config','split']:
        if digest(bundle/(name+'.json'))!=m[name+'_sha256']:
            raise ValueError(name+' drift')
    c=json.loads((bundle/'config.json').read_text()); validate_config(c)
    return c,m,json.loads((bundle/'split.json').read_text())


def evaluate(active, images, labels, indices, mean, std, batch):
    # Compiled no-grad BF16 forward failed same-weight parity on this stack.
    # Keep training compiled, but use the verified eager evaluation path.
    active=getattr(active,'_orig_mod',active)
    total=torch.zeros(3,device=images.device)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(indices),batch):
            ids=indices[start:start+batch]; n=len(ids)
            # Pad only for static compile shape; padded examples never enter metrics.
            if n<batch: ids=torch.cat([ids,ids[:1].expand(batch-n)])
            x=(images[ids].float()/255.-mean)/std
            logits=active(x)[:n].float(); y=labels[ids[:n]]
            total+=torch.stack([F.cross_entropy(logits,y,reduction='sum'),
                               (logits.argmax(1)==y).sum(),torch.tensor(n,device=images.device)])
    loss,correct,n=total.cpu().tolist()
    if not math.isfinite(loss): raise RuntimeError('Nonfinite evaluation')
    return dict(ce=loss/n,accuracy=correct/n,examples=int(n))


def parity_check(model, active, x, y):
    """Same weights/input BF16 eager versus compiled forward and gradients."""
    def observation(fn):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=fn(x); loss=F.cross_entropy(logits.float(),y)
        loss.backward()
        g=torch.cat([p.grad.detach().flatten() for p in model.parameters()])
        return logits.detach().float(),loss.detach(),g
    a=observation(model); b=observation(active)
    metrics=dict(logit_max_abs=float((a[0]-b[0]).abs().max()),
                 loss_abs=float((a[1]-b[1]).abs()),
                 gradient_relative_l2=float((a[2]-b[2]).norm()/a[2].norm().clamp_min(1e-12)),
                 gradient_cosine=float(F.cosine_similarity(a[2],b[2],dim=0)))
    model.zero_grad(set_to_none=True)
    if not all(math.isfinite(v) for v in metrics.values()) or not (
            metrics['logit_max_abs']<=.1 and metrics['loss_abs']<=.03 and
            metrics['gradient_relative_l2']<=.1 and metrics['gradient_cosine']>=.99):
        raise RuntimeError('Eager/compiled parity failed: '+str(metrics))
    return metrics


def evaluation_parity_check(model,x,y):
    """Eager evaluation must agree with the eager gradient-enabled reference."""
    prior=model.training
    model.eval()
    with torch.enable_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        ref=model(x).detach().float().clone()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        got=model(x).float().clone()
    model.train(prior)
    result=dict(max_logit_abs=float((ref-got).abs().max()),
                loss_abs=float((F.cross_entropy(ref,y)-F.cross_entropy(got,y)).abs()))
    if result['max_logit_abs']>.1 or result['loss_abs']>.03:
        raise RuntimeError('Eager evaluation parity failed: '+str(result))
    return result


def execute(bundle, mode):
    if not os.environ.get('GPU_CLAIM_INDICES'):
        raise RuntimeError('GPU execution requires gpu-claim')
    bundle=Path(bundle).resolve(); c,m,split=verify(bundle)
    if Path(__file__).resolve() != bundle/'source/cifar100_qualification.py':
        raise RuntimeError('Execute the frozen source, not live files')
    if mode=='train':
        gate=json.loads((bundle/'smoke/result.json').read_text())
        if gate['status']!='completed' or gate['prepared_sha256']!=digest(bundle/'prepared.json'):
            raise RuntimeError('Matching smoke gate required')
    out=bundle/mode; out.mkdir(exist_ok=False)
    write_json(out/'status.json',dict(status='running',started_unix=time.time()))
    try:
        run_training(bundle,out,c,m,split,mode)
    except BaseException as e:
        write_json(out/'status.json',dict(status='failed',error=repr(e),updated_unix=time.time()))
        raise


def run_training(bundle,out,c,m,split,mode):
    began=time.monotonic()
    torch.set_num_threads(c['cpu_threads'])
    torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=True
    raw_images,raw_labels,_=load_training_data(c)
    images=torch.from_numpy(raw_images).permute(0,3,1,2).contiguous().cuda()
    labels=torch.tensor(raw_labels,device='cuda')
    ids={k:torch.tensor(v,device='cuda') for k,v in split.items()}
    mean=torch.tensor(m['mean'],device='cuda',dtype=torch.float32)[None,:,None,None]
    std=torch.tensor(m['std'],device='cuda',dtype=torch.float32)[None,:,None,None]
    model=build_model(c).cuda()
    initial=model_digest(model)
    if mode=='train':
        gate=json.loads((bundle/'smoke/result.json').read_text())
        if initial!=gate['initial_model_sha256']: raise RuntimeError('Initial-state mismatch')
    opt=torch.optim.AdamW(parameter_groups(model,c['weight_decay']),lr=c['lr'],
                         betas=tuple(c['betas']),eps=1e-8,fused=True)
    active=torch.compile(model,mode='default',fullgraph=True,dynamic=False)
    aug=torch.compile(augment,mode='default',fullgraph=True,dynamic=False)
    strong=None
    if c.get('recipe')=='mixup_cutmix_randaugment':
        from cifar100_strong_augmentation import StrongAugmentation
        strong=StrongAugmentation(c,mean,std)
    gen=torch.Generator(device='cuda').manual_seed(c['seed']+1001)
    order_rng=np.random.default_rng(c['seed']+2001)
    per_epoch=m['steps_per_epoch']; total=per_epoch*c['epochs']
    limit=c['smoke_steps'] if mode=='smoke' else total
    write_json(out/'identity.json',dict(initial_model_sha256=initial,
        parameter_count=sum(p.numel() for p in model.parameters()),
        optimizer_groups=[{k:v for k,v in g.items() if k in ['param_names','weight_decay']} for g in opt.param_groups],
        torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        total_schedule_steps=total,actual_step_budget=limit,steps_per_epoch=per_epoch,
        evaluation_policy='eager_bf16',recipe=c.get('recipe','minimal'),
        horizons=m['horizons'],prepared_sha256=digest(bundle/'prepared.json')))
    records=[]; step=0; timed=[]; sample_digest=hashlib.sha256(); smoke_digest=None
    losses=[]; norms=[]; last_log=0
    # Untrained CE/accuracy is calibration evidence, not a selectable checkpoint.
    model.eval()
    initial_validation=evaluate(active,images,labels,ids['validation'],mean,std,c['eval_batch_size'])
    model.train()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    chunk_start=time.monotonic()
    with (out/'training.jsonl').open('x') as logs,(out/'validation.jsonl').open('x') as vals:
        vals.write(json.dumps(dict(step=0,epoch=0,validation=initial_validation))+'\n'); vals.flush()
        for epoch in range(1,c['epochs']+1):
            order=order_rng.permutation(np.asarray(split['train']))
            device_order=torch.from_numpy(order).cuda()
            for batch_no in range(per_epoch):
                if step>=limit: break
                step+=1
                lo=batch_no*c['batch_size']; hi=lo+c['batch_size']
                batch_ids=device_order[lo:hi]
                sample_digest.update(order[lo:hi].tobytes())
                offsets=torch.randint(0,9,(c['batch_size'],2),device='cuda',generator=gen)
                flips=torch.rand(c['batch_size'],device='cuda',generator=gen)<.5
                y=labels[batch_ids]
                if strong is None:
                    x=aug(images[batch_ids],offsets,flips,mean,std); target=y
                else:
                    x,target,_=strong(images[batch_ids],y,offsets,flips)
                lr=lr_at(step,total,per_epoch*c['warmup_epochs'],c['lr'],c['min_lr_ratio'])
                for group in opt.param_groups: group['lr']=lr
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=F.cross_entropy(active(x).float(),target,
                                         label_smoothing=c['label_smoothing'] if strong is None else 0.)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_norm'],foreach=True)
                opt.step(); losses.append(loss.detach()); norms.append(norm.detach())
                if step==c['smoke_steps']:
                    smoke_digest=sample_digest.hexdigest()
                    if mode=='train':
                        gate=json.loads((bundle/'smoke/result.json').read_text())
                        if smoke_digest!=gate['consumed_ids_sha256']: raise RuntimeError('Smoke/train stream mismatch')
                if step%c['log_every']==0 or batch_no==per_epoch-1 or step==limit:
                    stats=torch.stack([torch.stack(losses).mean(),torch.stack(norms).max()]).cpu().tolist()
                    torch.cuda.synchronize(); elapsed=time.monotonic()-chunk_start
                    n=step-last_log
                    if not all(math.isfinite(v) for v in stats): raise RuntimeError('Nonfinite training')
                    row=dict(step=step,epoch=epoch,lr=lr,smoothed_loss=stats[0],max_grad_norm=stats[1],
                             steps=n,seconds=elapsed,ms_per_step=1000*elapsed/n,
                             steady_state=last_log>=c['log_every'])
                    if row['steady_state']: timed.append(row)
                    logs.write(json.dumps(row,allow_nan=False)+'\n'); logs.flush()
                    losses.clear(); norms.clear(); last_log=step; chunk_start=time.monotonic()
                if time.monotonic()-began>c['max_wall_seconds']:
                    raise RuntimeError('Wall-time budget exceeded; no automatic resume')
            model.eval()
            validation=evaluate(active,images,labels,ids['validation'],mean,std,c['eval_batch_size'])
            train_panel=evaluate(active,images,labels,ids['train_panel'],mean,std,c['eval_batch_size'])
            record=dict(step=step,epoch=epoch,partial_epoch=(step%per_epoch!=0),
                        validation=validation,train_panel=train_panel)
            vals.write(json.dumps(record,allow_nan=False)+'\n'); vals.flush(); records.append(record)
            print(json.dumps(record),flush=True)
            model.train(); torch.cuda.synchronize(); chunk_start=time.monotonic()
            if step>=limit: break
    parity=None
    evaluation_parity=evaluation_parity_check(model,x,y)
    if mode=='smoke':
        # Verify at nontrivial trained weights, with identical current input.
        parity=parity_check(model,active,x,y)
        torch.cuda.synchronize()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError('Nonfinite final parameters')
    verify(bundle)
    if not timed: raise RuntimeError('No steady-state throughput window')
    result=dict(status='completed',mode=mode,steps=step,records=len(records),
        initial_model_sha256=initial,final_model_sha256=model_digest(model),
        consumed_ids_sha256=sample_digest.hexdigest(),smoke_prefix_ids_sha256=smoke_digest,
        prepared_sha256=digest(bundle/'prepared.json'),parity=parity,
        evaluation_policy='eager_bf16',evaluation_parity=evaluation_parity,
        recipe=c.get('recipe','minimal'),mix_counts=strong.counts if strong else None,
        initial_validation=initial_validation,final=records[-1],
        median_window_ms=float(np.median([r['ms_per_step'] for r in timed])),
        measured_training_seconds=sum(r['seconds'] for r in timed),
        projected_training_seconds=total*np.median([r['ms_per_step'] for r in timed])/1000,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.monotonic()-began,
        checkpoints=False,official_test=False)
    write_json(out/'result.json',result)
    write_json(out/'status.json',dict(status='completed',updated_unix=time.time()))
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','smoke','train'])
    p.add_argument('--bundle',required=True)
    p.add_argument('--config')
    a=p.parse_args()
    if a.mode=='prepare':
        if not a.config: p.error('--config required for prepare')
        prepare(a.bundle,a.config)
    else: execute(a.bundle,a.mode)


if __name__=='__main__': main()
