"""Fresh two-sided SOAP OpenWebText beta study, from frozen code and inputs.

CPU preparation is separate. This runner requires a shared lifetime GPU claim.
No weights, model checkpoints, or optimizer tensors are written to disk.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

PROCESS_STARTED = time.monotonic()
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from cifar_optimizers import grouped_parameters, SplitOptimizer, apply_lr_multiplier
from fast_soap import FastSOAP
from soap_reference import ResearchSOAP
from transformer import Transformer
from tensor_ownership import TensorOwnership

PROFILES = {'stock': (.95,.95,.95), 'previous': (.9,.999,.99)}
HORIZON = 10000
BATCH = 32
PHYSICAL = 8
WARMUP = 200
RATES = (.0005,.0015,.003)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def tensor_digest(items):
    h = hashlib.sha256()
    for name, t in items:
        h.update(str((name,str(t.dtype),tuple(t.shape))).encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def token_loss(logits, labels):
    return F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]), labels.reshape(-1))


def multiplier(call):
    if call <= WARMUP:
        return call/WARMUP
    return .1+.9*.5*(1+math.cos(math.pi*(call-WARMUP)/(HORIZON-WARMUP)))


def build_optimizer(model, profile, arm, lr, compile_warm=True):
    beta1,beta2,cov = PROFILES[profile]
    groups = grouped_parameters(model,dict(matrix_lr=lr,lr=.0006,weight_decay=.05))
    matrix_groups = [g for g in groups if g['role']=='matrix']
    common = dict(lr=lr,betas=(beta1,beta2),shampoo_beta=cov,precondition_frequency=10,hard_reset_interval=50)
    if arm == 'qr':
        matrix = ResearchSOAP(matrix_groups,basis_method='qr',**common)
    elif arm == 'warm':
        matrix = FastSOAP(matrix_groups,compile_warm=compile_warm,**common)
    else:
        raise ValueError(arm)
    auxiliary = torch.optim.AdamW([g for g in groups if g['role']=='auxiliary'],
                                  betas=(.9,.999),eps=1e-8,fused=next(model.parameters()).is_cuda)
    return SplitOptimizer(matrix,auxiliary)


def state_digest(model, opt):
    def values():
        yield from model.named_parameters()
        for name,p in model.named_parameters():
            if p.grad is not None:yield 'grad.'+name,p.grad
        for part in (opt.matrix,opt.auxiliary):
            for i,state in enumerate(part.state.values()):
                for key,value in state.items():
                    for j,t in enumerate(value if isinstance(value,list) else [value]):
                        if isinstance(t,torch.Tensor):yield f'{type(part).__name__}.{i}.{key}.{j}',t
    return tensor_digest(values())


def geometry(opt, expected_age):
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        rows = []
        for group in opt.matrix.param_groups:
            for name,p in zip(group['param_names'],group['params']):
                s = opt.matrix.state[p]
                assert s['step']==expected_age
                assert all(bool(torch.isfinite(t).all()) for t in [p,s['exp_avg'],s['exp_avg_sq'],*s['GG']])
                for axis,q in enumerate(s['Q']):
                    assert q is not None and bool(torch.isfinite(q).all())
                    gram=q.T@q-torch.eye(len(q),device=q.device)
                    error=float(gram.norm())
                    assert math.isfinite(error) and error<=.05 and error/math.sqrt(len(q))<=.005,(name,axis,error)
                    rows.append(dict(parameter=name,axis=axis,dimension=len(q),absolute_gram=error))
        return rows
    finally:
        torch.set_float32_matmul_precision(previous)


def attach_momentum_checks(opt, records):
    matrix=opt.matrix
    original=matrix._research_refresh

    @torch.no_grad()
    def refresh(state, group):
        age=state['step']
        if age not in (10,50):return original(state,group)
        prior=torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision('highest')
            before=matrix._project_back_with_basis(state['exp_avg'],state['Q'],group)
            torch.set_float32_matmul_precision(prior)
            original(state,group)
            torch.set_float32_matmul_precision('highest')
            after=matrix._project_back_with_basis(state['exp_avg'],state['Q'],group)
            error=float((after-before).norm()/before.norm().clamp_min(1e-12))
            records.append(dict(age=age,shape=list(before.shape),relative_physical_m_error=error))
            assert math.isfinite(error) and error<=.01,error
        finally:
            torch.set_float32_matmul_precision(prior)
    matrix._research_refresh=refresh


def run_arm(bundle, name, profile, arm, lr, calls, seed=204):
    output=bundle/'runs'/name
    output.mkdir(parents=True,exist_ok=False)
    start=PROCESS_STARTED
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.cuda.reset_peak_memory_stats()
    model=Transformer(dim=768,depth=12,heads=12,ff_mult=2.,ff_hidden_dim=1536,
                      vocab_size=50257,max_seq_len=1024,tie_embeddings=True,
                      use_rope=True,qk_norm=True,bias=False,dropout=0.,
                      fused_qkv=True,fused_swiglu=True,gradient_checkpointing=False).cuda()
    params=sum(p.numel() for p in model.parameters())
    initial=tensor_digest(model.named_parameters())
    opt=build_optimizer(model,profile,arm,lr)
    checks=[]
    attach_momentum_checks(opt,checks)
    loss_fn=torch.compile(token_loss,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True})
    train=np.load(bundle/'train.npy',mmap_mode='r')
    val=np.load(bundle/'validation.npy',mmap_mode='r')
    assert train.shape==(HORIZON*BATCH,1024) and val.shape==(1024,1024)
    events=[]

    def emit(event, **row):
        record={'event':event,'elapsed_seconds':time.monotonic()-start,**row}
        with (output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        if event!='update' or row['call']%50==0:print(json.dumps(dict(run=name,**record),allow_nan=False),flush=True)
        return record

    def budget():
        assert time.monotonic()-start<6*3600,'Per-arm six-hour limit'
        assert torch.cuda.max_memory_allocated()<24*2**30,'24GiB allocation limit'

    def backward(call):
        model.zero_grad(set_to_none=True)
        tokens=torch.from_numpy(np.array(train[(call-1)*BATCH:call*BATCH],dtype=np.int64)).cuda()
        values=[]
        for t in tokens.split(PHYSICAL):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):
                loss=loss_fn(model(t[:,:-1]),t[:,1:])/(BATCH//PHYSICAL)
            loss.backward()
            values.append(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
        return float(torch.stack(values).sum()),float(norm)

    def evaluate(call):
        checked=call in (0,51,calls)
        before=state_digest(model,opt) if checked else None
        rng=torch.cuda.get_rng_state().clone()
        grad_ids=[id(p.grad) for p in model.parameters()]
        model.eval()
        begin=time.monotonic()
        losses=[]
        try:
            with torch.no_grad():
                for begin_row in range(0,len(val),PHYSICAL):
                    t=torch.from_numpy(np.array(val[begin_row:begin_row+PHYSICAL],dtype=np.int64)).cuda()
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):
                        value=loss_fn(model(t[:,:-1]),t[:,1:])
                    losses.append(float(value))
                    budget()
        finally:
            model.train()
        assert torch.equal(rng,torch.cuda.get_rng_state())
        assert grad_ids==[id(p.grad) for p in model.parameters()]
        if checked:assert state_digest(model,opt)==before,'Evaluation changed model/optimizer/gradients'
        result=emit('validation',call=call,ce=statistics.mean(losses),
                    panel_ce=[statistics.mean(losses[:64]),statistics.mean(losses[64:])],
                    targets=len(val)*1023,seconds=time.monotonic()-begin,state_preservation_checked=checked)
        events.append(result)
        return result

    emit('started',profile=profile,betas=PROFILES[profile],arm=arm,peak_matrix_lr=lr,
         auxiliary_betas=(.9,.999),auxiliary_peak_lr=.0006,parameter_count=params,
         initial_parameters_sha256=initial,torch=torch.__version__,cuda=torch.version.cuda,
         gpu=torch.cuda.get_device_name(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
         claim=os.environ['GPU_CLAIM_INDICES'],calls=calls,schedule_horizon=HORIZON)
    first=[]
    for _ in range(2):
        loss,norm=backward(1)
        first.append(dict(loss=loss,norm=norm,hash=tensor_digest((n,p.grad) for n,p in model.named_parameters() if p.grad is not None)))
    emit('initial_backward_repeat',observations=first)
    assert first[0]==first[1],'Initial backward is not exactly repeatable'
    emit('initial_gradient',**first[0])
    initial_val=evaluate(0)
    updates=[]
    guards=[]
    validation_calls={50,51,100,200,500,*range(1000,calls+1,500),calls}
    guard_calls={1,11,21,51,61,*range(250,calls+1,250),calls}
    for call in range(1,calls+1):
        apply_lr_multiplier(opt,multiplier(call))
        torch.cuda.synchronize()
        begin=time.monotonic()
        loss,norm=backward(call)
        torch.cuda.synchronize()
        middle=time.monotonic()
        opt.step()
        torch.cuda.synchronize()
        finish=time.monotonic()
        assert math.isfinite(loss) and math.isfinite(norm)
        age=call-1
        kind='cold' if not age else 'qr' if age%10==0 and (arm=='qr' or age%50==0) else 'warm' if age%10==0 else 'ordinary'
        diagnostic=call in (11,51) # actual-M checks are inside these steps
        updates.append(emit('update',call=call,kind=kind,loss=loss,unclipped_norm=norm,
                             fb_ms=(middle-begin)*1000,optimizer_ms=(finish-middle)*1000,
                             step_ms=(finish-begin)*1000,diagnostic=diagnostic))
        if call in guard_calls:
            rows=geometry(opt,age)
            guards.append(dict(call=call,factors=len(rows),max_absolute_gram=max(r['absolute_gram'] for r in rows)))
            emit('geometry',**guards[-1])
        if call in validation_calls:evaluate(call)
        if call==61:
            median=statistics.median(r['step_ms'] for r in updates if 22<=r['call']<=61 and not r['diagnostic'])
            assert median<=3000,('Complete-step speed gate >3s',median)
            emit('speed_gate',median_step_ms=median,passed=True)
        budget()
    clean=[r for r in updates if r['call']>=62 and not r['diagnostic']]
    # Full 50-call cycles avoid bias from an incomplete tail. Diagnostic work
    # and all setup/evaluation remain included in elapsed-time observations.
    clean=[r for r in clean if 101<=r['call']<=calls-calls%50]
    result=dict(status='completed',name=name,profile=profile,arm=arm,peak_matrix_lr=lr,seed=seed,
                calls=calls,targets=calls*BATCH*1023,initial_parameters_sha256=initial,
                initial_gradient=first[0],initial_ce=initial_val['ce'],validation=events,
                final_ce=events[-1]['ce'],parameter_count=params,geometry=guards,momentum_checks=checks,
                clean_cycle_calls=len(clean),mean_step_ms=statistics.mean(r['step_ms'] for r in clean) if clean else None,
                mean_optimizer_ms=statistics.mean(r['optimizer_ms'] for r in clean) if clean else None,
                elapsed_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    owner=TensorOwnership()
    owner.watch_tree(model.state_dict(),'model')
    owner.watch_tree(opt.state_dict(),'optimizer')
    owner.watch_tree(dict((n,p.grad) for n,p in model.named_parameters() if p.grad is not None),'gradients')
    del loss_fn,opt,model,train,val
    gc.collect()
    torch.cuda.synchronize()
    result['registered_release']=owner.report()
    owner.close()
    result['allocated_before_library_workspace_clear']=torch.cuda.memory_allocated()
    # Deterministic cuBLAS leaves a 32MiB workspace after deleting all GEMM
    # tensors on this build (separately measured in the v3 diagnostic).
    # Clear only library workspace after measured work and registered release.
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    result['released_allocated_bytes']=torch.cuda.memory_allocated()
    write(output/'cleanup_audit.json',dict(registered=result['registered_release'],
          before_library_clear=result['allocated_before_library_workspace_clear'],
          after_library_clear=result['released_allocated_bytes']))
    assert result['registered_release']['all_registered_released'],'Registered model/optimizer/gradient tensors leaked'
    assert result['released_allocated_bytes']<2**20,'Resident tensors leaked at arm release'
    write(output/'result.json',result)
    emit('completed',final_ce=result['final_ce'],elapsed_seconds=result['elapsed_seconds'])
    return result


def worker(bundle, spec):
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--bundle',str(bundle),
                           '--worker',json.dumps(spec)],check=True)
    return json.loads((bundle/'runs'/spec['name']/'result.json').read_text())


def campaign(bundle, phase):
    provenance=json.loads((bundle/'provenance.json').read_text())
    for name,record in provenance['files'].items():assert sha(bundle/name)==record['sha256'],name
    for name,h in provenance['source_files'].items():assert sha(bundle/'source'/name)==h,name
    assert Path(__file__).resolve()==bundle/'source/benchmarks/owt_restart.py','Run frozen source only'
    if phase=='qualify':
        results=[worker(bundle,dict(name=f'qualify_{profile}_{arm}',profile=profile,arm=arm,lr=.0005,calls=61))
                 for profile in PROFILES for arm in ('qr','warm')]
        assert len({r['initial_parameters_sha256'] for r in results})==1
        assert len({r['initial_gradient']['hash'] for r in results})==1
        assert len({r['initial_ce'] for r in results})==1
        write(bundle/'qualification.json',dict(status='passed',runs=results))
    elif phase=='learn':
        assert json.loads((bundle/'qualification.json').read_text())['status']=='passed'
        calibration=[]
        selected={}
        for profile in PROFILES:
            rows=[worker(bundle,dict(name=f'calibrate_{profile}_{lr}',profile=profile,arm='qr',lr=lr,calls=500)) for lr in RATES]
            calibration.extend(rows)
            selected[profile]=min(rows,key=lambda r:r['final_ce'])['peak_matrix_lr']
        write(bundle/'calibration.json',dict(status='completed',runs=calibration,selected=selected,
              limitation='Single-seed 500-call baseline calibration, using the full 10000-call schedule; shared selected LR per QR/warm pair.'))
        primary=[]
        for profile in PROFILES:
            # Reverse within-pair order between profiles to expose, rather than
            # eliminate, temporal ordering as a single-seed limitation.
            for arm in (('qr','warm') if profile=='stock' else ('warm','qr')):
                primary.append(worker(bundle,dict(name=f'primary_{profile}_{arm}',profile=profile,arm=arm,lr=selected[profile],calls=HORIZON)))
        write(bundle/'comparison.json',dict(status='completed',runs=primary,
              limitations=['One model/data seed; no statistical equivalence claim',
                           'Beta profiles also differ in beta1',
                           'Stock beta profile on a declared matrix/auxiliary split, not an entire upstream training-recipe reproduction',
                           'Shared-node timing and fixed validation samples; use measured elapsed panels']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--phase',choices=['qualify','learn'])
    p.add_argument('--worker')
    a=p.parse_args()
    assert os.environ.get('GPU_CLAIM_INDICES') and torch.cuda.device_count()==1,'Shared single-GPU claim required'
    bundle=a.bundle.resolve()
    if a.worker:
        spec=json.loads(a.worker)
        try:run_arm(bundle,**spec)
        except BaseException as exc:
            write(bundle/'runs'/spec['name']/'failure.json',dict(error=repr(exc)))
            raise
    else:
        assert a.phase
        write(bundle/f'{a.phase}_status.json',dict(status='running',pid=os.getpid(),phase=a.phase))
        try:
            campaign(bundle,a.phase)
            write(bundle/f'{a.phase}_status.json',dict(status='completed',pid=os.getpid(),phase=a.phase))
        except BaseException as exc:
            write(bundle/f'{a.phase}_status.json',dict(status='failed',pid=os.getpid(),phase=a.phase,error=repr(exc)))
            raise


if __name__=='__main__':
    main()
