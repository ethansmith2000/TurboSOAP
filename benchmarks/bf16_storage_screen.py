"""Bounded real-model storage/compilation screen; requires a lifetime GPU claim.

Frozen inputs and the 10k-call LR schedule match owt_restart. Only 201 calls and
128 validation rows are used here: these are engineering/early-learning probes.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from bf16_soap import CompiledSOAP,STATE_KEYS,matrix_step
from cifar_optimizers import grouped_parameters,SplitOptimizer,apply_lr_multiplier
from owt_restart import Transformer,build_optimizer,token_loss,multiplier,tensor_digest,sha,write,state_digest

CASES=('original','compiled_fp32','bf16','bf16_fp32_basis')
CONTROL_CADENCES={'bf16_qr20':20,'bf16_qr50':50}


def event_kind(case,call):
    age=call-1
    if not age:return 'cold'
    if case in CONTROL_CADENCES:
        return 'qr' if age%CONTROL_CADENCES[case]==0 else 'ordinary'
    if age%10:return 'ordinary'
    return 'qr' if case=='qr10' or age%50==0 else 'warm'


def build_storage_optimizer(model,case,compile_kernels=True):
    if case in ('original','qr10'):
        return build_optimizer(model,'stock','qr' if case=='qr10' else 'warm',.0015,
                               compile_warm=compile_kernels)
    if case not in (*CASES,*CONTROL_CADENCES):raise ValueError(case)
    groups=grouped_parameters(model,dict(matrix_lr=.0015,lr=.0006,weight_decay=.05))
    matrix=CompiledSOAP([g for g in groups if g['role']=='matrix'],
        state_dtype=torch.float32 if case=='compiled_fp32' else torch.bfloat16,
        basis_dtype=torch.bfloat16 if case=='bf16' or case in CONTROL_CADENCES else torch.float32,
        precondition_frequency=CONTROL_CADENCES.get(case,10),
        hard_reset_interval=CONTROL_CADENCES.get(case,50),
        compile_kernels=compile_kernels)
    auxiliary=torch.optim.AdamW([g for g in groups if g['role']=='auxiliary'],
        betas=(.9,.999),eps=1e-8,fused=next(model.parameters()).is_cuda)
    return SplitOptimizer(matrix,auxiliary)


def observation_calls(calls):
    ages=(10,20,50,100,150,200,500,510,1000,1010,1500,1510,1990)
    return {age+1 for age in ages if age+1<=calls}


@torch.no_grad()
def coordinate_residuals(matrix):
    """Observation only: offdiagonal defect in each stored basis's coordinates."""
    previous=torch.get_float32_matmul_precision();rows=[]
    torch.set_float32_matmul_precision('highest')
    try:
        for name,p,s in states(matrix):
            for axis,(q,c) in enumerate(((s['ql'],s['cl']),(s['qr'],s['cr']))):
                rotated=q.float().T@c.float()@q.float()
                denominator=rotated.norm().clamp_min(1e-30)
                rotated.diagonal().zero_()
                rows.append(dict(name=name,axis=axis,relative_coordinate_offdiagonal=float(rotated.norm()/denominator)))
    finally:torch.set_float32_matmul_precision(previous)
    return rows


def states(matrix):
    """Views of every real matrix's state, independent of optimizer layout."""
    for group in matrix.param_groups:
        if isinstance(matrix,CompiledSOAP):
            state=matrix.state[group['params'][0]]
            for i,(name,p) in enumerate(zip(group['param_names'],group['params'])):
                yield name,p,{key:state[key][i] for key in STATE_KEYS}
        else:
            for name,p in zip(group['param_names'],group['params']):
                state=matrix.state[p]
                yield name,p,dict(zip(STATE_KEYS,(state['exp_avg'],state['exp_avg_sq'],
                    *state['GG'],*state['Q'])))


def momentum_before(matrix):
    # Reconstruct the actual next stored M before its basis transport. All
    # world-space audits themselves use strict FP32, outside measured steps.
    result={}
    previous=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        for name,p,s in states(matrix):
            dtype=torch.bfloat16 if isinstance(matrix,CompiledSOAP) else torch.float32
            projected=s['ql'].to(dtype).T@p.grad.to(dtype)@s['qr'].to(dtype)
            m=(.95*s['m'].float()+.05*projected.float()).to(s['m'].dtype).float()
            result[name]=s['ql'].float()@m@s['qr'].float().T
    finally:torch.set_float32_matmul_precision(previous)
    return result


def diagnostics(matrix,before=None,rounded_floor=False):
    previous=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    rows=[];moment=[]
    try:
        for name,p,s in states(matrix):
            assert all(bool(torch.isfinite(t).all()) for t in (p,*s.values())),name
            assert bool((s['v']>=0).all()),name
            for axis,key in enumerate(('ql','qr')):
                q=s[key].float();d=q.shape[-1]
                error=float((q.T@q-torch.eye(d,device=q.device)).norm())
                row=dict(name=name,axis=axis,dimension=d,absolute_gram=error,
                         normalized_gram=error/math.sqrt(d))
                if rounded_floor:
                    reference=torch.linalg.qr(q).Q.bfloat16().float()
                    row['fresh_bf16_rounded_qr_normalized_gram']=float(
                        (reference.T@reference-torch.eye(d,device=q.device)).norm())/math.sqrt(d)
                rows.append(row)
            if before is not None:
                world=s['ql'].float()@s['m'].float()@s['qr'].float().T
                error=float((world-before[name]).norm()/before[name].norm().clamp_min(1e-12))
                moment.append(dict(name=name,relative_physical_m_error=error))
    finally:torch.set_float32_matmul_precision(previous)
    return dict(factors=rows,momentum=moment,
        max_absolute_gram=max(r['absolute_gram'] for r in rows),
        max_normalized_gram=max(r['normalized_gram'] for r in rows),
        max_physical_m_error=max((r['relative_physical_m_error'] for r in moment),default=0))


@torch.no_grad()
def cuda_parity(case):
    """Real CUDA eager/compiled equality screen, plus a same-state FP32 probe."""
    from functools import partial
    torch.manual_seed(944)
    state_dtype=torch.float32 if case=='compiled_fp32' else torch.bfloat16
    basis_dtype=torch.bfloat16 if case=='bf16' or case in CONTROL_CADENCES else torch.float32
    p=torch.randn(2,48,32,device='cuda');g=torch.randn_like(p)
    ql=torch.linalg.qr(torch.randn(2,48,48,device='cuda')).Q.to(basis_dtype).contiguous()
    qr=torch.linalg.qr(torch.randn(2,32,32,device='cuda')).Q.to(basis_dtype).contiguous()
    state=(torch.randn_like(p).to(state_dtype),torch.rand_like(p).to(state_dtype),
           torch.eye(48,device='cuda').repeat(2,1,1).to(state_dtype),
           torch.eye(32,device='cuda').repeat(2,1,1).to(state_dtype),ql,qr)
    config=dict(beta1=.95,beta2=.95,cov_beta=.95,eps=1e-8,weight_decay=.05,
                compute_dtype=torch.bfloat16,basis_dtype=basis_dtype)
    args=(p,g,state,torch.tensor(10.,device='cuda'),torch.tensor(.001,device='cuda'))
    rows=[]
    for event in ('ordinary','warm','qr'):
        fn=partial(matrix_step,event=event,**config)
        eager=fn(*args)
        compiled=torch.compile(fn,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True})
        actual=compiled(*args)
        row=dict(event=event,parameter_max_absolute=float((actual[0]-eager[0]).abs().max()),
                 state_relative_errors={key:float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
                    for key,a,b in zip(STATE_KEYS,actual[1],eager[1])})
        assert row['parameter_max_absolute']<2e-5,row
        assert max(row['state_relative_errors'].values())<.025,row
        rows.append(row)
    # This probes low precision arithmetic with identical starting state, not
    # independent training trajectories. Tiny update cancellation is avoided.
    low=matrix_step(*args,event='ordinary',**config)[0]-p
    high_config={**config,'compute_dtype':torch.float32,'basis_dtype':torch.float32}
    high=matrix_step(p,g,tuple(t.float() for t in state),args[3],args[4],
                     event='ordinary',**high_config)[0]-p
    relative=float((low-high).norm()/high.norm())
    cosine=float(torch.nn.functional.cosine_similarity(low.flatten(),high.flatten(),dim=0))
    result=dict(eager_compiled=rows,same_state_update_relative_error=relative,
                same_state_update_cosine=cosine,
                arithmetic_screen_passed=relative<=.02 and cosine>=.999)
    del compiled,actual,eager,fn,args,state,p,g,ql,qr,low,high
    torch._dynamo.reset();gc.collect();torch.cuda.empty_cache()
    return result


def run(bundle,case,calls,protocol=None):
    protocol=protocol or {}
    start=time.monotonic();output=bundle/'runs'/case
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    def emit(event_name,**values):
        row=dict(event=event_name,case=case,elapsed_seconds=time.monotonic()-start,**values)
        with (output/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if event_name!='update' or values['call']%25==0:print(json.dumps(row,allow_nan=False),flush=True)
    parity=None if case in ('original','qr10') else cuda_parity(case)
    write(output/'cuda_parity.json',parity)
    emit('cuda_parity',result=parity)
    seed=protocol.get('model_seed',204)
    torch.manual_seed(seed)
    model=Transformer(dim=768,depth=12,heads=12,ff_mult=2.,ff_hidden_dim=1536,
        vocab_size=50257,max_seq_len=1024,tie_embeddings=True,use_rope=True,
        qk_norm=True,bias=False,dropout=0.,fused_qkv=True,fused_swiglu=True,
        gradient_checkpointing=False).cuda()
    initial=tensor_digest(model.named_parameters())
    opt=build_storage_optimizer(model,case)
    loss_fn=torch.compile(token_loss,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True})
    train=np.load(bundle/'train.npy',mmap_mode='r')
    val=np.load(bundle/'validation.npy',mmap_mode='r')[:protocol.get('validation_rows',128)]
    assert train.shape==(320000,1024) and val.shape==(protocol.get('validation_rows',128),1024)
    def backward(call):
        model.zero_grad(set_to_none=True)
        tokens=torch.from_numpy(np.array(train[(call-1)*32:call*32],dtype=np.int64)).cuda()
        losses=[]
        for t in tokens.split(8):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):
                loss=loss_fn(model(t[:,:-1]),t[:,1:])/4
            loss.backward();losses.append(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
        return float(torch.stack(losses).sum()),float(norm)
    validation=[]
    def evaluate(call):
        checked=call in protocol.get('evaluation_hash_calls',())
        before=state_digest(model,opt) if checked else None
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
        grad_ids=[id(p.grad) for p in model.parameters()]
        model.eval();losses=[]
        with torch.no_grad():
            for begin in range(0,len(val),8):
                t=torch.from_numpy(np.array(val[begin:begin+8],dtype=np.int64)).cuda()
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):
                    losses.append(float(loss_fn(model(t[:,:-1]),t[:,1:])))
        model.train()
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
        assert grad_ids==[id(p.grad) for p in model.parameters()]
        if checked:assert state_digest(model,opt)==before,'Evaluation changed listed model/gradient/optimizer tensors'
        row=dict(call=call,ce=statistics.mean(losses),targets=len(val)*1023,
                 listed_tensor_hash_preservation_checked=checked)
        validation.append(row);emit('validation',**row)
        return row
    emit('started',initial_parameters_sha256=initial,calls=calls,seed=seed,torch=torch.__version__,
         cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),claim=os.environ['GPU_CLAIM_INDICES'])
    first_loss,first_norm=backward(1)
    first_gradient=tensor_digest((n,p.grad) for n,p in model.named_parameters() if p.grad is not None)
    emit('initial_gradient',loss=first_loss,norm=first_norm,sha256=first_gradient)
    if protocol.get('repeat_initial_backward',False):
        repeated_loss,repeated_norm=backward(1)
        repeated_hash=tensor_digest((n,p.grad) for n,p in model.named_parameters() if p.grad is not None)
        assert (repeated_loss,repeated_norm,repeated_hash)==(first_loss,first_norm,first_gradient)
        emit('initial_backward_repeat',passed=True)
    expected=protocol.get('expected_startup')
    if expected:
        assert initial==expected['initial_parameters_sha256']
        assert first_gradient==expected['initial_gradient_sha256']
        if 'initial_loss' in expected:assert first_loss==expected['initial_loss'] and first_norm==expected['initial_norm']
        emit('startup_match',passed=True,reference=protocol['startup_reference'])
    initial_validation=evaluate(0)
    if expected and 'initial_validation_ce' in expected:
        assert initial_validation['ce']==expected['initial_validation_ce']
        emit('initial_validation_match',passed=True)
    updates=[];guards=[]
    diagnostic_calls=set(protocol.get('diagnostic_calls',(11,51,101,151,201)))
    validation_calls=set(protocol.get('validation_calls',(50,100,calls)))
    capture_calls=set(protocol.get('capture_calls',(51,101,151,calls)))
    torch.cuda.reset_peak_memory_stats()
    from torch._dynamo.utils import counters
    counters.clear()
    for call in range(1,calls+1):
        apply_lr_multiplier(opt,multiplier(call))
        torch.cuda.synchronize();begin=time.monotonic()
        loss,norm=backward(call)
        torch.cuda.synchronize();middle=time.monotonic()
        checked=call in diagnostic_calls
        with torch.no_grad():before=momentum_before(opt.matrix) if checked else None
        torch.cuda.synchronize();opt_begin=time.monotonic()
        opt.step();torch.cuda.synchronize();finish=time.monotonic()
        assert math.isfinite(loss) and math.isfinite(norm)
        event=event_kind(case,call)
        if case in CONTROL_CADENCES:assert opt.matrix.last_event==event
        row=dict(call=call,kind=event,loss=loss,fb_ms=(middle-begin)*1000,
                 optimizer_ms=(finish-opt_begin)*1000,
                 step_ms=((middle-begin)+(finish-opt_begin))*1000,diagnostic=checked)
        updates.append(row);emit('update',**row)
        if checked or call==1:
            with torch.no_grad():guard=diagnostics(opt.matrix,before,rounded_floor=call==1)
            if call in protocol.get('residual_calls',()):guard['coordinate_residuals']=coordinate_residuals(opt.matrix)
            guards.append(dict(call=call,**guard));write(output/'geometry.json',guards)
            emit('geometry',call=call,max_absolute_gram=guard['max_absolute_gram'],
                 max_normalized_gram=guard['max_normalized_gram'],max_physical_m_error=guard['max_physical_m_error'])
        del before
        if call in validation_calls:evaluate(call)
        if call in capture_calls:
            emit('capture_counters',call=call,stats=dict(counters['stats']),graph_breaks=dict(counters['graph_break']))
        assert time.monotonic()-start<protocol.get('arm_timeout_seconds',1800),'Per-arm time budget'
        assert torch.cuda.max_memory_allocated()<24*2**30,'24GiB allocation budget'
    # Retain diagnostics because they are outside timed optimizer/FB regions.
    # Full cycles only; the learning protocol extends this to calls 101..2000.
    if protocol.get('check_final_state_finite',False):
        for p in model.parameters():assert bool(torch.isfinite(p).all())
        for optimizer in (opt.matrix,opt.auxiliary):
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value,torch.Tensor):assert bool(torch.isfinite(value).all())
        emit('final_state_finite',passed=True)
    clean=[r for r in updates if 101<=r['call']<=calls-calls%50]
    state_bytes=sum(t.numel()*t.element_size() for s in opt.matrix.state.values()
                    for value in s.values() for t in (value if isinstance(value,list) else [value])
                    if isinstance(t,torch.Tensor))
    result=dict(status='completed',case=case,calls=calls,seed=seed,initial_parameters_sha256=initial,
        initial_gradient_sha256=first_gradient,validation=validation,parity=parity,
        matrix_state_bytes=state_bytes,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        mean_optimizer_ms=statistics.mean(r['optimizer_ms'] for r in clean),
        mean_step_ms=statistics.mean(r['step_ms'] for r in clean),
        optimizer_by_kind={kind:statistics.mean(r['optimizer_ms'] for r in clean if r['kind']==kind)
                           for kind in ('ordinary','warm','qr') if any(r['kind']==kind for r in clean)},
        max_absolute_gram=max(g['max_absolute_gram'] for g in guards),
        max_normalized_gram=max(g['max_normalized_gram'] for g in guards),
        max_physical_m_error=max(g['max_physical_m_error'] for g in guards),
        original_geometry_guard_passed=all(g['max_absolute_gram']<=.05 and g['max_normalized_gram']<=.005 for g in guards),
        precision_specific_guard_passed=all(g['max_normalized_gram']<=.005 and g['max_physical_m_error']<=.01 for g in guards),
        dynamo_counters={key:dict(value) for key,value in counters.items() if key in ('stats','graph_break','inductor')},
        elapsed_seconds=time.monotonic()-start)
    result['clean_cycle_calls']=len(clean)
    result['protocol']=protocol
    if protocol.get('repeat_initial_backward',False):
        result['initial_backward_repeat_passed']=True
    write(output/'result.json',result)
    emit('completed',result=result)
    # Each arm is a fresh bounded subprocess; all CUDA state is released at
    # process exit rather than trying to clear compiler-owned graph references.
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--worker',choices=CASES);args=parser.parse_args();bundle=args.bundle
    assert os.environ.get('GPU_CLAIM_UUIDS'),'A lifetime GPU claim is required'
    assert Path(__file__).resolve()==bundle/'source/benchmarks/bf16_storage_screen.py','Frozen source required'
    manifest=json.loads((bundle/'manifest.json').read_text())
    for name,digest in manifest['source_files'].items():assert sha(bundle/'source'/name)==digest,name
    for name,digest in manifest['inputs'].items():assert sha(bundle/name)==digest,name
    if args.worker:run(bundle,args.worker,manifest['calls']);return
    results=[]
    for case in manifest['cases']:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--bundle',str(bundle),'--worker',case],
                       check=True,timeout=1800)
        results.append(json.loads((bundle/'runs'/case/'result.json').read_text()))
    assert len({r['initial_parameters_sha256'] for r in results})==1
    assert len({r['initial_gradient_sha256'] for r in results})==1
    write(bundle/'comparison.json',dict(status='completed',results=results))


if __name__=='__main__':main()
