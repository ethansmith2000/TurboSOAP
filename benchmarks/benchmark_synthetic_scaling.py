"""Short synthetic trainer: measured event latencies, estimated cadence costs.

Forced optimizer clocks sample each event; these are not learning trajectories
or observed long-run schedule throughputs. No weights/checkpoints are saved.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from soap_reference import ResearchSOAP


class MockBlock(nn.Module):
    """Square projection plus 4x expansion/contraction; no attention or dataset."""
    def __init__(self,width):
        super().__init__()
        self.square=nn.Linear(width,width,bias=False)
        self.up=nn.Linear(width,4*width,bias=False)
        self.down=nn.Linear(4*width,width,bias=False)

    def forward(self,x):
        return self.down(F.gelu(self.up(F.gelu(self.square(x)))))


def event_fractions(method,frequency,reset=200):
    if frequency<1 or reset<frequency or reset%frequency:
        raise ValueError('Reset must be a positive multiple of refresh frequency')
    if method=='qr':return {'ordinary':1-1/frequency,'qr_refresh':1/frequency}
    if method=='warm':return {'ordinary':1-1/frequency,'warm_refresh':1/frequency-1/reset,'qr_reset':1/reset}
    raise ValueError('Unknown method')


def force_event(optimizer,event):
    # step() increments before refresh selection. Bias-correction age is also
    # forced; event timing cannot be interpreted as a valid learning history.
    before={'ordinary':200,'qr_refresh':209,'warm_refresh':209,'qr_reset':399}[event]
    for state in optimizer.state.values():state['step']=before


def tensor_bytes(value,seen=None):
    seen=set() if seen is None else seen
    if isinstance(value,torch.Tensor):
        storage=value.untyped_storage();key=(str(value.device),storage.data_ptr())
        if key in seen:return 0
        seen.add(key);return storage.nbytes()
    if isinstance(value,dict):return sum(tensor_bytes(v,seen) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(tensor_bytes(v,seen) for v in value)
    return 0


class StateGuardError(FloatingPointError):
    def __init__(self,diagnostics):
        self.diagnostics=diagnostics
        super().__init__('Numerical state guard: '+json.dumps(diagnostics))


def inspect_state(optimizer,mode):
    factor_dims=[];gram=[];finite=True
    prior=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
    try:
        for parameter,state in optimizer.state.items():
            finite=finite and all(bool(torch.isfinite(v).all()) for v in
                                  [parameter,state['exp_avg'],state['exp_avg_sq']])
            dimensions=[];factor_errors=[]
            for q in state['Q']:
                if q is None:continue
                dimensions.append(q.shape[0])
                error=float((q.T@q-torch.eye(q.shape[0],device=q.device)).norm()/q.shape[0]**.5)
                gram.append(error);factor_errors.append(error)
            factor_dims.append(dict(parameter_shape=list(parameter.shape),dimensions=dimensions,gram_errors=factor_errors))
            expected=list(parameter.shape) if mode=='all' else [min(parameter.shape)]
            if dimensions!=expected:raise RuntimeError('A requested factor was omitted')
    finally:
        torch.set_float32_matmul_precision(prior)
    diagnostics=dict(active_factors=factor_dims,maximum_gram_error=max(gram),finite=finite)
    if not finite or not all(math.isfinite(x) and x<.05 for x in gram):
        raise StateGuardError(diagnostics)
    return diagnostics


def measure_case(width,tokens,mode,method,round_index,config,check_budget):
    torch.manual_seed(config['seed']+width)
    model=MockBlock(width).cuda().train()
    generator=torch.Generator(device='cuda').manual_seed(config['seed']+tokens)
    batches=[(torch.randn(tokens,width,device='cuda',generator=generator),
              torch.randn(tokens,width,device='cuda',generator=generator)) for _ in range(4)]
    opt=ResearchSOAP(model.parameters(),lr=config['lr'],betas=tuple(config['adam_betas']),
        shampoo_beta=config['covariance_beta'],weight_decay=config['weight_decay'],eps=config['eps'],
        precondition_mode=mode,basis_method=method,precondition_frequency=10,
        hard_reset_interval=200,variance_policy=config['variance_policy'],
        covariance_compute_dtype='float32')

    def backward(index):
        opt.zero_grad(set_to_none=True);x,target=batches[index%len(batches)]
        with torch.autocast('cuda',dtype=torch.bfloat16):output=model(x)
        loss=F.mse_loss(output.float(),target);loss.backward()
        return loss

    # Warm the model path; discard gradients before creating any optimizer state.
    for index in range(3):backward(index)
    opt.zero_grad(set_to_none=True);torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats();before_state=torch.cuda.memory_allocated()

    def step(index,event=None):
        check_budget()
        if event:force_event(opt,event)
        start,middle,end=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        torch.cuda.synchronize();began=time.perf_counter();start.record()
        loss=backward(index);middle.record();opt.step();end.record();end.synchronize()
        wall=(time.perf_counter()-began)*1000
        value=float(loss.detach())
        if not math.isfinite(value):raise FloatingPointError('Nonfinite synthetic loss')
        n=len(list(model.parameters()))
        expected_refresh=0 if event in (None,'ordinary') else n
        expected_reset=n if event in ('qr_refresh','qr_reset') else 0
        if opt.last_basis_refreshes!=expected_refresh or opt.last_hard_reset_events!=expected_reset:
            raise RuntimeError(f'Wrong event dispatch: {event}')
        return dict(model_backward_cuda_ms=start.elapsed_time(middle),
                    optimizer_cuda_ms=middle.elapsed_time(end),
                    whole_step_cuda_ms=start.elapsed_time(end),whole_step_wall_ms=wall,
                    loss_sanity_only=value,refreshes=opt.last_basis_refreshes,
                    resets=opt.last_hard_reset_events)

    cold=step(0)
    peak=torch.cuda.max_memory_allocated()
    failure=None;geometry={}
    try:geometry['initialization']=inspect_state(opt,mode)
    except StateGuardError as error:
        geometry['initialization']=error.diagnostics;failure={'event':'initialization',**error.diagnostics}
    torch.cuda.reset_peak_memory_stats()
    events={}
    for event in event_fractions(method,10):
        if failure:break
        for index in range(config['discarded_event_warmups']):step(index,event)
        events[event]=[step(index,event) for index in range(config['samples_per_event'])]
        peak=max(peak,torch.cuda.max_memory_allocated())
        # Check the warm state before the subsequent QR reset can repair it.
        try:geometry[event]=inspect_state(opt,mode)
        except StateGuardError as error:
            geometry[event]=error.diagnostics;failure={'event':event,**error.diagnostics}
        torch.cuda.reset_peak_memory_stats()
    state_bytes=tensor_bytes(opt.state)
    result=dict(width=width,tokens=tokens,mode=mode,method=method,round=round_index,
        parameter_count=sum(p.numel() for p in model.parameters()),active_factors=geometry['initialization']['active_factors'],
        cold_state_initialization=cold,events=events,state_bytes=state_bytes,
        pre_state_allocated_bytes=before_state,peak_allocated_bytes=peak,
        peak_extra_over_pre_state_bytes=max(0,peak-before_state),
        event_geometry=geometry,maximum_gram_error=max(g['maximum_gram_error'] for g in geometry.values()),
        finite=all(g['finite'] for g in geometry.values()),status='guard_rejected' if failure else 'valid',failure=failure)
    result['effective_optimizer_group']={k:v for k,v in opt.param_groups[0].items() if k!='params'}
    del opt,model,batches
    gc.collect();torch.cuda.empty_cache()
    return result


def summarize(report):
    metrics=['model_backward_cuda_ms','optimizer_cuda_ms','whole_step_cuda_ms','whole_step_wall_ms']
    rows=[]
    for width in report['config']['widths']:
        for tokens in report['config']['tokens']:
            for mode in report['config']['modes']:
                group=[r for r in report['results'] if (r['width'],r['tokens'],r['mode'])==(width,tokens,mode)]
                for method,frequency,label in [('qr',10,'qr10'),('qr',20,'qr20'),('warm',10,'warm10_qr200')]:
                    rounds=[r for r in group if r['method']==method]
                    if len(rounds)!=report['config']['rounds']:raise ValueError('Incomplete rounds')
                    if any(r['status']!='valid' for r in rounds):continue
                    fractions=event_fractions(method,frequency)
                    estimates=[]
                    for r in rounds:
                        estimates.append({metric:sum(weight*statistics.mean(x[metric] for x in r['events'][event])
                                                   for event,weight in fractions.items()) for metric in metrics})
                    rows.append(dict(width=width,tokens=tokens,mode=mode,policy=label,
                        active_factors=rounds[0]['active_factors'],parameter_count=rounds[0]['parameter_count'],
                        event_fractions=fractions,estimated_cadence_ms={k:statistics.median(e[k] for e in estimates) for k in metrics},
                        round_estimates_ms=estimates,
                        initialization_optimizer_ms=[r['cold_state_initialization']['optimizer_cuda_ms'] for r in rounds],
                        state_bytes=rounds[0]['state_bytes'],peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rounds),
                        maximum_gram_error=max(r['maximum_gram_error'] for r in rounds)))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args();b=args.bundle.resolve();c=json.loads((b/'config.json').read_text())
    if not os.environ.get('GPU_CLAIM_INDICES') or torch.cuda.device_count()!=1:
        raise RuntimeError('Use exactly one lifetime GPU claim')
    manifest=json.loads((b/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((b/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Changed input: {name}')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    started=time.monotonic()
    def budget():
        if time.monotonic()-started>c['maximum_seconds']:raise RuntimeError('Declared budget exceeded')
    report=dict(status='running',config=c,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],results=[],
        execution='Eager BF16 mock model; FP32 parameters/optimizer/covariance products, TF32 permitted.',
        limitations=['Synthetic single MLP block, no attention, data pipeline, compiler or distributed execution.',
            'Forced event clocks are a timing probe, not valid optimizer history or a learning experiment.',
            'Cadence-weighted event means are estimates, not measured long-run schedule throughput.',
            'QR20 estimate reuses QR event costs; numerical histories and equal-quality schedules are not matched.',
            'Initialization includes covariance allocation/accumulation and eigh; library/model warmup is separate.',
            'CUDA elapsed intervals include host dispatch gaps; each sample synchronizes.',
            'No quality or target-model speedup claim; no pooling with older refresh/CIFAR timings.'])
    def save():
        temporary=b/'results.tmp';temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(b/'results.json')
    save()
    for width in c['widths']:
        for tokens in c['tokens']:
            for mode in c['modes']:
                for round_index in range(c['rounds']):
                    for method in (['qr','warm'] if round_index%2==0 else ['warm','qr']):
                        result=measure_case(width,tokens,mode,method,round_index,c,budget)
                        report['results'].append(result);save()
                        print(json.dumps({k:result[k] for k in ['width','tokens','mode','method','round','status','state_bytes','maximum_gram_error','failure']}),flush=True)
    report.update(status='completed_with_rejections' if any(r['status']!='valid' for r in report['results']) else 'completed',
                  summary=summarize(report),all_in_seconds=time.monotonic()-started)
    save();print('COMPLETE',report['all_in_seconds'],flush=True)


if __name__=='__main__':main()
