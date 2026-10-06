"""Bounded local attribution and closed-loop synthetic retraction screen."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import torch
from torch.nn import functional as F
from soap import SOAP
from soap_reference import ResearchSOAP
from warm_retraction import VARIANTS, Retraction, candidate, retract, warm
from benchmark_synthetic_scaling import MockBlock
from cifar_tracking_probe import project, world, relative


class Observed(ResearchSOAP):
    def __init__(self, params, variant, beta, **kwargs):
        super().__init__(params, lr=.0005, betas=(.9, .999), shampoo_beta=beta,
                         weight_decay=.05, precondition_frequency=20,
                         hard_reset_interval=200, covariance_compute_dtype='float32',
                         transport_precision='highest', **kwargs)
        self.variant = variant
        self.pending = []
        for group in self.param_groups:
            group['basis_ns_iterations'] = variant.iterations if variant else 6
            if variant:
                group.update(basis_rotation_cap_mode=variant.cap_mode,basis_rotation_cap=variant.cap)

    def _gauge_step_one(self, cov, q, group):
        return warm(cov, q, group, self.variant)

    def _research_refresh(self, state, group):
        old = dict(Q=state['Q'], m=state['exp_avg'], v=state['exp_avg_sq'], cov=state['GG'])
        super()._research_refresh(state, group)
        self.pending.append((state, group, old))


def gram_error(q):
    return float((q.T@q-torch.eye(q.shape[0], device=q.device)).norm()/q.shape[0]**.5)


def residual(cov, q):
    b = q.T@cov@q
    return float((b-torch.diag_embed(b.diagonal())).norm()/b.norm().clamp_min(1e-30))


def spectral_summary(gram):
    eigenvalues = torch.linalg.eigvalsh((gram+gram.T)*.5)
    return dict(minimum_gram_eigenvalue=float(eigenvalues[0]),
                maximum_gram_eigenvalue=float(eigenvalues[-1]),
                spectral_gram_error=float((eigenvalues-1).abs().max()))


@torch.no_grad()
def factor_attribution(cov, q, group):
    # Construct exactly the high-precision-setting candidate/scale used by the
    # timed algorithm, then evaluate its geometry under highest precision.
    torch.set_float32_matmul_precision('high')
    x, a, raw = candidate(cov, q, group)
    h = x.T@x
    scale = (1.25/h.abs().sum(-1).amax().clamp_min(1e-12).sqrt()).clamp(max=1.)
    ys = {}
    for order in (3, 5):
        for scaled in (False, True):
            for iterations in (1, 2, 6):
                label = f'order{order}_scaled{int(scaled)}_n{iterations}'
                ys[label] = retract(x, Retraction(label, order, iterations, scaled))
    raw_x, _, _ = candidate(cov, q, group, False)
    torch.set_float32_matmul_precision('highest')
    try:
        return dict(dimension=q.shape[0], scale=float(scale),
                    raw_rotation_rms=float(raw.norm()/q.shape[0]**.5),
                    clipped_rotation_rms=float(a.norm()/q.shape[0]**.5),
                    rotation_spectral_norm_squared=float(torch.linalg.eigvalsh(a.T@a)[-1]),
                    incoming_gram=gram_error(q), candidate_gram=gram_error(x),
                    candidate_spectrum=spectral_summary(x.T@x),
                    scaled_spectrum=spectral_summary((x.T@x)*scale.square()),
                    rawcov_candidate_relative=relative(raw_x, x),
                    curves={k:gram_error(y) for k,y in ys.items()})
    finally:
        torch.set_float32_matmul_precision('high')


@torch.no_grad()
def diagnostics(opt, ema, step, attribution):
    torch.set_float32_matmul_precision('highest')
    try:
        rows = []
        finite = True
        maximum = 0.
        for p,state in opt.state.items():
            finite = finite and all(bool(torch.isfinite(x).all()) for x in
                                   (p,state['exp_avg'],state['exp_avg_sq'],*state['Q']))
            gs = [gram_error(q) for q in state['Q']]
            maximum = max(maximum, *gs)
            rows.append(dict(shape=list(p.shape), gram=gs,
                residual=[residual(c,q) for c,q in zip(state['GG'],state['Q'])],
                global_world_m_error=relative(world(state['exp_avg'],state['Q']),ema[p])))
        refresh = []
        for state,group,old in opt.pending:
            qs=state['Q'];old_m=world(old['m'],old['Q'])
            record=dict(shape=list(state['exp_avg'].shape),
                world_m_transport_error=relative(world(state['exp_avg'],qs),old_m),
                movement_rms=[float((a-b).norm()/a.shape[0]**.5) for a,b in zip(qs,old['Q'])],
                residual_before=[residual(c,q) for c,q in zip(old['cov'],old['Q'])],
                residual_after=[residual(c,q) for c,q in zip(old['cov'],qs)])
            if attribution and step in (20,180,220,420):
                # Largest factor of the expansion matrix: real model trajectory,
                # including early, late, after-reset and second-reset states.
                if old['m'].shape[0] == 4*old['m'].shape[1]:
                    record['factor_attribution']=factor_attribution(old['cov'][0],old['Q'][0],group)
                    torch.set_float32_matmul_precision('highest')
            refresh.append(record)
        return dict(finite=finite,maximum_gram=maximum,matrices=rows,refreshes=refresh)
    finally:
        opt.pending.clear()
        torch.set_float32_matmul_precision('high')


def run_case(width, variant, beta, updates, budget, attribution=False, repeat=0):
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(20261005+width)
    model=MockBlock(width).cuda().train()
    gen=torch.Generator(device='cuda').manual_seed(20261005+width+64)
    batches=[(torch.randn(64,width,device='cuda',generator=gen),
              torch.randn(64,width,device='cuda',generator=gen)) for _ in range(4)]
    opt=Observed(model.parameters(), variant, beta, basis_method='warm' if variant else 'qr')
    ema={p:torch.zeros_like(p) for p in model.parameters()}
    result=dict(width=width,beta=beta,variant=asdict(variant) if variant else {'name':'qr20'},
                repeat=repeat,status='running',steps=[],observations=[],diagnostic_seconds=0.)
    def backward(step):
        opt.zero_grad(set_to_none=True)
        x,y=batches[step%4]
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x)
        loss=F.mse_loss(out.float(),y);loss.backward();return loss
    for step in range(3):backward(step)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    for step in range(updates+1):
        budget()
        start,middle,end=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        began=time.perf_counter();start.record();loss=backward(step);middle.record()
        opt.step();end.record();end.synchronize()
        wall_ms=1000*(time.perf_counter()-began)
        event=('init' if step==0 else 'ordinary' if step%20 else
               'qr' if variant is None or step%200==0 else 'warm')
        assert opt.last_basis_refreshes == 3*int(step>0 and step%20==0)
        result['steps'].append(dict(step=step,event=event,loss=float(loss.detach()),
            optimizer_ms=middle.elapsed_time(end),step_cuda_ms=start.elapsed_time(end),step_wall_ms=wall_ms))
        with torch.no_grad():
            if step:
                for p,m in ema.items():m.mul_(.9).add_(p.grad,alpha=.1)
        if step==0 or step%20==0:
            began=time.perf_counter();d=diagnostics(opt,ema,step,attribution)
            torch.cuda.synchronize();result['diagnostic_seconds']+=time.perf_counter()-began
            result['observations'].append(dict(step=step,event=event,**d))
            if not d['finite'] or d['maximum_gram']>=.05 or not math.isfinite(float(loss)):
                result.update(status='guard_rejected',failed_step=step);break
    if result['status']=='running':result['status']='completed'
    values=[x for x in result['steps'] if x['step']>0]
    result['timing']=dict(optimizer_ms=statistics.mean(x['optimizer_ms'] for x in values),
                         step_wall_ms=statistics.mean(x['step_wall_ms'] for x in values),
                         events={e:statistics.mean(x['optimizer_ms'] for x in values if x['event']==e)
                                 for e in set(x['event'] for x in values)})
    result['peak_bytes_including_diagnostics']=torch.cuda.max_memory_allocated()
    result['model_sha256']=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    del opt,model,batches,ema
    gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args();b=args.bundle
    assert os.environ.get('GPU_CLAIM_INDICES') and torch.cuda.device_count()==1
    manifest=json.loads((b/'manifest.json').read_text())
    for rel,h in manifest['source_sha256'].items():
        assert hashlib.sha256((b/'source'/rel).read_bytes()).hexdigest()==h,rel
    c=json.loads((b/'config.json').read_text())
    assert hashlib.sha256((b/'config.json').read_bytes()).hexdigest()==manifest['config_sha256']
    torch.set_num_threads(4);started=time.monotonic()
    def budget():
        if time.monotonic()-started>c['wall_budget_seconds']:raise TimeoutError('Bounded diagnostic budget exceeded')
    report=dict(status='running',config=c,torch=torch.__version__,cuda=torch.version.cuda,
                gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],cases=[])
    def serializable(value):
        if isinstance(value,float) and not math.isfinite(value):return str(value)
        if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
        if isinstance(value,list):return [serializable(v) for v in value]
        return value
    def save():
        report['wall_seconds']=time.monotonic()-started
        temp=b/'results.tmp';temp.write_text(json.dumps(serializable(report),indent=2,allow_nan=False)+'\n');temp.replace(b/'results.json')
    for beta in c.get('betas',[c.get('beta',.999)]):
        for width in c['widths']:
            for name in c['variants']:
                variant=None if name=='qr20' else next(v for v in VARIANTS if v.name==name)
                row=run_case(width,variant,beta,c['updates'],budget,
                             attribution=c.get('attribution',True) and name=='scaled_ns6')
                report['cases'].append(row);save()
                print('CASE '+json.dumps({k:row[k] for k in ['width','beta','variant','status','timing']}),flush=True)
    # Unchanged repeats validate state and elapsed variability without diagnostics.
    for name in c.get('repeat_variants',['scaled_ns6','local_quintic1']):
        variant=next(v for v in VARIANTS if v.name==name)
        row=run_case(c['widths'][-1],variant,c.get('betas',[c.get('beta',.999)])[-1],c['updates'],budget,repeat=1)
        report['cases'].append(row);save()
        print('REPEAT '+json.dumps(dict(name=name,status=row['status'],timing=row['timing'])),flush=True)
    report['status']='completed';save();print('COMPLETE',flush=True)


if __name__=='__main__':main()
