"""Natural rectangular trajectories with same-state momentum/precision oracles."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from soap import SOAP
from soap_reference import ResearchSOAP
from benchmark_synthetic_scaling import MockBlock, inspect_state, StateGuardError, tensor_bytes
from benchmark_large_precision import warm_six


def variants():
    base = dict(method='warm', frequency=10, iterations=6, cap='average', precision='high',
                covariance='float32', bf16_gauge=False, compiled=False)
    return [dict(base, **{'name': name, **overrides}) for name, overrides in [
        ('qr10', dict(method='qr')), ('qr20', dict(method='qr', frequency=20)),
        ('warm2', dict(iterations=2)), ('warm6', {}),
        ('warm6_fp32', dict(precision='highest')),
        ('warm6_bf16_cov', dict(covariance='bfloat16')),
        ('warm6_bf16_gauge', dict(bf16_gauge=True)),
        ('warm6_compile', dict(compiled=True)),
        ('warm2_spectral_cap', dict(iterations=2, cap='spectral_bound'))]]


def project(x, basis):
    left, right = basis
    if left is not None: x = left.T @ x
    if right is not None: x = x @ right
    return x


def world(x, basis):
    left, right = basis
    if left is not None: x = left @ x
    if right is not None: x = x @ right.T
    return x


def relative(actual, reference):
    return float((actual-reference).norm()/reference.norm().clamp_min(1e-30))


def cosine(actual, reference):
    return float((actual.flatten() @ reference.flatten()) /
                 (actual.norm()*reference.norm()).clamp_min(1e-30))


def factor_metrics(cov, before, after):
    def residual(q):
        b = q.T @ cov @ q
        return float((b-torch.diag_embed(b.diagonal())).norm()/b.norm().clamp_min(1e-30))
    return dict(dimension=after.shape[0], old_residual=residual(before), new_residual=residual(after),
        basis_change_rms=float((after-before).norm()/after.shape[0]**.5),
        gram_error=float((after.T@after-torch.eye(after.shape[0],device=after.device)).norm()/after.shape[0]**.5))


class ObservedSOAP(ResearchSOAP):
    """Retain references for diagnostics after the timed optimizer step."""
    def _research_refresh(self, state, group):
        old = dict(Q=state['Q'], m=state['exp_avg'], cov=state['GG'])
        super()._research_refresh(state, group)
        self.pending.append((state, group, old))


@torch.no_grad()
def diagnose(opt, world_ema):
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    rows = []
    try:
        for state, group, old in opt.pending:
            new = state['Q']; old_world = world(old['m'], old['Q'])
            actual_world = world(state['exp_avg'], new)
            strict_m = project(old_world, new)
            row = dict(shape=list(state['exp_avg'].shape),
                transport_world_relative_error=relative(actual_world, old_world),
                transport_compute_relative_error=relative(state['exp_avg'], strict_m),
                factors=[factor_metrics(c, q, n) for c, q, n in zip(old['cov'], old['Q'], new) if q is not None])
            reset = group['basis_method']=='qr' or state['step']%200==0
            if not reset:
                # SAME covariance, incoming Q, m and carried v: isolate gauge and
                # transport precision, not the different trajectories or v policies.
                reference_q = [SOAP._gauge_step_one(c, q, group) if q is not None else None
                               for c, q in zip(old['cov'], old['Q'])]
                reference_m = project(old_world, reference_q)
                denominator = state['exp_avg_sq'].sqrt().add(group['eps'])
                actual_update = world(state['exp_avg']/denominator, new)
                reference_update = world(reference_m/denominator, reference_q)
                row.update(update_shadow_relative_error=relative(actual_update, reference_update),
                    update_shadow_cosine=cosine(actual_update, reference_update),
                    shadow_maximum_gram_error=max(float((q.T@q-torch.eye(q.shape[0],device=q.device)).norm()/q.shape[0]**.5)
                                                  for q in reference_q if q is not None))
            rows.append(row)
        momentum = [dict(shape=list(p.shape), relative_error=relative(world(opt.state[p]['exp_avg'],opt.state[p]['Q']), ref))
                    for p, ref in world_ema.items()]
    finally:
        torch.set_float32_matmul_precision(previous)
        opt.pending.clear()
    return dict(refreshes=rows, global_world_momentum=momentum)


def make_optimizer(parameters, arm, mode):
    opt=ObservedSOAP(parameters,lr=.0005,betas=(.9,.999),shampoo_beta=arm.get('covariance_beta',.999),
        basis_method=arm['method'],precondition_mode=mode,precondition_frequency=arm['frequency'],
        covariance_compute_dtype=arm['covariance'],hard_reset_interval=200)
    opt.pending=[]
    for group in opt.param_groups:
        group.update(basis_ns_iterations=arm['iterations'],basis_rotation_cap_mode=arm['cap'])
    return opt


def measure(arm, mode, repeat, config, budget):
    torch.set_float32_matmul_precision(arm['precision'])
    torch.manual_seed(config['seed']+config['width']); model=MockBlock(config['width']).cuda()
    generator=torch.Generator(device='cuda').manual_seed(config['seed']+config['tokens'])
    batches=[(torch.randn(config['tokens'],config['width'],device='cuda',generator=generator),
              torch.randn(config['tokens'],config['width'],device='cuda',generator=generator)) for _ in range(4)]
    opt=make_optimizer(model.parameters(),arm,mode)
    if arm['bf16_gauge']:
        def bf16(cov,q,group):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return SOAP._gauge_step_one(cov,q,group).float()
        opt._gauge_step_one=bf16
    world_ema={p:torch.zeros_like(p) for p in model.parameters()}
    result=dict(arm=arm,mode=mode,repeat=repeat,status='running',samples=[],observations=[],
        compile_preflight=[],failure=None,
        effective_optimizer_group={k:v for k,v in opt.param_groups[0].items() if k!='params'})

    def backward(index):
        opt.zero_grad(set_to_none=True); x,target=batches[index%4]
        with torch.autocast('cuda',dtype=torch.bfloat16): prediction=model(x)
        loss=F.mse_loss(prediction.float(),target); loss.backward()
        return loss

    for i in range(3): backward(i)
    peak=0
    for step in range(config['updates']+1):
        budget(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start,middle,end=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        began=time.perf_counter(); start.record(); loss=backward(step); middle.record()
        opt.step(); end.record(); end.synchronize()
        wall_ms=1000*(time.perf_counter()-began); peak=max(peak,torch.cuda.max_memory_allocated())
        event=('initialization' if step==0 else 'ordinary' if not opt.last_basis_refreshes else
               'qr_refresh' if arm['method']=='qr' else 'qr_reset' if opt.last_hard_reset_events else 'warm_refresh')
        expected_refresh=3*int(step>0 and step%arm['frequency']==0)
        expected_reset=expected_refresh*int(arm['method']=='qr' or step%200==0)
        if (opt.last_basis_refreshes,opt.last_hard_reset_events)!=(expected_refresh,expected_reset):
            raise RuntimeError('Incorrect natural event dispatch')
        result['samples'].append(dict(step=step,event=event,loss_sanity_only=float(loss.detach()),
            model_cuda_ms=start.elapsed_time(middle),optimizer_cuda_ms=middle.elapsed_time(end),
            step_cuda_ms=start.elapsed_time(end),step_wall_ms=wall_ms))
        if not math.isfinite(result['samples'][-1]['loss_sanity_only']):
            result.update(status='guard_rejected',failure=dict(step=step,reason='nonfinite_loss')); break
        with torch.no_grad():
            if step:
                for p,ref in world_ema.items(): ref.mul_(.9).add_(p.grad,alpha=.1)
        if step==0 or opt.last_basis_refreshes or step==config['updates']:
            failure=None
            try: geometry=inspect_state(opt,mode)
            except StateGuardError as error: geometry=error.diagnostics; failure=error
            diagnostics=diagnose(opt,world_ema)
            result['observations'].append(dict(step=step,event=event,geometry=geometry,**diagnostics))
            if failure:
                result.update(status='guard_rejected',failure=dict(step=step,reason='state_guard')); break
        if step==0 and arm['compiled']:
            compiled=torch.compile(warm_six,fullgraph=True,dynamic=False)
            seen=set()
            for state in opt.state.values():
                for cov,q in zip(state['GG'],state['Q']):
                    if q is None or (q.shape,q.stride()) in seen: continue
                    seen.add((q.shape,q.stride())); budget(); torch.cuda.synchronize(); began=time.perf_counter()
                    with torch.no_grad(): expected=warm_six(cov,q); actual=compiled(cov,q)
                    torch.cuda.synchronize(); error=relative(actual,expected)
                    result['compile_preflight'].append(dict(dimension=q.shape[0],wall_seconds=time.perf_counter()-began,
                                                            relative_output_error=error))
                    if not math.isfinite(error) or error>.01: raise RuntimeError('Compile preflight failed')
                    del expected,actual
            opt._gauge_step_one=lambda cov,q,group:compiled(cov,q)
    if result['status']=='running': result['status']='valid'
    flat=[r for o in result['observations'] for r in o['refreshes']]
    def maximum(key): return max((r[key] for r in flat if key in r),default=0.)
    errors=dict(maximum_gram_error=max(o['geometry']['maximum_gram_error'] for o in result['observations']),
        maximum_global_momentum_error=max(r['relative_error'] for o in result['observations'] for r in o['global_world_momentum']),
        maximum_transport_world_error=maximum('transport_world_relative_error'),
        maximum_transport_compute_error=maximum('transport_compute_relative_error'),
        maximum_shadow_update_error=maximum('update_shadow_relative_error'),
        minimum_shadow_update_cosine=min((r['update_shadow_cosine'] for r in flat if 'update_shadow_cosine' in r),default=1.),
        maximum_shadow_gram_error=maximum('shadow_maximum_gram_error'))
    result.update(errors=errors,state_bytes=tensor_bytes(opt.state),peak_allocated_bytes_with_oracle=peak)
    gates=config['diagnostic_budgets']
    result['diagnostic_budget_passed']=(result['status']=='valid' and
        all(math.isfinite(v) for v in errors.values()) and
        errors['maximum_global_momentum_error']<=gates['momentum_relative'] and
        errors['maximum_transport_world_error']<=gates['momentum_relative'] and
        errors['maximum_shadow_update_error']<=gates['update_relative'] and
        errors['minimum_shadow_update_cosine']>=gates['update_cosine'] and
        errors['maximum_shadow_gram_error']<.05)
    result['timing']=None
    if result['status']=='valid':
        samples=result['samples'][1:]
        result['timing']=dict(mean_ms={k:statistics.mean(s[k] for s in samples) for k in
                             ['model_cuda_ms','optimizer_cuda_ms','step_cuda_ms','step_wall_ms']},
            event_counts=dict(Counter(s['event'] for s in samples)),
            event_optimizer_ms={event:statistics.mean(s['optimizer_cuda_ms'] for s in samples if s['event']==event)
                                for event in set(s['event'] for s in samples)})
    del opt,model,batches,world_ema
    gc.collect(); torch.cuda.empty_cache()
    return result


def prepare(bundle):
    if bundle.exists(): raise ValueError('Use a new bundle')
    if shutil.disk_usage(bundle.parent).free<5*2**30: raise RuntimeError('Require 5 GiB free')
    config=dict(width=768,tokens=64,seed=20261002,updates=200,repeats=2,modes=['all','smaller_side'],
        arms=variants(),maximum_seconds=900,checkpoint=False,artifact_budget_bytes=20*2**20,
        diagnostic_budgets=dict(momentum_relative=.01,update_relative=.01,update_cosine=.999),
        budget_scope='Provisional numerical screen, not a loss-equivalence guarantee')
    (bundle/'source/benchmarks').mkdir(parents=True)
    for name in ['soap.py','soap_reference.py','benchmarks/benchmark_synthetic_scaling.py',
                 'benchmarks/benchmark_large_precision.py','benchmarks/rectangular_precision_gate.py']:
        shutil.copy2(HERE.parent/name,bundle/'source'/name)
    shutil.copy2(HERE/'RECTANGULAR_PRECISION.md',bundle/'PROTOCOL.md')
    (bundle/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (bundle/'manifest.json').write_text(json.dumps({str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p in bundle.rglob('*') if p.is_file()},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true'); args=parser.parse_args(); bundle=args.bundle.resolve()
    if args.prepare: prepare(bundle); print(bundle); return
    if not os.environ.get('GPU_CLAIM_INDICES') or torch.cuda.device_count()!=1: raise RuntimeError('One lifetime GPU claim required')
    for name,digest in json.loads((bundle/'manifest.json').read_text()).items():
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=digest: raise RuntimeError('Changed input: '+name)
    config=json.loads((bundle/'config.json').read_text()); torch.set_num_threads(4)
    began=time.monotonic()
    def budget():
        if time.monotonic()-began>config['maximum_seconds']: raise RuntimeError('Declared budget exceeded')
    report=dict(status='running',config=config,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],results=[])
    def save():
        path=bundle/'results.tmp';path.write_text(json.dumps(report,indent=2)+'\n');path.replace(bundle/'results.json')
    save()
    for mode in config['modes']:
        for repeat in range(config['repeats']):
            for arm in config['arms'][::1 if repeat%2==0 else -1]:
                row=measure(arm,mode,repeat,config,budget); report['results'].append(row); save()
                print(json.dumps(dict(arm=arm['name'],mode=mode,repeat=repeat,status=row['status'],
                    failure=row['failure'],errors=row['errors'],diagnostic_budget_passed=row['diagnostic_budget_passed'],
                    timing=row['timing'])),flush=True)
    report.update(status='completed',all_in_seconds=time.monotonic()-began); save()
    print('COMPLETE',report['all_in_seconds'],flush=True)


if __name__=='__main__':main()
