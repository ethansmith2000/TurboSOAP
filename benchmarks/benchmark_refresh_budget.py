"""Measure dense SOAP refresh components including moment movement.

Fixed-state eager CUDA microbenchmark, not a standard-SOAP or training comparison.
QR includes this repository's alignment policy; report that overhead separately.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from soap import SOAP


def measure(fn, repeats=5):
    for _ in range(2):fn()
    torch.cuda.synchronize()
    device_times=[]
    wall_times=[]
    for _ in range(3):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        began=time.perf_counter()
        start.record()
        for _ in range(repeats):fn()
        end.record(); end.synchronize()
        wall_times.append((time.perf_counter()-began)*1000/repeats)
        device_times.append(start.elapsed_time(end)/repeats)
    return dict(median_cuda_ms=statistics.median(device_times),
                median_wall_ms=statistics.median(wall_times),
                cuda_rounds_ms=device_times,wall_rounds_ms=wall_times,repeats=repeats)


def case(shape, mode):
    m,n=shape
    param=torch.nn.Parameter(torch.zeros(m,n,device='cuda'))
    opt=SOAP([param],precondition_mode=mode,covariance_compute_dtype='bfloat16',
             basis_track_stats=False,basis_ns_iterations=2)
    group=opt.param_groups[0]
    grad=torch.randn_like(param)
    active=[True,True] if mode=='all' else [m<=n,n<m]
    covariances=[]; bases=[]
    for dim,enabled in zip(shape,active):
        if not enabled:
            covariances.append(None); bases.append(None); continue
        noise=torch.randn(dim,16,device='cuda')
        covariance=torch.diag(torch.linspace(.5,2.,dim,device='cuda'))+.01*(noise@noise.T)/dim
        covariances.append(covariance)
        bases.append(torch.eye(dim,device='cuda'))
    first=torch.randn_like(param)
    second=torch.rand_like(param)+.1

    def update(kind):
        return [None if q is None else
                (opt._hard_reset_one(c,q,'qr') if kind=='qr' else opt._gauge_step_one(c,q,group))
                for c,q in zip(covariances,bases)]

    qr_basis=update('qr'); warm_basis=update('warm')

    def transport(new,include_v):
        state={'exp_avg':first,'exp_avg_sq':second}
        opt._transport_first_moment(state,bases,new,group)
        if include_v:opt._transport_second_moment_for_reset(state,bases,new,group)
        return state

    def full(kind,include_v):
        new=update(kind)
        return transport(new,include_v)

    scratch={'GG':[None if c is None else c.clone() for c in covariances]}
    functions={
        'covariance_accumulation':lambda:opt._accumulate_covariance(grad,scratch,group),
        'gradient_project_and_update_project_back':lambda:opt._project_back_with_basis(opt._project_with_basis(grad,bases,group),bases,group),
        'qr_core_without_alignment':lambda:[torch.linalg.qr(c@q,mode='reduced') for c,q in zip(covariances,bases) if q is not None],
        'qr_basis_with_alignment':lambda:update('qr'),
        'warm_basis_with_retraction':lambda:update('warm'),
        'm_transport_qr':lambda:transport(qr_basis,False),
        'm_and_v_transport_qr':lambda:transport(qr_basis,True),
        'm_transport_warm':lambda:transport(warm_basis,False),
        'm_and_v_transport_warm':lambda:transport(warm_basis,True),
        'full_qr_refresh_m':lambda:full('qr',False),
        'full_qr_refresh_m_v':lambda:full('qr',True),
        'full_warm_refresh_m':lambda:full('warm',False),
        'full_warm_refresh_m_v':lambda:full('warm',True),
    }
    timings={name:measure(fn) for name,fn in functions.items()}
    # Geometry checks must not inherit reduced-precision TF32 matmuls from the
    # timed optimizer path: otherwise the check itself adds orthogonality error.
    timed_precision=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    geometry={}
    for kind,new in [('qr',qr_basis),('warm',warm_basis)]:
        errors=[]
        for c,q in zip(covariances,new):
            if q is None:continue
            b=q.T@c@q
            errors.append(dict(dimension=q.shape[0],
                relative_offdiagonal=float((b-torch.diag_embed(b.diagonal())).norm()/b.norm()),
                normalized_gram_error=float((q.T@q-torch.eye(q.shape[0],device=q.device)).norm()/q.shape[0]**.5)))
        geometry[kind]=errors
    torch.set_float32_matmul_precision(timed_precision)
    return dict(shape=shape,mode=mode,active_factor_dimensions=[d for d,a in zip(shape,active) if a],
                timings=timings,geometry=geometry,geometry_matmul_precision='highest',
                full_warm_over_qr_m=timings['full_warm_refresh_m']['median_wall_ms']/timings['full_qr_refresh_m']['median_wall_ms'],
                full_warm_over_qr_m_v=timings['full_warm_refresh_m_v']['median_wall_ms']/timings['full_qr_refresh_m_v']['median_wall_ms'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('GPU_CLAIM_INDICES'),'Use a lifetime GPU claim'
    torch.set_num_threads(4)
    torch.manual_seed(123)
    torch.set_float32_matmul_precision('high')
    report=dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name(),
        execution='eager',basis_dtype='float32',covariance_compute_dtype='bfloat16',
        float32_matmul_precision='high',seed=123,results=[],
        limitations=['Fixed synthetic covariance and basis; state allocation/host alignment included, no model training.',
                    'QR uses local sign/permutation alignment, not upstream sorting semantics.',
                    'Warm includes dense retraction; full refresh includes specified moment handling.',
                    'Component medians need not add to full-path medians; no quality-equivalent cost claim.',
                    'No compiled/batched optimizer optimization; small eager paths can be launch dominated.',
                    'Ordinary covariance/projection costs shown separately; full optimizer step and training time remain to benchmark.'])
    for shape in [(192,192),(768,192),(768,768),(1536,768),(1536,1536),(3072,768)]:
        for mode in ['all','smaller_side']:
            result=case(shape,mode)
            report['results'].append(result)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:result[k] for k in ['shape','mode','full_warm_over_qr_m','full_warm_over_qr_m_v']}),flush=True)
            gc.collect(); torch.cuda.empty_cache()


if __name__=='__main__':main()
