"""Same-state full-refresh cost: eager versus compilation and graph replay."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
import torch
from soap_reference import ResearchSOAP
from warm_retraction import VARIANTS,warm
from cifar_strict_transport import shadow_refresh
from cifar_tracking_probe import world,project,relative


@torch.no_grad()
def run(b):
    assert os.environ.get('GPU_CLAIM_INDICES') and torch.cuda.device_count()==1
    m=json.loads((b/'manifest.json').read_text())
    for rel,h in m['source_sha256'].items():assert hashlib.sha256((b/'source'/rel).read_bytes()).hexdigest()==h
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.manual_seed(917)
    torch._dynamo.config.recompile_limit=32
    started=time.monotonic();rows=[]
    for a,z in [(768,3072),(1536,6144)]:
        grad=torch.randn(a,z,device='cuda')/z**.5
        old=dict(q=[torch.eye(a,device='cuda'),torch.eye(z,device='cuda')],
                 cov=[grad@grad.T,grad.T@grad],m=.01*torch.randn_like(grad),v=torch.rand_like(grad)+.01)
        policies=[('qr20','eager')]+[(name,mode) for name in ['scaled_ns6','spectral_cubic2_cap05']
                                   for mode in ['eager','default','reduce-overhead']]
        for name,mode in policies:
            opt=ResearchSOAP([torch.nn.Parameter(torch.zeros_like(grad))],basis_method='qr' if name=='qr20' else 'warm',transport_precision='highest')
            group=opt.param_groups[0];group['hard_reset_interval']=0
            compile_seconds=0.;preflight=[]
            if name!='qr20':
                variant=next(v for v in VARIANTS if v.name==name)
                group.update(basis_ns_iterations=variant.iterations,basis_rotation_cap_mode=variant.cap_mode,basis_rotation_cap=variant.cap)
                def kernel(c,q):return warm(c,q,group,variant)
                began=time.perf_counter()
                fn=kernel if mode=='eager' else torch.compile(kernel,fullgraph=True,dynamic=False,mode=mode)
                compile_seconds+=time.perf_counter()-began
                for cov,q in zip(old['cov'],old['q']):
                    expected=kernel(cov,q)
                    torch.cuda.synchronize();began=time.perf_counter()
                    for _ in range(3):
                        torch.compiler.cudagraph_mark_step_begin()
                        actual=fn(cov,q).clone()
                    torch.cuda.synchronize();compile_seconds+=time.perf_counter()-began
                    torch.set_float32_matmul_precision('highest')
                    error=relative(actual,expected);assert error<.01
                    preflight.append(dict(dimension=q.shape[0],relative_output_error=error))
                    torch.set_float32_matmul_precision('high')
                    del actual,expected
                # Graph outputs need owned storage across subsequent invocations;
                # include the clone in the measured graph path.
                opt._gauge_step_one=(lambda cov,q,g:fn(cov,q).clone()) if mode=='reduce-overhead' else (lambda cov,q,g:fn(cov,q))
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin();actual=shadow_refresh(opt,old,group,20)
                del actual
            torch.cuda.synchronize();device=[];wall=[]
            for _ in range(5):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                began=time.perf_counter();start.record()
                torch.compiler.cudagraph_mark_step_begin();actual=shadow_refresh(opt,old,group,20)
                end.record();end.synchronize();device.append(start.elapsed_time(end));wall.append(1000*(time.perf_counter()-began))
            torch.set_float32_matmul_precision('highest')
            gram=max(float((q.T@q-torch.eye(q.shape[0],device=q.device)).norm()/q.shape[0]**.5) for q in actual['Q'])
            expected_m=project(world(old['m'],old['q']),actual['Q'])
            transport=relative(actual['exp_avg'],expected_m)
            preservation=relative(world(actual['exp_avg'],actual['Q']),world(old['m'],old['q']))
            torch.set_float32_matmul_precision('high')
            rows.append(dict(shape=[a,z],policy=name,mode=mode,setup_seconds=compile_seconds,
                preflight=preflight,device_ms=device,wall_ms=wall,median_device_ms=statistics.median(device),
                median_wall_ms=statistics.median(wall),maximum_gram=gram,selected_basis_m_error=transport,
                world_m_error=preservation,geometry_and_transport_pass=gram<.05 and transport<1e-6 and preservation<.01,
                graph_output_clone_charged=mode=='reduce-overhead'))
            print('CASE '+json.dumps(rows[-1]),flush=True)
            (b/'results.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2)+'\n')
            del actual,opt
            if name!='qr20':del fn,kernel
            torch._dynamo.reset();gc.collect();torch.cuda.empty_cache()
            if time.monotonic()-started>480:raise TimeoutError('Compile diagnostic budget exceeded')
        del old,grad
    (b/'results.json').write_text(json.dumps(dict(status='completed',rows=rows,wall_seconds=time.monotonic()-started,
        torch=torch.__version__,cuda=torch.version.cuda,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']),indent=2)+'\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);run(p.parse_args().bundle)
