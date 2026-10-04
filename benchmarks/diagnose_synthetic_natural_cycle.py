"""Check the synthetic guard failure under a natural 200-update cadence.

Read-only shadow retractions diagnose a failure; returned optimizer bases always
come from the unchanged implementation. No timings or learning claims.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
import soap
from soap_reference import ResearchSOAP
from benchmark_synthetic_scaling import MockBlock,inspect_state,StateGuardError


def gram_error(q):
    prior=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
    try:return float((q.T@q-torch.eye(q.shape[0],device=q.device)).norm()/q.shape[0]**.5)
    finally:torch.set_float32_matmul_precision(prior)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle
    if not os.environ.get('GPU_CLAIM_INDICES') or torch.cuda.device_count()!=1:raise RuntimeError('One claimed GPU required')
    manifest=json.loads((b/'manifest.json').read_text())
    for name,digest in manifest.items():
        if hashlib.sha256((b/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Changed source: {name}')
    config=json.loads((b/'config.json').read_text());torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    original=soap._orthogonalize_ns
    report=dict(status='running',config=config,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,cases=[])
    started=time.monotonic()
    def save():
        tmp=b/'results.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(b/'results.json')
    for method,mode in [('qr','all'),('warm','all'),('warm','smaller_side')]:
        torch.manual_seed(config['seed']+config['width']);model=MockBlock(config['width']).cuda()
        gen=torch.Generator(device='cuda').manual_seed(config['seed']+config['tokens'])
        batches=[(torch.randn(config['tokens'],config['width'],device='cuda',generator=gen),
                  torch.randn(config['tokens'],config['width'],device='cuda',generator=gen)) for _ in range(4)]
        opt=ResearchSOAP(model.parameters(),lr=.0005,betas=(.9,.999),shampoo_beta=.999,
                         basis_method=method,precondition_mode=mode,precondition_frequency=10,hard_reset_interval=200)
        row=dict(method=method,mode=mode,status='running',observations=[],retraction_diagnostics=[])
        report['cases'].append(row);save()
        step=0
        def observed_retraction(candidate,iterations=2):
            output=original(candidate,iterations)
            error=gram_error(output)
            gram=candidate.T@candidate
            upper=gram.abs().sum(-1).amax().clamp_min(1e-12)
            scale=float((soap._NS_MAX_SINGULAR/upper.sqrt()).clamp(max=1.))
            item=dict(step=step,dimension=candidate.shape[0],iterations=iterations,
                      infinity_bound=float(upper),initial_scale=scale,output_gram_error=error)
            if error>.05:
                item['shadow_extra_iterations_gram_error']={str(n):gram_error(original(candidate,n)) for n in [4,6]}
            row['retraction_diagnostics'].append(item)
            return output
        soap._orthogonalize_ns=observed_retraction
        for step in range(config['updates']+1):
            if time.monotonic()-started>config['maximum_seconds']:raise RuntimeError('Diagnostic budget exceeded')
            opt.zero_grad(set_to_none=True);x,target=batches[step%4]
            with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(x)
            loss=F.mse_loss(prediction.float(),target);loss.backward();opt.step()
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite diagnostic loss')
            if step==0 or opt.last_basis_refreshes or step==config['updates']:
                try:geometry=inspect_state(opt,mode)
                except StateGuardError as error:
                    row.update(status='guard_rejected',failure_step=step,failure=error.diagnostics);break
                row['observations'].append(dict(step=step,**geometry))
        else:row['status']='passed'
        row['last_step']=step;save()
        print(json.dumps({k:v for k,v in row.items() if k not in ['observations','retraction_diagnostics']}),flush=True)
        soap._orthogonalize_ns=original
        del opt,model,batches;gc.collect();torch.cuda.empty_cache()
    report.update(status='completed',all_in_seconds=time.monotonic()-started);save()


if __name__=='__main__':main()
