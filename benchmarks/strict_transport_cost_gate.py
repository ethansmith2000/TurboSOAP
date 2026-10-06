"""Synthetic shape cost and implementation gate; no learning proxy."""
import argparse
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
from cifar_strict_transport import shadow_refresh,diagnose

SHAPES=[(192,192),(576,192),(768,192),(192,768),(768,3072),(1536,6144)]
VARIANTS=[('qr','inherit',.1),('qr','highest',.1),('warm','inherit',.1),
          ('warm','highest',.1),('warm','inherit',.2),('warm','highest',.2)]


@torch.no_grad()
def run(bundle):
    assert torch.cuda.device_count()==1 and os.environ.get('GPU_CLAIM_INDICES')
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(917)
    torch.set_float32_matmul_precision('high');rows=[]
    for shape_index,(a,b) in enumerate(SHAPES):
        # PSD, deliberately rank-limited rectangular covariance. Synthetic only.
        grad=torch.randn(a,b,device='cuda')/max(a,b)**.5
        old=dict(q=[torch.eye(a,device='cuda'),torch.eye(b,device='cuda')],
                 cov=[grad@grad.T,grad.T@grad],m=torch.randn_like(grad)*.01,
                 v=torch.rand_like(grad)+.01)
        opt=ResearchSOAP([torch.nn.Parameter(torch.zeros_like(grad))])
        base=opt.param_groups[0];states={};records={}
        repeats=3 if max(a,b)<=768 else 1
        for method,precision,cap in VARIANTS:
            label=f'{method}_{precision}_cap{cap}'
            g={**base,'basis_method':method,'transport_precision':precision,'basis_rotation_cap':cap,
               'basis_ns_iterations':6 if method=='warm' else 2,'hard_reset_interval':0}
            actual=shadow_refresh(opt,old,g,20)
            try:d=diagnose(opt,old,g,20,actual)
            except FloatingPointError as error:d=dict(targeted_pass=False,error=str(error))
            states[label]=actual
            records[label]=dict(shape=[a,b],method=method,transport_precision=precision,cap=cap,
                                synthetic=True,cifar_shape=max(a,b)<=768,repetitions_per_round=repeats,
                                diagnostic=d,round_ms=[])
        for round_index in range(3):
            shift=(shape_index+round_index)%len(VARIANTS);order=VARIANTS[shift:]+VARIANTS[:shift]
            for method,precision,cap in order:
                label=f'{method}_{precision}_cap{cap}'
                g={**base,'basis_method':method,'transport_precision':precision,'basis_rotation_cap':cap,
                   'basis_ns_iterations':6 if method=='warm' else 2,'hard_reset_interval':0}
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):shadow_refresh(opt,old,g,20)
                end.record();end.synchronize()
                records[label]['round_ms'].append(start.elapsed_time(end)/repeats)
                if time.monotonic()-started>480:raise RuntimeError('Cost-gate wall budget exceeded')
        for method,precision,cap in VARIANTS:
            label=f'{method}_{precision}_cap{cap}';record=records[label]
            record['median_ms']=statistics.median(record['round_ms'])
            control=states[f'{method}_inherit_cap{cap}'];actual=states[label]
            record['basis_and_v_match_inherit']=(all(torch.equal(x,y) for x,y in zip(control['Q'],actual['Q']))
                                               and torch.equal(control['exp_avg_sq'],actual['exp_avg_sq']))
            assert record['basis_and_v_match_inherit']
            rows.append(record)
        print('COST_SHAPE '+json.dumps(dict(shape=[a,b],elapsed=time.monotonic()-started)),flush=True)
        del grad,old,opt,base,states,records,actual,control,g
    small=[r for r in rows if r['cifar_shape'] and r['transport_precision']=='highest']
    result=dict(status='completed',rows=rows,cifar_gate_pass=all(r['diagnostic']['targeted_pass'] for r in small),
                gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,cuda=torch.version.cuda,
                wall_seconds=time.monotonic()-started,
                scope='Synthetic rank-limited covariance, identity incoming bases. Eager refresh costs only; no learning or amortized trainer-speed inference.')
    (bundle/'cost_gate.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    result=run(p.parse_args().bundle)
    if not result['cifar_gate_pass']:raise SystemExit('CIFAR implementation gate failed')
