"""Fixed synthetic same-state QR product attribution; no training or tuning."""
import argparse
from datetime import datetime,timezone
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

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
import torch
from bf16_soap import qr_basis,transport_moment,matrix_step

ROOT=HERE.parent
DIMENSIONS=(768,1536,2304,3072)
BATCHES=(1,12)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def preqr(c,q,policy):
    if policy=='original':
        scores=(q.transpose(-1,-2)@c@q).diagonal(dim1=-2,dim2=-1)
        order=scores.argsort(dim=-1,descending=True)
        return c@q.gather(-1,order.unsqueeze(-2).expand_as(q)),order,scores
    if policy!='reuse':raise ValueError(policy)
    z=c@q;scores=(q*z).sum(-2);order=scores.argsort(dim=-1,descending=True)
    return z.gather(-1,order.unsqueeze(-2).expand_as(z)),order,scores


def basis(c,q,policy):
    z,order,_=preqr(c,q,policy)
    return torch.linalg.qr(z).Q,order


def relative(a,b):
    return float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-30))


def next_probe(c,old,new,order,m,v,g):
    """One refreshed left factor, fixed 32D right identity; actual BF16 kernels."""
    if c.ndim==2:c,old,new,order=c[None],old[None],new[None],order[None]
    identity=torch.eye(32,device=c.device).expand(c.shape[0],32,32).bfloat16().contiguous()
    new=new.bfloat16().contiguous();old=old.bfloat16().contiguous()
    moved=transport_moment(m,old,identity,new,identity,torch.bfloat16)
    variance=v.gather(-2,order.unsqueeze(-1).expand_as(v))
    state=(moved,variance,c.bfloat16(),identity,new,identity)
    update,next_state=matrix_step(torch.zeros_like(g),g,state,
        torch.tensor(1001.,device=c.device),torch.tensor(.001,device=c.device),
        beta1=.95,beta2=.95,cov_beta=.95,eps=1e-8,weight_decay=.05,event='ordinary',
        compute_dtype=torch.bfloat16,basis_dtype=torch.bfloat16)
    world=torch.bmm(new,moved)
    assert all(bool(torch.isfinite(t).all()) for t in (update,*state,*next_state))
    assert bool((variance>=0).all()) and bool((next_state[1]>=0).all())
    return dict(moment=moved,variance=variance,world_m=world,following_update=update)


def numerical(c,q):
    outputs={p:basis(c,q,p) for p in ('original','reuse')}
    n,no=outputs['original'];r,ro=outputs['reuse']
    scores=preqr(c,q,'original')[2];rscores=preqr(c,q,'reuse')[2]
    d=q.shape[-1];batch=1 if q.ndim==2 else q.shape[0]
    m=torch.randn(batch,d,32,device=c.device).bfloat16()
    v=(.1+torch.rand(batch,d,32,device=c.device)).bfloat16()
    g=torch.randn(batch,d,32,device=c.device)
    probes={p:next_probe(c,q,*outputs[p],m,v,g) for p in outputs}
    previous=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
    try:
        signs=(n*r).sum(-2).sign();signs=torch.where(signs==0,torch.ones_like(signs),signs)
        gram={p:float((a.transpose(-1,-2)@a-torch.eye(d,device=c.device)).norm()/math.sqrt(batch*d)) for p,(a,_) in outputs.items()}
        a=probes['original']['following_update'];b=probes['reuse']['following_update']
        cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))
        band=float((scores.gather(-1,no)-scores.gather(-1,ro)).abs().max()/scores.abs().max().clamp_min(1e-30))
        metrics=dict(order_disagreements=int((no!=ro).sum()),total_columns=batch*d,
            score_max_relative_difference=float((scores-rscores).abs().max()/scores.abs().max().clamp_min(1e-30)),
            reordered_score_band_relative=band,negative_same_column_overlaps=int((signs<0).sum()),
            raw_basis_relative=relative(n,r),sign_aligned_basis_relative=relative(n,r*signs.unsqueeze(-2)),
            normalized_gram=gram,following_update_relative=relative(a,b),following_update_cosine=cosine,
            state_relative={k:relative(probes['original'][k],probes['reuse'][k]) for k in ('moment','variance','world_m')},
            finite=True,variance_nonnegative=True,
            probe_scope='Synthetic one-sided refresh; fixed 32D right identity, BF16 storage/transport and actual next ordinary matrix_step. No training trajectory.')
        assert all(math.isfinite(v) for k,v in metrics.items() if isinstance(v,float))
    finally:torch.set_float32_matmul_precision(previous)
    return metrics


def measure(functions,warmup=2,rounds=10):
    for fn in functions.values():
        for _ in range(warmup):fn()
    torch.cuda.synchronize();samples={n:[] for n in functions}
    for turn in range(rounds):
        names=list(functions)
        if turn%2:names.reverse()
        for name in names:
            torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            wall=time.perf_counter();start.record();value=functions[name]();end.record();end.synchronize()
            samples[name].append(dict(cuda_ms=start.elapsed_time(end),wall_ms=(time.perf_counter()-wall)*1000))
            del value
    return {n:dict(median_cuda_ms=statistics.median(v['cuda_ms'] for v in s),
        median_wall_ms=statistics.median(v['wall_ms'] for v in s),samples=s) for n,s in samples.items()}


def tensor_hash(t):
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


@torch.no_grad()
def run(bundle,m):
    start=time.monotonic();rows=[];torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    write(bundle/'progress.json',dict(status='running',completed=[]))
    try:
        for batch in BATCHES:
            for dimension in DIMENSIONS:
                torch.manual_seed(20261027+batch+dimension)
                q=torch.linalg.qr(torch.randn(batch,dimension,dimension,device='cuda')).Q
                spectrum=torch.linspace(1.,4.,dimension,device='cuda')
                c=(q*spectrum)@q.transpose(-1,-2)
                perturb=torch.randn(batch,dimension,32,device='cuda')
                c=c+.02/dimension*(perturb@perturb.transpose(-1,-2))
                c=(.5*(c+c.transpose(-1,-2))).bfloat16().float().contiguous()
                q=q.bfloat16().float().contiguous()
                if batch==1:c,q=c[0],q[0]
                before={k:tensor_hash(v) for k,v in [('cov',c),('basis',q)]}
                metrics=numerical(c,q)
                fixed=preqr(c,q,'reuse')[0]
                timing=measure(dict(original=lambda:basis(c,q,'original'),reuse=lambda:basis(c,q,'reuse'),
                    original_products_sort=lambda:preqr(c,q,'original'),reuse_product_sort=lambda:preqr(c,q,'reuse'),
                    same_input_qr_solver=lambda:torch.linalg.qr(fixed).Q))
                after={k:tensor_hash(v) for k,v in [('cov',c),('basis',q)]};assert before==after
                row=dict(batch=batch,dimension=dimension,input_sha256=before,inputs_preserved=True,
                    numerical=metrics,timing=timing,peak_gpu_bytes=torch.cuda.max_memory_allocated())
                assert row['peak_gpu_bytes']<24*2**30,'24GiB allocation cap'
                rows.append(row);write(bundle/'partial_results.json',rows)
                write(bundle/'progress.json',dict(status='running',completed=[[r['batch'],r['dimension']] for r in rows],elapsed_seconds=time.monotonic()-start))
                assert time.monotonic()-start<580,'Internal profile execution budget'
                del c,q,fixed,perturb;gc.collect();torch.cuda.empty_cache()
        result=dict(status='completed',rows=rows,elapsed_seconds=time.monotonic()-start,
            dtype='FP32 QR arithmetic and output, BF16-rounded fixed input C/Q promoted before timing',
            matmul_precision='high (same for both); QR/eigh solver FP32',compiled=False,
            attribution='Three original dense products versus one reused CQ. Batches1 and12 matched within each pair. No precision/compilation/training speed claim.',
            synthetic=True,training_calls=0,model_or_optimizer_tensor_checkpoints_written=False,
            environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                physical_claim=os.environ['GPU_CLAIM_INDICES'],allow_tf32=torch.backends.cuda.matmul.allow_tf32))
        write(bundle/'result.json',result)
        d=Path(m['compact_evidence_directory']);d.mkdir(parents=True,exist_ok=True)
        for name in ('result.json','manifest.json','PROTOCOL.md','cpu_checks.xml'):shutil.copy2(bundle/name,d/name)
        write(bundle/'progress.json',dict(status='completed',elapsed_seconds=time.monotonic()-start))
    except BaseException as error:
        write(bundle/'progress.json',dict(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-start));raise


def verify(bundle):
    m=json.loads((bundle/'manifest.json').read_text())
    assert m['dimensions']==list(DIMENSIONS) and m['batches']==list(BATCHES) and m['gpu_execution_seconds_cap']==600
    for n,h in m['source_files'].items():assert sha(bundle/'source'/n)==h,n
    for n,h in m['support_files'].items():assert sha(bundle/n)==h,n
    for n,h in m['production_source_sha256'].items():assert sha(bundle/'source'/n)==h,n
    return m


def prepare(bundle,checks):
    assert not bundle.exists(),'Fresh bundle required'
    ledger=json.loads((ROOT/'automation/remaining_experiments_20261027.json').read_text())
    assert ledger['stages']['refresh_controls']['status']=='completed'
    assert 'failures="0"' in checks.read_text() and 'errors="0"' in checks.read_text()
    for name,h in ledger['production_source_sha256'].items():assert sha(ROOT/name)==h,name
    names=[*ledger['production_source_sha256'],'benchmarks/bf16_qr_attribution.py','tests/test_bf16_qr_attribution.py']
    bundle.mkdir();(bundle/'source').mkdir()
    for n in names:
        target=bundle/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,target)
    shutil.copy2(checks,bundle/'cpu_checks.xml');shutil.copy2(ROOT/'BF16_QR_ATTRIBUTION_20261027.md',bundle/'PROTOCOL.md')
    m=dict(dimensions=list(DIMENSIONS),batches=list(BATCHES),gpu_execution_seconds_cap=600,
        source_files={n:sha(bundle/'source'/n) for n in names},
        support_files={n:sha(bundle/n) for n in ('PROTOCOL.md','cpu_checks.xml')},
        production_source_sha256=ledger['production_source_sha256'],
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        compact_evidence_directory=str(ROOT/'reports/bf16_qr_attribution_20261027'))
    write(bundle/'manifest.json',m);verify(bundle)
    print(json.dumps(dict(status='prepared',bundle=str(bundle))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--checks',type=Path);a=p.parse_args()
    if a.prepare:prepare(a.bundle,a.checks);return
    assert os.environ.get('GPU_CLAIM_UUIDS'),'Lifetime GPU claim required'
    assert Path(__file__).resolve()==a.bundle/'source/benchmarks/bf16_qr_attribution.py'
    run(a.bundle,verify(a.bundle))


if __name__=='__main__':main()
