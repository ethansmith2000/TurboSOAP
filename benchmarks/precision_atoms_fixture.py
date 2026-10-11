"""Fresh managed common-input diagnostic; no trainer, tensor files or old rerun."""
import argparse
import functools
import gc
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
STARTED=time.monotonic()
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
import torch
from warm_compile_kernel import warm_kernel,classify,LIMITS
from warm_compile_fixture import highest,metrics,relative,tensor_hash,graph_counts,sha
from precision_atoms_kernel import warm_split,warm_stages

PROTOCOL=dict(dimensions=[1536,6144],spectra=['broad','clustered'],seed=101118,
    beta=.99,rank=32,perturbation=2.,basis_lr=.5,damping=.01,cap=.5,cubic_iterations=2,
    variants=['eager_high','eager_highest','compiled_high','compiled_highest',
              'compiled_no_patterns','compiled_emulate_casts','compiled_bf16_cov','compiled_bf16_candidate'],
    controls={'compiled_no_patterns':{'pattern_matcher':False},
              'compiled_emulate_casts':{'emulate_precision_casts':True}},
    default_compile_mode='default',fullgraph=True,dynamic=False,
    first_setup_calls=2,warmup_calls=3,samples=9,recurrent_updates=12,recurrent_observe=[1,4,8,12],
    stage_observer='default fullgraph high, ten FP32 stage outputs; causal use requires endpoint relative<=1e-6 versus unchanged whole compiled kernel',
    stage_endpoint_limit=1e-6,limits=LIMITS,compiled_vs_own_eager_limit=1e-4,
    internal_seconds=900,external_seconds=960,allocated_bytes=8*2**30,evidence_bytes=30*2**20,
    compiler_cache_limit_bytes=512*2**20,expected_compiler_cache_bytes=150*2**20,
    expected_evidence_bytes=2*2**20,cpu_hash_chunk_bytes=2**20,
    bf16_reduced_precision_reduction=False,
    scope='fresh synthetic fixed-covariance stage and arithmetic controls; skinny highest-FP32 M probes; not trainer or full SOAP timing',
    artifacts='compact JSON, source, logs and isolated generated compiler cache; no checkpoint, weights, tensor files or profiler traces',
    cache_retention='through next precision review; purpose compiler arithmetic/generated-code audit, not model recovery',
    candidate_policy='same old limits; own-reference highest is distinct control; preserve every failed candidate; no default change')


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(bundle):
    m=json.loads((bundle/'manifest.json').read_text())
    assert json.loads((bundle/'protocol.json').read_text())==PROTOCOL
    assert sha(bundle/'protocol.json')==m['protocol_sha256']
    for name,h in m['source_sha256'].items():assert sha(bundle/'source'/name)==h,name
    return m


def functions():
    fp=functools.partial(warm_kernel,mixed=False)
    def compiled(fn,options=None):
        if options is None:
            return torch.compile(fn,fullgraph=True,dynamic=False,mode='default')
        return torch.compile(fn,fullgraph=True,dynamic=False,options=options)
    return dict(eager_high=(fp,'high',False),eager_highest=(fp,'highest',False),
        compiled_high=(compiled(fp),'high',False),compiled_highest=(compiled(fp),'highest',False),
        compiled_no_patterns=(compiled(fp,PROTOCOL['controls']['compiled_no_patterns']),'high',False),
        compiled_emulate_casts=(compiled(fp,PROTOCOL['controls']['compiled_emulate_casts']),'high',False),
        compiled_bf16_cov=(compiled(functools.partial(warm_split,cov_mixed=True,candidate_mixed=False)),'high',True),
        compiled_bf16_candidate=(compiled(functools.partial(warm_split,cov_mixed=False,candidate_mixed=True)),'high',True))


def invoke(spec,c,q):
    fn,precision,_=spec
    torch.set_float32_matmul_precision(precision)
    return fn(c,q)


def own_eager(name,c,q):
    return warm_split(c,q,True,False) if name=='compiled_bf16_cov' else warm_split(c,q,False,True) if name=='compiled_bf16_candidate' else warm_kernel(c,q)


@torch.no_grad()
def run(bundle):
    m=verify(bundle)
    assert os.environ.get('GPU_CLAIM_INDICES') and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    assert not torch.is_autocast_enabled()
    torch.cuda.reset_peak_memory_stats()
    rng=torch.Generator(device='cuda').manual_seed(PROTOCOL['seed'])
    rows=[];setup=[];initial=graph_counts()
    info=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        uuid=os.environ['CUDA_VISIBLE_DEVICES'],claim_indices=os.environ['GPU_CLAIM_INDICES'],
        environment={k:os.environ.get(k) for k in ('CUDA_HOME','CUDA_VERSION','OMP_NUM_THREADS','MKL_NUM_THREADS',
        'TORCHINDUCTOR_COMPILE_THREADS','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','PYTHONDONTWRITEBYTECODE')},
        cpu_affinity=sorted(os.sched_getaffinity(0)),bf16_reduced_precision_reduction=False,
        numerical_options=PROTOCOL['controls'],protocol_sha256=m['protocol_sha256'])
    write(bundle/'actual_environment.json',info)
    def guard():
        assert time.monotonic()-STARTED<PROTOCOL['internal_seconds'],'time budget'
        assert torch.cuda.max_memory_allocated()<PROTOCOL['allocated_bytes'],'allocated memory budget'
        assert sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())<PROTOCOL['evidence_bytes'],'evidence budget'
        cache=Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])
        assert sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())<PROTOCOL['compiler_cache_limit_bytes'],'cache budget'
    def save(status='running'):
        write(bundle/'results.json',dict(status=status,rows=rows,setup=setup,environment=info,
            all_in_seconds=time.monotonic()-STARTED,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            counts_initial=initial,counts_final=graph_counts(),scope=PROTOCOL['scope'],
            qualified_training_timing=False,qualified_learning=False))
    for n in PROTOCOL['dimensions']:
        with highest():
            q=torch.linalg.qr(torch.randn(n,n,device='cuda',generator=rng)).Q.contiguous()
            h=torch.randn(n,PROTOCOL['rank'],device='cuda',generator=rng)/PROTOCOL['rank']**.5
            noise=h@h.T
            world_m=q@(torch.randn(n,16,device='cuda',generator=rng)/n**.5)
        specs=functions()
        staged=torch.compile(warm_stages,fullgraph=True,dynamic=False,mode='default')
        for spectrum in PROTOCOL['spectra']:
            with highest():
                d=torch.linspace(.5,2.,n,device='cuda') if spectrum=='broad' else torch.linspace(1.,1.001,n,device='cuda')
                base=(q*d)@q.T
                cov=.99*base+.01*(base+PROTOCOL['perturbation']*noise)
                cov=.5*(cov+cov.T)
            del base
            hashes=dict(cov=tensor_hash(cov),q=tensor_hash(q))
            print('INPUT '+json.dumps(dict(dimension=n,spectrum=spectrum,hashes=hashes)),flush=True)
            reference={key:invoke(specs[key],cov,q) for key in ('eager_high','eager_highest')}
            preflight={}
            for name,spec in specs.items():
                torch.set_float32_matmul_precision(spec[1]);torch.cuda.synchronize();began=time.monotonic()
                before=graph_counts()
                for _ in range(PROTOCOL['first_setup_calls']):actual=invoke(spec,cov,q)
                torch.cuda.synchronize();seconds=time.monotonic()-began
                expected=own_eager(name,cov,q)
                same_error=relative(actual,expected)
                setup.append(dict(dimension=n,spectrum=spectrum,variant=name,first_two_calls_seconds=seconds,
                    own_eager_relative=same_error,own_eager_pass=same_error<=PROTOCOL['compiled_vs_own_eager_limit'],
                    counts_before=before,counts_after=graph_counts()))
                ref=reference['eager_highest' if spec[1]=='highest' else 'eager_high']
                mm=metrics(actual,ref,q,cov,world_m)
                mm['gate']=classify(mm,spec[2]);mm['own_eager_pass']=same_error<=PROTOCOL['compiled_vs_own_eager_limit']
                preflight[name]=mm
                for _ in range(PROTOCOL['warmup_calls']):actual=invoke(spec,cov,q)
                del actual,expected;guard()
            # Returning stages may inhibit/reorganize fusion: explicitly test observer impact.
            torch.set_float32_matmul_precision('high');torch.cuda.synchronize();began=time.monotonic()
            for _ in range(2):observed=staged(cov,q)
            torch.cuda.synchronize();stage_setup_seconds=time.monotonic()-began
            whole=invoke(specs['compiled_high'],cov,q)
            endpoint=relative(observed['qnew'],whole)
            eager_stages=warm_stages(cov,q)
            assert torch.equal(eager_stages['qnew'],reference['eager_high'])
            stage_errors={key:dict(relative=relative(observed[key],eager_stages[key]),
                eager_norm=float(eager_stages[key].norm()),compiled_norm=float(observed[key].norm())) for key in observed}
            observer=dict(endpoint_relative=endpoint,causal_eligible=endpoint<=PROTOCOL['stage_endpoint_limit'],
                setup_seconds=stage_setup_seconds,errors=stage_errors,
                output_bytes=sum(x.numel()*x.element_size() for x in observed.values()),
                interpretation='upstream propagation included; stage-return changes cannot be attributed to original whole kernel if endpoint gate fails')
            del observed,whole,eager_stages
            torch.cuda.synchronize();before=graph_counts()
            samples={name:[] for name in specs};order_log=[]
            order_rng=random.Random(PROTOCOL['seed']+n+(spectrum=='clustered'))
            for _ in range(PROTOCOL['samples']):
                order=list(specs);order_rng.shuffle(order);order_log.append(order)
                for name in order:
                    # Precision flag changes are outside timed basis execution.
                    torch.set_float32_matmul_precision(specs[name][1]);torch.cuda.synchronize()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    began=time.perf_counter();start.record();actual=specs[name][0](cov,q)
                    end.record();end.synchronize()
                    samples[name].append(dict(device_ms=start.elapsed_time(end),wall_ms=1000*(time.perf_counter()-began)))
                    del actual
                guard()
            after=graph_counts();assert before==after,'steady compilation or graph break'
            qs=dict.fromkeys(specs,q);recurrent={name:[] for name in specs}
            for step in range(1,PROTOCOL['recurrent_updates']+1):
                for name,spec in specs.items():qs[name]=invoke(spec,cov,qs[name])
                if step in PROTOCOL['recurrent_observe']:
                    for name,spec in specs.items():
                        ref=qs['eager_highest' if spec[1]=='highest' else 'eager_high']
                        mm=metrics(qs[name],ref,q,cov,world_m)
                        mm.update(step=step,gate=classify(mm,spec[2]));recurrent[name].append(mm)
                guard()
            assert hashes==dict(cov=tensor_hash(cov),q=tensor_hash(q)),'common inputs mutated'
            row=dict(dimension=n,spectrum=spectrum,hashes=hashes,preflight=preflight,recurrent=recurrent,
                stage_observer=observer,common_inputs_unchanged=True,steady_counts_before=before,steady_counts_after=after,
                timing_order=order_log,samples=samples,
                high_vs_highest=metrics(reference['eager_high'],reference['eager_highest'],q,cov,world_m),
                medians={name:{key:statistics.median(s[key] for s in values) for key in ('device_ms','wall_ms')} for name,values in samples.items()},
                eligible={name:preflight[name]['gate']['passed'] and preflight[name]['own_eager_pass'] and all(x['gate']['passed'] for x in recurrent[name]) for name in specs})
            rows.append(row);save()
            print('CASE '+json.dumps({key:row[key] for key in ('dimension','spectrum','medians','eligible')}),flush=True)
            print('OBSERVER '+json.dumps(dict(dimension=n,spectrum=spectrum,endpoint_relative=endpoint,causal_eligible=observer['causal_eligible'])),flush=True)
            del cov,reference,qs,d
        del specs,staged,q,h,noise,world_m
        torch._dynamo.reset();gc.collect();torch.cuda.empty_cache()
    guard();verify(bundle);save('completed')
    print('COMPLETE '+json.dumps(dict(all_in_seconds=time.monotonic()-STARTED,peak_allocated_bytes=torch.cuda.max_memory_allocated())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    b=parser.parse_args().bundle
    assert not (b/'execution_status.json').exists(),'spent bundle: never retry'
    write(b/'execution_status.json',dict(status='running',pid=os.getpid(),utc=time.time()))
    try:run(b)
    except BaseException as exc:
        write(b/'execution_status.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc()));raise
    else:write(b/'execution_status.json',dict(status='completed',utc=time.time()))
