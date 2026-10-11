"""Identical-candidate retraction replay and bounded hybrid precision fixture."""
import argparse
from collections import Counter
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
from precision_atoms_kernel import warm_stages
from common_y_kernel import warm_prefix,retract_two

PROTOCOL=dict(dimensions=[1536,6144],spectra=['broad','clustered'],seed=101119,
    beta=.99,rank=32,perturbation=2.,basis_lr=.5,damping=.01,cap=.5,cubic_iterations=2,
    variants=['eager_high','eager_highest','compiled_high','compiled_highest','compiled_split_high',
              'eager_hybrid_fp32','compiled_hybrid_fp32','eager_hybrid_bf16','compiled_hybrid_bf16'],
    compile_mode='default',fullgraph=True,dynamic=False,first_setup_calls=2,warmup_calls=3,samples=9,
    recurrent_updates=12,recurrent_observe=[1,4,8,12],limits=LIMITS,own_eager_limit=1e-4,
    observer_endpoint_limit=1e-6,compiled_prefix_vs_observed_y_limit=1e-7,split_endpoint_limit=1e-6,
    retraction_modes=['eager_high','compiled_high','eager_highest','compiled_highest'],
    reference_policy='high and highest against respective eager; hybrids against eager hybrid FP32; BF16 own-eager parity separate',
    hybrid_policy='prefix high, only optional Q@A BF16; two cubic retraction passes highest; precision context and boundary charged',
    internal_seconds=900,external_seconds=960,allocated_bytes=8*2**30,evidence_bytes=30*2**20,
    logical_warm_call_limit=1200,prefix_call_limit=800,retraction_call_limit=900,stage_call_limit=16,
    expected_compiler_cache_bytes=150*2**20,compiler_cache_limit_bytes=512*2**20,
    expected_evidence_bytes=2*2**20,cpu_hash_chunk_bytes=2**20,bf16_reduced_precision_reduction=False,
    artifacts='compact JSON/source/logs/generated compiler code only; Y/stages GPU-RAM-only, released per case; no checkpoints/tensors/weights/traces',
    cache_retention='through next precision review for generated-code/arithmetic provenance, not model recovery',
    scope='fresh synthetic same-Y retraction and hybrid cost/fidelity; no trainer/whole SOAP timing/learning or production changes')


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(bundle):
    m=json.loads((bundle/'manifest.json').read_text())
    assert json.loads((bundle/'protocol.json').read_text())==PROTOCOL
    assert sha(bundle/'protocol.json')==m['protocol_sha256']
    for name,h in m['source_sha256'].items():assert sha(bundle/'source'/name)==h,name
    return m


def make_functions(counts):
    def counted(fn,kind):
        def call(*args):counts[kind]+=1;return fn(*args)
        return call
    def compile_default(fn):return torch.compile(fn,fullgraph=True,dynamic=False,mode='default')
    fp=functools.partial(warm_kernel,mixed=False)
    prefixes={name:counted(fn,'prefix') for name,fn in dict(
        eager_fp32=functools.partial(warm_prefix,cov_mixed=False,candidate_mixed=False),
        compiled_fp32=compile_default(functools.partial(warm_prefix,cov_mixed=False,candidate_mixed=False)),
        eager_bf16=functools.partial(warm_prefix,cov_mixed=False,candidate_mixed=True),
        compiled_bf16=compile_default(functools.partial(warm_prefix,cov_mixed=False,candidate_mixed=True))).items()}
    rets={name:(counted(fn,'retraction'),precision) for name,fn,precision in (
        ('eager_high',retract_two,'high'),('compiled_high',compile_default(retract_two),'high'),
        ('eager_highest',retract_two,'highest'),('compiled_highest',compile_default(retract_two),'highest'))}
    def pipeline(prefix,ret,hybrid):
        def call(c,q):
            y=prefix(c,q)
            if hybrid:
                with highest():return ret(y)
            return ret(y)
        return counted(call,'pipeline')
    specs=dict(eager_high=(counted(fp,'whole_direct'),'high','eager_high',False),
        eager_highest=(counted(fp,'whole_direct'),'highest','eager_highest',False),
        compiled_high=(counted(compile_default(fp),'whole_direct'),'high','eager_high',False),
        compiled_highest=(counted(compile_default(fp),'whole_direct'),'highest','eager_highest',False),
        compiled_split_high=(pipeline(prefixes['compiled_fp32'],rets['compiled_high'][0],False),'high','eager_high',False),
        eager_hybrid_fp32=(pipeline(prefixes['eager_fp32'],rets['eager_highest'][0],True),'high','eager_hybrid_fp32',False),
        compiled_hybrid_fp32=(pipeline(prefixes['compiled_fp32'],rets['compiled_highest'][0],True),'high','eager_hybrid_fp32',False),
        eager_hybrid_bf16=(pipeline(prefixes['eager_bf16'],rets['eager_highest'][0],True),'high','eager_hybrid_fp32',True),
        compiled_hybrid_bf16=(pipeline(prefixes['compiled_bf16'],rets['compiled_highest'][0],True),'high','eager_hybrid_fp32',True))
    observer=counted(compile_default(warm_stages),'stage')
    eager_observer=counted(warm_stages,'stage')
    return specs,prefixes,rets,observer,eager_observer


def invoke(spec,c,q):
    torch.set_float32_matmul_precision(spec[1]);return spec[0](c,q)


def eager_name(name):
    if 'hybrid_bf16' in name:return 'eager_hybrid_bf16'
    if 'hybrid_fp32' in name:return 'eager_hybrid_fp32'
    return 'eager_highest' if name.endswith('highest') else 'eager_high'


@torch.no_grad()
def run(bundle):
    m=verify(bundle)
    assert os.environ.get('GPU_CLAIM_INDICES') and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    assert not torch.is_autocast_enabled()
    torch.cuda.reset_peak_memory_stats();rng=torch.Generator(device='cuda').manual_seed(PROTOCOL['seed'])
    counts=Counter();rows=[];setup=[];initial=graph_counts()
    info=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        uuid=os.environ['CUDA_VISIBLE_DEVICES'],claim_indices=os.environ['GPU_CLAIM_INDICES'],
        environment={k:os.environ.get(k) for k in ('CUDA_HOME','CUDA_VERSION','OMP_NUM_THREADS','MKL_NUM_THREADS',
        'TORCHINDUCTOR_COMPILE_THREADS','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','PYTHONDONTWRITEBYTECODE')},
        cpu_affinity=sorted(os.sched_getaffinity(0)),bf16_reduced_precision_reduction=False,protocol_sha256=m['protocol_sha256'])
    write(bundle/'actual_environment.json',info)
    def guard():
        assert time.monotonic()-STARTED<900,'internal time budget'
        assert torch.cuda.max_memory_allocated()<PROTOCOL['allocated_bytes'],'allocated memory budget'
        assert counts['whole_direct']+counts['pipeline']<=PROTOCOL['logical_warm_call_limit'],'warm call budget'
        for kind in ('prefix','retraction','stage'):assert counts[kind]<=PROTOCOL[kind+'_call_limit'],kind+' call budget'
        assert sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())<PROTOCOL['evidence_bytes'],'evidence budget'
        cache=Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])
        assert sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())<PROTOCOL['compiler_cache_limit_bytes'],'cache budget'
    def save(status='running'):
        write(bundle/'results.json',dict(status=status,rows=rows,setup=setup,counts=dict(counts),environment=info,
            all_in_seconds=time.monotonic()-STARTED,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            compiler_initial=initial,compiler_final=graph_counts(),scope=PROTOCOL['scope'],qualified_learning=False,qualified_training_timing=False))
    for n in PROTOCOL['dimensions']:
        with highest():
            q=torch.linalg.qr(torch.randn(n,n,device='cuda',generator=rng)).Q.contiguous();counts['fixture_qr']+=1
            h=torch.randn(n,32,device='cuda',generator=rng)/32**.5;noise=h@h.T
            world_m=q@(torch.randn(n,16,device='cuda',generator=rng)/n**.5)
        specs,prefixes,rets,observer,eager_observer=make_functions(counts)
        for spectrum in PROTOCOL['spectra']:
            with highest():
                d=torch.linspace(.5,2.,n,device='cuda') if spectrum=='broad' else torch.linspace(1.,1.001,n,device='cuda')
                base=(q*d)@q.T;cov=.99*base+.01*(base+2*noise);cov=.5*(cov+cov.T)
            del base
            hashes=dict(cov=tensor_hash(cov),q=tensor_hash(q))
            print('INPUT '+json.dumps(dict(dimension=n,spectrum=spectrum,hashes=hashes)),flush=True)
            refs={name:invoke(specs[name],cov,q) for name in ('eager_high','eager_highest','eager_hybrid_fp32','eager_hybrid_bf16')}
            preflight={}
            for name,spec in specs.items():
                torch.set_float32_matmul_precision(spec[1]);torch.cuda.synchronize();began=time.monotonic();before=graph_counts()
                for _ in range(2):actual=invoke(spec,cov,q)
                torch.cuda.synchronize();seconds=time.monotonic()-began
                own=invoke(specs[eager_name(name)],cov,q)
                error=relative(actual,own)
                setup.append(dict(dimension=n,spectrum=spectrum,variant=name,first_two_calls_seconds=seconds,
                    own_eager_relative=error,own_eager_pass=error<=PROTOCOL['own_eager_limit'],before=before,after=graph_counts()))
                mm=metrics(actual,refs[spec[2]],q,cov,world_m);counts['metric']+=1
                mm.update(gate=classify(mm,spec[3]),own_eager_pass=error<=PROTOCOL['own_eager_limit']);preflight[name]=mm
                for _ in range(3):actual=invoke(spec,cov,q)
                del actual,own;guard()
            torch.set_float32_matmul_precision('high');torch.cuda.synchronize();began=time.monotonic()
            for _ in range(2):observed=observer(cov,q)
            torch.cuda.synchronize();observer_setup_seconds=time.monotonic()-began
            eager_stages=eager_observer(cov,q);whole=invoke(specs['compiled_high'],cov,q)
            observer_error=relative(observed['qnew'],whole)
            ye,yc=eager_stages['candidate'],observed['candidate']
            pure_e=prefixes['eager_fp32'](cov,q);pure_c=prefixes['compiled_fp32'](cov,q)
            assert torch.equal(pure_e,ye),'eager candidate identity'
            prefix_error=relative(pure_c,yc)
            split=invoke(specs['compiled_split_high'],cov,q);split_error=relative(split,whole)
            boundary=dict(observer_endpoint_relative=observer_error,prefix_y_relative=prefix_error,
                split_endpoint_relative=split_error,observer_setup_seconds=observer_setup_seconds,
                represents_whole_high=observer_error<=1e-6 and prefix_error<=1e-7 and split_error<=1e-6,
                same_y_probe_is_valid_independently=True)
            del eager_stages,observed,whole,pure_e,pure_c,split
            yhash={name:tensor_hash(y) for name,y in [('eager',ye),('compiled',yc)]}
            outputs={}
            for source,y in [('eager',ye),('compiled',yc)]:
                for name,(fn,precision) in rets.items():
                    torch.set_float32_matmul_precision(precision);outputs[source,name]=fn(y)
            same_input={}
            for source,y in [('eager',ye),('compiled',yc)]:
                for name,refname in [('compiled_high','eager_high'),('compiled_highest','eager_highest')]:
                    mm=metrics(outputs[source,name],outputs[source,refname],q,cov,world_m);counts['metric']+=1
                    mm['gate']=classify(mm,False);same_input[source+'_'+name]=mm
            with highest():ydifference=float((yc-ye).norm())
            sensitivity={}
            for name in rets:
                with highest():difference=float((outputs['compiled',name]-outputs['eager',name]).norm())
                mm=metrics(outputs['compiled',name],outputs['eager',name],q,cov,world_m);counts['metric']+=1
                sensitivity[name]=dict(absolute_input_difference=ydifference,absolute_output_difference=difference,
                    amplification=difference/ydifference if ydifference>0 else None,metrics=mm)
            assert yhash==dict(eager=tensor_hash(ye),compiled=tensor_hash(yc)),'candidate input mutated'
            replay=dict(boundary=boundary,candidate_hashes=yhash,same_input=same_input,sensitivity=sensitivity,
                candidate_inputs_unchanged=True,candidate_pair_bytes=ye.numel()*ye.element_size()+yc.numel()*yc.element_size())
            del ye,yc,outputs,y
            torch.cuda.synchronize();before=graph_counts();samples={name:[] for name in specs};order_log=[]
            order_rng=random.Random(PROTOCOL['seed']+n+(spectrum=='clustered'))
            for _ in range(9):
                order=list(specs);order_rng.shuffle(order);order_log.append(order)
                for name in order:
                    torch.set_float32_matmul_precision(specs[name][1]);torch.cuda.synchronize()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    began=time.perf_counter();start.record();actual=specs[name][0](cov,q);end.record();end.synchronize()
                    samples[name].append(dict(device_ms=start.elapsed_time(end),wall_ms=1000*(time.perf_counter()-began)));del actual
                guard()
            after=graph_counts();assert before==after,'steady compilation or graph break'
            qs=dict.fromkeys(specs,q);recurrent={name:[] for name in specs}
            for step in range(1,13):
                for name,spec in specs.items():qs[name]=invoke(spec,cov,qs[name])
                if step in [1,4,8,12]:
                    for name,spec in specs.items():
                        mm=metrics(qs[name],qs[spec[2]],q,cov,world_m);counts['metric']+=1
                        mm.update(step=step,gate=classify(mm,spec[3]));recurrent[name].append(mm)
                guard()
            assert hashes==dict(cov=tensor_hash(cov),q=tensor_hash(q)),'common inputs mutated'
            row=dict(dimension=n,spectrum=spectrum,hashes=hashes,common_inputs_unchanged=True,preflight=preflight,
                same_y=replay,recurrent=recurrent,samples=samples,timing_order=order_log,
                steady_counts_before=before,steady_counts_after=after,
                hybrid_vs_high=metrics(refs['eager_hybrid_fp32'],refs['eager_high'],q,cov,world_m),
                medians={name:{key:statistics.median(x[key] for x in values) for key in ('device_ms','wall_ms')} for name,values in samples.items()},
                eligible={name:preflight[name]['gate']['passed'] and preflight[name]['own_eager_pass'] and all(x['gate']['passed'] for x in recurrent[name]) for name in specs})
            counts['metric']+=1
            rows.append(row);save();print('CASE '+json.dumps({k:row[k] for k in ('dimension','spectrum','medians','eligible')}),flush=True)
            print('REPLAY '+json.dumps(dict(dimension=n,spectrum=spectrum,boundary=boundary,
                amplification={name:value['amplification'] for name,value in sensitivity.items()})),flush=True)
            del cov,refs,qs,d
        del specs,prefixes,rets,observer,eager_observer,q,h,noise,world_m
        torch._dynamo.reset();gc.collect();torch.cuda.empty_cache()
    guard();verify(bundle);save('completed')
    print('COMPLETE '+json.dumps(dict(all_in_seconds=time.monotonic()-STARTED,counts=dict(counts),peak_allocated_bytes=torch.cuda.max_memory_allocated())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);b=parser.parse_args().bundle
    assert not (b/'execution_status.json').exists(),'spent bundle: no retry'
    write(b/'execution_status.json',dict(status='running',pid=os.getpid(),utc=time.time()))
    try:run(b)
    except BaseException as exc:
        write(b/'execution_status.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc()));raise
    else:write(b/'execution_status.json',dict(status='completed',utc=time.time()))
