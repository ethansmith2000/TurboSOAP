"""Bounded common-input prefix sensitivity diagnosis; no production/default change."""
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
import weakref
STARTED=time.monotonic()
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
import torch
from common_y_kernel import warm_prefix, retract_two
from residual_retraction import residual_retract_two
from prefix_sensitivity_kernel import prefix_stages, independent_direction, scaled_perturbation
from warm_compile_fixture import highest, metrics, relative, tensor_hash, graph_counts, sha
from warm_compile_kernel import LIMITS, classify

PROTOCOL=dict(seed=149,dimensions=[2048,8192],spectra=['broad','clustered'],layouts=['row','column'],
    column_scale_defects=[0.,.0007],candidate_mixed=True,
    prefix_precisions=['high','highest'],implementations=['eager','compiled'],stage_implementations=['eager'],retractions=['old','residual_bf16'],
    perturbations=[dict(name='fp32_rounding_direction',relative_norm=2e-7),
                   dict(name='bf16_rounding_direction',relative_norm=5e-7),
                   dict(name='independent_skinny_direction',relative_norm=1e-6)],
    direction_policy='actual compiled residual-FP32/BF16 minus compiled-old outputs on common high-prefix Y; normalized to declared common relative norms; independent skinny rank16 control',
    beta=.99,covariance_rank=32,covariance_noise_weight=.02,basis_lr=.5,damping=.01,cap=.5,
    cubic_iterations=2,gram_precision='highest',storage_dtype='float32',
    compile_mode='default',fullgraph=True,dynamic=False,setup_calls=2,warmup_calls=3,timing_samples=7,
    recurrent_updates=8,recurrent_observe=[1,4,8],observer_candidate_limit=1e-6,observer_endpoint_limit=1e-6,
    own_eager_basis_limit=1e-4,common_strict_basis_limit=1e-4,limits=LIMITS,
    reference_policy='each precision against its compiled-old complete policy; eager/compiled own-policy agreement separate; highest-vs-high change descriptive, not QR equivalence',
    attribution_policy='ONLY eager observers provide stage data after candidate/endpoint1e-6 validation; compiled paths return pure Y and have no compiled stage observer or stage attribution; prior failed observer gate remains preserved',
    internal_seconds=1800,external_seconds=1860,allocated_bytes=24*2**30,evidence_bytes=40*2**20,
    compiler_cache_limit_bytes=512*2**20,expected_compiler_cache_bytes=100*2**20,expected_evidence_bytes=5*2**20,
    whole_call_limit=2800,prefix_call_limit=3500,retraction_call_limit=4000,stage_call_limit=600,metric_call_limit=1000,
    bf16_reduced_precision_reduction=False,
    artifacts='dimension-local RAM-only Q/C/stages/directions/skinny M; compact configs/seeds/source/hash/metrics/logs; no weights/checkpoints/tensorfiles/traces',
    retention='compact evidence and isolated generated-code cache retained through next prefix review for arithmetic provenance',
    scope='observer-free compiled prefix perturbation/precision/cap/tracking diagnosis and synthetic complete warm basis cost; no full SOAP/QR/learning qualification')


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def verify(bundle):
    m=json.loads((bundle/'manifest.json').read_text())
    assert json.loads((bundle/'protocol.json').read_text())==PROTOCOL
    assert sha(bundle/'protocol.json')==m['protocol_sha256']
    for name,h in m['source_sha256'].items():assert sha(bundle/'source'/name)==h,name
    for name,h in m['prepared_sha256'].items():assert sha(bundle/name)==h,name
    return m


def make_functions(counts):
    def counted(fn,kind):
        def call(*args):counts[kind]+=1;return fn(*args)
        return call
    def compile_default(fn):return torch.compile(fn,mode='default',fullgraph=True,dynamic=False)
    prefix=functools.partial(warm_prefix,cov_mixed=False,candidate_mixed=True)
    observer=functools.partial(prefix_stages,cov_mixed=False,candidate_mixed=True)
    prefixes={};stages={}
    for impl in PROTOCOL['implementations']:
        for precision in PROTOCOL['prefix_precisions']:
            prefixes[impl+'_'+precision]=counted(compile_default(prefix) if impl=='compiled' else prefix,'prefix')
            if impl=='eager':stages[impl+'_'+precision]=counted(observer,'stage')
    rets={}
    for impl in PROTOCOL['implementations']:
        for ret,fn in dict(old=retract_two,residual_bf16=functools.partial(residual_retract_two,correction_mixed=True),
                           residual_fp32=functools.partial(residual_retract_two,correction_mixed=False)).items():
            rets[impl+'_'+ret]=counted(compile_default(fn) if impl=='compiled' else fn,'retraction')
    def pipeline(pre,ret,precision):
        def call(cov,q):
            torch.set_float32_matmul_precision(precision)
            y=pre(cov,q)
            with highest():return ret(y)
        return counted(call,'whole')
    whole={impl+'_'+precision+'_'+ret:pipeline(prefixes[impl+'_'+precision],rets[impl+'_'+ret],precision)
           for impl in PROTOCOL['implementations'] for precision in PROTOCOL['prefix_precisions'] for ret in PROTOCOL['retractions']}
    return prefixes,stages,rets,whole


def invoke_stage(fn,precision,cov,q):
    torch.set_float32_matmul_precision(precision)
    return fn(cov,q)


def invoke_ret(fn,y):
    with highest():return fn(y)


def compare_stages(actual,reference,input_relative):
    with highest():
        differences={name:dict(relative=relative(actual[name],reference[name]),
                               absolute=float((actual[name]-reference[name]).norm()))
                     for name in ('c','b','raw','a','delta','candidate')}
        for v in differences.values():v['relative_amplification']=v['relative']/max(input_relative,1e-24)
    return differences


def stage_scalars(stage):
    with highest():
        bound=float(stage['bound']);applied_cap=min(1.,.5/max(bound,1e-12))
        return dict(covariance_scale=float(stage['scale']),raw_norm=float(stage['raw'].norm()),
                    raw_bound=bound,cap_active=bound>.5,applied_cap=applied_cap,
                    applied_bound=float(stage['a'].norm()/2**.5),candidate_delta_norm=float(stage['delta'].norm()))


def geometry_and_tracking(q,cov):
    with highest():
        gram=q.T@q;gram.diagonal().sub_(1.)
        b=q.T@cov@q;diagonal=b.diagonal().clone();b.diagonal().zero_()
        return dict(gram=float(gram.norm()/q.shape[0]**.5),covariance_offdiag=float(b.norm()),
                    covariance_diagonal_norm=float(diagonal.norm()))


def checked_metrics(actual,reference,initial,cov,world,counts):
    counts['metric']+=1
    mm=metrics(actual,reference,initial,cov,world)
    mm['mixed_gate']=classify(mm,True)
    mm['strict_basis_pass']=mm['basis_relative']<=PROTOCOL['common_strict_basis_limit']
    return mm


def timed(fn,args):
    torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    began=time.perf_counter();start.record();output=fn(*args);end.record();end.synchronize()
    return output,dict(device_ms=start.elapsed_time(end),wall_ms=1000*(time.perf_counter()-began))


@torch.no_grad()
def run_dimension(n,generator,counts,guard,owners,journal):
    def own(x):owners.append(weakref.ref(x));return x
    with highest():
        qbase=own(torch.linalg.qr(torch.randn(n,n,device='cuda',generator=generator)).Q.contiguous())
        counts['fixture_qr']+=1
        h=torch.randn(n,32,device='cuda',generator=generator)/32**.5;noise=h@h.T
        world=own(qbase@(torch.randn(n,16,device='cuda',generator=generator)/n**.5))
        independent=own(independent_direction(qbase,torch.randn(n,16,device='cuda',generator=generator),
                                               torch.randn(n,16,device='cuda',generator=generator)))
    prefixes,stages,rets,whole=make_functions(counts);rows=[]
    # Compile every declared Q layout/precision before any steady windows or recurrence.
    for spectrum in PROTOCOL['spectra']:
        with highest():
            d=torch.linspace(.5,2.,n,device='cuda') if spectrum=='broad' else torch.linspace(1.,1.001,n,device='cuda')
            base=(qbase*d)@qbase.T;cov=base+.02*noise;cov=own(.5*(cov+cov.T))
        for layout in PROTOCOL['layouts']:
            for defect in PROTOCOL['column_scale_defects']:
                with highest():q=qbase*(1+defect*torch.linspace(-1,1,n,device='cuda'))
                q=own(q.contiguous() if layout=='row' else q.T.contiguous().T)
                hashes=dict(q=tensor_hash(q),cov=tensor_hash(cov));setup=[]
                for name,fn in whole.items():
                    torch.cuda.synchronize();began=time.monotonic();before=graph_counts()
                    for _ in range(2):output=fn(cov,q)
                    torch.cuda.synchronize()
                    setup.append(dict(variant=name,first_two_seconds=time.monotonic()-began,
                                      compiler_before=before,compiler_after=graph_counts()))
                    for _ in range(3):output=fn(cov,q)
                    del output;guard()
                # First common prefix produces real arithmetic-scale retraction directions.
                y=invoke_stage(prefixes['compiled_high'],'high',cov,q)
                old=own(invoke_ret(rets['compiled_old'],y))
                fp=own(invoke_ret(rets['compiled_residual_fp32'],y));bf=own(invoke_ret(rets['compiled_residual_bf16'],y))
                with highest():
                    fp_direction=own(fp-old);bf_direction=own(bf-old)
                    direction_norms=dict(fp32_actual_relative=relative(fp,old),bf16_actual_relative=relative(bf,old),
                                         fp32_direction_norm=float(fp_direction.norm()),bf16_direction_norm=float(bf_direction.norm()))
                assert direction_norms['fp32_direction_norm']>0 and direction_norms['bf16_direction_norm']>0,'zero arithmetic perturbation direction'
                del y,old,fp,bf
                directions=dict(fp32_rounding_direction=fp_direction,bf16_rounding_direction=bf_direction,
                                independent_skinny_direction=independent)
                observer_rows={};sensitivity={}
                for name in prefixes:
                    stage_fn=stages.get(name)
                    impl,precision=name.split('_')
                    pure=invoke_stage(prefixes[name],precision,cov,q)
                    observed=None;candidate_error=None;endpoint_error=None;observer_valid=False
                    if stage_fn is not None:
                        torch.cuda.synchronize();began=time.monotonic();before=graph_counts()
                        for _ in range(2):observed=invoke_stage(stage_fn,precision,cov,q)
                        torch.cuda.synchronize()
                        setup.append(dict(variant='observer_'+name,first_two_seconds=time.monotonic()-began,
                                          compiler_before=before,compiler_after=graph_counts()))
                        for _ in range(3):observed=invoke_stage(stage_fn,precision,cov,q)
                        with highest():candidate_error=relative(observed['candidate'],pure)
                    pure_endpoint=invoke_ret(rets[impl+'_old'],pure)
                    if observed is not None:
                        observed_endpoint=invoke_ret(rets[impl+'_old'],observed['candidate'])
                        with highest():endpoint_error=relative(observed_endpoint,pure_endpoint)
                        del observed_endpoint
                        observer_valid=candidate_error<=1e-6 and endpoint_error<=1e-6
                    observer_rows[name]=dict(executed=observed is not None,valid=observer_valid,
                                             candidate_relative=candidate_error,endpoint_relative=endpoint_error,
                                             scalars=stage_scalars(observed) if observed is not None else None,
                                             before=geometry_and_tracking(q,cov),after=geometry_and_tracking(pure_endpoint,cov),
                                             stage_attribution='eager-only' if observed is not None else 'unavailable: pure compiled prefix')
                    journal(dict(kind='observer_gate',dimension=n,spectrum=spectrum,layout=layout,defect=defect,
                                 prefix=name,observer=observer_rows[name]))
                    assert observed is None or observer_valid,'eager observer invalid'
                    base_ret={ret:invoke_ret(rets[impl+'_'+ret],pure) for ret in PROTOCOL['retractions']}
                    perturb_rows=[]
                    before=graph_counts()
                    for perturb in PROTOCOL['perturbations']:
                        with highest():perturbed=scaled_perturbation(q,directions[perturb['name']],perturb['relative_norm'])
                        perturbed=own(perturbed.contiguous() if layout=='row' else perturbed.T.contiguous().T)
                        q_hash=tensor_hash(perturbed)
                        with highest():input_relative=relative(perturbed,q);input_distance=float((perturbed-q).norm())
                        pstages=invoke_stage(stage_fn,precision,cov,perturbed) if stage_fn is not None else None
                        py=invoke_stage(prefixes[name],precision,cov,perturbed)
                        with highest():pobserver_error=relative(pstages['candidate'],py) if pstages is not None else None
                        journal(dict(kind='perturbed_observer_gate',dimension=n,spectrum=spectrum,layout=layout,defect=defect,
                                     prefix=name,perturbation=perturb['name'],observer_error=pobserver_error,input_relative=input_relative))
                        assert pstages is None or pobserver_error<=1e-6,'perturbed eager observer changed candidate'
                        with highest():y_distance=float((py-pure).norm())
                        endpoints={}
                        for ret in PROTOCOL['retractions']:
                            actual=invoke_ret(rets[impl+'_'+ret],py)
                            # Same base Y plus precisely the observed candidate difference, to isolate retraction.
                            with highest():endpoint_distance=float((actual-base_ret[ret]).norm())
                            mm=checked_metrics(actual,base_ret[ret],q,cov,world,counts)
                            endpoints[ret]=dict(endpoint_relative=mm['basis_relative'],metrics=mm,
                                composition_absolute_amplification=endpoint_distance/input_distance,
                                retraction_absolute_amplification=endpoint_distance/max(y_distance,1e-24))
                            del actual
                        perturb_rows.append(dict(name=perturb['name'],requested_relative=perturb['relative_norm'],
                            actual_input_relative=input_relative,input_distance=input_distance,
                            observer_valid_for_stages=observer_valid and pobserver_error is not None and pobserver_error<=1e-6,
                            perturbed_observer_relative=pobserver_error,stages=compare_stages(pstages,observed,input_relative) if pstages is not None else None,
                            base_scalars=stage_scalars(observed) if observed is not None else None,perturbed_scalars=stage_scalars(pstages) if pstages is not None else None,
                            pure_candidate_relative=relative(py,pure),prefix_absolute_amplification=y_distance/input_distance,
                            endpoints=endpoints,input_unchanged=q_hash==tensor_hash(perturbed)))
                        journal(dict(kind='sensitivity',dimension=n,spectrum=spectrum,layout=layout,defect=defect,
                                     prefix=name,result=perturb_rows[-1]))
                        assert perturb_rows[-1]['input_unchanged'],'perturbed Q mutated'
                        del perturbed,pstages,py
                    # Perturbations keep the declared input layouts; shape/precision is already compiled.
                    assert before==graph_counts(),'steady perturbation compilation/graphbreak'
                    sensitivity[name]=perturb_rows
                    del observed,pure,pure_endpoint,base_ret
                    guard()
                # Compare high vs highest on exactly common Q, with observer validity explicitly retained.
                high_stage=invoke_stage(stages['eager_high'],'high',cov,q)
                highest_stage=invoke_stage(stages['eager_highest'],'highest',cov,q)
                stage_policy_difference=dict(implementation='eager-only',valid=observer_rows['eager_high']['valid'] and observer_rows['eager_highest']['valid'],
                    stages=compare_stages(highest_stage,high_stage,1.),high=stage_scalars(high_stage),highest=stage_scalars(highest_stage))
                del high_stage,highest_stage
                before=graph_counts();samples={name:[] for name in whole};orders=[]
                order_rng=random.Random(149+n+int(defect*1e6)+(layout=='column')+(spectrum=='clustered'))
                for _ in range(7):
                    order=list(whole);order_rng.shuffle(order);orders.append(order)
                    for name in order:
                        output,sample=timed(whole[name],(cov,q));samples[name].append(sample);del output
                    guard()
                assert before==graph_counts(),'steady complete-call compilation/graphbreak'
                qs=dict.fromkeys(whole,q);recurrent={name:[] for name in whole};policy_recurrent=[]
                # Warm row Q prefixes explicitly for column-first safety (all current outputs are row).
                before_recurrent=graph_counts()
                for step in range(1,9):
                    qs={name:own(whole[name](cov,value)) for name,value in qs.items()}
                    if step in PROTOCOL['recurrent_observe']:
                        for name,actual in qs.items():
                            impl,precision,ret=name.split('_',2)
                            mm=checked_metrics(actual,qs['compiled_'+precision+'_old'],q,cov,world,counts)
                            with highest():mm['own_eager_relative']=relative(actual,qs['eager_'+precision+'_'+ret])
                            mm.update(step=step,own_eager_pass=mm['own_eager_relative']<=1e-4)
                            recurrent[name].append(mm)
                        policy_recurrent.append(dict(step=step,metrics=checked_metrics(qs['compiled_highest_old'],qs['compiled_high_old'],q,cov,world,counts)))
                        del actual
                    guard()
                assert before_recurrent==graph_counts(),'recurrent compilation/graphbreak'
                assert hashes==dict(q=tensor_hash(q),cov=tensor_hash(cov)),'common Q/C mutated'
                row=dict(dimension=n,spectrum=spectrum,layout=layout,defect=defect,input_hashes=hashes,inputs_unchanged=True,
                    q_stride=list(q.stride()),direction_norms=direction_norms,observer=observer_rows,sensitivity=sensitivity,
                    stage_policy_difference=stage_policy_difference,setup=setup,samples=samples,timing_order=orders,
                    recurrent=recurrent,policy_recurrent=policy_recurrent,steady_compiler_before=before,
                    steady_compiler_after=before_recurrent,recurrent_compiler_before=before_recurrent,recurrent_compiler_after=graph_counts())
                row['medians']={name:{k:statistics.median(v[k] for v in values) for k in ('device_ms','wall_ms')} for name,values in samples.items()}
                row['eligible']={name:all(v['mixed_gate']['passed'] and v['own_eager_pass'] for v in values)
                    and all(v['valid'] for v in observer_rows.values() if v['executed']) for name,values in recurrent.items()}
                journal(dict(kind='completed_case',row=row,counts=dict(counts)))
                rows.append(row)
                print('CASE '+json.dumps({k:row[k] for k in ('dimension','spectrum','layout','defect','medians','eligible')}),flush=True)
                del q,qs,fp_direction,bf_direction,directions
        del cov,base,d
    return rows


@torch.no_grad()
def run(bundle):
    m=verify(bundle);launch=json.loads((bundle/'launch_manifest.json').read_text())
    assert os.environ['GPU_CLAIM_INDICES']==str(launch['gpu']['index']) and os.environ['CUDA_VISIBLE_DEVICES']==launch['gpu']['uuid']
    assert torch.cuda.device_count()==1
    prefixes=('TORCH','TRITON','CUDA','CUBLAS','CUDNN','NVIDIA_TF32','OMP','MKL','PYTHONDONTWRITEBYTECODE')
    observed={k:v for k,v in os.environ.items() if k.startswith(prefixes)}
    assert observed==dict(launch['environment'],CUDA_VISIBLE_DEVICES=launch['gpu']['uuid'])
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    assert not torch.is_autocast_enabled()
    info=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),environment=observed,
              claim_indices=os.environ['GPU_CLAIM_INDICES'],uuid=os.environ['CUDA_VISIBLE_DEVICES'],
              bf16_reduced_precision_reduction=False,affinity=sorted(os.sched_getaffinity(0)),pid=os.getpid(),protocol_sha256=m['protocol_sha256'])
    assert info['torch']=='2.11.0+cu128' and info['cuda']=='12.8'
    write(bundle/'actual_environment.json',info)
    torch.cuda.reset_peak_memory_stats();generator=torch.Generator(device='cuda').manual_seed(149)
    counts=Counter();rows=[];releases=[];initial=graph_counts()
    def guard():
        assert time.monotonic()-STARTED<PROTOCOL['internal_seconds'],'runtime bound'
        assert torch.cuda.max_memory_allocated()<PROTOCOL['allocated_bytes'],'GPU memory bound'
        for name in ('whole','prefix','retraction','stage','metric'):assert counts[name]<=PROTOCOL[name+'_call_limit'],name+' call bound'
        cache=bundle/'compile_cache'
        assert sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())<PROTOCOL['compiler_cache_limit_bytes']
        assert sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file() and cache not in p.parents)<PROTOCOL['evidence_bytes']
    def save(status):write(bundle/'results.json',dict(status=status,rows=rows,counts=dict(counts),releases=releases,environment=info,
        all_in_seconds=time.monotonic()-STARTED,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        compiler_initial=initial,compiler_final=graph_counts(),scope=PROTOCOL['scope']))
    def journal(v):
        with (bundle/'probe_journal.jsonl').open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
    save('running')
    for n in PROTOCOL['dimensions']:
        owners=[];rows.extend(run_dimension(n,generator,counts,guard,owners,journal))
        torch._dynamo.reset();gc.collect();torch.cuda.empty_cache()
        alive=sum(ref() is not None for ref in owners);assert alive==0,'registered owner survived dimension release'
        releases.append(dict(dimension=n,registered_owners=len(owners),alive_after_release=alive,allocated_after_release=torch.cuda.memory_allocated()))
        guard();save('running')
    verify(bundle);guard();save('completed')
    print('COMPLETE '+json.dumps(dict(seconds=time.monotonic()-STARTED,counts=dict(counts),peak_allocated_bytes=torch.cuda.max_memory_allocated())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);b=parser.parse_args().bundle
    assert Path(__file__).resolve()==(b/'source/benchmarks/observer_free_prefix_fixture.py').resolve()
    assert not (b/'execution_status.json').exists(),'spent bundle: no retry'
    verify(b);write(b/'execution_status.json',dict(status='running',pid=os.getpid(),utc=time.time()))
    try:run(b)
    except BaseException as exc:
        write(b/'execution_status.json',dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc()));raise
    else:write(b/'execution_status.json',dict(status='completed',pid=os.getpid(),utc=time.time()))
