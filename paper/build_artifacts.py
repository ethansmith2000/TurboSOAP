"""Generate and check manuscript tables/provenance from saved evidence, CPU only.

No optimizer imports, training, data downloads, or GPU admission.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCES={
    'seed204':'reports/bf16_replicate_20261027_seed204/comparison.json',
    'seed205':'reports/bf16_full_horizon_20261026_seed205/comparison.json',
    'controls':'reports/bf16_refresh_controls_20261027/comparison.json',
    'profile':'reports/bf16_qr_attribution_20261027/result.json',
    'pilot':'reports/bf16_learning_20261026/comparison.json',
    'replication_manifest':'reports/bf16_replicate_20261027_seed204/manifest.json',
    'reference_data':'reports/owt_restart_20261026/provenance.json',
    'replication_audit':'reports/bf16_replicate_20261027_seed204/completion_audit.json',
    'controls_audit':'reports/bf16_refresh_controls_20261027/completion_audit.json',
    'profile_audit':'automation/qr_profile_review.json',
    'supervisor_audit':'reports/bf16_completion_20261027/final_supervisor_audit.json',
    'stall_example':'reports/review_20261026/equal_diagonal_counterexample.json',
}
FIGURES={
    'figures/full_horizon.svg':'reports/bf16_completion_20261027/full_horizon.svg',
    'figures/controls_and_profile.svg':'reports/bf16_completion_20261027/controls_and_profile.svg',
}


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads((ROOT/path).read_text())


def encode(value):return json.dumps(value,indent=2,allow_nan=False)+'\n'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
        '| '+' | '.join(['---']+['---:']*(len(headers)-1))+' |',
        *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def calculate():
    data={k:read(v) for k,v in SOURCES.items()}
    for k in ('seed204','seed205','controls','profile','pilot'):assert data[k]['status']=='completed',k
    for k in ('replication_audit','controls_audit','profile_audit'):assert data[k]['status']=='passed',k
    assert all(i['statename']=='EXITED' and i['exitstatus']==0 for i in data['supervisor_audit'])
    pins={n:data['replication_manifest']['source_files'][n] for n in ('bf16_soap.py','soap_reference.py','fast_soap.py','soap.py')}
    for n,h in pins.items():assert digest(ROOT/n)==h,f'Production source changed: {n}'
    pairs=[(204,data['seed204']['cached_quality_reference'],data['seed204']['candidate']),
        (205,*data['seed205']['results'])]
    quality=[]
    for seed,r,c in pairs:
        assert r['seed']==c['seed']==seed and r['calls']==c['calls']==10000
        assert r['initial_parameters_sha256']==c['initial_parameters_sha256']
        assert r['initial_gradient_sha256']==c['initial_gradient_sha256']
        assert r['validation'][0]['ce']==c['validation'][0]['ce']
        delta=c['validation'][-1]['ce']-r['validation'][-1]['ce']
        rv={v['call']:v['ce'] for v in r['validation']}
        peak=max((dict(call=v['call'],delta=v['ce']-rv[v['call']]) for v in c['validation']),key=lambda v:v['delta'])
        quality.append(dict(seed=seed,stock_ce=r['validation'][-1]['ce'],bf16_ce=c['validation'][-1]['ce'],
            delta=delta,perplexity_increase_percent=100*math.expm1(delta),peak_gap=peak))
    timing=[]
    p=data['seed204']['paired_prefix_timing'];assert p['calls']==[101,200] and p['number_of_calls']==100
    timing.append(dict(seed=204,calls=p['calls'],call_count=100,stock_ms=p['mean_step_ms']['qr10'],bf16_ms=p['mean_step_ms']['bf16']))
    a,b=data['seed205']['results'];assert a['clean_cycle_calls']==b['clean_cycle_calls']==9900
    timing.append(dict(seed=205,calls=[101,10000],call_count=9900,stock_ms=a['mean_step_ms'],bf16_ms=b['mean_step_ms']))
    for t in timing:t['time_reduction_percent']=100*(1-t['bf16_ms']/t['stock_ms'])
    controls=[]
    warm=data['controls']['cached_warm_prefix'];warm_end=warm['validation'][-1]['ce']
    assert warm_end==next(v['ce'] for v in data['seed204']['candidate']['validation'] if v['call']==2000)
    labels={'cached_bf16_warm10_qr50':'Warm10/QR50 (cached)','bf16_qr20':'QR20','bf16_qr50':'QR50'}
    for r in [warm,*data['controls']['results']]:
        assert r['clean_cycle_calls']==1900 and r['validation'][-1]['call']==2000
        controls.append(dict(policy=labels[r['case']],final_ce=r['validation'][-1]['ce'],
            delta_vs_warm=r['validation'][-1]['ce']-warm_end,step_ms=r['mean_step_ms'],optimizer_ms=r['mean_optimizer_ms']))
    profile=[]
    for r in data['profile']['rows']:
        assert r['inputs_preserved'] and r['numerical']['finite'] and r['numerical']['variance_nonnegative']
        if r['batch']!=12:continue
        t=r['timing'];n=t['original']['median_cuda_ms'];c=t['reuse']['median_cuda_ms']
        profile.append(dict(dimension=r['dimension'],batch=12,original_cuda_ms=n,reuse_cuda_ms=c,
            time_reduction_percent=100*(1-c/n),order_disagreements=r['numerical']['order_disagreements'],
            total_columns=r['numerical']['total_columns']))
    pilot=[]
    plabels={'qr10':'Stock QR10','original':'Original warm, FP32 state',
        'compiled_fp32':'Compiled FP32 storage / BF16 arithmetic','bf16':'Full BF16',
        'bf16_fp32_basis':'BF16 M/V/C, FP32 bases'}
    for r in data['pilot']['results']:
        assert r['calls']==2000 and r.get('seed',data['pilot']['seed'])==204
        pilot.append(dict(policy=plabels[r['case']],final_ce=r['validation'][-1]['ce'],
            step_ms=r['mean_step_ms'],state_gib=r['matrix_state_bytes']/2**30))
    # Keep the original-reference prefix explicit as a mixed-implementation
    # context row; its timing is from the development pilot, not the controls.
    stock_pilot=data['pilot']['results'][0]
    controls.insert(0,dict(policy='Stock QR10 (pilot)',final_ce=stock_pilot['validation'][-1]['ce'],
        delta_vs_warm=stock_pilot['validation'][-1]['ce']-warm_end,
        step_ms=stock_pilot['mean_step_ms'],optimizer_ms=stock_pilot['mean_optimizer_ms']))
    for row,source in zip(controls,[stock_pilot,warm,*data['controls']['results']]):
        row['ce_at_500']=next(v['ce'] for v in source['validation'] if v['call']==500)
    assert warm_end==data['pilot']['results'][3]['validation'][-1]['ce'],'Repeated prefix is not independent replication'
    shapes=[(2304,768),(768,768),(3072,768),(768,1536)]
    tensor_elements=12*sum(2*m*n+2*m*m+2*n*n for m,n in shapes)
    assert a['matrix_state_bytes']==4*tensor_elements
    assert b['matrix_state_bytes']==2*tensor_elements+32
    tables={
        'quality':table(['Seed','Stock QR10 CE','Full BF16 CE','CE increase','Perplexity increase'],
            [[r['seed'],f"{r['stock_ce']:.6f}",f"{r['bf16_ce']:.6f}",f"+{r['delta']:.6f}",f"+{r['perplexity_increase_percent']:.3f}%"] for r in quality]),
        'timing':table(['Seed','Matched calls','Stock step ms','BF16 step ms','Time reduction'],
            [[r['seed'],f"{r['calls'][0]}–{r['calls'][1]}",f"{r['stock_ms']:.3f}",f"{r['bf16_ms']:.3f}",f"{r['time_reduction_percent']:.2f}%"] for r in timing]),
        'controls':table(['Policy','CE at 500','CE at 2000','Step ms','Optimizer ms'],
            [[r['policy'],f"{r['ce_at_500']:.6f}",f"{r['final_ce']:.6f}",f"{r['step_ms']:.3f}",f"{r['optimizer_ms']:.3f}"] for r in controls]),
        'profile':table(['Dimension (batch 12)','Original CUDA ms','Reuse CUDA ms','Time reduction','Reordered columns'],
            [[r['dimension'],f"{r['original_cuda_ms']:.3f}",f"{r['reuse_cuda_ms']:.3f}",f"{r['time_reduction_percent']:.2f}%",f"{r['order_disagreements']}/{r['total_columns']}"] for r in profile]),
        'pilot':table(['Development arm','CE at 2000','Step ms','Matrix-state GiB'],
            [[r['policy'],f"{r['final_ce']:.6f}",f"{r['step_ms']:.3f}",f"{r['state_gib']:.3f}"] for r in pilot]),
    }
    evidence=dict(status='draft_evidence_verified',model_seeds=[204,205],quality=quality,timing=timing,controls=controls,
        batch12_profile=profile,development_pilot=pilot,production_source_sha256=pins,
        memory=dict(stock_matrix_state_bytes=a['matrix_state_bytes'],bf16_matrix_state_bytes=b['matrix_state_bytes'],
            stock_peak_gpu_bytes=a['peak_gpu_bytes'],bf16_peak_gpu_bytes=b['peak_gpu_bytes']),
        mean_paired_ce_delta=sum(r['delta'] for r in quality)/2,
        inputs={k:dict(path=v,sha256=digest(ROOT/v)) for k,v in SOURCES.items()},
        figures={n:dict(source=v,sha256=digest(ROOT/v)) for n,v in FIGURES.items()},
        review_figure=dict(path='figures/review_tradeoff.svg',
            sha256=digest(HERE/'figures/review_tradeoff.svg'),
            manifest='review_figure.json',manifest_sha256=digest(HERE/'review_figure.json')),
        numerical_error_role='diagnostic',training_or_gpu_execution_in_this_build=False)
    dm=data['reference_data']['source_manifest']
    config=dict(model=dict(parameters=109396992,width=768,layers=12,heads=12,ff_hidden=1536,
        vocab=50257,sequence_length=1024,tied_embeddings=True,rope=True,qk_norm=True,bias=False,dropout=0),
        data={k:dm[k] for k in ('dataset','dataset_revision','tokenizer','tokenizer_revision','tokenizer_sha256','split_policy','packing','causal_policy')},
        selected_rows=dict(train=320000,validation=1024),row_selection_seeds=dict(train=2205,validation=4205),
        data_sha256=data['replication_manifest']['inputs'],
        optimization=dict(matrix_betas=[.95,.95],covariance_beta=.95,matrix_peak_lr=.0015,
            auxiliary_betas=[.9,.999],auxiliary_peak_lr=.0006,epsilon=1e-8,decay_for_declared_groups=.05,clip_norm=1.,
            global_batch=32,microbatch=8,accumulation=4,calls=10000,matrix_updates=9999,auxiliary_updates=10000,
            targets_per_call=32*1023,warmup_calls=200,min_lr_ratio=.1,schedule='linear warmup then cosine',
            state_dtype='bfloat16',basis_dtype='bfloat16',matrix_compute_dtype='bfloat16',master_parameter_dtype='float32',
            refresh_every=10,qr_repair_every=50,warm_damping=.01,warm_motion_cap=.5,residual_corrections=2),
        recorded_runtime=dict(torch='2.11.0+cu128',cuda='12.8',gpu='NVIDIA GeForce RTX 5090',
            float32_matmul_precision='high',allow_bf16_reduced_precision_reduction=False,
            geometry_float32_matmul_precision='highest',deterministic_algorithms=True,
            torch_num_threads=4,
            service_environment=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',
                OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',
                TORCHINDUCTOR_COMPILE_THREADS='2',HF_HUB_OFFLINE='1',
                PYTHONDONTWRITEBYTECODE='1'),
            compiler_cache_policy='Bundle-local compiler_cache, shared sequentially within each campaign'),
        production_source_sha256=pins)
    return tables,evidence,config


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    tables,evidence,config=calculate();draft_path=HERE/'DRAFT.md';draft=draft_path.read_text()
    for name,t in tables.items():
        pattern=r'<!-- BEGIN TABLE '+name+r' -->.*?<!-- END TABLE '+name+r' -->'
        body=f'<!-- BEGIN TABLE {name} -->\n{t}\n<!-- END TABLE {name} -->'
        assert len(re.findall(pattern,draft,flags=re.S))==1,name
        if args.check:assert re.search(pattern,draft,flags=re.S).group()==body,f'Manuscript table changed: {name}'
        else:draft=re.sub(pattern,lambda _:body,draft,flags=re.S)
    for name,source in FIGURES.items():assert digest(HERE/name)==digest(ROOT/source),name
    review=json.loads((HERE/'review_figure.json').read_text())
    assert digest(HERE/review['generator'])==review['generator_sha256']
    for path,h in review['inputs'].items():assert digest(ROOT/path)==h,path
    for path,h in review['outputs'].items():assert digest(HERE/path)==h,path
    if not args.check:draft_path.write_text(draft)
    outputs={'evidence.json':encode(evidence),'configuration.json':encode(config),
        'tables.md':'# Generated tables\n\n'+'\n\n'.join(f'## {name}\n\n{t}' for name,t in tables.items())+'\n'}
    for name,value in outputs.items():
        target=HERE/name
        if args.check:assert target.read_text()==value,f'Generated artifact changed: {name}'
        else:target.write_text(value)
    print(encode(dict(status='passed',mode='check' if args.check else 'build',tables=len(tables),
        source_artifacts=len(SOURCES),frozen_figures=len(FIGURES),review_figures=1,production_source_pins=4,
        gpu_execution=False,mean_paired_ce_delta=evidence['mean_paired_ce_delta'])).strip())


if __name__=='__main__':main()
