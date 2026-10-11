"""Fail-closed numerical, observer, replay and local timing qualification."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
try:
    from .timing_diagnostics import range_over_mean
    from .analyze_timing_traces import analyze_trace
except ImportError:
    from timing_diagnostics import range_over_mean
    from analyze_timing_traces import analyze_trace

KEYS=['step','validation','train_panel','model_sha256','rng_sha256','consumed_ids_sha256',
      'augmented_batches','optimizer_counters','maximum_gram_error','mean_gram_error']
def controls(run):return [{k:x[k] for k in [*KEYS,'optimizer_sha256']} for x in run['observations']]
def probes(run):return [{k:v for k,v in p.items() if k!='diagnostic_seconds'} for p in run['tracking_probes']]
def signature(trace):
    return sorted((list(k),n) for k,n in Counter((e['cat'],e['name']) for e in trace['traceEvents']
        if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset'] and e.get('ph')=='X').items())

def material_effect(conditions, pairs, exact, repeats):
    a,z=conditions
    effect=z['model_mean_ms']/a['model_mean_ms']-1
    separated=a['model_max_ms']<z['model_min_ms'] or z['model_max_ms']<a['model_min_ms']
    direction=all(x*effect>0 for x in pairs)
    return effect, exact and repeats and abs(effect)>=.05 and separated and direction

def summarize(b):
    d=json.loads((b/'screen.json').read_text());m=json.loads((b/'manifest.json').read_text())
    assert d['status']=='completed' and len(d['runs'])==6
    for field,digest in [('audit_path','audit_sha256')]:
        assert hashlib.sha256(Path(m[field]).read_bytes()).hexdigest()==m[digest]
    history=Path(m['historical_bundle']);variance=Path(m['variance_bundle'])
    for p,key in [(history/'screen.json','historical_screen_sha256'),(variance/'screen.json','variance_screen_sha256')]:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==m[key]
    old=json.loads((history/'screen.json').read_text());ref=next(x for x in old['runs'] if x['label']=='near_0')
    vv=json.loads((variance/'screen.json').read_text());vref=next(x for x in vv['runs'] if x['label']=='qr_carry_beta0.99')
    assert d['initial_model_sha256']==vv['initial_model_sha256']==old['initial_model_sha256']
    for name in ['soap.py','soap_reference.py','benchmarks/cifar_optimizers.py','benchmarks/cifar_radius_learning_runner.py']:
        assert m['source_sha256'][name]==json.loads((variance/'manifest.json').read_text())['source_sha256'][name]
    assert d['torch']==vv['torch'] and d['cuda']==vv['cuda']
    runs=d['runs'];expected=d['config']['memory_screen']['arms'];assert [x['label'] for x in runs]==[x['label'] for x in expected]
    records=json.loads((b/'affinity_records.json').read_text());restored=json.loads((b/'affinity_restored.json').read_text())
    assert len(records)==12 and len(restored)==2
    labels=[x['label'] for x in runs]+[runs[4]['label']]*3+[runs[5]['label']]*3
    assert [x['label'] for x in records]==labels
    for record in records:
        run=next(x for x in runs if x['label']==record['label']);assert [x[1] for x in record['threads']]==[116,116 if run['kind']=='same' else 117]
    numerical=True;anchors=[];thread_rows=[];observer_complete=True
    for run,record in zip(runs,records[:6]):
        assert run['status']=='completed' and run['steps']==1000
        assert [x['step'] for x in run['observations']]==[875,1000]
        for a,z in zip(run['observations'],ref['observations']):
            anchors.append(dict(label=run['label'],step=a['step'],exact=all(a[k]==z[k] for k in KEYS)))
        numerical &= run['optimizer_sha256']==ref['optimizer_sha256'] and probes(run)==probes(ref)
        numerical &= run['observations'][0]['optimizer_sha256']==vref['observations'][0]['optimizer_sha256']
        gs=run['refresh_guards'];assert len(gs)==50
        mm=[x for p in run['tracking_probes'] for x in p['matrices']];assert len(mm)==120
        numerical &= all(x['factor_count']==48 and x['maximum_gram_error']<.05 for x in gs) and all(x['targeted_pass'] for x in mm)
        hh=run['history'];assert len(hh)==40 and all(x['host']['steps']==25 for x in hh)
        assert sum('gpu_sample' in x['host'] for x in hh)==4
        assert all(d['gpu_uuid'] in x['host']['gpu_sample'] for x in hh if 'gpu_sample' in x['host'])
        for x in hh:
            h=x['host'];assert h['observation_seconds']>=0
            assert all(v['main_user_cpu_ms']>=0 and v['main_system_cpu_ms']>=0 for v in h['phases'].values())
            tids={q['tid']:q for q in h['thread_interval']}
            for tid,cpu in record['threads']:
                q=tids[tid];observer_complete &= q['affinity']==[cpu] and q['last_cpu']==cpu and q['runtime_ns']>0
        main,worker=[x[0] for x in record['threads']]
        all_tids={q['tid'] for x in hh for q in x['host']['thread_interval']}
        def total(tid,key):return sum(q[key] for x in hh for q in x['host']['thread_interval'] if q['tid']==tid)
        thread_rows.append(dict(label=run['label'],kind=run['kind'],main_runtime_seconds=total(main,'runtime_ns')/1e9,
            autograd_runtime_seconds=total(worker,'runtime_ns')/1e9,main_runqueue_wait_seconds=total(main,'runqueue_wait_ns')/1e9,
            autograd_runqueue_wait_seconds=total(worker,'runqueue_wait_ns')/1e9,
            other_runtime_seconds=sum(total(t,'runtime_ns') for t in all_tids-{main,worker})/1e9,
            observation_seconds=sum(x['host']['observation_seconds'] for x in hh),
            phase_cpu={phase:{key:sum(x['host']['phases'][phase][key] for x in hh)/1000 for key in ['host_wall_ms','thread_cpu_ms','process_cpu_ms','main_user_cpu_ms','main_system_cpu_ms']} for phase in hh[0]['host']['phases']}))
    exact=numerical and all(x['exact'] for x in anchors) and all(controls(x)==controls(runs[0]) for x in runs)
    compiler=all(x['telemetry_start']['compiler_counters']==x['telemetry_end']['compiler_counters'] for x in runs)
    conditions=[]
    for kind in ['near','same']:
        xs=[x for x in runs if x['kind']==kind];assert len(xs)==3
        wall=range_over_mean([x['training_seconds'] for x in xs]);opt=range_over_mean([x['optimizer_ms'] for x in xs])
        conditions.append(dict(kind=kind,wall_range_over_mean=wall,optimizer_range_over_mean=opt,repeat_gate=wall<=.03 and opt<=.02,
            training_mean_seconds=st.mean(x['training_seconds'] for x in xs),model_mean_ms=st.mean(x['model_backward_cuda_ms'] for x in xs),
            model_min_ms=min(x['model_backward_cuda_ms'] for x in xs),model_max_ms=max(x['model_backward_cuda_ms'] for x in xs),
            optimizer_mean_ms=st.mean(x['optimizer_ms'] for x in xs)))
    pairs=[]
    for i,j in [(0,1),(3,2),(4,5)]:pairs.append(runs[j]['model_backward_cuda_ms']/runs[i]['model_backward_cuda_ms']-1)
    repeats=all(x['repeat_gate'] for x in conditions)
    effect,material=material_effect(conditions,pairs,exact and compiler and observer_complete,repeats)
    replays=d['profile_replays'];assert len(replays)==6
    replay_exact=all(x['exact'] and x['numerical_control']==replays[0]['numerical_control'] for x in replays)
    traces=[];signatures=[]
    for run in replays:
        if 'profiler' not in run:continue
        p=b/run['profiler']['trace']
        with gzip.open(p,'rt') as f:trace=json.load(f)
        row=analyze_trace(trace);assert row['gpu_work']['count']>0 and row['unattributed_gpu_events']==0
        assert len(row['phases'])==3 and all(x['host_calls']==4 and x['gpu_work']['count']>0 for x in row['phases'].values())
        traces.append(dict(kind=run['kind'],mode=run['mode'],trace=p.name,**row));signatures.append(signature(trace))
    assert len(traces)==4
    signature_exact=all(x==signatures[0] for x in signatures)
    # Trace signature differences are diagnostic; exact numerical replay/completeness remain mandatory.
    qualified=exact and compiler and observer_complete and replay_exact and repeats
    result=dict(status='completed',numerical_exact=exact,historical_anchors=anchors,observer_complete=observer_complete,
        compiler_stable=compiler,local_repeat_qualified=qualified,material_placement_effect=material and replay_exact,
        same_over_near_model_effect=effect,adjacent_pair_effects=pairs,conditions=conditions,thread_rows=thread_rows,
        profile_replays_exact=replay_exact,profile_gpu_operation_signatures_exact=signature_exact,
        traces=traces,refresh_guards=300,matrix_probes=720,
        full_strict_reference_failures=sum(not x['full_strict_pass'] for r in runs for p in r['tracking_probes'] for x in p['matrices']),
        primary_steps=6000,all_in_seconds=d['all_in_seconds'],scope='Instrumented1000-update QR placement diagnostic; no3500/full-policy/wall-quality/default promotion')
    (b/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(b/'trace_summary.json').write_text(json.dumps(traces,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['traces','thread_rows','historical_anchors']},indent=2));return result

if __name__=='__main__':summarize(Path(sys.argv[1]))
