"""Fail-closed full-horizon, two-policy co-location timing qualification."""
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
from timing_diagnostics import range_over_mean

KEYS=['step','validation','train_panel','model_sha256','optimizer_sha256','rng_sha256',
      'consumed_ids_sha256','augmented_batches','optimizer_counters','maximum_gram_error','mean_gram_error']
def observations(run):return [{k:o[k] for k in KEYS} for o in run['observations']]
def probes(run,strip_variance=False):
    return [{**{k:v for k,v in p.items() if k not in ['diagnostic_seconds','matrices']},
        'matrices':[{k:v for k,v in x.items() if not(strip_variance and k=='variance')} for x in p['matrices']]}
        for p in run['tracking_probes']]

def summarize(b):
    d=json.loads((b/'screen.json').read_text());m=json.loads((b/'manifest.json').read_text())
    assert d['status']=='completed' and len(d['runs'])==6
    history=Path(m['historical_bundle']);assert hashlib.sha256((history/'screen.json').read_bytes()).hexdigest()==m['historical_screen_sha256']
    prior=Path(m['prerequisite']);assert hashlib.sha256((prior/'summary.json').read_bytes()).hexdigest()==m['prerequisite_summary_sha256']
    old=json.loads((history/'screen.json').read_text());om=json.loads((history/'manifest.json').read_text())
    for k in ['split_sha256','base_prepared_sha256']:assert m[k]==om[k]
    for k in ['soap.py','soap_reference.py','benchmarks/cifar_optimizers.py','benchmarks/cifar_radius_learning_runner.py']:
        assert m['source_sha256'][k]==om['source_sha256'][k]
    for k in ['torch','cuda','initial_model_sha256']:assert d[k]==old[k]
    runs=d['runs'];assert [r['label'] for r in runs]==[a['label'] for a in d['config']['memory_screen']['arms']]
    records=json.loads((b/'affinity_records.json').read_text());restored=json.loads((b/'affinity_restored.json').read_text())
    assert len(records)==12 and len(restored)==2 and all([x[1] for x in r['threads']]==[116,116] for r in records)
    assert [r['label'] for r in records]==[r['label'] for r in runs]+[runs[4]['label']]*3+[runs[5]['label']]*3
    anchors=[];catalog={};observer=True;numerical=True;rows=[]
    for r,record in zip(runs,records[:6]):
        assert r['status']=='completed' and r['steps']==3500 and [o['step'] for o in r['observations']]==[875,1750,3500]
        ref=next(x for x in old['runs'] if x['label']==f"{'qr' if r['kind']=='qr' else 'fro'}_carry_beta0.99")
        for a,z in zip(r['observations'],ref['observations']):anchors.append(dict(label=r['label'],step=a['step'],exact=all(a[k]==z[k] for k in KEYS)))
        numerical &= probes(r)==probes(ref,True)
        assert len(r['refresh_guards'])==175
        mm=[x for p in r['tracking_probes'] for x in p['matrices']];assert len(mm)==168
        numerical &= all(x['targeted_pass'] for x in mm) and all(x['factor_count']==48 and x['maximum_gram_error']<.05 for x in r['refresh_guards'])
        hh=r['history'];assert len(hh)==140 and all(x['host']['steps']==25 for x in hh)
        assert sum('gpu_sample' in x['host'] for x in hh)==14
        assert all(d['gpu_uuid'] in x['host']['gpu_sample'] for x in hh if 'gpu_sample' in x['host'])
        for x in hh:
            h=x['host']
            for k,v in h.get('affinity_definitions',{}).items():assert k not in catalog or catalog[k]==v;catalog[k]=v
            tids={t['tid']:t for t in h['thread_interval']}
            assert all(t['affinity_ref'] in catalog for t in tids.values())
            for tid,_ in record['threads']:
                t=tids[tid];observer &= catalog[t['affinity_ref']]==[116] and t['last_cpu']==116 and t['runtime_ns']>0
        main,worker=[x[0] for x in record['threads']]
        total=lambda tid,key:sum(t[key] for x in hh for t in x['host']['thread_interval'] if t['tid']==tid)
        rows.append(dict(label=r['label'],kind=r['kind'],training_seconds=r['training_seconds'],optimizer_ms=r['optimizer_ms'],
            model_event_ms=r['model_backward_cuda_ms'],ce=r['observations'][-1]['validation']['ce'],
            main_runtime_seconds=total(main,'runtime_ns')/1e9,autograd_runtime_seconds=total(worker,'runtime_ns')/1e9,
            autograd_runqueue_wait_seconds=total(worker,'runqueue_wait_ns')/1e9,
            observer_seconds=sum(x['host']['observation_seconds'] for x in hh)))
    conditions=[]
    for kind in ['qr','fro']:
        xs=[r for r in runs if r['kind']==kind];assert len(xs)==3
        exact=all(observations(x)==observations(xs[0]) and probes(x)==probes(xs[0]) and x['optimizer_sha256']==xs[0]['optimizer_sha256'] for x in xs)
        numerical &= exact
        wall=range_over_mean([x['training_seconds'] for x in xs]);opt=range_over_mean([x['optimizer_ms'] for x in xs])
        conditions.append(dict(kind=kind,exact_controls=exact,wall_range_over_mean=wall,optimizer_range_over_mean=opt,
            repeat_gate=wall<=.03 and opt<=.02,mean_training_seconds=st.mean(x['training_seconds'] for x in xs),
            mean_optimizer_ms=st.mean(x['optimizer_ms'] for x in xs),mean_model_ms=st.mean(x['model_backward_cuda_ms'] for x in xs)))
    replays=d['profile_replays'];assert len(replays)==6
    replay_exact=all(len(xs:=[r for r in replays if r['kind']==kind])==3 and all(x['exact'] and x['numerical_control']==xs[0]['numerical_control'] for x in xs) for kind in ['qr','fro'])
    profiles=[r['profiler'] for r in replays if 'profiler' in r];assert len(profiles)==4
    profile_complete=all(p['device_events']>0 and p['device_sum_us']>0 and p['unattributed_gpu_events']==0 and
        all(p['phases'].get('timing/'+phase,{}).get('host_calls')==4 and p['phases']['timing/'+phase]['gpu_work']['count']>0 for phase in ['augmentation','model_backward','optimizer']) for p in profiles)
    compiler=all(r['telemetry_start']['compiler_counters']==r['telemetry_end']['compiler_counters'] for r in runs)
    exact=numerical and len(anchors)==18 and all(x['exact'] for x in anchors)
    result=dict(status='completed',qualified=exact and observer and compiler and replay_exact and profile_complete and all(x['repeat_gate'] for x in conditions),
        numerical_exact=exact,observer_complete=observer,compiler_stable=compiler,profile_replays_exact=replay_exact,
        corrected_profiles_complete=profile_complete,conditions=conditions,rows=rows,historical_anchors=anchors,
        refresh_guards=1050,matrix_probes=1008,primary_updates=21000,all_in_seconds=d['all_in_seconds'],
        full_strict_reference_failures=sum(not x['full_strict_pass'] for r in runs for p in r['tracking_probes'] for x in p['matrices']),
        scope='Full3500 QR/Fro same-core instrumented timing qualification only; no measured-wall endpoint or permanent default')
    (b/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','historical_anchors']},indent=2));return result

if __name__=='__main__':summarize(Path(sys.argv[1]))
