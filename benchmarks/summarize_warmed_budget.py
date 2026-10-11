"""Fail-closed qualification of the new warmed 90-second campaign."""
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
from timing_diagnostics import range_over_mean

KEYS=['step','validation','train_panel','model_sha256','optimizer_sha256','rng_sha256',
      'consumed_ids_sha256','augmented_batches','optimizer_counters','maximum_gram_error','mean_gram_error']

def fixed_observations(run):
    return [{k:o[k] for k in KEYS} for o in run['observations'] if 'fixed_updates' in o['reasons']]

def probes(run):
    return [{**{k:v for k,v in p.items() if k!='diagnostic_seconds'}} for p in run['tracking_probes']]

def repeat_gate(runs):
    # Qualification must use identical3500 work, never variable final lengths.
    fixed=[next(o for o in r['observations'] if o['step']==3500) for r in runs]
    walls=[o['training_seconds'] for o in fixed]
    opts=[o['timing']['optimizer_ms'] for o in fixed]
    wall,opt=range_over_mean(walls),range_over_mean(opts)
    return dict(wall_range_over_mean=wall,optimizer_range_over_mean=opt,
                mean_fixed_training_seconds=st.mean(walls),mean_fixed_optimizer_ms=st.mean(opts),
                repeat_gate=wall<=.03 and opt<=.02)

def endpoint_gate(run,target=90.,maximum_overshoot=1.):
    endpoints=[o for o in run['observations'] if 'equal_wall' in o['reasons']]
    if len(endpoints)!=1:return False
    e=endpoints[0]
    before=[h['training_seconds'] for h in run['history'] if h['step']<e['step']]
    return (run['wall_target_reached'] and target<=e['training_seconds']<=target+maximum_overshoot
            and (before[-1] if before else 0.)<target)

def summarize(b):
    d=json.loads((b/'screen.json').read_text());m=json.loads((b/'manifest.json').read_text())
    assert d['status']=='completed' and len(d['runs'])==6
    prior=Path(m['historical_bundle']);old=json.loads((prior/'screen.json').read_text());om=json.loads((prior/'manifest.json').read_text())
    assert hashlib.sha256((prior/'screen.json').read_bytes()).hexdigest()==m['historical_screen_sha256']
    assert hashlib.sha256((prior/'summary.json').read_bytes()).hexdigest()==m['prerequisite_summary_sha256']
    for k in ['split_sha256','base_prepared_sha256']:assert m[k]==om[k]
    for k in ['soap.py','soap_reference.py','benchmarks/cifar_optimizers.py','benchmarks/cifar_radius_learning_runner.py']:
        assert m['source_sha256'][k]==om['source_sha256'][k]
    for k in ['torch','cuda','initial_model_sha256']:assert d[k]==old[k]
    runs=d['runs'];assert [r['label'] for r in runs]==[x['label'] for x in d['config']['memory_screen']['arms']]
    records=json.loads((b/'affinity_records.json').read_text());restored=json.loads((b/'affinity_restored.json').read_text())
    assert len(records)==12 and len(restored)==2
    assert [r['label'] for r in records]==[r['label'] for r in runs for _ in range(2)]
    assert all([x[1] for x in r['threads']]==[116,116] for r in records)
    anchors=[];catalog={};observer=True;exact=True;warm=True;chunks=0;rows=[]
    for i,r in enumerate(runs):
        assert r['status']=='completed' and 3500<=r['steps']<=6000
        ref=next(x for x in old['runs'] if x['kind']==r['kind'])
        fo=fixed_observations(r);assert [o['step'] for o in fo]==[875,1750,3500]
        for x,z in zip(fo,ref['observations']):anchors.append(dict(label=r['label'],step=x['step'],exact=all(x[k]==z[k] for k in KEYS)))
        exact &= probes(r)==probes(ref)
        guard=r['refresh_guards'];assert sum(x['step']<=3500 for x in guard)==175
        exact &= all(x['factor_count']==48 and x['maximum_gram_error']<.05 for x in guard)
        mm=[x for p in r['tracking_probes'] for x in p['matrices']];assert len(mm)==168
        exact &= all(x['targeted_pass'] for x in mm)
        w=r['execution_warmup'];warm &= (w['steps']==400 and w['seconds']>0 and
            w['restored_model_sha256']==d['initial_model_sha256'] and w['guard']['factor_count']==48
            and w['guard']['maximum_gram_error']<.05)
        hh=r['history'];assert len(hh)==r['steps']//25 and [x['step'] for x in hh]==list(range(25,r['steps']+1,25))
        assert all(x['host']['steps']==25 for x in hh)
        assert sum('gpu_sample' in x['host'] for x in hh)==(chunks+len(hh))//10-chunks//10
        chunks+=len(hh)
        assert all(d['gpu_uuid'] in x['host']['gpu_sample'] for x in hh if 'gpu_sample' in x['host'])
        for x in hh:
            h=x['host']
            for k,v in h.get('affinity_definitions',{}).items():assert k not in catalog or catalog[k]==v;catalog[k]=v
            tids={t['tid']:t for t in h['thread_interval']};assert all(t['affinity_ref'] in catalog for t in tids.values())
            for tid,_ in records[2*i+1]['threads']:
                t=tids[tid];observer &= catalog[t['affinity_ref']]==[116] and t['last_cpu']==116 and t['runtime_ns']>0
        e=next((o for o in r['observations'] if 'equal_wall' in o['reasons']),None)
        rows.append(dict(label=r['label'],kind=r['kind'],steps=r['steps'],fixed_ce=fo[-1]['validation']['ce'],
            wall_ce=e['validation']['ce'] if e else None,endpoint_seconds=e['training_seconds'] if e else None,
            endpoint_step=e['step'] if e else None,overshoot_seconds=e['training_seconds']-90 if e else None,
            warmup_seconds=w['seconds'],training_seconds=r['training_seconds'],all_in_seconds=r['all_in_seconds']))
    conditions=[]
    for kind in ['qr','fro']:
        xs=[r for r in runs if r['kind']==kind];assert len(xs)==3
        exact &= all(fixed_observations(x)==fixed_observations(xs[0]) and probes(x)==probes(xs[0]) for x in xs)
        warm &= all(all(x['execution_warmup'][k]==xs[0]['execution_warmup'][k] for k in
                    ['model_sha256','optimizer_sha256','optimizer_counters','restored_model_sha256','restored_rng_sha256']) for x in xs)
        cc=repeat_gate(xs);ends=[x for x in rows if x['kind']==kind]
        cc.update(kind=kind,endpoint_steps=[x['endpoint_step'] for x in ends],endpoint_ces=[x['wall_ce'] for x in ends],
            mean_wall_ce=st.mean(x['wall_ce'] for x in ends) if all(x['wall_ce'] is not None for x in ends) else None,
            all_endpoints_qualified=all(endpoint_gate(x) for x in xs));conditions.append(cc)
    compiler=all(r['telemetry_start']['compiler_counters']==r['telemetry_end']['compiler_counters'] for r in runs)
    exact &= len(anchors)==18 and all(x['exact'] for x in anchors)
    qualified=exact and warm and observer and compiler and all(x['repeat_gate'] and x['all_endpoints_qualified'] for x in conditions)
    s=dict(status='completed',qualified=qualified,numerical_exact=exact,warmup_exact_and_restored=warm,
        observer_complete=observer,compiler_stable=compiler,conditions=conditions,rows=rows,historical_anchors=anchors,
        primary_updates=sum(r['steps'] for r in runs),warmup_updates=2400,
        refresh_guards=sum(len(r['refresh_guards']) for r in runs),matrix_probes=1008,
        full_strict_reference_failures=sum(not x['full_strict_pass'] for r in runs for p in r['tracking_probes'] for x in p['matrices']),
        all_in_seconds=d['all_in_seconds'],scope='New90-second warmed training-time endpoints, one seed, instrumented scope only')
    (b/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps({k:v for k,v in s.items() if k not in ['rows','historical_anchors']},indent=2));return s


if __name__=='__main__':summarize(Path(sys.argv[1]))
