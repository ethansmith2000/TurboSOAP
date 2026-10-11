"""Fail-closed wide fixed-work learning and separate three-repeat timing gates."""
import hashlib,json,math
from pathlib import Path
import sys
try:
    from .wide_learning_checks import chunk_boundaries,timing_repeat
except ImportError:
    from wide_learning_checks import chunk_boundaries,timing_repeat
KEYS=['step','validation','train_panel','evaluation_parity','model_sha256','optimizer_sha256',
    'rng_sha256','consumed_ids_sha256','augmented_batches','optimizer_counters','maximum_gram_error','mean_gram_error']

def observations(r):return [{k:o[k] for k in KEYS} for o in r['observations']]
def probes(r):return [{k:v for k,v in p.items() if k!='diagnostic_seconds'} for p in r['tracking_probes']]
def comparable(r):
    return dict(observations=observations(r),optimizer=r['optimizer_sha256'],probes=probes(r),
        guards=[{k:v for k,v in g.items() if k!='diagnostic_seconds'} for g in r['refresh_guards']],
        chunks=[(h['step'],h['loss'],h['maximum_gradient_norm']) for h in r['history']])

def summarize(b):
    d=json.loads((b/'screen.json').read_text());m=json.loads((b/'manifest.json').read_text())
    for name,digest in m['source_sha256'].items():assert hashlib.sha256((b/'source'/name).read_bytes()).hexdigest()==digest,name
    for name,key in [('config.json','config_sha256'),('PROTOCOL.md','protocol_sha256'),('split.json','split_sha256'),('data_prepared.json','base_prepared_sha256')]:assert hashlib.sha256((b/name).read_bytes()).hexdigest()==m[key],name
    prior=Path(m['historical_bundle']);old=json.loads((prior/'screen.json').read_text());pm=json.loads((prior/'manifest.json').read_text())
    for name,key in [('screen.json','historical_screen_sha256'),('summary.json','historical_summary_sha256')]:assert hashlib.sha256((prior/name).read_bytes()).hexdigest()==m[key],name
    prerequisite=Path(m['prerequisite_bundle']);assert hashlib.sha256((prerequisite/'summary.json').read_bytes()).hexdigest()==m['prerequisite_summary_sha256']
    for name in ['soap.py','soap_reference.py','benchmarks/cifar_optimizers.py','benchmarks/wide_probe.py','benchmarks/movement_retraction.py','benchmarks/warm_retraction.py']:
        assert m['source_sha256'][name]==pm['source_sha256'][name],name
    for k in ['split_sha256','base_prepared_sha256']:assert m[k]==pm[k]
    for k in ['torch','cuda','initial_model_sha256']:assert d[k]==old[k]
    assert d['status']=='completed' and len(d['runs'])==6 and d['parameter_count']==28660324 and d['smoke']['matrix_released']
    assert sorted(tuple(x['shape']) for x in d['matrix_inventory'])==sorted([(4608,1536),(1536,1536),(6144,1536),(1536,6144)])
    runs=d['runs'];assert [r['label'] for r in runs]==['qr_0','fro_1','fro_2','qr_3','qr_4','fro_5']
    rec=json.loads((b/'affinity_records.json').read_text());rest=json.loads((b/'affinity_restored.json').read_text())
    assert len(rec)==6 and len(rest)==2 and [x['label'] for x in rec]==[r['label'] for r in runs]
    assert all([x[1] for x in z['threads']]==[116,116] for z in rec)
    study=d['config']['memory_screen'];boundaries=chunk_boundaries(3500,study['evaluation_updates'])
    lengths=[y-x for x,y in zip([0,*boundaries[:-1]],boundaries)]
    catalog={};observer=True;numeric=True;anchors=[];rows=[];all_mm=[];early_exact=True;chunks=0
    for r,record in zip(runs,rec):
        assert r['status']=='completed' and r['steps']==3500 and [o['step'] for o in r['observations']]==[100,200,400,420,875,1750,3500]
        ref=next(x for x in old['runs'] if x['kind']==r['kind'])
        for x,y in zip(observations(r)[:4],observations(ref)):
            anchors.append(dict(label=r['label'],step=x['step'],exact=x==y))
        early_exact &= probes(r)[:6]==probes(ref)
        assert [g['step'] for g in r['refresh_guards']]==[1]+list(range(21,3501,20))
        numeric &= all(g['factor_count']==8 and g['maximum_gram_error']<.05 for g in r['refresh_guards'])
        ps=r['tracking_probes'];assert [p['step'] for p in ps]==[21,181,201,221,381,401,861,1741,3481]
        mm=[x for p in ps for x in p['matrices']];ff=[f for x in mm for f in x['factors']]
        assert len(mm)==36 and len(ff)==72 and all(not x['full_reference_collected'] for x in mm)
        numeric &= all(x['targeted_pass'] and x['selected_basis_update_error']<1e-6 and x['world_m_preservation_error']<.01
            and x['maximum_gram']<.05 and math.isfinite(x['gradient_rms']) for x in mm)
        assert r['peak_allocated_bytes']<=24*2**30 and r['lifecycle']['matrix_released'] and r['lifecycle']['optimizer_released']
        hh=r['history'];assert [h['step'] for h in hh]==boundaries and [h['host']['steps'] for h in hh]==lengths
        assert sum('gpu_sample' in h['host'] for h in hh)==(chunks+len(hh))//10-chunks//10;chunks+=len(hh)
        for h in hh:
            host=h['host']
            for k,v in host.get('affinity_definitions',{}).items():
                assert k not in catalog or catalog[k]==v
                catalog[k]=v
            ts={t['tid']:t for t in host['thread_interval']};assert all(t['affinity_ref'] in catalog for t in ts.values())
            for tid,_ in record['threads']:
                t=ts[tid];observer &= catalog[t['affinity_ref']]==[116] and t['last_cpu']==116 and t['runtime_ns']>0
            if 'gpu_sample' in host:assert d['gpu_uuid'] in host['gpu_sample']
        all_mm.extend(mm)
        rows.append(dict(label=r['label'],kind=r['kind'],training_seconds=r['training_seconds'],optimizer_ms=r['optimizer_ms'],
            model_ms=r['model_backward_cuda_ms'],ce=r['observations'][-1]['validation']['ce'],accuracy=r['observations'][-1]['validation']['accuracy'],
            observations=[dict(step=o['step'],ce=o['validation']['ce'],accuracy=o['validation']['accuracy']) for o in r['observations']],
            peak_allocated_gib=r['peak_allocated_bytes']/2**30,
            allocated_before_updates_gib=r['allocated_before_measured_updates_bytes']/2**30,
            allocated_after_cleanup_gib=r['lifecycle']['allocated_after_cleanup_bytes']/2**30,
            cleanup_seconds=r['lifecycle']['cleanup_seconds'],
            offline_probe_seconds=sum(p['diagnostic_seconds'] for p in ps),guard_seconds=sum(g['diagnostic_seconds'] for g in r['refresh_guards'])))
    conditions=[]
    for kind in ['qr','fro']:
        xs=[r for r in runs if r['kind']==kind];assert len(xs)==3
        exact=all(comparable(x)==comparable(xs[0]) for x in xs)
        gate=timing_repeat(xs);gate.update(kind=kind,exact_controls=exact)
        conditions.append(gate)
    anchors_exact=len(anchors)==24 and all(x['exact'] for x in anchors) and early_exact
    compiler=all(r['telemetry_start']['compiler_counters']==r['telemetry_end']['compiler_counters'] for r in runs)
    fixed=numeric and anchors_exact and compiler and observer and all(x['exact_controls'] for x in conditions)
    s=dict(status='completed',fixed_work_qualified=fixed,timing_qualified=fixed and all(x['repeat_gate'] for x in conditions),
        numerical_pass=numeric,historical_anchors_exact=anchors_exact,compiler_stable=compiler,observer_complete=observer,
        owned_teardown_released=True,wall_quality_endpoint_collected=False,full_strict_reference_collected=False,
        conditions=conditions,rows=rows,historical_anchors=anchors,primary_updates=21000,smoke_updates=25,
        refresh_guards=1050,matrix_probes=216,factor_rows=432,early_matrix_probes=144,
        timing_chunks=chunks,own_gpu_samples=84,
        maximum_physical_m_error=max(x['world_m_preservation_error'] for x in all_mm),
        maximum_selected_update_error=max(x['selected_basis_update_error'] for x in all_mm),
        maximum_gram=max(x['maximum_gram'] for x in all_mm),all_in_seconds=d['all_in_seconds'],
        scope='ActualViT1536/depth1 fixed3500 early learning at20/200epochs and separate local instrumented three-repeat timing gates')
    (b/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps({k:v for k,v in s.items() if k not in ['rows','historical_anchors']},indent=2));return s

if __name__=='__main__':summarize(Path(sys.argv[1]))
