"""Fail-closed actual-model feasibility and selected-transport qualification."""
import hashlib
import json
import math
from pathlib import Path
import sys
from timing_diagnostics import range_over_mean

KEYS=['step','validation','train_panel','model_sha256','optimizer_sha256','rng_sha256',
      'consumed_ids_sha256','augmented_batches','optimizer_counters','maximum_gram_error','mean_gram_error']

def comparable(run):
    return dict(observations=[{k:o[k] for k in KEYS} for o in run['observations']],
        optimizer=run['optimizer_sha256'],
        probes=[{k:v for k,v in p.items() if k!='diagnostic_seconds'} for p in run['tracking_probes']],
        guards=[{k:v for k,v in g.items() if k!='diagnostic_seconds'} for g in run['refresh_guards']],
        chunk_losses=[(h['step'],h['loss'],h['maximum_gradient_norm']) for h in run['history']])

def summarize(b):
    d=json.loads((b/'screen.json').read_text());m=json.loads((b/'manifest.json').read_text())
    for name,digest in m['source_sha256'].items():
        assert hashlib.sha256((b/'source'/name).read_bytes()).hexdigest()==digest,name
    for name,key in [('config.json','config_sha256'),('PROTOCOL.md','protocol_sha256'),
        ('split.json','split_sha256'),('data_prepared.json','base_prepared_sha256')]:
        assert hashlib.sha256((b/name).read_bytes()).hexdigest()==m[key],name
    assert d['status']=='completed' and len(d['runs'])==3 and d['parameter_count']==28660324
    assert sorted(tuple(x['shape']) for x in d['matrix_inventory'])==sorted([(4608,1536),(1536,1536),(6144,1536),(1536,6144)])
    runs=d['runs'];assert [r['label'] for r in runs]==['qr_0','fro_1','qr_2']
    rec=json.loads((b/'affinity_records.json').read_text());rest=json.loads((b/'affinity_restored.json').read_text())
    assert len(rec)==3 and len(rest)==2 and [x['label'] for x in rec]==[r['label'] for r in runs]
    assert all([x[1] for x in z['threads']]==[116,116] for z in rec)
    catalog={};observer=True;numeric=True;rows=[];all_mm=[]
    for r,record in zip(runs,rec):
        assert r['status']=='completed' and r['steps']==420 and [o['step'] for o in r['observations']]==[100,200,400,420]
        assert [g['step'] for g in r['refresh_guards']]==[1]+list(range(21,421,20))
        numeric &= all(g['factor_count']==8 and g['maximum_gram_error']<.05 for g in r['refresh_guards'])
        ps=r['tracking_probes'];assert [p['step'] for p in ps]==[21,181,201,221,381,401]
        mm=[x for p in ps for x in p['matrices']];ff=[x for z in mm for x in z['factors']]
        assert len(mm)==24 and len(ff)==48
        assert all(not x['full_reference_collected'] for x in mm)
        numeric &= all(x['targeted_pass'] and x['selected_basis_update_error']<1e-6 and
            x['world_m_preservation_error']<.01 and x['maximum_gram']<.05 and math.isfinite(x['gradient_rms']) for x in mm)
        assert r['peak_allocated_bytes']<=24*2**30
        hh=r['history'];assert [h['step'] for h in hh]==list(range(10,421,10))
        assert all(h['host']['steps']==10 for h in hh)
        for h in hh:
            host=h['host']
            for k,v in host.get('affinity_definitions',{}).items():
                assert k not in catalog or catalog[k]==v
                catalog[k]=v
            ts={t['tid']:t for t in host['thread_interval']}
            assert all(t['affinity_ref'] in catalog for t in ts.values())
            for tid,_ in record['threads']:
                t=ts[tid];observer &= catalog[t['affinity_ref']]==[116] and t['last_cpu']==116 and t['runtime_ns']>0
            if 'gpu_sample' in host:assert d['gpu_uuid'] in host['gpu_sample']
        all_mm.extend(mm)
        rows.append(dict(label=r['label'],kind=r['kind'],training_seconds=r['training_seconds'],
            all_in_seconds=r['all_in_seconds'],optimizer_ms=r['optimizer_ms'],model_ms=r['model_backward_cuda_ms'],
            peak_allocated_gib=r['peak_allocated_bytes']/2**30,ce=r['observations'][-1]['validation']['ce'],
            observations=[dict(step=o['step'],ce=o['validation']['ce']) for o in r['observations']],
            offline_probe_seconds=sum(p['diagnostic_seconds'] for p in ps),
            guard_seconds=sum(g['diagnostic_seconds'] for g in r['refresh_guards'])))
    exact=comparable(runs[0])==comparable(runs[2])
    compiler=all(r['telemetry_start']['compiler_counters']==r['telemetry_end']['compiler_counters'] for r in runs)
    s=dict(status='completed',feasibility_qualified=numeric and exact and compiler and observer,
        numerical_pass=numeric,qr_repeat_exact=exact,compiler_stable=compiler,observer_complete=observer,
        wall_qualification_collected=False,full_strict_reference_collected=False,
        qr_pair_wall_range_over_mean=range_over_mean([runs[i]['training_seconds'] for i in [0,2]]),
        qr_pair_optimizer_range_over_mean=range_over_mean([runs[i]['optimizer_ms'] for i in [0,2]]),
        primary_updates=1260,smoke_updates=25,refresh_guards=63,matrix_probes=72,factor_rows=144,
        maximum_physical_m_error=max(x['world_m_preservation_error'] for x in all_mm),
        maximum_selected_update_error=max(x['selected_basis_update_error'] for x in all_mm),
        maximum_gram=max(x['maximum_gram'] for x in all_mm),rows=rows,
        all_in_seconds=d['all_in_seconds'],parameter_count=d['parameter_count'],matrix_inventory=d['matrix_inventory'],
        scope='Actualwidth1536 depth1 fixed420 feasibility/numerics/cost; no wall-quality or converged-learning claim')
    (b/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2));return s

if __name__=='__main__':summarize(Path(sys.argv[1]))
