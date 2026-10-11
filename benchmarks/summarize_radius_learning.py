"""Validate learning pairing and preserve failures of prospective wall gates."""
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
from timing_diagnostics import range_over_mean

CONTROL_KEYS=['step','validation','train_panel','model_sha256','rng_sha256',
              'consumed_ids_sha256','augmented_batches','optimizer_counters',
              'maximum_gram_error','mean_gram_error']

def fixed(run):
    return [o for o in run['observations'] if 'fixed_updates' in o['reasons']]

def probes(run):
    return [{k:v for k,v in p.items() if k!='diagnostic_seconds'} for p in run['tracking_probes']]

def summarize(b):
    r=json.loads((b/'screen.json').read_text());m=json.loads((b/'manifest.json').read_text())
    assert r['status']=='completed' and len(r['runs'])==16
    prior=Path(m['historical_bundle']); old_path=prior/'screen.json'
    assert hashlib.sha256(old_path.read_bytes()).hexdigest()==m['historical_screen_sha256']
    old=json.loads(old_path.read_text());om=json.loads((prior/'manifest.json').read_text())
    for k in ['split_sha256','base_prepared_sha256']:assert m[k]==om[k]
    for k in ['soap.py','soap_reference.py','benchmarks/cifar_optimizers.py']:
        assert m['source_sha256'][k]==om['source_sha256'][k]
    for k in ['torch','cuda','initial_model_sha256']:assert r[k]==old[k]
    study=r['config']['memory_screen']; runs=r['runs'];rows=[];anchors=[];diagnostics=[];streams={}
    assert len({x['label'] for x in runs})==16 and [x['label'] for x in runs]==[x['label'] for x in study['arms']]
    numerical=True;guards=0
    for run in runs:
        assert run['status']=='completed' and 3500<=run['steps']<=5000
        observations=fixed(run);assert [o['step'] for o in observations]==[875,1750,3500]
        assert all(len(o.get('optimizer_sha256',''))==64 for o in observations)
        assert len(run['refresh_guards'])==1+(run['steps']-1)//20
        guards+=len(run['refresh_guards'])
        guard_pass=all(g['factor_count']==48 and g['maximum_gram_error']<.05 for g in run['refresh_guards'])
        assert [p['step'] for p in run['tracking_probes']]==[age+1 for age in study['tracking_ages']]
        matrices=[x for p in run['tracking_probes'] for x in p['matrices']]
        assert len(matrices)==168
        targeted=all(x['targeted_pass'] for x in matrices); numerical &= guard_pass and targeted
        for g in run['effective_groups']:
            if g['role']=='matrix':
                assert g['transport_precision']=='highest' and g['shampoo_beta']==run['covariance_beta']
                assert g['precondition_frequency']==20 and g['hard_reset_interval']==(0 if run['kind']=='qr' else 200)
                if run['kind']!='qr':
                    assert g['basis_rotation_cap']==(.5 if run['kind']=='fro05' else .6)
                    assert g['basis_ns_iterations']==(3 if run['kind']=='fro06strong' else 2)
        for o in observations:
            stream={k:o[k] for k in ['rng_sha256','consumed_ids_sha256','augmented_batches']}
            assert o['step'] not in streams or streams[o['step']]==stream
            streams[o['step']]=stream
        if run['kind'] in ('qr','fro05'):
            label=('qr20' if run['kind']=='qr' else 'fro_cubic2')+f"_beta{run['covariance_beta']}"
            reference=next(x for x in old['runs'] if x['label']==label)
            for obs in observations:
                previous=next(x for x in fixed(reference) if x['step']==obs['step'])
                anchors.append(dict(label=run['label'],step=obs['step'],exact=all(obs[k]==previous[k] for k in CONTROL_KEYS)))
        for event in ['warm','qr']:
            ms=[x for x in matrices if x['executed_method']==event]
            if ms:diagnostics.append(dict(label=run['label'],event=event,count=len(ms),
                max_world_m=max(x['world_m_preservation_error'] for x in ms),
                max_selected_basis_error=max(x['selected_basis_update_error'] for x in ms),
                max_full_strict_difference=max(x['full_strict_update_difference'] for x in ms),
                full_strict_failures=sum(not x['full_strict_pass'] for x in ms)))
        endpoints=[o for o in run['observations'] if 'equal_wall' in o['reasons']];assert len(endpoints)<=1
        budget=None
        if endpoints:
            o=endpoints[0];assert o['step']%25==0 and o['training_seconds']>=90
            before=[h for h in run['history'] if h['step']<o['step']]
            assert not before or before[-1]['training_seconds']<90
            budget=dict(step=o['step'],ce=o['validation']['ce'],accuracy=o['validation']['accuracy'],
                        seconds=o['training_seconds'],overshoot_seconds=o['training_seconds']-90)
        end=observations[-1]
        rows.append(dict(label=run['label'],kind=run['kind'],beta=run['covariance_beta'],repeat=bool(run.get('repeat_of')),
            ce=end['validation']['ce'],accuracy=end['validation']['accuracy'],optimizer_ms=end['timing']['optimizer_ms'],
            training_seconds=end['training_seconds'],total_steps=run['steps'],budget=budget,
            guards_pass=guard_pass,targeted_pass=targeted))
    controls=[]
    for kind in ['qr','fro05','fro06','fro06strong']:
        group=[x for x in runs if x['kind']==kind and x['covariance_beta']==.99];assert len(group)==3
        controls_exact=all([{k:o[k] for k in CONTROL_KEYS+['optimizer_sha256']} for o in fixed(x)]==
                           [{k:o[k] for k in CONTROL_KEYS+['optimizer_sha256']} for o in fixed(group[0])]
                           and probes(x)==probes(group[0]) for x in group)
        end=[fixed(x)[-1] for x in group]
        wr=range_over_mean([o['training_seconds'] for o in end]); ore=range_over_mean([o['timing']['optimizer_ms'] for o in end])
        controls.append(dict(kind=kind,exact=controls_exact,wall_range_over_mean=wr,optimizer_range_over_mean=ore,
            repeat_qualified=controls_exact and wr<=.03 and ore<=.02))
    records=json.loads((b/'affinity_records.json').read_text());restored=json.loads((b/'affinity_restored.json').read_text())
    placement=(len(records)==16 and [x['label'] for x in records]==[x['label'] for x in runs]
               and all([x[1] for x in record['threads']]==[116,117] for record in records) and len(restored)==2)
    compiler=all(x['telemetry_start']['compiler_counters']==x['telemetry_end']['compiler_counters'] for x in runs)
    exact_history=len(anchors)==24 and all(x['exact'] for x in anchors)
    fixed_qualified=numerical and exact_history and all(x['exact'] for x in controls) and placement and compiler
    result=dict(status='completed',fixed_update_qualified=fixed_qualified,rows=rows,controls=controls,
        historical_anchors=anchors,diagnostics=diagnostics,placement_recorded=placement,compiler_stable=compiler,
        beta099_wall_qualified=fixed_qualified and all(x['repeat_qualified'] for x in controls)
            and all(x['budget'] for x in rows if x['beta']==.99),
        beta0999_wall_qualified=False,beta0999_wall_scope='Exploratory: one arm per policy, no beta-specific repeats',
        refresh_guards=guards,matrix_probes=2688,trainer_steps=sum(x['steps'] for x in runs),wall_seconds=r['all_in_seconds'])
    (b/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','historical_anchors','diagnostics']},indent=2))
    return result

if __name__=='__main__':summarize(Path(sys.argv[1]))
