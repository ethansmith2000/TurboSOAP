"""Fail-closed fixed-update qualification for the explicit V ablation."""
import hashlib
import json
from pathlib import Path
import sys
from timing_diagnostics import range_over_mean

KEYS=['step','validation','train_panel','model_sha256','optimizer_sha256','rng_sha256',
      'consumed_ids_sha256','augmented_batches','optimizer_counters','maximum_gram_error','mean_gram_error']
def fixed(run):return [x for x in run['observations'] if 'fixed_updates' in x['reasons']]
def probes(run):return [{k:v for k,v in p.items() if k!='diagnostic_seconds'} for p in run['tracking_probes']]

def summarize(b):
    r=json.loads((b/'screen.json').read_text());m=json.loads((b/'manifest.json').read_text())
    assert r['status']=='completed' and len(r['runs'])==12
    history=Path(m['historical_bundle']);p=history/'screen.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==m['historical_screen_sha256']
    oracle=Path(m['moment_oracle_path']);assert hashlib.sha256(oracle.read_bytes()).hexdigest()==m['moment_oracle_sha256']
    old=json.loads(p.read_text());om=json.loads((history/'manifest.json').read_text())
    for k in ['split_sha256','base_prepared_sha256']:assert m[k]==om[k]
    for k in ['soap.py','soap_reference.py','benchmarks/cifar_optimizers.py','benchmarks/cifar_radius_learning_runner.py']:
        assert m['source_sha256'][k]==om['source_sha256'][k]
    for k in ['torch','cuda','initial_model_sha256']:assert r[k]==old[k]
    runs=r['runs'];study=r['config']['memory_screen']
    assert len({x['label'] for x in runs})==12 and [x['label'] for x in runs]==[x['label'] for x in study['arms']]
    rows=[];anchors=[];streams={};diagnostics=[];guard_count=0;numerical=True
    for run in runs:
        assert run['status']=='completed' and run['steps']==3500
        obs=fixed(run);assert len(run['observations'])==3 and [x['step'] for x in obs]==[875,1750,3500]
        assert len(run['refresh_guards'])==175
        guard=all(x['factor_count']==48 and x['maximum_gram_error']<.05 and x['minimum_variance']>=0 for x in run['refresh_guards'])
        guard_count+=len(run['refresh_guards'])
        assert [p['step'] for p in run['tracking_probes']]==[a+1 for a in study['tracking_ages']]
        mm=[x for p in run['tracking_probes'] for x in p['matrices']];assert len(mm)==168
        targeted=all(x['targeted_pass'] for x in mm)
        variance=all(x['variance']['formula_exact'] and x['variance']['implementation_relative_error']==0
                     and x['variance']['minimum']>=0 and x['variance']['policy']==run['variance_policy'] for x in mm)
        numerical &= guard and targeted and variance
        groups=[g for g in run['effective_groups'] if g['role']=='matrix'];assert len(groups)==1
        g=groups[0];assert g['variance_policy']==run['variance_policy'] and g['transport_precision']=='highest'
        assert g['shampoo_beta']==run['covariance_beta'] and g['precondition_frequency']==20
        assert g['hard_reset_interval']==(0 if run['kind'].startswith('qr') else 200)
        if run['kind'].startswith('fro'):assert g['basis_rotation_cap']==.5 and g['basis_ns_iterations']==2
        for o in obs:
            assert len(o['optimizer_sha256'])==64
            stream={k:o[k] for k in ['rng_sha256','consumed_ids_sha256','augmented_batches']}
            assert o['step'] not in streams or streams[o['step']]==stream
            streams[o['step']]=stream
        if run['variance_policy']=='permutation':
            oldkind='qr' if run['kind'].startswith('qr') else 'fro05'
            ref=next(x for x in old['runs'] if x['label']==f"{oldkind}_beta{run['covariance_beta']}_r0")
            for o,z in zip(obs,fixed(ref)):
                anchors.append(dict(label=run['label'],step=o['step'],exact=all(o[k]==z[k] for k in KEYS)))
            a=probes(run);z=probes(ref)
            assert [{**p,'matrices':[{k:v for k,v in x.items() if k!='variance'} for x in p['matrices']]} for p in a]==z
        for event in ['warm','qr']:
            xs=[x for x in mm if x['executed_method']==event]
            if xs:diagnostics.append(dict(label=run['label'],event=event,count=len(xs),
                max_world_m=max(x['world_m_preservation_error'] for x in xs),
                max_selected_basis_error=max(x['selected_basis_update_error'] for x in xs),
                max_full_strict_difference=max(x['full_strict_update_difference'] for x in xs),
                full_strict_failures=sum(not x['full_strict_pass'] for x in xs),
                max_same_basis_v_difference=max(x['variance']['relative_v_difference'] for x in xs),
                max_same_basis_update_difference=max(x['variance']['same_basis_update_relative_difference'] for x in xs),
                minimum_update_cosine=min(x['variance']['same_basis_update_cosine'] for x in xs),
                variance_mass_ratio_range=[min(x['variance']['mass_ratio'] for x in xs),max(x['variance']['mass_ratio'] for x in xs)]))
        o=obs[-1];rows.append(dict(label=run['label'],kind=run['kind'],beta=run['covariance_beta'],
            variance_policy=run['variance_policy'],repeat=bool(run.get('repeat_of')),ce=o['validation']['ce'],
            accuracy=o['validation']['accuracy'],optimizer_ms=o['timing']['optimizer_ms'],training_seconds=o['training_seconds'],
            guards_pass=guard,targeted_pass=targeted,variance_formula_pass=variance))
    controls=[]
    for kind in ['qr_carry','qr_overlap','fro_carry','fro_overlap']:
        xs=[x for x in runs if x['kind']==kind and x['covariance_beta']==.99];assert len(xs)==2
        exact=([{k:o[k] for k in KEYS} for o in fixed(xs[0])]==[{k:o[k] for k in KEYS} for o in fixed(xs[1])]
               and probes(xs[0])==probes(xs[1]))
        controls.append(dict(kind=kind,exact=exact,wall_range_over_mean=range_over_mean([x['training_seconds'] for x in xs]),
            optimizer_range_over_mean=range_over_mean([x['optimizer_ms'] for x in xs])))
    records=json.loads((b/'affinity_records.json').read_text());restored=json.loads((b/'affinity_restored.json').read_text())
    placement=(len(records)==12 and [x['label'] for x in records]==[x['label'] for x in runs]
               and all([p[1] for p in x['threads']]==[116,117] for x in records) and len(restored)==2)
    compiler=all(x['telemetry_start']['compiler_counters']==x['telemetry_end']['compiler_counters'] for x in runs)
    result=dict(status='completed',fixed_update_qualified=numerical and len(anchors)==18 and all(x['exact'] for x in anchors)
        and all(x['exact'] for x in controls) and placement and compiler,rows=rows,controls=controls,
        anchors=anchors,diagnostics=diagnostics,placement_recorded=placement,compiler_stable=compiler,
        refresh_guards=guard_count,matrix_probes=2016,variance_formula_checks=2016,trainer_steps=42000,
        wall_seconds=r['all_in_seconds'],wall_quality_scope='No wall endpoint or qualification; fixed-update study only')
    (b/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','anchors','diagnostics']},indent=2));return result

if __name__=='__main__':summarize(Path(sys.argv[1]))
