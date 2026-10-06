"""Validate the opt-in precision/cap/beta factorial and report paired contrasts."""
import argparse
import json
from pathlib import Path
import statistics

KEYS=['model_sha256','rng_sha256','consumed_ids_sha256','augmented_batches','validation','train_panel']


def summarize(bundle,prior):
    r=json.loads((bundle/'screen.json').read_text());old=json.loads((prior/'screen.json').read_text())
    gate=json.loads((bundle/'cost_gate.json').read_text());study=r['config']['memory_screen']
    by={a['label']:a for a in r['runs']};arms={a['label']:a for a in study['arms']}
    assert r['status']=='completed' and len(by)==len(r['runs'])==14 and set(by)==set(arms)
    assert gate['status']=='completed' and gate['cifar_gate_pass'] and gate['gpu_uuid']==r['gpu_uuid']
    assert len(gate['rows'])==36 and all(x['basis_and_v_match_inherit'] for x in gate['rows'])
    cost=[]
    for x in gate['rows']:
        assert len(x['round_ms'])==3 and x['median_ms']==statistics.median(x['round_ms']) and min(x['round_ms'])>0
        if x['transport_precision']=='highest':
            base=next(y for y in gate['rows'] if y['shape']==x['shape'] and y['method']==x['method'] and y['cap']==x['cap'] and y['transport_precision']=='inherit')
            cost.append(dict(shape=x['shape'],method=x['method'],cap=x['cap'],inherit_ms=base['median_ms'],strict_ms=x['median_ms'],
                             relative_change=x['median_ms']/base['median_ms']-1,targeted_pass=x['diagnostic']['targeted_pass'],
                             selected_basis_update_error=x['diagnostic'].get('selected_basis_update_error'),
                             maximum_gram=x['diagnostic'].get('maximum_gram')))
            if x['cifar_shape']:assert x['diagnostic']['targeted_pass']
    rows=[];diagnostics=[];anchors=[];streams={};guard_count=0
    for label,run in by.items():
        arm=arms[label];assert all(run[k]==v for k,v in arm.items())
        assert run['status']=='completed' and run['steps']==3500
        warm=arm['policy']=='warm20';strict=arm['transport_precision']=='highest'
        for g in run['effective_groups']:
            assert g['peak_lr']==.0005 and g['betas']==[.9,.999]
            if g['role']=='matrix':
                expected=dict(shampoo_beta=arm['covariance_beta'],basis_method='warm' if warm else 'qr',
                    precondition_frequency=20,hard_reset_interval=200 if warm else 0,variance_policy='permutation',
                    basis_ns_iterations=6 if warm else 2,transport_precision=arm['transport_precision'],basis_rotation_cap=arm['basis_rotation_cap'])
                assert all(g[k]==v for k,v in expected.items())
        guards=run['refresh_guards'];guard_count+=len(guards)
        assert [g['step'] for g in guards]==[1]+list(range(21,3501,20))
        assert all(g['factor_count']==48 and 0<=g['maximum_gram_error']<.05 for g in guards)
        assert [o['step'] for o in run['observations']]==[875,1750,3500]
        for o in run['observations']:
            inp={k:o[k] for k in KEYS[1:4]}
            assert o['step'] not in streams or streams[o['step']]==inp
            streams[o['step']]=inp;age=o['step']-1
            expected=dict(research_qr_refreshes=24*(age//200 if warm else age//20),research_warm_refreshes=24*(age//20-age//200 if warm else 0))
            if strict:expected['research_strict_transports']=24*(age//20)
            assert o['optimizer_counters']==expected
        if not strict and arm['basis_rotation_cap']==.1 and not arm.get('repeat_of'):
            previous=next(a for a in old['runs'] if a['policy']==arm['policy'] and a['covariance_beta']==arm['covariance_beta'] and not a.get('repeat_of'))
            for o,p in zip(run['observations'],previous['observations']):
                exact=all(o[k]==p[k] for k in KEYS);assert exact
                anchors.append(dict(label=label,step=o['step'],exact=exact))
        probes=run['tracking_probes'];assert [p['step'] for p in probes]==[x+1 for x in study['tracking_ages']]
        matrices=[]
        for p in probes:
            assert len(p['matrices'])==24 and len({m['matrix'] for m in p['matrices']})==24
            for m in p['matrices']:
                assert m['age']==p['step']-1 and m['strict_transport']==strict
                assert m['full_strict_pass']==(m['full_strict_update_difference']<.01)
                if strict:assert m['targeted_pass'] and m['selected_basis_update_error']<1e-6 and m['world_m_preservation_error']<.01
                matrices.append(m)
        for event in sorted({m['executed_method'] for m in matrices}):
            values=[m for m in matrices if m['executed_method']==event]
            diagnostics.append(dict(label=label,event=event,samples=len(values),
                full_strict_rejections=sum(not m['full_strict_pass'] for m in values),
                maximum_full_strict_difference=max(m['full_strict_update_difference'] for m in values),
                maximum_selected_basis_update_error=max(m['selected_basis_update_error'] for m in values),
                maximum_world_m_preservation_error=max(m['world_m_preservation_error'] for m in values),
                mean_residual=statistics.mean(m['mean_residual'] for m in values)))
        end=run['observations'][-1]
        rows.append(dict(label=label,policy=arm['policy'],precision=arm['transport_precision'],cap=arm['basis_rotation_cap'],
            beta=arm['covariance_beta'],repeated=bool(arm.get('repeat_of')),ce=end['validation']['ce'],accuracy=end['validation']['accuracy'],
            train_ce=end['train_panel']['ce'],optimizer_ms=end['timing']['optimizer_ms'],phase_timing=end['timing'],
            training_seconds=end['training_seconds'],maximum_gram_error=max(g['maximum_gram_error'] for g in guards)))
    controls=[]
    for first,last in study['control_pairs']:
        for a,b in zip(by[first]['observations'],by[last]['observations']):assert all(a[k]==b[k] for k in KEYS)
        assert [p['matrices'] for p in by[first]['tracking_probes']]==[p['matrices'] for p in by[last]['tracking_probes']]
        pair=[next(x for x in rows if x['label']==name) for name in [first,last]]
        variation={}
        for key in ['optimizer_ms','model_backward_cuda_ms','iteration_cuda_ms']:
            values=[x['phase_timing'][key] for x in pair];variation[key]=(max(values)-min(values))/statistics.mean(values)
        values=[x['training_seconds'] for x in pair];variation['training_wall']=(max(values)-min(values))/statistics.mean(values)
        controls.append(dict(original=first,repeated=last,exact_observations=3,exact_probe_records=True,timing_relative_range=variation))
    primary=[x for x in rows if not x['repeated']]
    def get(policy,precision,cap,beta):
        return next(x for x in primary if (x['policy'],x['precision'],x['cap'],x['beta'])==(policy,precision,cap,beta))
    contrasts=[]
    def contrast(kind,control,intervention):
        contrasts.append(dict(kind=kind,control=control['label'],intervention=intervention['label'],
            ce_delta=intervention['ce']-control['ce'],accuracy_pp_delta=100*(intervention['accuracy']-control['accuracy']),
            optimizer_ms_delta=intervention['optimizer_ms']-control['optimizer_ms']))
    for beta in [.99,.999]:
        for policy,cap in [('qr20',.1),('warm20',.1),('warm20',.2)]:contrast('strict_transport',get(policy,'inherit',cap,beta),get(policy,'highest',cap,beta))
        for precision in ['inherit','highest']:contrast('cap_0.1_to_0.2',get('warm20',precision,.1,beta),get('warm20',precision,.2,beta))
    for policy,precision,cap in [('qr20','inherit',.1),('qr20','highest',.1),('warm20','inherit',.1),('warm20','highest',.1),('warm20','inherit',.2),('warm20','highest',.2)]:
        contrast('covariance_0.999_to_0.99',get(policy,precision,cap,.999),get(policy,precision,cap,.99))
    interactions={str(beta):(get('warm20','highest',.2,beta)['ce']-get('warm20','highest',.1,beta)['ce'])-(get('warm20','inherit',.2,beta)['ce']-get('warm20','inherit',.1,beta)['ce']) for beta in [.99,.999]}
    nm=json.loads((bundle/'manifest.json').read_text());om=json.loads((prior/'manifest.json').read_text())
    provenance={k:nm[k]==om[k] for k in ['split_sha256','base_prepared_sha256']}
    for name in ['soap.py',*[k for k in nm['source_sha256'] if k.startswith('benchmarks/cifar_support/')]]:provenance[name]=nm['source_sha256'][name]==om['source_sha256'][name]
    for k in ['torch','cuda','initial_model_sha256']:provenance[k]=r[k]==old[k]
    assert all(provenance.values())
    return dict(status='completed',rows=rows,diagnostics=diagnostics,contrasts=contrasts,cap_precision_interaction_ce=interactions,
        cost=cost,cost_gate_pass=True,prior_anchors=anchors,prior_provenance=provenance,all_prior_anchors_exact=True,
        controls=controls,refresh_guard_count=guard_count,probe_matrix_records=sum(x['samples'] for x in diagnostics),
        gpu_uuid=r['gpu_uuid'],training_all_in_seconds=r['all_in_seconds'],cost_gate_seconds=gate['wall_seconds'],
        limitations=['One seed and20-epoch endpoint on200-epoch schedule; fixed LR, no convergence or tuned-method ranking.',
                     'Strict m changes arithmetic only at refresh, including QR resets; basis/v policy remains explicit.',
                     'Selected-basis implementation gate is distinct from full strict-refresh sensitivity, whose failures are retained.',
                     'Synthetic eager event costs are separate from natural CIFAR optimizer timing and do not establish large-model training speed.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    a=p.parse_args();s=summarize(a.bundle,a.prior);(a.bundle/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps({k:s[k] for k in ['status','refresh_guard_count','probe_matrix_records','training_all_in_seconds']}))


if __name__=='__main__':main()
