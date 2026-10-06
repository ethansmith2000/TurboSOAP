"""Validate historical anchors and summarize precision counterfactual paths."""
import argparse
import json
from pathlib import Path
import statistics

VARIANTS=('qr','warm1','warm2_final','warm2_sequential','warm1_cap02')
KEYS=('model_sha256','rng_sha256','consumed_ids_sha256','augmented_batches','validation','train_panel')
METRICS=('full_update_difference','fixed_basis_transport_difference','strict_transport_remaining_difference',
         'fixed_basis_m_difference','fixed_basis_world_m_difference')


def aggregate(values):
    return dict(samples=len(values),
        metrics={k:dict(mean=statistics.mean(v[k] for v in values),maximum=max(v[k] for v in values),
                        over_one_percent=sum(v[k]>=.01 for v in values)) for k in METRICS},
        stages={k:dict(mean=statistics.mean(v['stages'][k] for v in values),maximum=max(v['stages'][k] for v in values))
                for k in values[0]['stages']},
        full_failures_resolved_by_strict_transport=sum(v['full_update_difference']>=.01 and
                                                       v['strict_transport_remaining_difference']<.01 for v in values),
        qr_order_changed=sum(v['qr_order'] is not None and not all(v['qr_order']['axes_equal']) for v in values),
        variance_changed=sum(not v['variance_equal'] for v in values),
        maximum_chain_closure=max(v['chain_closure_relative'] for v in values))


def summarize(bundle,prior):
    r=json.loads((bundle/'screen.json').read_text());old=json.loads((prior/'screen.json').read_text())
    s=r['config']['memory_screen'];arms={a['label']:a for a in s['arms']};by={a['label']:a for a in r['runs']}
    assert r['status']=='completed' and len(by)==len(r['runs'])==3 and set(by)==set(arms)
    anchors=[];comparisons=0;matrices=[];streams={}
    for label,run in by.items():
        assert run['status']=='completed' and run['steps']==3500
        assert all(run[k]==v for k,v in arms[label].items())
        warm=run['policy']=='warm20'
        for g in run['effective_groups']:
            assert g['peak_lr']==.0005 and g['betas']==[.9,.999]
            if g['role']=='matrix':
                expected=dict(shampoo_beta=.99,basis_method='warm' if warm else 'qr',precondition_frequency=20,
                              hard_reset_interval=200 if warm else 0,variance_policy='permutation',basis_ns_iterations=6 if warm else 2)
                assert all(g[k]==v for k,v in expected.items())
        assert [g['step'] for g in run['refresh_guards']]==[1]+list(range(21,3501,20))
        assert all(g['factor_count']==48 and 0<=g['maximum_gram_error']<.05 for g in run['refresh_guards'])
        previous=next(a for a in old['runs'] if a['label']==label)
        assert [o['step'] for o in run['observations']]==[875,1750,3500]
        for o,p in zip(run['observations'],previous['observations']):
            exact=all(o[k]==p[k] for k in KEYS);assert exact
            anchors.append(dict(label=label,step=o['step'],exact=exact))
            inputs={k:o[k] for k in KEYS[1:4]}
            assert o['step'] not in streams or streams[o['step']]==inputs
            streams[o['step']]=inputs;age=o['step']-1
            assert o['optimizer_counters']==dict(research_qr_refreshes=24*(age//200 if warm else age//20),
                                                 research_warm_refreshes=24*(age//20-age//200 if warm else 0))
        assert [p['step'] for p in run['tracking_probes']]==[age+1 for age in s['tracking_ages']]
        old_matrices={(m['age'],m['matrix']):m for p in previous['tracking_probes'] for m in p['matrices']}
        for p in run['tracking_probes']:
            assert len(p['matrices'])==24 and len({m['matrix'] for m in p['matrices']})==24
            for m in p['matrices']:
                assert m['age']==p['step']-1 and m['executed_control_exact']
                q=old_matrices[m['age'],m['matrix']]
                assert m['shape']==q['shape'] and m['executed_method']==q['executed_method']
                assert set(m['variants'])==set(VARIANTS)
                for name,v in m['variants'].items():
                    assert v['full_update_difference']==q['variants'][name]['update_shadow_error'],(label,m['age'],m['matrix'],name)
                    assert 0<=v['chain_closure_relative']<1e-5
                    for flag,metric in [('full_update_pass','full_update_difference'),
                                        ('fixed_basis_transport_pass','fixed_basis_transport_difference'),
                                        ('strict_transport_remaining_pass','strict_transport_remaining_difference')]:
                        assert v[flag]==(v[metric]<.01)
                    if name!='qr':assert v['variance_equal'] and v['qr_order'] is None
                    comparisons+=1
                matrices.append((label,m))
    first,last=s['control_pairs'][0]
    assert [p['matrices'] for p in by[first]['tracking_probes']]==[p['matrices'] for p in by[last]['tracking_probes']]
    for a,b in zip(by[first]['observations'],by[last]['observations']):assert all(a[k]==b[k] for k in KEYS)
    nm=json.loads((bundle/'manifest.json').read_text());om=json.loads((prior/'manifest.json').read_text())
    provenance={k:nm[k]==om[k] for k in ['split_sha256','base_prepared_sha256']}
    for name in ['soap.py','soap_reference.py','benchmarks/cifar_optimizers.py','benchmarks/cifar_refresh_efficiency.py',
                 *[k for k in nm['source_sha256'] if k.startswith('benchmarks/cifar_support/')]]:
        provenance[name]=nm['source_sha256'][name]==om['source_sha256'][name]
    for k in ['torch','cuda','initial_model_sha256']:provenance[k]=r[k]==old[k]
    assert all(provenance.values())
    rows=[];shape_rows=[]
    for label in by:
        for event in sorted({m['executed_method'] for lab,m in matrices if lab==label}):
            subset=[m for lab,m in matrices if lab==label and m['executed_method']==event]
            for name in VARIANTS:
                rows.append(dict(label=label,executed_method=event,variant=name,
                                 **aggregate([m['variants'][name] for m in subset])))
                for shape in sorted({tuple(m['shape']) for m in subset}):
                    shape_rows.append(dict(label=label,executed_method=event,variant=name,shape=list(shape),
                        **aggregate([m['variants'][name] for m in subset if tuple(m['shape'])==shape])))
    totals={name:aggregate([m['variants'][name] for _,m in matrices]) for name in VARIANTS}
    return dict(status='completed',all_prior_anchors_exact=True,prior_anchors=anchors,prior_provenance=provenance,
                repeated_numerical_probes_exact=True,exact_previous_full_refresh_comparisons=comparisons,
                refresh_guard_count=sum(len(a['refresh_guards']) for a in by.values()),matrix_records=len(matrices),
                totals=totals,rows=rows,shape_rows=shape_rows,gpu_uuid=r['gpu_uuid'],all_in_seconds=r['all_in_seconds'],
                limitations=['Identical incoming states and local precision counterfactuals; no candidate changes training.',
                             'One seed, covariance .99, selected ages and small factors only.',
                             'Vector differences telescope; stage norms are path-dependent and do not add.',
                             'No timing or learning conclusion for stricter precision; strict FP32 is a numerical reference.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    a=p.parse_args();s=summarize(a.bundle,a.prior)
    (a.bundle/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps({k:s[k] for k in ['status','exact_previous_full_refresh_comparisons','matrix_records','all_in_seconds']}))


if __name__=='__main__':main()
