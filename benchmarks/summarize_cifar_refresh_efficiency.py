"""Validate unchanged learning trajectories and compare local refresh efficiency."""
import argparse
import copy
import json
from pathlib import Path
import statistics

VARIANTS = ('qr', 'warm1', 'warm2_final', 'warm2_sequential', 'warm1_cap02')
KEYS = ('model_sha256', 'rng_sha256', 'consumed_ids_sha256', 'augmented_batches',
        'validation', 'train_panel')


def without_timing(probes):
    rows=copy.deepcopy([m for p in probes for m in p['matrices']])
    for row in rows:
        for value in row['variants'].values():value.pop('timing')
    return rows


def summarize(bundle, prior):
    r=json.loads((bundle/'screen.json').read_text())
    old=json.loads((prior/'screen.json').read_text())
    study=r['config']['memory_screen'];arms={a['label']:a for a in study['arms']}
    runs={a['label']:a for a in r['runs']}
    assert r['status']=='completed' and len(runs)==len(r['runs'])==3 and set(runs)==set(arms)
    anchors=[];streams={};all_records=[]
    for label,run in runs.items():
        arm=arms[label];assert all(run[k]==v for k,v in arm.items())
        assert run['status']=='completed' and run['steps']==3500
        warm=run['policy']=='warm20'
        for g in run['effective_groups']:
            assert g['peak_lr']==.0005 and g['betas']==[.9,.999]
            if g['role']=='matrix':
                expected=dict(shampoo_beta=.99,basis_method='warm' if warm else 'qr',
                              precondition_frequency=20,hard_reset_interval=200 if warm else 0,
                              variance_policy='permutation',basis_ns_iterations=6 if warm else 2)
                assert all(g[k]==v for k,v in expected.items())
        assert [g['step'] for g in run['refresh_guards']]==[1]+list(range(21,3501,20))
        assert all(g['factor_count']==48 and 0<=g['maximum_gram_error']<.05 for g in run['refresh_guards'])
        assert [o['step'] for o in run['observations']]==[875,1750,3500]
        previous=next(a for a in old['runs'] if a['label']==label)
        for o,p in zip(run['observations'],previous['observations']):
            exact=all(o[k]==p[k] for k in KEYS);assert exact
            anchors.append(dict(label=label,step=o['step'],exact=exact))
            stream={k:o[k] for k in KEYS[1:4]}
            assert o['step'] not in streams or streams[o['step']]==stream
            streams[o['step']]=stream
            age=o['step']-1
            assert o['optimizer_counters']==dict(research_qr_refreshes=24*(age//200 if warm else age//20),
                                                 research_warm_refreshes=24*(age//20-age//200 if warm else 0))
        assert [p['step'] for p in run['tracking_probes']]==[age+1 for age in study['tracking_ages']]
        for p in run['tracking_probes']:
            assert len(p['matrices'])==24 and len({m['matrix'] for m in p['matrices']})==24
            for m in p['matrices']:
                assert m['age']==p['step']-1 and m['executed_control_exact']
                assert set(m['variants'])==set(VARIANTS)
                assert len(m['incoming'])==len(m['shape'])==2
                for v in m['variants'].values():
                    assert len(v['factors'])==2
                    assert len(v['timing']['round_ms'])==3 and all(x>0 for x in v['timing']['round_ms'])
                    assert v['timing']['median_ms']==statistics.median(v['timing']['round_ms'])
                all_records.append((label,m))
    first,last=study['control_pairs'][0]
    assert without_timing(runs[first]['tracking_probes'])==without_timing(runs[last]['tracking_probes'])
    for x,y in zip(runs[first]['observations'],runs[last]['observations']):
        assert all(x[k]==y[k] for k in KEYS)
    nm=json.loads((bundle/'manifest.json').read_text());om=json.loads((prior/'manifest.json').read_text())
    provenance={k:nm[k]==om[k] for k in ['split_sha256','base_prepared_sha256']}
    for name in ['soap.py','soap_reference.py','benchmarks/cifar_optimizers.py',
                 *[k for k in nm['source_sha256'] if k.startswith('benchmarks/cifar_support/')]]:
        provenance[name]=nm['source_sha256'][name]==om['source_sha256'][name]
    for k in ['torch','cuda','initial_model_sha256']:provenance[k]=r[k]==old[k]
    assert all(provenance.values())
    rows=[];factors=[]
    for label in runs:
        for event in sorted({m['executed_method'] for lab,m in all_records if lab==label}):
            matrices=[m for lab,m in all_records if lab==label and m['executed_method']==event]
            for shape in sorted({tuple(m['shape']) for m in matrices}):
                subset=[m for m in matrices if tuple(m['shape'])==shape]
                incoming=statistics.mean(statistics.mean(f['residual'] for f in m['incoming']) for m in subset)
                for name in VARIANTS:
                    values=[m['variants'][name] for m in subset]
                    after=statistics.mean(statistics.mean(f['residual'] for f in v['factors']) for v in values)
                    cost=statistics.mean(v['timing']['median_ms'] for v in values)
                    rows.append(dict(label=label,executed_method=event,shape=list(shape),variant=name,
                                     samples=len(values),rejections=sum(not v['passes'] for v in values),
                                     incoming_residual=incoming,after_residual=after,refresh_ms=cost,
                                     reduction_per_ms=(incoming-after)/cost,
                                     maximum_gram=max(v['maximum_gram'] for v in values),
                                     max_world_m_preservation_error=max(v['world_m_preservation_error'] for v in values),
                                     max_world_m_shadow_error=max(v['world_m_shadow_error'] for v in values),
                                     max_update_shadow_error=max(v['update_shadow_error'] for v in values)))
            for dim in sorted({d for m in matrices for d in m['shape']}):
                pairs=[(m,axis) for m in matrices for axis,d in enumerate(m['shape']) if d==dim]
                factors.append(dict(label=label,executed_method=event,dimension=dim,samples=len(pairs),
                    incoming_residual=statistics.mean(m['incoming'][i]['residual'] for m,i in pairs),
                    residual={name:statistics.mean(m['variants'][name]['factors'][i]['residual'] for m,i in pairs)
                              for name in VARIANTS}))
    variation={}
    for name in VARIANTS:
        values=[sum(m['variants'][name]['timing']['median_ms'] for lab,m in all_records if lab==label)
                for label in [first,last]]
        variation[name]=(max(values)-min(values))/statistics.mean(values)
    failures={name:sum(not m['variants'][name]['passes'] for _,m in all_records) for name in VARIANTS}
    fidelity={}
    for name in VARIANTS:
        values=[m['variants'][name] for _,m in all_records]
        fidelity[name]={}
        for key,budget in [('maximum_gram',.05),('world_m_preservation_error',.01),
                           ('world_m_shadow_error',.01),('update_shadow_error',.01)]:
            fidelity[name][key]=dict(maximum=max(v[key] for v in values),
                                    rejections=sum(v[key]>=budget for v in values),budget=budget)
    return dict(status='completed',prior_anchors=anchors,all_prior_anchors_exact=True,
                repeated_numerical_probes_exact=True,prior_provenance=provenance,
                refresh_guard_count=sum(len(a['refresh_guards']) for a in r['runs']),
                probe_matrix_records=len(all_records),candidate_checks=len(all_records)*len(VARIANTS),
                candidate_rejections=failures,fidelity=fidelity,rows=rows,factors=factors,
                maximum_sequential_vs_final_update_difference=max(m['sequential_vs_final_update_difference'] for _,m in all_records),
                repeated_event_timing_relative_range=variation,gpu_uuid=r['gpu_uuid'],prior_gpu_uuid=old['gpu_uuid'],
                all_in_seconds=r['all_in_seconds'],
                limitations=['Local alternatives only; no candidate changes training.',
                             'Covariance .99, one seed, selected ages and small CIFAR factors only.',
                             'Refresh-event latency excludes per-update covariance accumulation and Adam update.',
                             'Residual decrease per ms is a local diagnostic, not learning progress per second.',
                             'No compilation, BF16 gauge, inference_mode or large-factor cost extrapolation.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    a=p.parse_args();s=summarize(a.bundle,a.prior)
    (a.bundle/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps({k:s[k] for k in ['status','candidate_checks','candidate_rejections','all_in_seconds']}))


if __name__=='__main__':main()
