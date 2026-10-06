"""Paired CIFAR validation for the isolated retraction entry point."""
import argparse
import json
from pathlib import Path
import statistics

KEYS=['model_sha256','rng_sha256','consumed_ids_sha256','augmented_batches','validation','train_panel']


def summarize(b,prior):
    r=json.loads((b/'screen.json').read_text());old=json.loads((prior/'screen.json').read_text())
    groups={x['label']:x['groups'] for x in json.loads((b/'experimental_groups.json').read_text())}
    study=r['config']['memory_screen'];by={a['label']:a for a in r['runs']}
    assert r['status']=='completed' and len(by)==len(r['runs'])==10
    assert set(by)=={a['label'] for a in study['arms']}
    rows=[];diags=[];anchors=[];streams={};guards=0
    for label,run in by.items():
        assert run['status']=='completed' and run['steps']==3500
        warm=run['policy']=='warm20';custom='retraction_variant' in run
        for g in run['effective_groups']:
            assert g['peak_lr']==.0005 and g['betas']==[.9,.999]
            if g['role']=='matrix':
                assert g['shampoo_beta']==run['covariance_beta']
                assert g['basis_ns_iterations']==(run['basis_ns_iterations'] if custom else 6 if warm else 2)
                assert g['transport_precision']=='highest'
                assert g['variance_policy']=='permutation' and g['precondition_frequency']==20
        for g in groups[label]:
            assert g['basis_rotation_cap_mode']==('spectral_bound' if custom else 'average')
            assert g['basis_rotation_cap']==(.5 if custom else .1)
            if custom:assert g['research_retraction_variant']==run['retraction_variant']
        gs=run['refresh_guards'];guards+=len(gs)
        assert [g['step'] for g in gs]==[1]+list(range(21,3501,20))
        assert all(g['factor_count']==48 and g['maximum_gram_error']<.05 for g in gs)
        assert [o['step'] for o in run['observations']]==[875,1750,3500]
        for o in run['observations']:
            inp={k:o[k] for k in KEYS[1:4]}
            assert o['step'] not in streams or streams[o['step']]==inp
            streams[o['step']]=inp;age=o['step']-1
            assert o['optimizer_counters']==dict(research_qr_refreshes=24*(age//200 if warm else age//20),
                research_warm_refreshes=24*(age//20-age//200 if warm else 0),research_strict_transports=24*(age//20))
        if not custom and not run.get('repeat_of'):
            p=next(x for x in old['runs'] if x['policy']==run['policy'] and x['covariance_beta']==run['covariance_beta']
                   and x['transport_precision']=='highest' and x['basis_rotation_cap']==.1 and not x.get('repeat_of'))
            for o,z in zip(run['observations'],p['observations']):
                assert all(o[k]==z[k] for k in KEYS),(label,o['step'])
                anchors.append(dict(label=label,step=o['step'],exact=True))
        assert [p['step'] for p in run['tracking_probes']]==[x+1 for x in study['tracking_ages']]
        matrices=[]
        for p in run['tracking_probes']:
            assert len(p['matrices'])==24
            for m in p['matrices']:
                assert m['targeted_pass'] and m['selected_basis_update_error']<1e-6
                assert m['world_m_preservation_error']<.01
                assert m['full_strict_pass']==(m['full_strict_update_difference']<.01)
                matrices.append(m)
        for event in sorted({m['executed_method'] for m in matrices}):
            vals=[m for m in matrices if m['executed_method']==event]
            diags.append(dict(label=label,event=event,samples=len(vals),
                full_strict_failures=sum(not m['full_strict_pass'] for m in vals),
                max_full_strict_difference=max(m['full_strict_update_difference'] for m in vals),
                max_selected_basis_error=max(m['selected_basis_update_error'] for m in vals),
                max_world_m_error=max(m['world_m_preservation_error'] for m in vals),
                mean_residual=statistics.mean(m['mean_residual'] for m in vals)))
        end=run['observations'][-1]
        rows.append(dict(label=label,kind=label.split('_beta')[0],beta=run['covariance_beta'],
            repeated=bool(run.get('repeat_of')),ce=end['validation']['ce'],accuracy=end['validation']['accuracy'],
            optimizer_ms=end['timing']['optimizer_ms'],timing=end['timing'],training_seconds=end['training_seconds'],
            maximum_gram=max(g['maximum_gram_error'] for g in gs)))
    controls=[]
    for first,last in study['control_pairs']:
        for a,z in zip(by[first]['observations'],by[last]['observations']):assert all(a[k]==z[k] for k in KEYS)
        assert [p['matrices'] for p in by[first]['tracking_probes']]==[p['matrices'] for p in by[last]['tracking_probes']]
        a,z=[next(x for x in rows if x['label']==v) for v in [first,last]]
        variation={k:abs(a['timing'][k]-z['timing'][k])/statistics.mean([a['timing'][k],z['timing'][k]])
                   for k in ['optimizer_ms','model_backward_cuda_ms','iteration_cuda_ms']}
        variation['training_wall']=abs(a['training_seconds']-z['training_seconds'])/statistics.mean([a['training_seconds'],z['training_seconds']])
        controls.append(dict(original=first,repeated=last,observations_exact=3,probes_exact=True,timing_relative_range=variation))
    nm=json.loads((b/'manifest.json').read_text());om=json.loads((prior/'manifest.json').read_text())
    for k in ['split_sha256','base_prepared_sha256']:assert nm[k]==om[k]
    for k in ['soap.py','soap_reference.py','benchmarks/cifar_lr_screen.py','benchmarks/cifar_optimizers.py']:
        assert nm['source_sha256'][k]==om['source_sha256'][k]
    for k in ['torch','cuda','initial_model_sha256']:assert r[k]==old[k]
    result=dict(status='completed',rows=rows,diagnostics=diags,prior_anchors=anchors,controls=controls,
                refresh_guards=guards,matrix_probes=sum(x['samples'] for x in diags),
                all_in_seconds=r['all_in_seconds'],unchanged_runtime_and_data_provenance=True)
    (b/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Bounded one-step retraction: paired CIFAR learning','',
           'Ten arms, 3500updates each, seed139, 20epochs on the original200-epoch schedule. Both covariance betas .99/.999, fixed LR.0005, Adam(.9,.999), high basis matmuls and strict m transport. Cubic2/quintic1 use row-sum cap.5 with no candidate rescaling; NS6 uses the historical average cap.1 and scaled retraction. Cubic1 failed the earlier unchanged momentum guard; that failed experiment is retained separately. This is a joint movement/retraction intervention, not an iteration-count-only ablation.','',
           '| Covariance beta | Policy | Repeat | Validation CE | Accuracy (%) | Optimizer ms/update | Max Gram |',
           '|---:|---|---|---:|---:|---:|---:|']
    for x in rows:lines.append(f"| {x['beta']} | {x['kind']} | {x['repeated']} | {x['ce']:.6f} | {100*x['accuracy']:.2f} | {x['optimizer_ms']:.3f} | {x['maximum_gram']:.6g} |")
    lines+=['','## Selected-basis and full-refresh diagnostics','',
            '| Arm | Event | Samples | Full strict-refresh failures (1%) | Max full difference | Max world-M error | Mean residual |',
            '|---|---|---:|---:|---:|---:|---:|']
    for x in diags:lines.append(f"| {x['label']} | {x['event']} | {x['samples']} | {x['full_strict_failures']} | {x['max_full_strict_difference']:.5g} | {x['max_world_m_error']:.5g} | {x['mean_residual']:.5g} |")
    lines+=['',f"All {guards:,} refresh guards and {result['matrix_probes']:,} targeted transport probes pass. Full strict-refresh differences remain a separate diagnostic; QR/reset failures are not hidden or treated as proof of bad learning.",'',
            f"All {len(anchors)} historical QR/NS6 observations reproduce exactly. Both repeated controls reproduce weights, metrics, data/RNG hashes and numerical probes. Total training-screen elapsed {r['all_in_seconds']:.1f}s.",'',
            '## Timing repeat variability','',
            '| Control | Optimizer range/mean | Model/backward range/mean | Training wall range/mean |','|---|---:|---:|---:|']
    for x in controls:
        v=x['timing_relative_range'];lines.append(f"| {x['original']} | {100*v['optimizer_ms']:.2f}% | {100*v['model_backward_cuda_ms']:.2f}% | {100*v['training_wall']:.2f}% |")
    lines+=['','This is a one-seed fixed-update screen, not convergence or a tuned-method ranking. Do not combine its losses with synthetic timings to invent a time-to-quality comparison. No optimizer/default changes or saved weights. Numerical probes are offline and excluded from timing. See PROTOCOL.md, experimental_groups.json and source/ for exact scope.']
    (b/'README.md').write_text('\n'.join(lines)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    args=p.parse_args();s=summarize(args.bundle,args.prior)
    print(json.dumps({k:v for k,v in s.items() if k not in ['rows','diagnostics']},indent=2))
