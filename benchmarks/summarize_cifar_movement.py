"""Validate common-state attribution and preserve every candidate's failures."""
import argparse
import json
import math
from pathlib import Path
import statistics as st

KEYS=('model_sha256','rng_sha256','consumed_ids_sha256','augmented_batches','validation','train_panel')


def summarize(bundle, prior):
    r=json.loads((bundle/'screen.json').read_text())
    previous=json.loads((prior/'screen.json').read_text())
    assert r['status']=='completed' and len(r['runs'])==5
    study=r['config']['memory_screen'];by={a['label']:a for a in r['runs']}
    assert set(by)=={a['label'] for a in study['arms']}
    factors=[];effects=[];anchors=[];guards=0
    for a in r['runs']:
        assert a['status']=='completed' and a['steps']==1000
        guards+=len(a['refresh_guards'])
        assert [p['step'] for p in a['tracking_probes']]==[age+1 for age in study['tracking_ages']]
        assert all(p['factor_count']==48 and p['maximum_gram_error']<.05 for p in a['refresh_guards'])
        assert len(a['refresh_guards'])==50
        if not a.get('repeat_of'):
            oldkind='cubic2' if a['label'].startswith('row_cubic2') else 'ns6'
            old=next(x for x in previous['runs'] if x['label']==f"{oldkind}_beta{a['covariance_beta']}")
            obs=a['observations'][0];assert obs['step']==875
            assert all(obs[k]==old['observations'][0][k] for k in KEYS),a['label']
            anchors.append(dict(label=a['label'],step=875,exact=True))
        for probe in a['tracking_probes']:
            assert len(probe['matrices'])==24
            for m in probe['matrices']:
                assert m['targeted_pass'] and m['selected_basis_update_error']<1e-6
                if a.get('repeat_of'):continue
                context=dict(label=a['label'],age=m['age'],matrix=m['matrix'],beta=a['covariance_beta'])
                for f in m['movement_factors']:
                    for name,z in f['alternatives'].items():
                        assert all(math.isfinite(v) for v in z.values())
                        factors.append(dict(**context,policy=name,axis=f['axis'],dimension=f['dimension'],
                            raw_spectral=f['raw_spectral'],row_bound=f['row_bound'],skew_fro_bound=f['skew_fro_bound'],**z))
                for name,z in m['movement_effects'].items():
                    assert all(math.isfinite(v) for v in z.values())
                    effects.append(dict(**context,policy=name,**z))
    for first,last in study['control_pairs']:
        a,z=by[first],by[last]
        assert all(all(x[k]==y[k] for k in KEYS) for x,y in zip(a['observations'],z['observations']))
        assert [p['matrices'] for p in a['tracking_probes']]==[p['matrices'] for p in z['tracking_probes']]
    rows=[]
    for name in sorted({x['policy'] for x in factors}):
        fs=[x for x in factors if x['policy']==name];es=[x for x in effects if x['policy']==name]
        rows.append(dict(policy=name,factors=len(fs),matrices=len(es),
            mean_scale=st.mean(x['scale'] for x in fs),median_scale=st.median(x['scale'] for x in fs),
            binding_fraction=st.mean(x['scale']<.99999 for x in fs),
            mean_movement=st.mean(x['movement_rms'] for x in fs),
            mean_residual_change=st.mean(x['residual_change'] for x in fs),
            maximum_gram=max(x['gram'] for x in fs),maximum_world_m=max(x['world_m_error'] for x in es),
            world_m_failures=sum(x['world_m_error']>=.01 for x in es),
            maximum_rotation_spectral=max(x['rotation_spectral'] for x in fs)))
    groups=[]
    for age in study['tracking_ages']:
        for name in sorted({x['policy'] for x in factors}):
            fs=[x for x in factors if x['policy']==name and x['age']==age]
            groups.append(dict(age=age,policy=name,mean_movement=st.mean(x['movement_rms'] for x in fs),
                mean_residual_change=st.mean(x['residual_change'] for x in fs),mean_scale=st.mean(x['scale'] for x in fs)))
    result=dict(status='completed',rows=rows,by_age=groups,anchors=anchors,
        exact_repeated_observations=2,exact_repeated_probes=True,refresh_guards=guards,
        matrices_including_repeat=5*7*24,all_in_seconds=r['all_in_seconds'],
        interpretation='Common-state intervention; oracle includes expensive eigensolve and is not a runtime candidate.')
    (bundle/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# CIFAR movement bounds: common-state attribution','',
        'Four source trajectories and one exact repeat, 1000 updates each. Alternatives are evaluated on identical incoming states and never feed back into training. Covariance .99/.999; high basis arithmetic and highest diagnostic precision.','',
        '| Policy | Mean retained scale | Mean movement RMS | Mean residual change | Max Gram | Max world-M error | M failures ≥1% |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for x in rows:lines.append(f"| {x['policy']} | {x['mean_scale']:.4f} | {x['mean_movement']:.5f} | {x['mean_residual_change']:.5f} | {x['maximum_gram']:.5g} | {x['maximum_world_m']:.5g} | {x['world_m_failures']} |")
    lines+=['',f"All {guards} actual-control refresh guards pass. Four historical step-875 observations and both repeated observations match exactly; repeated numerical probes are exact. {result['matrices_including_repeat']} matrices sampled including the repeat.",'',
        'The skew-Frobenius bound uses paired singular values of a real skew matrix. It is not guaranteed tighter than row sum for every matrix. The exact spectral bound is an offline oracle; its eigensolve cost is excluded from deployed-policy proposals. A shadow policy passing these checks still requires natural-trajectory qualification. Residual improvement is not a learning guarantee. See PROTOCOL.md and frozen source.']
    (bundle/'README.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    a=p.parse_args();print(json.dumps(summarize(a.bundle,a.prior),indent=2))
