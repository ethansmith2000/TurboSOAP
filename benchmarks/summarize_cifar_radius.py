"""Summarize common-state diagnostics without treating shadows as learning runs."""
import copy
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

KEYS=('step','model_sha256','rng_sha256','consumed_ids_sha256','augmented_batches','validation','train_panel')


def numerical(probes):
    result=copy.deepcopy(probes)
    for p in result:
        p.pop('diagnostic_seconds',None)
        for m in p['matrices']:
            for e in m['radius_effects'].values():e.pop('refresh_timing',None)
    return result


def main():
    b=Path(sys.argv[1]);r=json.loads((b/'screen.json').read_text());manifest=json.loads((b/'manifest.json').read_text())
    oldpath=Path(manifest['historical_bundle'])/'screen.json'
    assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==manifest['historical_screen_sha256']
    previous=json.loads(oldpath.read_text());runs=r['runs'];assert r['status']=='completed' and len(runs)==3
    anchors=[];records=[]
    for run in runs:
        assert run['status']=='completed' and run['steps']==1000
        assert len(run['refresh_guards'])==50 and all(x['maximum_gram_error']<.05 for x in run['refresh_guards'])
        assert [p['step'] for p in run['tracking_probes']]==[age+1 for age in r['config']['memory_screen']['tracking_ages']]
        prior=next(a for a in previous['runs'] if a['label']==f"fro_cubic2_beta{run['covariance_beta']}")
        obs=run['observations'][0];expected=next(o for o in prior['observations'] if o['step']==875)
        assert all(obs[k]==expected[k] for k in KEYS)
        anchors.append(dict(label=run['label'],step=875,exact=True))
        for p in run['tracking_probes']:
            assert len(p['matrices'])==24
            for matrix in p['matrices']:
                assert matrix['targeted_pass'] and matrix['radius_control_exact']
                if not run.get('repeat_of'):
                    for policy,effects in matrix['radius_effects'].items():
                        records.append(dict(policy=policy,age=matrix['age'],matrix=matrix['matrix'],shape=matrix['shape'],
                            beta=run['covariance_beta'],**effects))
    a,z=runs[0],runs[2]
    assert all(all(x[k]==y[k] for k in KEYS) for x,y in zip(a['observations'],z['observations']))
    assert a['optimizer_sha256']==z['optimizer_sha256'] and numerical(a['tracking_probes'])==numerical(z['tracking_probes'])
    rows=[]
    for policy in sorted({x['policy'] for x in records}):
        selected=[x for x in records if x['policy']==policy];factors=[f for x in selected for f in x['factors']]
        chains=[x['frozen_covariance_chain'] for x in selected if 'frozen_covariance_chain' in x]
        costs=[x for x in selected if 'refresh_timing' in x]
        rows.append(dict(policy=policy,matrices=len(selected),factors=len(factors),
            mean_movement=st.mean(x['movement_rms'] for x in factors),mean_residual_change=st.mean(x['residual_change'] for x in factors),
            maximum_gram=max(x['gram'] for x in factors),maximum_world_m=max(x['world_m_error'] for x in selected),
            single_refresh_failures=sum(not x['passes'] for x in selected),
            chains=len(chains),maximum_chain_world_m=max(x['maximum_world_m_error'] for x in chains),
            chain_failures=sum(not x['passes'] for x in chains),
            mean_sampled_refresh_ms=st.mean(x['refresh_timing']['median_ms'] for x in costs),
            shadow_qualified=all(x['passes'] for x in selected) and all(x['passes'] for x in chains)))
    costs_by_shape=[]
    for policy in sorted({x['policy'] for x in records}):
        for shape in sorted({tuple(x['shape']) for x in records}):
            for phase in ['strong','cheap']:
                sel=[x for x in records if x['policy']==policy and tuple(x['shape'])==shape and 'refresh_timing' in x and
                     ((x['age']//20)%2==1)==(phase=='strong')]
                costs_by_shape.append(dict(policy=policy,shape=shape,phase=phase,samples=len(sel),
                    mean_refresh_ms=st.mean(x['refresh_timing']['median_ms'] for x in sel)))
    result=dict(status='completed',rows=rows,costs_by_shape=costs_by_shape,historical_matches=anchors,
        exact_repeated_observations=True,exact_repeated_probes=True,refresh_guards=150,targeted_probes=864,
        interpretation='Offline common-state and frozen-covariance chain diagnostics; not candidate learning or natural-cycle qualification.')
    (b/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
