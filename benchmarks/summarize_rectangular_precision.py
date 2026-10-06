"""Validate and summarize the frozen rectangular precision gate."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def expected_events(arm, updates):
    counts=Counter()
    for step in range(1,updates+1):
        event=('ordinary' if step%arm['frequency'] else 'qr_refresh' if arm['method']=='qr'
               else 'qr_reset' if step%200==0 else 'warm_refresh')
        counts[event]+=1
    return dict(counts)


def summarize(report):
    c=report['config'];results=report['results']
    expected={(m,a['name'],r) for m in c['modes'] for a in c['arms'] for r in range(c['repeats'])}
    actual=[(r['mode'],r['arm']['name'],r['repeat']) for r in results]
    if report['status']!='completed' or len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Incomplete or duplicate grid')
    for r in results:
        if r['arm']!=next(a for a in c['arms'] if a['name']==r['arm']['name']):
            raise ValueError('Arm changed')
        if [s['step'] for s in r['samples']]!=list(range(len(r['samples']))):
            raise ValueError('Discontinuous natural trajectory')
        if r['status']=='valid':
            if len(r['samples'])!=c['updates']+1:raise ValueError('Truncated accepted trajectory')
            counts=Counter(s['event'] for s in r['samples'][1:])
            correct=expected_events(r['arm'],c['updates'])
            if dict(counts)!=correct or r['timing']['event_counts']!=correct:raise ValueError('Wrong cadence')
        elif r['timing'] is not None:raise ValueError('Rejected state has a cost summary')
        if any(not math.isfinite(v) for v in r['errors'].values()):raise ValueError('Nonfinite diagnostic')
        for o in r['observations']:
            for factor in o['geometry']['active_factors']:
                expected_dims=factor['parameter_shape'] if r['mode']=='all' else [min(factor['parameter_shape'])]
                if factor['dimensions']!=expected_dims:raise ValueError('Missing active factor')
        budgets=c['diagnostic_budgets'];e=r['errors']
        passed=(r['status']=='valid' and e['maximum_global_momentum_error']<=budgets['momentum_relative'] and
                e['maximum_transport_world_error']<=budgets['momentum_relative'] and
                e['maximum_shadow_update_error']<=budgets['update_relative'] and
                e['minimum_shadow_update_cosine']>=budgets['update_cosine'] and e['maximum_shadow_gram_error']<.05)
        if r['diagnostic_budget_passed']!=passed:raise ValueError('Diagnostic gate mismatch')
    rows=[]
    for mode in c['modes']:
        for arm in c['arms']:
            cases=sorted([r for r in results if r['mode']==mode and r['arm']==arm],key=lambda r:r['repeat'])
            row=dict(mode=mode,arm=arm['name'],geometry_passed=all(r['status']=='valid' for r in cases),
                diagnostic_passed=all(r['diagnostic_budget_passed'] for r in cases),
                errors={k:(min if k.startswith('minimum') else max)(r['errors'][k] for r in cases)
                        for k in cases[0]['errors']},
                failures=[dict(repeat=r['repeat'],**r['failure']) for r in cases if r['failure']],
                compile_preflights=[r['compile_preflight'] for r in cases],
                initialization_ms=[r['samples'][0]['optimizer_cuda_ms'] for r in cases],
                state_bytes=cases[0]['state_bytes'])
            warm=[f for r in cases for o in r['observations'] if o['event']=='warm_refresh'
                  for refresh in o['refreshes'] for f in refresh['factors']]
            row['warm_factor_diagnostics']={str(dim):dict(samples=sum(f['dimension']==dim for f in warm),
                mean_basis_change_rms=statistics.mean(f['basis_change_rms'] for f in warm if f['dimension']==dim),
                mean_residual_change=statistics.mean(f['new_residual']-f['old_residual'] for f in warm if f['dimension']==dim))
                for dim in sorted(set(f['dimension'] for f in warm))}
            if row['geometry_passed']:
                row['repeat_mean_ms']=[r['timing']['mean_ms'] for r in cases]
                row['median_mean_ms']={k:statistics.median(r['timing']['mean_ms'][k] for r in cases)
                                       for k in cases[0]['timing']['mean_ms']}
            rows.append(row)
    return dict(case_count=len(results),geometry_rejections=sum(r['status']!='valid' for r in results),
        diagnostic_budget_rejections=sum(r['status']=='valid' and not r['diagnostic_budget_passed'] for r in results),
        passing_cases=sum(r['diagnostic_budget_passed'] for r in results),
        all_in_seconds=report['all_in_seconds'],rows=rows,
        limitations=['Numerical budgets are provisional; no learning-quality claim.',
            'Same-state shadow does not measure prior BF16 covariance error or diagonal-v approximation.',
            'Full-cycle costs exclude diagnostic and compiler preflight overhead; raw first events are retained.',
            'Residual/motion comparisons across arms have different trajectories; no counterfactual tracking claim.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle
    s=summarize(json.loads((b/'results.json').read_text()))
    (b/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps({k:v for k,v in s.items() if k not in ['rows','limitations']}))


if __name__=='__main__':main()
