"""Validate matrix-LR attribution and aggregate paired local tracking scores."""
import argparse
import json
from pathlib import Path
import statistics


KEYS=['model_sha256','rng_sha256','consumed_ids_sha256','augmented_batches','validation','train_panel']


def summarize(bundle,prior):
    r=json.loads((bundle/'screen.json').read_text());old=json.loads((prior/'screen.json').read_text())
    if r['status']!='completed':raise ValueError('Require completed run')
    study=r['config']['memory_screen'];arms={a['label']:a for a in study['arms']};by={x['label']:x for x in r['runs']}
    if set(by)!=set(arms) or len(r['runs'])!=len(arms):raise ValueError('Incomplete grid')
    rows=[];streams={};tracking=[]
    for label,run in by.items():
        a=arms[label]
        if any(run[k]!=v for k,v in a.items()):raise ValueError('Changed arm')
        if run['status']!='completed' or run['steps']!=3500:raise ValueError('Wrong endpoint')
        warm=a['policy']=='warm20'
        for g in run['effective_groups']:
            if g['betas']!=[.9,.999] or g['peak_lr']!=(a['matrix_lr'] if g['role']=='matrix' else .0005):
                raise ValueError('Confounded memory/LR')
            if g['role']=='matrix':
                expected=dict(shampoo_beta=.99,basis_method='warm' if warm else 'qr',precondition_frequency=20,
                    hard_reset_interval=200 if warm else 0,basis_ns_iterations=6 if warm else 2,variance_policy='permutation')
                if any(g[k]!=v for k,v in expected.items()):raise ValueError('Confounded matrix setting')
        guards=run['refresh_guards']
        if [g['step'] for g in guards]!=[1]+list(range(21,3501,20)):raise ValueError('Missing guards')
        if any(g['factor_count']!=48 or not 0<=g['maximum_gram_error']<.05 for g in guards):raise ValueError('Failed guard')
        if [o['step'] for o in run['observations']]!=[875,1750,3500]:raise ValueError('Missing observations')
        for o in run['observations']:
            age=o['step']-1;expected=dict(research_qr_refreshes=24*(age//200 if warm else age//20),
                research_warm_refreshes=24*(age//20-age//200 if warm else 0))
            if o['optimizer_counters']!=expected:raise ValueError('Wrong actual cadence')
            stream={k:o[k] for k in KEYS[1:4]}
            if o['step'] in streams and streams[o['step']]!=stream:raise ValueError('Unpaired input stream')
            streams[o['step']]=stream
        probes=run.get('tracking_probes',[])
        if a['tracking_probe']:
            if [p['step'] for p in probes]!=[age+1 for age in study['tracking_ages']]:raise ValueError('Missing tracking age')
            for p in probes:
                if len(p['matrices'])!=24 or len({m['matrix'] for m in p['matrices']})!=24:raise ValueError('Missing tracked matrices')
                for m in p['matrices']:
                    if m['age']!=p['step']-1 or [f['dimension'] for f in m['factors']]!=m['shape']:raise ValueError('Wrong tracking shape/age')
        elif probes:raise ValueError('Unplanned probes')
        endpoint=run['observations'][-1]
        rows.append(dict(label=label,policy=a['policy'],matrix_lr=a['matrix_lr'],repeated=bool(a.get('repeat_of')),
            ce=endpoint['validation']['ce'],accuracy=endpoint['validation']['accuracy'],train_ce=endpoint['train_panel']['ce'],
            optimizer_ms=endpoint['timing']['optimizer_ms'],training_seconds=endpoint['training_seconds'],
            phase_timing=endpoint['timing'],maximum_gram_error=max(g['maximum_gram_error'] for g in guards)))
        if probes:
            # Keep actual QR reset separate; equal-weight factor summaries are
            # descriptive, not weighted by layer importance or covariance energy.
            for event in sorted({m['executed_method'] for p in probes for m in p['matrices']}):
                matrices=[m for p in probes for m in p['matrices'] if m['executed_method']==event]
                factors=[f for m in matrices for f in m['factors']]
                for dim in sorted({f['dimension'] for f in factors}):
                    subset=[f for f in factors if f['dimension']==dim]
                    tracking.append(dict(label=label,executed_method=event,dimension=dim,samples=len(subset),
                        cap_binding_fraction=statistics.mean(f['cap_binding'] for f in subset),
                        mean_residual={k:statistics.mean(f['scores'][k]['residual'] for f in subset)
                                       for k in ['before','executed','qr','warm1','warm2']},
                        maximum_gram={k:max(f['scores'][k]['gram'] for f in subset) for k in ['executed','qr','warm1','warm2']},
                        qr_lower_residual_fraction=statistics.mean(f['scores']['qr']['residual']<f['scores']['warm1']['residual'] for f in subset),
                        second_warm_improves_fraction=statistics.mean(f['scores']['warm2']['residual']<f['scores']['warm1']['residual'] for f in subset)))
    control=[]
    for first,last in study['control_pairs']:
        for x,y in zip(by[first]['observations'],by[last]['observations']):
            control.append(dict(step=x['step'],exact=all(x[k]==y[k] for k in KEYS)))
        if [p['matrices'] for p in by[first]['tracking_probes']] != [p['matrices'] for p in by[last]['tracking_probes']]:
            raise ValueError('Repeated-control tracking diagnostics differ')
    if not all(c['exact'] for c in control):raise ValueError('Control mismatch')
    anchors=[]
    for policy in ['qr20','warm20']:
        new=by[policy+'_lr0.0005'];previous=next(x for x in old['runs'] if x['policy']==policy and x['covariance_beta']==.99 and not x.get('repeat_of'))
        for x,y in zip(new['observations'],previous['observations']):
            anchors.append(dict(policy=policy,step=x['step'],exact=all(x[k]==y[k] for k in KEYS)))
    nm=json.loads((bundle/'manifest.json').read_text());om=json.loads((prior/'manifest.json').read_text())
    provenance={k:nm[k]==om[k] for k in ['split_sha256','base_prepared_sha256']}
    for name in ['soap.py','soap_reference.py',*[k for k in nm['source_sha256'] if k.startswith('benchmarks/cifar_support/')]]:
        provenance[name]=nm['source_sha256'][name]==om['source_sha256'][name]
    for key in ['torch','cuda','initial_model_sha256']:provenance[key]=r[key]==old[key]
    primary=[x for x in rows if not x['repeated']]
    best={policy:min((x for x in primary if x['policy']==policy),key=lambda x:x['ce']) for policy in ['qr20','warm20']}
    controls=[x for x in rows if x['label'] in study['control_pairs'][0]]
    variation={}
    for key in ['optimizer_ms','model_backward_cuda_ms','iteration_cuda_ms']:
        values=[x['phase_timing'][key] for x in controls];variation[key]=(max(values)-min(values))/statistics.mean(values)
    values=[x['training_seconds'] for x in controls];variation['training_wall']=(max(values)-min(values))/statistics.mean(values)
    return dict(status='completed',rows=rows,best_tested=best,best_warm_minus_best_qr_ce=best['warm20']['ce']-best['qr20']['ce'],
        tracking=tracking,control_checks=control,prior_anchors=anchors,prior_provenance=provenance,
        all_prior_anchors_exact=all(provenance.values()) and all(x['exact'] for x in anchors),
        paired_inputs_and_effective_settings_verified=True,guards_verified=True,tracking_coverage_verified=True,
        gpu_uuid=r['gpu_uuid'],prior_gpu_uuid=old['gpu_uuid'],control_timing_relative_ranges=variation,
        all_in_seconds=r['all_in_seconds'],
        limitations=['One seed and 20-epoch endpoint on 200-epoch schedule; only matrix LR tuned over three values.',
            'Covariance .99 fixed; auxiliary LR .0005 fixed. The tested bracket may not contain the optimum.',
            'Tracking alternatives share incoming state but are never executed; lower residual is not a learning guarantee.',
            'QR alternative permutes v while warm carries it; update differences are algorithmic, not ground-truth errors.',
            'Different physical GPU from prior study; numerical anchors verified, timings not pooled.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    a=p.parse_args();s=summarize(a.bundle,a.prior);(a.bundle/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps({k:s[k] for k in ['all_prior_anchors_exact','best_warm_minus_best_qr_ce','all_in_seconds']}))


if __name__=='__main__':main()
