"""Validate effective cadence, guard coverage, paired streams and CIFAR anchors."""
import argparse
import json
from pathlib import Path
import statistics


IDENTITY_KEYS=['model_sha256','rng_sha256','consumed_ids_sha256','augmented_batches','validation','train_panel']


def summarize(bundle,prior):
    r=json.loads((bundle/'screen.json').read_text());old=json.loads((prior/'screen.json').read_text())
    if r['status']!='completed' or not r['control_repeat']['model_hash_equal']:raise ValueError('Incomplete or mismatched study')
    c=r['config'];study=c['memory_screen'];expected={a['label']:a for a in study['arms']}
    if len(r['runs'])!=len(expected) or {x['label'] for x in r['runs']}!=set(expected):raise ValueError('Incomplete grid')
    rows=[];pairs={};guard_max=0.
    for run in r['runs']:
        arm=expected[run['label']]
        if any(run[k]!=v for k,v in arm.items()):raise ValueError('Arm settings changed')
        warm=arm['policy'].startswith('warm');frequency=int(arm['policy'][4:] if warm else arm['policy'][2:])
        if run['status']!='completed' or run['steps']!=3500 or run['wall_target_reached'] is not None:raise ValueError('Wrong endpoint')
        for g in run['effective_groups']:
            if g['betas']!=[.9,.999] or g['peak_lr']!=.0005:raise ValueError('Confounded Adam/LR')
            if g['role']=='matrix':
                settings=dict(shampoo_beta=arm['covariance_beta'],basis_method='warm' if warm else 'qr',
                    precondition_frequency=frequency,hard_reset_interval=200 if warm else 0,
                    basis_ns_iterations=6 if warm else 2,variance_policy='permutation')
                if any(g[k]!=v for k,v in settings.items()):raise ValueError('Wrong effective matrix settings')
        guards=run['refresh_guards'];steps=[1]+list(range(1+frequency,3501,frequency))
        if [g['step'] for g in guards]!=steps:raise ValueError('Missing refresh guard')
        matrices=c['depth']*4
        if any(g['factor_count']!=2*matrices or not 0<=g['maximum_gram_error']<.05 for g in guards):raise ValueError('Invalid basis guard')
        guard_max=max(guard_max,max(g['maximum_gram_error'] for g in guards))
        if [o['step'] for o in run['observations']]!=[875,1750,3500]:raise ValueError('Missing observation')
        for o in run['observations']:
            age=o['step']-1
            counts=dict(research_qr_refreshes=matrices*(age//200 if warm else age//frequency),
                        research_warm_refreshes=matrices*(age//frequency-age//200 if warm else 0))
            if o['optimizer_counters']!=counts:raise ValueError('Wrong actual event count')
            stream={k:o[k] for k in ['rng_sha256','consumed_ids_sha256','augmented_batches']}
            if o['step'] in pairs and pairs[o['step']]!=stream:raise ValueError('Unpaired input stream')
            pairs[o['step']]=stream
        endpoint=run['observations'][-1]
        rows.append(dict(label=run['label'],policy=run['policy'],covariance_beta=run['covariance_beta'],
            repeated=bool(run.get('repeat_of')),validation_ce=endpoint['validation']['ce'],
            accuracy=endpoint['validation']['accuracy'],train_panel_ce=endpoint['train_panel']['ce'],
            optimizer_ms=endpoint['timing']['optimizer_ms'],training_seconds=endpoint['training_seconds'],
            phase_timing=endpoint['timing'],maximum_refresh_gram_error=max(g['maximum_gram_error'] for g in guards),
            diagnostic_seconds=sum(g['diagnostic_seconds'] for g in guards),refresh_guard_count=len(guards)))
    by_label={run['label']:run for run in r['runs']};control_checks=[]
    for first,last in study['control_pairs']:
        for x,y in zip(by_label[first]['observations'],by_label[last]['observations']):
            equal=all(x[k]==y[k] for k in IDENTITY_KEYS)
            control_checks.append(dict(step=x['step'],exact=equal))
    if not all(x['exact'] for x in control_checks):raise ValueError('Control trajectory mismatch')
    nm=json.loads((bundle/'manifest.json').read_text());om=json.loads((prior/'manifest.json').read_text())
    provenance={k:nm[k]==om[k] for k in ['split_sha256','base_prepared_sha256']}
    for name in ['soap.py','soap_reference.py',*[k for k in nm['source_sha256'] if k.startswith('benchmarks/cifar_support/')]]:
        provenance[name]=nm['source_sha256'][name]==om['source_sha256'][name]
    for key in ['torch','cuda','initial_model_sha256']:provenance[key]=r[key]==old[key]
    anchors=[]
    for beta,old_label in [(.999,'qr20_base'),(.99,'qr20_cov99')]:
        current=next(x for x in r['runs'] if x['policy']=='qr20' and x['covariance_beta']==beta and not x.get('repeat_of'))
        previous=next(x for x in old['runs'] if x['label']==old_label)
        for x,y in zip(current['observations'],previous['observations']):
            anchors.append(dict(covariance_beta=beta,step=x['step'],exact=all(x[k]==y[k] for k in IDENTITY_KEYS)))
    primary=[row for row in rows if not row['repeated']];comparisons=[]
    for beta in [.999,.99]:
        lookup={x['policy']:x for x in primary if x['covariance_beta']==beta}
        for candidate,baseline in [('warm20','qr20'),('warm40','qr40'),('warm20','qr40'),
                                    ('warm40','qr20'),('warm40','warm20'),('qr40','qr20')]:
            a,b=lookup[candidate],lookup[baseline]
            comparisons.append(dict(covariance_beta=beta,candidate=candidate,baseline=baseline,
                ce_delta=a['validation_ce']-b['validation_ce'],accuracy_delta=a['accuracy']-b['accuracy'],
                optimizer_cost_ratio=a['optimizer_ms']/b['optimizer_ms']))
    effects=[]
    for policy in ['qr20','qr40','warm20','warm40']:
        a=next(x for x in primary if x['policy']==policy and x['covariance_beta']==.99)
        b=next(x for x in primary if x['policy']==policy and x['covariance_beta']==.999)
        effects.append(dict(policy=policy,ce_delta_cov99_minus_cov999=a['validation_ce']-b['validation_ce']))
    controls=[x for x in rows if x['label'] in study['control_pairs'][0]]
    variation={}
    for key in ['optimizer_ms','model_backward_cuda_ms','iteration_cuda_ms']:
        values=[x['phase_timing'][key] for x in controls]
        variation[key]=(max(values)-min(values))/statistics.mean(values)
    times=[x['training_seconds'] for x in controls]
    variation['training_wall']=(max(times)-min(times))/statistics.mean(times)
    return dict(status='completed',rows=rows,comparisons=comparisons,covariance_effects=effects,
        control_checks=control_checks,prior_anchors=anchors,prior_provenance=provenance,
        all_prior_anchors_exact=all(provenance.values()) and all(x['exact'] for x in anchors),
        all_paired_streams_verified=True,all_effective_groups_verified=True,all_refresh_guards_verified=True,
        maximum_refresh_gram_error=guard_max,control_timing_relative_ranges=variation,
        all_in_seconds=r['all_in_seconds'],
        limitations=['One seed and early20-epoch endpoint on200-epoch schedule; no convergence/replication claim.',
            'Fixed LR.0005 for every policy/beta; practical optimizer rankings would need LR calibration.',
            'Offline per-refresh geometry probes excluded from timings; no online fallback tested.',
            'CIFAR timings only; no equal-wall observations or attachment of synthetic times to these losses.'])


def write_report(bundle,summary):
    s=summary
    lines=['# CIFAR-100: NS6 warm cadence and covariance quality\n',
        'Completed October 3, 2026 (America/New_York; directory uses UTC date). '
        'Nine arms: eight policy/beta combinations and one repeated QR20/.999 control. '
        'Each arm trains for 3500 updates (20 epochs) on the original 200-epoch schedule, '
        'with validation at epochs 5/10/20. Seed139; matrix and auxiliary peak LR.0005; '
        'Adam(.9,.999). This is a fixed-LR early-learning screen, not a tuned or converged ranking.\n',
        'At this fixed LR and early endpoint, QR has lower validation CE and lower optimizer '
        'cost than NS6 warm at matched cadence and covariance beta. All warm arms pass the '
        'per-refresh geometry guard. Covariance .99 improves all four policies in this screen. '
        'These results support keeping sparse QR as the control; they do not select a '
        'universally best beta or establish a tuned optimizer ranking.\n',
        '## Endpoint evidence\n',
        '| Covariance beta | Policy | Validation CE | Accuracy | Optimizer ms/update | Maximum refresh Gram error |\n'
        '|---:|---|---:|---:|---:|---:|']
    for beta in [.999,.99]:
        for policy in ['qr20','qr40','warm20','warm40']:
            row=next(r for r in s['rows'] if r['covariance_beta']==beta and r['policy']==policy and not r['repeated'])
            lines.append(f"| {beta} | {policy} | {row['validation_ce']:.6f} | {100*row['accuracy']:.2f}% | "
                         f"{row['optimizer_ms']:.3f} | {row['maximum_refresh_gram_error']:.6g} |")
    lines.extend(['\nWarm policies use six retraction iterations and scheduled QR every 200 matrix updates. '
        'QR-only policies do not use warm retractions. Covariance/Adam update every trainer step; '
        'bases and first moments change coordinates only on refreshes. Warm carries v; QR permutes it.\n',
        '## Fixed-beta comparisons\n',
        '| Covariance beta | Candidate minus baseline | CE difference (negative is better) | CIFAR optimizer-cost ratio |\n'
        '|---:|---|---:|---:|'])
    for comparison in s['comparisons']:
        lines.append(f"| {comparison['covariance_beta']} | {comparison['candidate']} − {comparison['baseline']} | "
                     f"{comparison['ce_delta']:+.6f} | {comparison['optimizer_cost_ratio']:.3f} |")
    lines.extend(['\n## Covariance effect within each policy\n',
                  '| Policy | CE(.99) minus CE(.999) |\n|---|---:|'])
    for effect in s['covariance_effects']:
        lines.append(f"| {effect['policy']} | {effect['ce_delta_cov99_minus_cov999']:+.6f} |")
    lines.extend(['\n## Integrity and numerical checks\n',
        'All input/RNG/augmentation streams match across arms at each observation. '
        'The repeated QR20/.999 control exactly matches weights, validation/train-panel metrics '
        'and input hashes at all three observations. Both QR20 betas exactly match the earlier '
        'memory study at all six corresponding observations; core optimizer, data/support, split '
        'and initial-weight provenance match. These checks validate the extended adapter and '
        'added probes against established numerical trajectories; timing is not pooled.\n',
        'Every matrix refresh is checked before later QR can hide drift. Because the first '
        'trainer step initializes matrix state and skips its update, matrix age is 3499 at '
        'the endpoint. Interval20/40 has 174/87 refreshes per matrix. Warm includes 17 QR '
        'resets and 157/70 warm events per matrix; QR-only has 174/87 QR events. All 24 '
        'matrix parameters and 48 factors are covered. Per-arm guard counts are 175/88, '
        'including initialization. Effective group settings and counters are checked explicitly.\n',
        f"Maximum Gram error over all refresh probes: {s['maximum_refresh_gram_error']:.6g}, "
        'below the predeclared .05 guard. The guard checks finite matrix parameters/moments '
        'and basis orthogonality. It does not establish covariance eigenspace accuracy or '
        'the adequacy of diagonal-v carry. No adaptive online error-triggered fallback was added.\n',
        '## Timing scope\n',
        'Component times include full optimizer work and exclude the first 50 updates, '
        'matching prior CIFAR reporting. Per-refresh diagnostics, evaluation, hash probes '
        'and compile qualification are outside training-time accounting. CUDA intervals '
        'include host dispatch gaps. These are descriptive costs for this CIFAR trainer, '
        'not equal-wall quality observations or large-factor throughput predictions.\n',
        '| Repeated-control timing component | Range / mean |\n|---|---:|'])
    for key,value in s['control_timing_relative_ranges'].items():lines.append(f'| {key} | {100*value:.2f}% |')
    lines.extend(['\n![Learning curves](learning_curves.png)\n',
        '## Limits and follow-up\n',
        'Single seed, early endpoint, and one fixed LR for every policy/beta. '
        'Any apparent ranking is conditional on those settings. The earlier memory study '
        'showed that some early beta gains shrink by 100 epochs; do not extrapolate these '
        '20-epoch effects to convergence. Next, diagnose warm basis tracking on the actual '
        'training covariances and check a bounded LR bracket for the new NS6 schedules '
        'before committing to larger training comparisons. Numerical geometry alone '
        'has not resolved the learning gap. Synthetic timing and CIFAR quality cannot be '
        'combined into a measured time-to-quality result. Preserve sparse QR controls '
        'in follow-up. No optimizer defaults change from this screen.\n',
        '## Resources and reproduction\n',
        f"Total job time: {s['all_in_seconds']/60:.2f} minutes. One lifetime GPU claim, released; "
        'private supervisor stopped. No model weights, checkpoints or moment snapshots. '
        '145 tests passed before launch, including actual adapter cadence/retraction calls. '
        'Frozen config/source/data provenance: `manifest.json`, `config.json`, `PROTOCOL.md`, '
        '`source/`. Raw curves, every refresh probe, hashes and timing: `screen.json`, '
        '`run.log`. Machine-validated comparisons and controls: `summary.json`. '
        'Launch command: `launch.conf`. Existing cached datasets and old campaigns remain intact.\n'])
    (bundle/'README.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True);a=p.parse_args()
    s=summarize(a.bundle,a.prior);(a.bundle/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    if not s['all_prior_anchors_exact']:raise ValueError('Historical QR20 anchor mismatch; investigate before ranking')
    write_report(a.bundle,s)
    print(json.dumps({k:s[k] for k in ['status','all_prior_anchors_exact','maximum_refresh_gram_error','all_in_seconds']}))


if __name__=='__main__':main()
