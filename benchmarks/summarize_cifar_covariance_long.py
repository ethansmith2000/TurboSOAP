"""Validate and summarize the longer, fixed-LR covariance-only experiment."""
import argparse
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def early_anchors(bundle, prior, *, complete=True):
    read=lambda root,name:json.loads((root/name).read_text())
    current,old=read(bundle,'screen.json'),read(prior,'screen.json')
    cm,om=read(bundle,'manifest.json'),read(prior,'manifest.json')
    for key in ['torch','cuda','initial_model_sha256','parameter_count']:
        require(current[key]==old[key],f'Changed anchor environment: {key}')
    for key in ['split_sha256','base_prepared_sha256']:
        require(cm[key]==om[key],f'Changed data: {key}')
    ignored={'memory_screen','branch_steps','max_wall_seconds'}
    for key in (set(current['config'])|set(old['config']))-ignored:
        require(current['config'].get(key)==old['config'].get(key),f'Changed config: {key}')
    core=['soap.py','soap_reference.py','benchmarks/cifar_optimizers.py',
          'benchmarks/cifar_lr_screen.py','benchmarks/cifar_optimizer_benchmark.py']
    core+=sorted(k for k in cm['source_sha256'] if k.startswith('benchmarks/cifar_support/'))
    for key in core:
        require(cm['source_sha256'][key]==om['source_sha256'][key],f'Changed source: {key}')
    old_runs={r['label']:r for r in old['runs']}
    anchors=[]
    for run in current['runs']:
        if run.get('repeat_of'):
            continue
        reference=old_runs[run['label']]
        require(run['effective_groups']==reference['effective_groups'],f'Changed groups: {run["label"]}')
        observations={o['step']:o for o in run['observations']}
        for step in [875,1750,3500]:
            if step not in observations and not complete:
                continue
            require(step in observations,f'Missing anchor: {run["label"]}/{step}')
            actual=observations[step]
            expected=next(o for o in reference['observations'] if o['step']==step)
            for key in ['model_sha256','rng_sha256','consumed_ids_sha256',
                        'augmented_batches','validation','train_panel']:
                require(actual[key]==expected[key],f'Anchor mismatch: {run["label"]}/{step}/{key}')
            anchors.append(dict(label=run['label'],step=step,exact_match=True))
    if complete:
        require(len(anchors)==18,'Require all six trajectories and three early anchors')
    return dict(passed=True,complete=len(anchors)==18,anchors=anchors,prior_bundle=str(prior),
                scope='Same seed extended; exact six-arm continuity. No timing pooling.')


def summarize(bundle, prior):
    report=json.loads((bundle/'screen.json').read_text())
    require(report['status']=='completed','Require completed study')
    require(report['control_repeat']['model_hash_equal'],'Control repeat mismatch')
    require(report['paired_fixed_update_checks_passed'],'Unpaired training streams')
    c=report['config'];study=c['memory_screen'];runs=report['runs']
    require(len(runs)==7 and len({r['label'] for r in runs})==7,'Require seven distinct arms')
    require(study['equal_updates']==17500 and study['wall_seconds'] is None,'Wrong endpoint or timing scope')
    require({(r['policy'],r['covariance_beta']) for r in runs}==
            {(p,b) for p in ['qr10','qr20','warm10'] for b in [.99,.999]},'Incomplete covariance pairs')
    reference={o['step']:o for o in runs[0]['observations']}
    rows=[]
    for run in runs:
        label=run['label']
        require(run['status']=='completed' and run['steps']==17500,f'Incomplete run: {label}')
        require(run['wall_target_reached'] is None,f'Unexpected wall target: {label}')
        require(run['matrix_beta2']==.999,f'Confounded matrix memory: {label}')
        require([o['step'] for o in run['observations']]==study['evaluation_updates'],f'Missing observations: {label}')
        require({g['role'] for g in run['effective_groups']}=={'matrix','auxiliary'},f'Wrong group roles: {label}')
        for group in run['effective_groups']:
            require(group['betas']==[.9,.999] and group['peak_lr']==.0005,f'Confounded group: {label}')
            if group['role']=='matrix':
                require(group['shampoo_beta']==run['covariance_beta'],f'Wrong covariance memory: {label}')
        trajectory=[]
        for observation in run['observations']:
            step=observation['step'];ref=reference[step]
            require(observation['reasons']==['fixed_updates'],f'Unexpected observation: {label}')
            for key in ['rng_sha256','consumed_ids_sha256','augmented_batches']:
                require(observation[key]==ref[key],f'Unpaired stream: {label}/{step}/{key}')
            if run.get('repeat_of'):
                for key in ['model_sha256','validation','train_panel']:
                    require(observation[key]==ref[key],f'Nonrepeating control: {step}/{key}')
            require(all(value==0 for value in observation['evaluation_parity'].values()),f'Evaluation parity: {label}/{step}')
            for metric in ['validation','train_panel']:
                require(math.isfinite(observation[metric]['ce']) and
                        0<=observation[metric]['accuracy']<=1,f'Invalid metric: {label}/{step}/{metric}')
            gram=observation['maximum_gram_error']
            require(math.isfinite(gram) and gram<=study['maximum_gram_error'],f'Geometry guard: {label}/{step}')
            trajectory.append(dict(step=step,epoch=observation['epoch'],
                validation_ce=observation['validation']['ce'],accuracy=observation['validation']['accuracy'],
                train_panel_ce=observation['train_panel']['ce'],maximum_gram_error=gram,
                timing=observation['timing'],training_seconds=observation['training_seconds']))
        rows.append(dict(label=label,policy=run['policy'],covariance_beta=run['covariance_beta'],
                         repeated=bool(run.get('repeat_of')),trajectory=trajectory))
    pairs=[]
    for policy in ['qr10','qr20','warm10']:
        base=next(r for r in rows if r['label']==policy+'_base')
        candidate=next(r for r in rows if r['label']==policy+'_cov99')
        deltas=[dict(step=a['step'],epoch=a['epoch'],ce_delta=b['validation_ce']-a['validation_ce'],
                     accuracy_delta=b['accuracy']-a['accuracy'])
                for a,b in zip(base['trajectory'],candidate['trajectory'])]
        pairs.append(dict(policy=policy,trajectory_deltas=deltas,endpoint=deltas[-1]))
    controls=[r['trajectory'][-1] for r in rows if r['label'] in ['qr10_base','qr10_base_repeat_end']]
    def relative_range(values):
        return (max(values)-min(values))/(sum(values)/len(values))
    summary=dict(status='completed',rows=rows,paired_effects=pairs,
        early_anchor_bridge=early_anchors(bundle,prior),all_in_seconds=report['all_in_seconds'],
        gpu_uuid=report['gpu_uuid'],paired_and_effective_group_checks_passed=True,
        control_timing=dict(training_seconds=[r['training_seconds'] for r in controls],
            wall_relative_range=relative_range([r['training_seconds'] for r in controls]),
            phase_relative_ranges={key:relative_range([r['timing'][key] for r in controls]) for key in
                ['augmentation_cuda_ms','model_backward_cuda_ms','optimizer_ms','iteration_cuda_ms']}),
        limitations=['Same seed extended to 100 of 200 scheduled epochs; no new-seed or convergence claim.',
            'Fixed LR .0005 was selected using earlier baseline memory; no beta-specific retuning.',
            'Elapsed CUDA intervals include dispatch gaps; no equal-wall quality observations.',
            'Joint beta changes, v transport and one-sided factors are separate untested axes.'])
    (bundle/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--check-anchors-only',action='store_true')
    args=parser.parse_args();bundle=args.bundle.resolve();prior=args.prior.resolve()
    result=early_anchors(bundle,prior,complete=False) if args.check_anchors_only else summarize(bundle,prior)
    print(json.dumps(result,indent=2))
