"""Summarize independent covariance/matrix-v memory interventions at fixed LR."""
import argparse
import json
from pathlib import Path


def baseline_bridge(bundle,prior):
    read=lambda b,n:json.loads((b/n).read_text())
    new,old=read(bundle,'screen.json'),read(prior,'screen.json')
    nm,om=read(bundle,'manifest.json'),read(prior,'manifest.json')
    failures=[]
    def compare(x,y,name):
        if x!=y:failures.append(name)
    for key in ['torch','cuda','initial_model_sha256']:compare(new[key],old[key],key)
    for key in ['split_sha256','base_prepared_sha256']:compare(nm[key],om[key],key)
    excluded={'memory_screen','lr_screen','max_wall_seconds','matrix_lr'}
    for key in (set(new['config'])|set(old['config']))-excluded:
        compare(new['config'][key],old['config'][key],f'config:{key}')
    source_keys=['soap.py','soap_reference.py']
    source_keys+=sorted(k for k in nm['source_sha256'] if k.startswith('benchmarks/cifar_support/'))
    for key in source_keys:compare(nm['source_sha256'][key],om['source_sha256'][key],key)
    anchors=[]
    for policy in ['qr10','qr20','warm10']:
        a=next(x for x in new['runs'] if x['label']==policy+'_base')
        b=next(x for x in old['runs'] if x['label']==policy+'_lr0.0005')
        compare(a['matrix_lr'],b['matrix_lr'],policy+':matrix_lr')
        before=len(failures)
        for step in [875,1750,3500]:
            x=next(o for o in a['observations'] if o['step']==step)
            y=next(o for o in b['observations'] if o['step']==step)
            for key in ['model_sha256','rng_sha256','consumed_ids_sha256','augmented_batches','validation','train_panel']:
                compare(x[key],y[key],f'{policy}:{step}:{key}')
        anchors.append(dict(policy=policy,exact_match=len(failures)==before))
    return dict(passed=not failures,failures=failures,anchors=anchors,prior_bundle=str(prior),
        scope='Core SOAP/data/config plus exact baseline trajectory checks; adapter adds tested matrix-only beta selection. No timing pooling.')


def summarize(bundle,prior):
    report=json.loads((bundle/'screen.json').read_text())
    if report['status']!='completed' or not report['control_repeat']['model_hash_equal']:
        raise ValueError('Require a complete study and exact control repeats')
    c=report['config'];study=c['memory_screen'];rows=[]
    for run in report['runs']:
        # Reject a confounded summary even if training itself completed.
        groups=run['effective_groups'];assert {g['role'] for g in groups}=={'matrix','auxiliary'}
        for group in groups:
            expected=run['matrix_beta2'] if group['role']=='matrix' else study['auxiliary_beta2_fixed']
            assert group['betas']==[study['matrix_beta1_fixed'],expected],(run['label'],group)
            assert group['peak_lr']==.0005
            if group['role']=='matrix':assert group['shampoo_beta']==run['covariance_beta']
        fixed=next(o for o in run['observations'] if o['step']==study['equal_updates'])
        assert run['steps']==study['equal_updates'] and run['wall_target_reached'] is None
        assert not any('equal_wall' in o['reasons'] for o in run['observations'])
        rows.append(dict(label=run['label'],policy=run['policy'],variant=run['variant'],
            repeated=bool(run.get('repeat_of')),covariance_beta=run['covariance_beta'],matrix_beta2=run['matrix_beta2'],
            validation_ce=fixed['validation']['ce'],accuracy=fixed['validation']['accuracy'],
            train_panel_ce=fixed['train_panel']['ce'],maximum_gram_error=fixed['maximum_gram_error'],
            optimizer_ms=fixed['timing']['optimizer_ms'],training_seconds=fixed['training_seconds'],
            phase_timing=fixed['timing']))
    base={x['policy']:x for x in rows if x['variant']=='base' and not x['repeated']}
    for row in rows:
        ref=base[row['policy']]
        row['ce_delta_vs_own_baseline']=row['validation_ce']-ref['validation_ce']
        row['accuracy_delta_vs_own_baseline']=row['accuracy']-ref['accuracy']
    controls=[x for x in rows if x['policy']=='qr10' and x['variant']=='base']
    times=[x['training_seconds'] for x in controls]
    phase_ranges={key:(max(x['phase_timing'][key] for x in controls)-min(x['phase_timing'][key] for x in controls))/
                      (sum(x['phase_timing'][key] for x in controls)/len(controls))
        for key in ['augmentation_cuda_ms','model_backward_cuda_ms','optimizer_ms','iteration_cuda_ms','process_cpu_seconds']}
    summary=dict(status='completed',rows=rows,control_repeat=report['control_repeat'],
        paired_fixed_update_checks_passed=report['paired_fixed_update_checks_passed'],
        effective_group_memory_checks_passed=True,baseline_bridge=baseline_bridge(bundle,prior),
        control_timing=dict(count=len(controls),training_wall_relative_range=(max(times)-min(times))/(sum(times)/len(times)),
            phase_relative_ranges=phase_ranges,scope='Descriptive timing only; no equal-wall quality observations in this study.'),
        all_in_seconds=report['all_in_seconds'],gpu_uuid=report['gpu_uuid'],
        limitations=['One seed, 20 of 200 scheduled epochs; no convergence or replication claim.',
            'Both LRs fixed .0005; beta/LR interactions are not retuned here.',
            'Only matrix memory changes; auxiliary AdamW beta2 fixed .999.',
            'One-factor interventions only; combined .99/.99 and v-transport policies not tested.',
            'Timing intervals include host dispatch gaps; no wall-quality ranking.'])
    (bundle/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True);a=p.parse_args()
    print(json.dumps(summarize(a.bundle.resolve(),a.prior.resolve()),indent=2))
