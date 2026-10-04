"""Summarize a completed frozen LR screen without choosing new experiments."""
import argparse
import json
from pathlib import Path


def summarize(bundle):
    report=json.loads((bundle/'screen.json').read_text())
    if report['status']!='completed':raise ValueError('Require completed, reproducible screen')
    study=report['config']['lr_screen'];rows=[]
    for run in report['runs']:
        equal=next(x for x in run['observations'] if x['step']==study['equal_updates'])
        wall=next((x for x in run['observations'] if 'equal_wall' in x['reasons']),None)
        rows.append(dict(label=run['label'],policy=run['policy'],matrix_lr=run['matrix_lr'],
            repeated=bool(run.get('repeat_of')) or run['label'].endswith('_repeat'),validation_ce=equal['validation']['ce'],
            accuracy=equal['validation']['accuracy'],training_seconds=equal['training_seconds'],
            optimizer_ms=equal.get('timing',{}).get('optimizer_ms',run['optimizer_ms']),
            phase_timing=equal.get('timing',{}),maximum_gram_error=equal['maximum_gram_error'],
            wall_step=wall['step'] if wall else None,
            wall_seconds=wall['training_seconds'] if wall else None,
            wall_overshoot_seconds=wall['training_seconds']-study['wall_seconds'] if wall else None,
            wall_ce=wall['validation']['ce'] if wall else None,
            wall_accuracy=wall['validation']['accuracy'] if wall else None))
    selections=[]
    for policy in ['qr10','warm10','qr20']:
        candidates=[x for x in rows if x['policy']==policy and not x['repeated']]
        best=min(candidates,key=lambda x:x['validation_ce'])
        rates=sorted(x['matrix_lr'] for x in candidates)
        selections.append(dict(policy=policy,label=best['label'],matrix_lr=best['matrix_lr'],
            validation_ce=best['validation_ce'],boundary_winner=best['matrix_lr'] in (rates[0],rates[-1])))
    timing_repeats=[]
    by_label={x['label']:x for x in rows}
    for original,repeated in study.get('control_pairs',[['qr10_lr0.001','qr10_lr0.001_repeat']]):
        first,repeat=by_label[original],by_label[repeated]
        timing_repeats.append(dict(original=original,repeated=repeated,
            first_training_seconds=first['training_seconds'],repeat_training_seconds=repeat['training_seconds'],
            relative_training_time_change=repeat['training_seconds']/first['training_seconds']-1,
            first_optimizer_ms=first['optimizer_ms'],repeat_optimizer_ms=repeat['optimizer_ms'],
            relative_optimizer_time_change=repeat['optimizer_ms']/first['optimizer_ms']-1,
            first_phases=first['phase_timing'],repeat_phases=repeat['phase_timing']))
    timing_gate=None
    if study.get('bracket')=='lower':
        labels={name for pair in study['control_pairs'] for name in pair}
        controls=[by_label[name] for name in labels]
        times=[x['training_seconds'] for x in controls]
        spread=(max(times)-min(times))/(sum(times)/len(times))
        phases={key:(max(x['phase_timing'][key] for x in controls)-min(x['phase_timing'][key] for x in controls))/
                     (sum(x['phase_timing'][key] for x in controls)/len(controls))
                for key in ['augmentation_cuda_ms','model_backward_cuda_ms','optimizer_ms','iteration_cuda_ms','process_cpu_seconds']
                if all(x['phase_timing'].get(key,0)>0 for x in controls)}
        timing_gate=dict(control_count=len(controls),relative_wall_range=spread,maximum_relative_wall_range=.03,
            passed=len(controls)==3 and spread<=.03 and report['control_repeat'].get('model_hash_equal',False),
            relative_phase_ranges=phases,scope='Predeclared heuristic for preliminary wall-time interpretation; not significance.')
    summary=dict(status='completed',rows=rows,selected_by_equal_update_ce=selections,
        timing_interpretation_gate=timing_gate,
        control_timing_repeat=timing_repeats[-1],control_timing_repeats=timing_repeats,
        control_repeat=report['control_repeat'],paired_fixed_update_checks_passed=report['paired_fixed_update_checks_passed'],
        all_in_seconds=report['all_in_seconds'],gpu_uuid=report['gpu_uuid'],
        limitations=['One seed; 20 of 200 scheduled epochs, not converged training.',
            'Validation selects LRs; these validation scores are not unbiased held-out generalization estimates.',
            'Boundary winners require a new predeclared LR bracket before practical optimizer ranking.',
            'Equal-wall is the first 25-update boundary after 75 training seconds; overshoot is reported.',
            'Training wall excludes compile smoke, diagnostics, hash probes and logging; full run time is separate.',
            'Timing can vary with device/system load; this campaign uses one UUID; cross-campaign timing is not a paired measurement.',
            'Both second-moment betas .999, two-sided factors, QR-reindexed/warm-carried v. Other policies untested.'])
    (bundle/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args();summary=summarize(a.bundle)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
