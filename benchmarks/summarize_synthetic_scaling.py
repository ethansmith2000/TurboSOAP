"""Report synthetic cadence estimates without treating them as throughput runs."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(report):
    if report['status'] not in ['completed','completed_with_rejections']:raise ValueError('Require completed event map')
    c=report['config'];expected=len(c['widths'])*len(c['tokens'])*len(c['modes'])*c['rounds']*2
    if len(report['results'])!=expected:raise ValueError('Incomplete case grid')
    keys=[(r['width'],r['tokens'],r['mode'],r['method'],r['round']) for r in report['results']]
    if len(set(keys))!=expected:raise ValueError('Duplicate case')
    comparisons=[];unavailable=[]
    for width in c['widths']:
        for tokens in c['tokens']:
            for mode in c['modes']:
                rows={r['policy']:r for r in report['summary'] if
                      (r['width'],r['tokens'],r['mode'])==(width,tokens,mode)}
                for baseline in ['qr10','qr20']:
                    if 'warm10_qr200' not in rows or baseline not in rows:
                        unavailable.append(dict(width=width,tokens=tokens,mode=mode,baseline=baseline,
                            reason='At least one required event round failed the numerical guard; no estimate promoted.'))
                        continue
                    warm=rows['warm10_qr200'];reference=rows[baseline];ratios={}
                    for metric in ['optimizer_cuda_ms','whole_step_wall_ms']:
                        values=[w[metric]/q[metric] for w,q in zip(warm['round_estimates_ms'],reference['round_estimates_ms'])]
                        ratios[metric]=dict(median=statistics.median(values),minimum=min(values),maximum=max(values),rounds=values)
                    comparisons.append(dict(width=width,tokens=tokens,mode=mode,baseline=baseline,
                        ratio_warm_over_baseline=ratios,parameter_count=warm['parameter_count'],
                        maximum_active_factor=max(max(x['dimensions']) for x in warm['active_factors']),
                        warm_state_bytes=warm['state_bytes']))
    return dict(status=report['status'],scope='Ratios of cadence-weighted event cost estimates, not observed schedule throughput.',
                comparisons=comparisons,unavailable=unavailable,
                rejected_cases=[{k:r[k] for k in ['width','tokens','mode','method','round','failure']} for r in report['results'] if r['status']!='valid'],
                all_in_seconds=report['all_in_seconds'],limitations=report['limitations'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle
    summary=summarize(json.loads((b/'results.json').read_text()))
    (b/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
