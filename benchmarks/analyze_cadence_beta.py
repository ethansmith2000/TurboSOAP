"""Cost comparisons and exact prior-run anchors for the covariance/cadence screen."""
import argparse
import json
from pathlib import Path
import statistics

from summarize_rectangular_precision import summarize


def analyze(report, prior):
    summary=summarize(report)
    definitions={a['name']:a for a in report['config']['arms']}
    for row in summary['rows']:
        arm=definitions[row['arm']]
        row.update(covariance_beta=arm['covariance_beta'],frequency=arm['frequency'],method=arm['method'],
                   new_covariance_weight_between_refreshes=1-arm['covariance_beta']**arm['frequency'])
    anchors=[]
    aliases={'qr10_cov0p999':'qr10','qr20_cov0p999':'qr20','warm10_cov0p999':'warm6'}
    for r in report['results']:
        if r['arm']['name'] not in aliases:continue
        old=next(c for c in prior['results'] if c['mode']==r['mode'] and c['repeat']==r['repeat']
                 and c['arm']['name']==aliases[r['arm']['name']])
        anchors.append(dict(arm=r['arm']['name'],repeat=r['repeat'],
            loss_records_exact=[s['loss_sanity_only'] for s in r['samples']]==[s['loss_sanity_only'] for s in old['samples']],
            diagnostic_records_exact=r['observations']==old['observations'],
            effective_optimizer_group_exact=r['effective_optimizer_group']==old['effective_optimizer_group']))
    comparisons=[]
    rows=summary['rows']
    for beta in [.999,.99]:
        warm=[r for r in rows if r['method']=='warm' and r['covariance_beta']==beta]
        qr=[r for r in rows if r['method']=='qr' and r['covariance_beta']==beta]
        for w in warm:
            for q in qr:
                if not(w['diagnostic_passed'] and q['diagnostic_passed']):continue
                ratios=[cw['optimizer_cuda_ms']/cq['optimizer_cuda_ms']
                        for cw,cq in zip(w['repeat_mean_ms'],q['repeat_mean_ms'])]
                comparisons.append(dict(covariance_beta=beta,warm_frequency=w['frequency'],qr_frequency=q['frequency'],
                    optimizer_cost_ratios=ratios,median_ratio=statistics.median(ratios)))
    return dict(summary=summary,anchors=anchors,cost_comparisons=comparisons,
        all_anchors_exact=all(a[k] for a in anchors for k in
                             ['loss_records_exact','diagnostic_records_exact','effective_optimizer_group_exact']),
        interpretation='Numerical/cost evidence only; no synthetic-loss ranking or equal-quality speed claim.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True);a=p.parse_args()
    result=analyze(json.loads((a.bundle/'results.json').read_text()),json.loads((a.prior/'results.json').read_text()))
    (a.bundle/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(all_anchors_exact=result['all_anchors_exact'],anchors=len(result['anchors']),
        case_count=result['summary']['case_count'],passing_cases=result['summary']['passing_cases'],
        all_in_seconds=result['summary']['all_in_seconds'])))


if __name__=='__main__':main()
