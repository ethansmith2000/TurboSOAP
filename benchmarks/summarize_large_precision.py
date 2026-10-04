"""Validate the complete large-factor screen and retain repeat-level comparisons."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from benchmark_large_precision import event_name


def summarize(report):
    config = report['config']; results = report['results']
    expected = {(width, arm['name'], repeat) for width in config['widths']
                for arm in config['arms'] for repeat in range(config['rounds'])}
    actual = [(r['width'], r['arm']['name'], r['round']) for r in results]
    if set(actual) != expected or len(actual) != len(expected): raise ValueError('Incomplete or duplicate grid')
    rows = []
    for width in config['widths']:
        for arm in config['arms']:
            cases = sorted([r for r in results if r['width'] == width and r['arm']['name'] == arm['name']],
                           key=lambda r: r['round'])
            for case in cases:
                if case['arm'] != arm: raise ValueError('Arm settings changed')
                if case['status'] == 'valid':
                    counts = Counter(event_name(i, arm) for i in range(1, config['updates'] + 1))
                    if case['summary']['event_counts'] != dict(counts): raise ValueError('Wrong natural events')
                    if len(case['samples']) != config['updates'] + 1: raise ValueError('Missing updates')
                    if case['maximum_gram_error'] >= .05: raise ValueError('Accepted failed state')
                elif case['summary'] is not None: raise ValueError('Rejected result has timings summary')
            row = dict(width=width, arm=arm['name'], valid=all(c['status'] == 'valid' for c in cases),
                maximum_gram_error=max(c['maximum_gram_error'] for c in cases),
                failures=[dict(round=c['round'], **c['failure']) for c in cases if c['failure']],
                state_bytes=cases[0]['state_bytes'], peak_allocated_bytes=max(c['peak_allocated_bytes'] for c in cases),
                initialization_optimizer_ms=[c['samples'][0]['optimizer_cuda_ms'] for c in cases],
                compile_preflight=[c['compile_preflight'] for c in cases if c['compile_preflight']],
                inference_exit_probes=[c['inference_exit_probe'] for c in cases if 'inference_exit_probe' in c])
            if row['valid']:
                row['repeat_mean_ms'] = [c['summary']['mean_ms'] for c in cases]
                row['median_mean_ms'] = {key: statistics.median(c['summary']['mean_ms'][key] for c in cases)
                                         for key in cases[0]['summary']['mean_ms']}
                row['event_optimizer_ms'] = {key: [c['summary']['event_optimizer_ms'][key] for c in cases]
                                             for key in cases[0]['summary']['event_optimizer_ms']}
            rows.append(row)
    comparisons = []
    for width in config['widths']:
        for candidate, baseline in [('qr10_high', 'qr10_fp32'), ('qr10_bf16_cov', 'qr10_high'),
                ('qr20_high', 'qr10_high'), ('warm6_high', 'warm6_fp32'),
                ('warm6_bf16_cov', 'warm6_high'), ('warm6_bf16_gauge', 'warm6_high'),
                ('warm6_compile', 'warm6_high'), ('warm6_inference', 'warm6_high'),
                ('warm6_high', 'qr10_high'), ('warm6_high', 'qr20_high')]:
            c = next(r for r in rows if r['width'] == width and r['arm'] == candidate)
            b = next(r for r in rows if r['width'] == width and r['arm'] == baseline)
            if not (c['valid'] and b['valid']): continue
            ratios = {metric: [cr[metric] / br[metric] for cr, br in zip(c['repeat_mean_ms'], b['repeat_mean_ms'])]
                      for metric in c['median_mean_ms']}
            comparisons.append(dict(width=width, candidate=candidate, baseline=baseline, repeat_ratios=ratios,
                                    median_ratios={k: statistics.median(v) for k, v in ratios.items()}))
    return dict(rows=rows, comparisons=comparisons, rejected_cases=sum(r['status'] != 'valid' for r in results),
                case_count=len(results), all_in_seconds=report['all_in_seconds'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--bundle', type=Path, required=True)
    bundle = parser.parse_args().bundle
    report = json.loads((bundle/'results.json').read_text())
    if report['status'] not in ('completed', 'completed_with_rejections'): raise ValueError('Run incomplete')
    summary = summarize(report)
    (bundle/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(case_count=summary['case_count'], rejected_cases=summary['rejected_cases'],
                          all_in_seconds=summary['all_in_seconds'])))


if __name__ == '__main__': main()
