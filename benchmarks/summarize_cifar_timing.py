"""Report repeat limits without discarding timing or numerical failures."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
from timing_diagnostics import range_over_mean


def main():
    b=Path(sys.argv[1]);r=json.loads((b/'screen.json').read_text())
    m=json.loads((b/'manifest.json').read_text())
    history=Path(m['historical_bundle'])/'screen.json'
    assert hashlib.sha256(history.read_bytes()).hexdigest()==m['historical_screen_sha256']
    old=json.loads(history.read_text())
    s=dict(status=r['status'],policies=[],historical_matches=[],profile_replays_exact=all(x['exact'] for x in r.get('profile_replays',[])))
    def controls(o):
        return {k:o[k] for k in ['step','validation','train_panel','model_sha256','rng_sha256',
                                 'consumed_ids_sha256','augmented_batches','optimizer_counters',
                                 'maximum_gram_error','mean_gram_error']}
    def probes(run):
        return [{k:v for k,v in item.items() if k!='diagnostic_seconds'} for item in run['tracking_probes']]
    for kind,old_label in [('qr','qr20_beta0.99'),('fro','fro_cubic2_beta0.99')]:
        runs=[x for x in r['runs'] if x['kind']==kind]
        ref=runs[0];oldrun=next(x for x in old['runs'] if x['label']==old_label)
        for run in runs:
            for obs in run['observations']:
                expected=next(x for x in oldrun['observations'] if x['step']==obs['step'])
                s['historical_matches'].append(dict(label=run['label'],step=obs['step'],exact=controls(obs)==controls(expected)))
        exact=all([controls(o) for o in run['observations']]==[controls(o) for o in ref['observations']] and
                  run['optimizer_sha256']==ref['optimizer_sha256'] and probes(run)==probes(ref) for run in runs)
        wall=[run['training_seconds'] for run in runs];opt=[run['optimizer_ms'] for run in runs]
        rows=[]
        for run in runs:
            phases={phase:{key:sum(row['host']['phases'][phase][key] for row in run['history'])/1000
                            for key in ('host_wall_ms','thread_cpu_ms','process_cpu_ms')}
                    for phase in ('augmentation','model_backward','optimizer')}
            gc_events=[event for row in run['history'] for event in row['host']['gc']]
            rows.append(dict(label=run['label'],training_seconds=run['training_seconds'],
                model_event_ms=run['model_backward_cuda_ms'],optimizer_event_ms=run['optimizer_ms'],
                host_phase_seconds=phases,gc_seconds=sum(x['wall_ms'] for x in gc_events)/1000,
                gc_count=len(gc_events),involuntary_context_switches=sum(x['host']['involuntary_context_switches'] for x in run['history'])))
        guards=all(g['maximum_gram_error']<.05 for run in runs for g in run['refresh_guards'])
        # Targeted implementations keep QR full-reference checks separate.
        targeted=all(mat['targeted_pass'] for run in runs for probe in run['tracking_probes'] for mat in probe['matrices'])
        wr,ore=range_over_mean(wall),range_over_mean(opt)
        s['policies'].append(dict(kind=kind,exact_controls=exact,guards_pass=guards,targeted_pass=targeted,
            wall_range_over_mean=wr,optimizer_range_over_mean=ore,repeat_gate=exact and guards and targeted and wr<=.03 and ore<=.02,runs=rows))
    s['qualified']=r['status']=='completed' and len(s['historical_matches'])==18 and all(x['exact'] for x in s['historical_matches']) and s['profile_replays_exact'] and all(x['repeat_gate'] for x in s['policies'])
    (b/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps(s,indent=2))


if __name__=='__main__':main()
