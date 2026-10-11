"""Analyze the declared local CPU placement intervention; no training."""
import json
from pathlib import Path
import statistics
import sys
from timing_diagnostics import range_over_mean


def controls(o):
    return {k:o[k] for k in ['step','validation','train_panel','model_sha256','rng_sha256',
        'consumed_ids_sha256','augmented_batches','optimizer_counters','maximum_gram_error','mean_gram_error']}


def probes(run):
    return [{k:v for k,v in item.items() if k!='diagnostic_seconds'} for item in run['tracking_probes']]


def main():
    b=Path(sys.argv[1]);r=json.loads((b/'screen.json').read_text());runs=r['runs'];ref=runs[0]
    old=json.loads(Path('/workspace/optimizer_replay_results/cifar_soap_movement_budget_20261006/screen.json').read_text())
    oldobs=next(o for run in old['runs'] if run['label']=='qr20_beta0.99' for o in run['observations'] if o['step']==875)
    exact=len(runs)==6 and all([controls(o) for o in run['observations']]==[controls(o) for o in ref['observations']] and
        run['optimizer_sha256']==ref['optimizer_sha256'] and probes(run)==probes(ref) for run in runs)
    historical=[dict(label=run['label'],exact=controls(next(o for o in run['observations'] if o['step']==875))==controls(oldobs)) for run in runs]
    compiler_stable=all(run['telemetry_start']['compiler_counters']==run['telemetry_end']['compiler_counters'] for run in runs)
    s=dict(status=r['status'],exact_across_placements=exact,historical_matches=historical,compiler_stable=compiler_stable,conditions=[])
    for kind in ['near','far']:
        group=[run for run in runs if run['kind']==kind]
        wall=[x['training_seconds'] for x in group];opt=[x['optimizer_ms'] for x in group];model=[x['model_backward_cuda_ms'] for x in group]
        s['conditions'].append(dict(kind=kind,runs=[dict(label=x['label'],training_seconds=x['training_seconds'],
            model_event_ms=x['model_backward_cuda_ms'],optimizer_event_ms=x['optimizer_ms'],
            model_host_seconds=sum(h['host']['phases']['model_backward']['host_wall_ms'] for h in x['history'])/1000) for x in group],
            wall_range_over_mean=range_over_mean(wall),optimizer_range_over_mean=range_over_mean(opt),
            model_mean_ms=statistics.mean(model),model_min_ms=min(model),model_max_ms=max(model),
            repeat_gate=len(group)==3 and range_over_mean(wall)<=.03 and range_over_mean(opt)<=.02))
    near,far=s['conditions'];effect=far['model_mean_ms']/near['model_mean_ms']-1
    pairs=[]
    for a,z in zip(runs[::2],runs[1::2]):
        n=a if a['kind']=='near' else z;f=a if a['kind']=='far' else z
        pairs.append(f['model_backward_cuda_ms']/n['model_backward_cuda_ms']-1)
    separated=near['model_max_ms']<far['model_min_ms'] or far['model_max_ms']<near['model_min_ms']
    s.update(model_relative_effect_far_over_near_minus_one=effect,adjacent_pair_effects=pairs,
        material_placement_effect=exact and compiler_stable and abs(effect)>=.05 and separated and all(x*effect>0 for x in pairs),
        refresh_guards=sum(len(x['refresh_guards']) for x in runs),
        targeted_probes=sum(len(p['matrices']) for x in runs for p in x['tracking_probes']),
        guards_pass=all(g['maximum_gram_error']<.05 for x in runs for g in x['refresh_guards']),
        targeted_pass=all(m['targeted_pass'] for x in runs for p in x['tracking_probes'] for m in p['matrices']))
    replay=r.get('profile_replays',[])
    s['profile_replays_exact']=len(replay)==6 and all(x['exact'] for x in replay) and all(x['numerical_control']==replay[0]['numerical_control'] for x in replay)
    s['profile_device_events_present']=sum('profiler' in x and x['profiler']['device_events']>0 for x in replay)==4
    s['local_repeat_qualified']=r['status']=='completed' and exact and all(x['exact'] for x in historical) and s['guards_pass'] and s['targeted_pass'] and all(x['repeat_gate'] for x in s['conditions']) and s['profile_replays_exact'] and s['profile_device_events_present']
    (b/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))


if __name__=='__main__':main()
