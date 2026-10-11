"""Fail-closed scope/trajectory/code/backend qualification for four new contexts."""
import itertools
import json
from pathlib import Path
import sys
try:
    from .cudnn_deterministic_contract import validate_config,finite_records,backend_delta
    from .summarize_execution_diagnostic import compare
except ImportError:
    from cudnn_deterministic_contract import validate_config,finite_records,backend_delta
    from summarize_execution_diagnostic import compare


def qualify(runs):
    completed=len(runs)==4 and all(r.get('status')=='completed' for r in runs)
    if not completed:return dict(completed=False,qualified=False)
    pairs=[dict(left=a['context']['label'],right=b['context']['label'],first_difference=compare(a,b),
                code_exact=a['code_after_measured']==b['code_after_measured'],
                backend_exact=a['backend_after_measured']==b['backend_after_measured'],
                runtime_names_counts_exact=a['runtime_profile']['kernel_names_counts']==b['runtime_profile']['kernel_names_counts'])
           for a,b in itertools.combinations(runs,2)]
    checks=dict(prescribed_fingerprints=all([x['step'] for x in r['measured']]==[1,2,20,21,25] and
        [x['step'] for x in r['smoke_fingerprints']]==[1,25] for r in runs),
        sampled_trajectories_exact=all(x['first_difference'] is None for x in pairs),
        loaded_code_exact=all(x['code_exact'] for x in pairs),backend_metadata_exact=all(x['backend_exact'] for x in pairs),
        benchmark_disabled_and_other_flags_fixed=all(r['backend_before_smoke']['cudnn_flags']['benchmark'] is False and
            r['backend_after_measured']['cudnn_flags']['benchmark'] is False and
            r['backend_before_smoke']['cudnn_flags']['deterministic'] is True and
            r['backend_after_measured']['cudnn_flags']['deterministic'] is True and
            r['backend_after_measured']['deterministic_algorithms'] is False for r in runs),
        finite_records=all(finite_records(r['measured']) and finite_records(r['smoke_fingerprints']) for r in runs),
        geometry=all([x['step'] for x in r['geometry']]==[1,21] and all(x['maximum_gram_error']<.05 for x in r['geometry']) for r in runs),
        initial_state_restoration=all(r['initial_state_replay']['final_invariants_before']==r['initial_state_replay']['final_invariants_after'] and
            r['initial_state_replay']['held_invariants_before']==r['initial_state_replay']['held_invariants_after'] and
            r['initial_state_replay']['rebuilt_input']==r['measured'][0]['input'] and
            r['initial_state_replay']['rebuilt_target']==r['measured'][0]['target'] and
            r['initial_state_replay']['retained_tensor_bytes']<120*2**20 for r in runs),
        replay=all(r['runtime_profile']['plain_profile_exact'] and r['runtime_profile']['invariants_before']==r['runtime_profile']['invariants_after'] for r in runs),
        release_affinity=all(r['owned_optimizer_released'] and len(r['affinity_restored'])==2 for r in runs))
    return dict(completed=True,qualified=all(checks.values()),checks=checks,pairs=pairs,
                runtime_names_counts_exact=all(x['runtime_names_counts_exact'] for x in pairs))


def main():
    b=Path(sys.argv[1]).resolve();c=json.loads((b/'config.json').read_text());validate_config(c)
    m=json.loads((b/'manifest.json').read_text());baseline=Path(m['baseline_bundle'])
    prior=[json.loads(p.read_text()) for p in sorted((baseline/'contexts').glob('*/result.json'))]
    runs=[]
    for ctx in c['execution_diagnostic']['contexts']:
        p=b/'contexts'/ctx['label']/'result.json'
        if p.exists():runs.append(json.loads(p.read_text()))
    result=qualify(runs);deltas=[]
    for r in runs:
        for phase in ['backend_before_smoke','backend_after_measured']:
            if phase in r:deltas.append(dict(label=r['context']['label'],phase=phase,
                fields=backend_delta(prior[0][phase],r[phase])))
    result['baseline_backend_deltas']=deltas
    result['only_declared_backend_metadata_change']=bool(deltas) and all(sorted(x['fields'])==sorted(m['declared_baseline_backend_deltas']) for x in deltas)
    result['qualified']=result['qualified'] and result['only_declared_backend_metadata_change']
    result['baseline_comparisons']=[dict(control=r['context']['label'],baseline=a['context']['label'],
        first_sampled_difference=compare(a,r)) for r in runs for a in prior if a['status']==r['status']=='completed']
    result['early_replay_comparisons']=[dict(label=r['context']['label'],actual_first_vs_plain=r['initial_state_replay']['actual_first_vs_plain'],
        actual_first_vs_profiled=r['initial_state_replay']['actual_first_vs_profiled'],plain_vs_profiled=r['initial_state_replay']['plain_vs_profiled']) for r in runs if 'initial_state_replay' in r]
    result['processes']=len(runs);result['measured_updates']=sum(25 for r in runs if r['status']=='completed')
    (b/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['pairs','baseline_comparisons','baseline_backend_deltas']},indent=2))


if __name__=='__main__':main()
