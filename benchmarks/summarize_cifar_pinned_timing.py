"""Apply predeclared timing, numerical, completeness, and placement gates."""
import json
from pathlib import Path
import sys
import summarize_cifar_timing


def main():
    summarize_cifar_timing.main()
    b=Path(sys.argv[1]);r=json.loads((b/'screen.json').read_text());s=json.loads((b/'summary.json').read_text())
    study=r['config']['memory_screen'];runs=r['runs'];replays=r.get('profile_replays',[])
    s['complete_records']=len(runs)==6 and all(run['status']=='completed' and run['steps']==3500 and
        len(run['refresh_guards'])==study['expected_refresh_guards_per_arm'] and
        sum(len(p['matrices']) for p in run['tracking_probes'])==study['expected_targeted_matrices_per_arm']
        for run in runs)
    s['compiler_stable']=all(x['telemetry_start']['compiler_counters']==x['telemetry_end']['compiler_counters'] for x in runs)
    records=json.loads((b/'affinity_records.json').read_text())
    restored=json.loads((b/'affinity_restored.json').read_text())
    s['placement_recorded']=len(records)==12 and all([x[1] for x in record['threads']]==[116,117] for record in records) and len(restored)==2
    s['profile_exact_recomputed']=len(replays)==6 and all(
        len(group:=[x for x in replays if x['kind']==kind])==3 and
        all(x['numerical_control']==group[0]['numerical_control'] for x in group) for kind in ['qr','fro'])
    profiles=[x['profiler'] for x in replays if 'profiler' in x]
    s['corrected_profiles_complete']=len(profiles)==4 and all(
        p['device_events']>0 and p['device_sum_us']>0 and p['unattributed_gpu_events']==0 and
        all(p['phases'].get('timing/'+phase,{}).get('host_calls')==4 and
            p['phases']['timing/'+phase]['host_wall_us']>0 for phase in ['augmentation','model_backward','optimizer'])
        for p in profiles)
    s['qualified']=s['qualified'] and all(s[k] for k in ['complete_records','compiler_stable','placement_recorded','profile_exact_recomputed','corrected_profiles_complete'])
    (b/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print('FINAL_GATE '+json.dumps({k:v for k,v in s.items() if k not in ['policies','historical_matches']}))


if __name__=='__main__':main()
