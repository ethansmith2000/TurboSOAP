"""Exact sampled numerical comparisons, with execution metadata kept separate."""
import itertools
import json
from pathlib import Path
import sys
try:
    from .execution_fingerprints import first_difference
except ImportError:
    from execution_fingerprints import first_difference

STAGES=['ids_sha256','input','target','rng_sha256','output','loss','grad_before_clip',
        'grad_after_clip','gradient_norm','model_sha256','basis','optimizer_sha256','counters']


def compare(a,b):
    if a['initial_model_sha256']!=b['initial_model_sha256']:return dict(phase='initialization',field='model')
    if a['batch_plan']['first25_ids_sha256']!=b['batch_plan']['first25_ids_sha256']:
        return dict(phase='allocation',field='first25_ids')
    for phase in ['smoke_fingerprints','measured']:
        if [r['step'] for r in a[phase]]!=[r['step'] for r in b[phase]]:
            return dict(phase=phase,field='sampled_steps')
        for left,right in zip(a[phase],b[phase]):
            for field in STAGES:
                found=first_difference(left[field],right[field])
                if found is not None:return dict(phase=phase,step=left['step'],field=field,detail=found)
    return None


def main():
    b=Path(sys.argv[1]).resolve();c=json.loads((b/'config.json').read_text());runs=[]
    for ctx in c['execution_diagnostic']['contexts']:
        path=b/'contexts'/ctx['label']/'result.json'
        if path.exists():runs.append(json.loads(path.read_text()))
    comparisons=[dict(left=a['context']['label'],right=z['context']['label'],first_difference=compare(a,z))
                 for a,z in itertools.combinations(runs,2) if a['status']==z['status']=='completed']
    def all_finite(value):
        if isinstance(value,dict):return value.get('finite',True) and all(all_finite(v) for v in value.values())
        if isinstance(value,list):return all(all_finite(v) for v in value)
        return True
    complete=len(runs)==8 and all(r['status']=='completed' for r in runs)
    summary=dict(completed=complete,processes=len(runs),measured_updates=sum(25 for r in runs if r['status']=='completed'),
        all_sampled_trajectories_exact=complete and all(x['first_difference'] is None for x in comparisons),
        finite=all(all_finite(r['measured']) and all_finite(r['smoke_fingerprints']) for r in runs),
        own_release_and_affinity=complete and all(r['owned_optimizer_released'] and len(r['affinity_restored'])==2 for r in runs),
        comparisons=comparisons,
        code_hashes={r['context']['label']:[x['source_sha256'] for x in r.get('code_after_measured',[])] for r in runs},
        kernel_names={r['context']['label']:r.get('runtime_profile',{}).get('kernel_names_counts',{}) for r in runs})
    (b/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['comparisons','code_hashes','kernel_names']},indent=2))


if __name__=='__main__':main()
