"""Keep primary repetition and feature/head interpretation as separate gates."""
import itertools
import json
from pathlib import Path
import sys
try:
    from .summarize_cudnn_deterministic import qualify as primary_qualify
    from .summarize_execution_diagnostic import compare
    from .feature_head_contract import validate_config,backend_delta,finite_records
    from .execution_fingerprints import first_difference
    from .feature_head_replay import interpretation
except ImportError:
    from summarize_cudnn_deterministic import qualify as primary_qualify
    from summarize_execution_diagnostic import compare
    from feature_head_contract import validate_config,backend_delta,finite_records
    from execution_fingerprints import first_difference
    from feature_head_replay import interpretation


def qualify(runs):
    primary=primary_qualify(runs)
    if not primary['completed']:return dict(completed=False,primary=primary,observer_valid=False,interpretation_supported=False)
    rows=[]
    for r in runs:
        f=r.get('feature_head_replay',{});features=f.get('features',[]);heads=f.get('heads',[])
        complete=f.get('status')=='completed' and len(features)==3 and len(heads)==3
        exact_input=f.get('rebuilt_input')==r['measured'][0]['input'] and f.get('rebuilt_target')==r['measured'][0]['target']
        restored=complete and f['final_invariants_before']==f['final_invariants_after'] and f['held_invariants_before']==f['held_invariants_after']
        memory=complete and f['retained_tensor_bytes']<128*2**20
        valid=complete and exact_input and restored and memory and finite_records(features) and finite_records(heads) and bool(f.get('feature_kernel_names_counts')) and bool(f.get('head_kernel_names_counts'))
        interp=None
        if complete:
            a=r['measured'][0];actual=dict(output=a['output'],loss=a['loss'],grad=a['grad_before_clip'])
            interp=interpretation(actual,r['initial_state_replay']['plain'],features,heads)
            valid=valid and interp==f['interpretation'] and f['held_operand']==features[0]['head_operand']
        rows.append(dict(label=r['context']['label'],valid=bool(valid),interpretation=interp))
    execution_ok=all(v for k,v in primary['checks'].items() if k!='sampled_trajectories_exact')
    valid=execution_ok and all(r['valid'] for r in rows)
    pairs=[]
    for a,b in itertools.combinations(runs,2):
        fa=a['feature_head_replay'];fb=b['feature_head_replay']
        if fa.get('status')==fb.get('status')=='completed':
            pairs.append(dict(left=a['context']['label'],right=b['context']['label'],primary_difference=compare(a,b),
                cls_features_exact=fa['features'][0]['cls_features']==fb['features'][0]['cls_features'],
                head_operand_exact=fa['features'][0]['head_operand']==fb['features'][0]['head_operand'],
                held_head_difference=first_difference(fa['heads'][0],fb['heads'][0]),
                feature_code_exact=a['code_after_feature_head_replay']==b['code_after_feature_head_replay'],
                feature_kernel_names_counts_exact=fa['feature_kernel_names_counts']==fb['feature_kernel_names_counts'],
                head_kernel_names_counts_exact=fa['head_kernel_names_counts']==fb['head_kernel_names_counts']))
    code_ok=len(pairs)==6 and all(p['feature_code_exact'] for p in pairs)
    return dict(completed=True,primary=primary,observer_valid=valid,observer_code_exact=code_ok,
                interpretation_supported=valid and code_ok and all(r['interpretation']['supported'] for r in rows),
                contexts=rows,feature_head_pairs=pairs)


def main():
    b=Path(sys.argv[1]);c=json.loads((b/'config.json').read_text());validate_config(c)
    m=json.loads((b/'manifest.json').read_text());baseline=Path(m['baseline_bundle'])
    old=[json.loads(p.read_text()) for p in sorted((baseline/'contexts').glob('*/result.json'))]
    runs=[]
    for ctx in c['execution_diagnostic']['contexts']:
        p=b/'contexts'/ctx['label']/'result.json'
        if p.exists():runs.append(json.loads(p.read_text()))
    result=qualify(runs)
    delta=[dict(label=r['context']['label'],phase=phase,fields=backend_delta(old[0][phase],r[phase])) for r in runs for phase in ['backend_before_smoke','backend_after_measured'] if phase in r]
    exact=bool(delta) and all(x['fields']==m['declared_baseline_backend_deltas'] for x in delta)
    result.update(baseline_backend_deltas=delta,baseline_backend_exact=exact)
    result['interpretation_supported']=result['interpretation_supported'] and exact
    result['baseline_comparisons']=[dict(control=r['context']['label'],baseline=a['context']['label'],first_sampled_difference=compare(a,r)) for r in runs for a in old if a['status']==r['status']=='completed']
    (b/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['primary','feature_head_pairs','baseline_backend_deltas','baseline_comparisons']},indent=2))


if __name__=='__main__':main()
