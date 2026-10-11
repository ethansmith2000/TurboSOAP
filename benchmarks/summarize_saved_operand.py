"""Separate primary repetition, saved capture validity and causal interpretation."""
import itertools
import json
from pathlib import Path
import sys
try:
    from .summarize_cudnn_deterministic import qualify as primary_qualify
    from .summarize_execution_diagnostic import compare
    from .saved_operand_contract import validate_config,backend_delta,finite_records
    from .saved_operand_replay import interpretation
    from .execution_fingerprints import first_difference
except ImportError:
    from summarize_cudnn_deterministic import qualify as primary_qualify
    from summarize_execution_diagnostic import compare
    from saved_operand_contract import validate_config,backend_delta,finite_records
    from saved_operand_replay import interpretation
    from execution_fingerprints import first_difference


def qualify(runs):
    primary=primary_qualify(runs)
    if not primary['completed']:return dict(completed=False,primary=primary,observer_valid=False,interpretation_supported=False)
    rows=[]
    for r in runs:
        f=r.get('saved_operand_replay',{});captures=f.get('captures',[]);heads=f.get('heads',[])
        complete=f.get('status')=='completed' and len(captures)==3 and len(heads) in [0,3]
        valid=complete and f['final_invariants_before']==f['final_invariants_after'] and f['held_invariants_before']==f['held_invariants_after'] and f['rebuilt_input']==r['measured'][0]['input'] and f['rebuilt_target']==r['measured'][0]['target'] and f['retained_tensor_bytes']<128*2**20 and finite_records(captures) and finite_records(heads) and bool(f.get('capture_kernel_names_counts'))
        interp=None
        if complete:
            a=r['measured'][0];actual=dict(output=a['output'],loss=a['loss'],grad=a['grad_before_clip'])
            interp=interpretation(actual,r['initial_state_replay']['plain'],captures,heads)
            valid=valid and interp==f['interpretation']
            if f['held_operand'] is not None:
                valid=valid and len(heads)==3 and f['held_operand']==captures[0]['operand'] and bool(f.get('head_kernel_names_counts'))
                abi=f.get('abi',{})
                for capture in captures:
                    selection=capture.get('selection') or {};metadata=selection.get('saved_metadata',[])
                    candidates=[row['index'] for row in metadata if row['dtype']==abi.get('dtype') and row['shape']==abi.get('shape') and row['stride']==abi.get('stride') and row['storage_offset']==0]
                    valid=valid and selection.get('node_type')=='CompiledFunctionBackward' and selection.get('selected_index')==abi.get('saved_index') and len(metadata)==abi.get('saved_count') and selection.get('version_checked_count',0)+selection.get('no_version_check_count',0)==len(metadata) and candidates==[abi.get('saved_index')]
            else:valid=valid and len(heads)==0 and all(c['operand'] is None and c['selection_failure'] for c in captures)
        rows.append(dict(label=r['context']['label'],valid=bool(valid),interpretation=interp))
    valid=all(v for k,v in primary['checks'].items() if k!='sampled_trajectories_exact') and all(r['valid'] for r in rows)
    pairs=[]
    for a,b in itertools.combinations(runs,2):
        fa=a['saved_operand_replay'];fb=b['saved_operand_replay']
        if fa.get('status')==fb.get('status')=='completed':
            pairs.append(dict(left=a['context']['label'],right=b['context']['label'],primary_difference=compare(a,b),
                original_operand_exact=fa['captures'][0]['operand']==fb['captures'][0]['operand'],
                original_replay_difference=first_difference(fa['captures'][0]['execution'],fb['captures'][0]['execution']),
                held_head_difference=first_difference(fa['heads'][0],fb['heads'][0]) if fa['heads'] and fb['heads'] else 'unavailable',
                observer_code_exact=a['code_after_saved_operand_replay']==b['code_after_saved_operand_replay'],
                capture_kernel_names_counts_exact=fa['capture_kernel_names_counts']==fb['capture_kernel_names_counts'],
                head_kernel_names_counts_exact=fa.get('head_kernel_names_counts')==fb.get('head_kernel_names_counts')))
    code_ok=len(pairs)==6 and all(p['observer_code_exact'] for p in pairs)
    return dict(completed=True,primary=primary,observer_valid=valid,observer_code_exact=code_ok,interpretation_supported=valid and code_ok and all(r['interpretation']['supported'] for r in rows),contexts=rows,saved_operand_pairs=pairs)


def main():
    b=Path(sys.argv[1]);c=json.loads((b/'config.json').read_text());validate_config(c)
    m=json.loads((b/'manifest.json').read_text());old=[json.loads(p.read_text()) for p in sorted((Path(m['baseline_bundle'])/'contexts').glob('*/result.json'))]
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
    print(json.dumps({k:v for k,v in result.items() if k not in ['primary','contexts','saved_operand_pairs','baseline_backend_deltas','baseline_comparisons']},indent=2))


if __name__=='__main__':main()
