"""Fail-closed new pilot qualification; old failures are outside this scope."""
import hashlib
import json
from pathlib import Path
import sys
try:
    from .fresh420_contract import validate_config,numerical_complete
    from .benchmark_off_contract import backend_delta,finite_records
    from .saved_operand_replay import interpretation
except ImportError:
    from fresh420_contract import validate_config,numerical_complete
    from benchmark_off_contract import backend_delta,finite_records
    from saved_operand_replay import interpretation

def replay_valid(r):
    try:
        first=r['measured'][0];warm=r['initial_state_replay'];s=r['saved_operand_replay']
        for v in [warm,s]:
            assert v['final_invariants_before']==v['final_invariants_after']
            assert v['held_invariants_before']==v['held_invariants_after']
            assert v['rebuilt_input']==first['input'] and v['rebuilt_target']==first['target']
            assert v['retained_tensor_bytes']<128*2**20
        assert s['status']=='completed' and len(s['captures'])==len(s['heads'])==3
        actual=dict(output=first['output'],loss=first['loss'],grad=first['grad_before_clip'])
        i=interpretation(actual,warm['plain'],s['captures'],s['heads'])
        assert i==s['interpretation'] and i['supported']
        assert s['held_operand']==s['captures'][0]['operand']
        assert s['capture_kernel_names_counts'] and s['head_kernel_names_counts']
        abi=s['abi']
        for row in s['captures']:
            sel=row['selection'];meta=sel['saved_metadata']
            assert sel['node_type']=='CompiledFunctionBackward'
            assert sel['selected_index']==abi['saved_index'] and len(meta)==abi['saved_count']
            assert sel['version_checked_count']+sel['no_version_check_count']==len(meta)
            candidates=[x['index'] for x in meta if x['dtype']==abi['dtype'] and x['shape']==abi['shape'] and x['stride']==abi['stride'] and x['storage_offset']==0]
            assert candidates==[abi['saved_index']]
        assert finite_records(s['captures']) and finite_records(s['heads'])
        return True
    except (AssertionError,KeyError,TypeError):return False

def qualify(runs):
    completed=len(runs)==3 and all(r.get('status')=='completed' for r in runs)
    labels=[r['context']['label'] for r in runs]
    checks=dict(completed=completed,correct_order=labels==['qr_0','fro_1','qr_2'],
        numerical=completed and all(numerical_complete(r) for r in runs),
        replay=completed and all(replay_valid(r) for r in runs),
        qr_repeat_exact=completed and runs[0]['comparison_records']==runs[2]['comparison_records'])
    return dict(qualified=all(checks.values()),checks=checks,
        wall_qualification_collected=False,full_strict_reference_collected=False,
        scope='Fresh-process actual1536 fixed420 sampled QR repetition/numerics; early LR warmup')

def summarize(b):
    m=json.loads((b/'manifest.json').read_text());c=json.loads((b/'config.json').read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    for n,h in m['source_sha256'].items():assert sha(b/'source'/n)==h,n
    for n,k in [('config.json','config_sha256'),('split.json','split_sha256'),('data_prepared.json','base_prepared_sha256'),('PROTOCOL.md','protocol_sha256')]:assert sha(b/n)==m[k],n
    baseline=Path(m['baseline_bundle']);assert sha(baseline/'manifest.json')==m['baseline_manifest_sha256']
    original=Path(m['original_config_bundle']);assert sha(original/'config.json')==m['original_config_sha256']
    validate_config(c,json.loads((original/'config.json').read_text()))
    runs=[json.loads(p.read_text()) for arm in c['memory_screen']['arms'] if (p:=b/'contexts'/arm['label']/'result.json').exists()]
    s=qualify(runs);old=json.loads(next((baseline/'contexts').glob('*/result.json')).read_text())
    deltas=[dict(label=r['context']['label'],phase=k,fields=backend_delta(old[k],r[k])) for r in runs for k in ['backend_before_smoke','backend_after_measured'] if k in r]
    s['baseline_backend_deltas']=deltas
    s['checks']['baseline_backend']=len(deltas)==6 and all(x['fields']==m['declared_baseline_backend_deltas'] for x in deltas)
    s['qualified']=all(s['checks'].values())
    s['rows']=[dict(label=r['context']['label'],status=r['status'],failure=r.get('failure'),
        observations=[dict(step=o['step'],ce=o['validation']['ce']) for o in r['observations']],
        training_seconds=r.get('training_seconds'),optimizer_ms=r.get('optimizer_ms'),
        all_in_seconds=r.get('all_in_seconds'),peak_allocated_bytes=r.get('peak_allocated_bytes'),
        guards=len(r['refresh_guards']),matrix_probes=sum(len(x['matrices']) for x in r['tracking_probes']),
        comparison_records=len(r['comparison_records'])) for r in runs]
    s['execution_status']=json.loads((b/'execution_status.json').read_text())
    (b/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2));return s

if __name__=='__main__':summarize(Path(sys.argv[1]))
