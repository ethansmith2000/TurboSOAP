import copy
import json
from pathlib import Path
import pytest
from benchmarks.fresh420_contract import validate_config,canonical,check_record,audit_primary_math,numerical_complete,reconstruct_stream
from benchmarks.summarize_fresh420 import qualify,replay_valid

ROOT=Path('/workspace/optimizer_replay_results')

def config():
    old=json.loads((ROOT/'cifar_soap_wide_pilot_20261007/config.json').read_text())
    c=copy.deepcopy(old);c['max_wall_seconds']=1200
    c['fresh420']=dict(per_process_seconds=390,smoke_steps=25,measured_steps=420,fingerprint_steps=[1,2,20,21,25])
    return old,c

def test_scope_exact_and_no_old_config_mutation():
    old,c=config();before=copy.deepcopy(old)
    assert validate_config(c,old) and old==before

@pytest.mark.parametrize('key,value',[('lr',.001),('checkpoint',True),('shampoo_beta',.999),('width',2048),('max_wall_seconds',1400)])
def test_scope_rejects_extra_changes(key,value):
    old,c=config();c[key]=value
    with pytest.raises(ValueError):validate_config(c,old)

def test_comparison_excludes_only_declared_wall_fields():
    r=dict(step=21,diagnostic_seconds=1.,factor_count=8,maximum_gram_error=.001)
    changed={**r,'diagnostic_seconds':2.}
    assert canonical('guard',r)==canonical('guard',changed)
    changed['maximum_gram_error']=.002
    with pytest.raises(ValueError):check_record([canonical('guard',r)],0,canonical('guard',changed))
    with pytest.raises(ValueError):check_record([],0,canonical('guard',r))

def test_chunk_loss_and_probe_state_cannot_be_dropped():
    r=dict(step=10,samples=[],host={},loss=2.,gradient_norm=1.)
    assert canonical('chunk',r)==canonical('chunk',{**r,'host':{'pid':123}})
    assert canonical('chunk',r)!=canonical('chunk',{**r,'loss':2.1})
    with pytest.raises(ValueError):canonical('chunk',{**r,'unexpected_state':'x'})
    p=dict(step=21,diagnostic_seconds=1.,matrices=[{'world_m_preservation_error':.001}])
    q=copy.deepcopy(p);q['matrices'][0]['world_m_preservation_error']=.002
    assert canonical('probe',p)!=canonical('probe',q)

def test_primary_math_audit_and_unchanged_replay_regions():
    here=Path(__file__).resolve().parents[1]/'benchmarks'
    before=(here/'cifar_saved_operand_child.py').read_text();after=(here/'cifar_fresh420_child.py').read_text()
    assert audit_primary_math(before,after)
    with pytest.raises(ValueError):audit_primary_math(before,after.replace("dtype=torch.bfloat16","dtype=torch.float16"))
    a='# INITIAL_STATE_REPLAY_START';z='# INITIAL_STATE_REPLAY_END'
    assert before[before.index(a):before.index(z)]==after[after.index(a):after.index(z)]
    a='# SAVED_OPERAND_REPLAY_START'
    # Added code provenance follows the unchanged saved-operand call.
    z="report['code_after_saved_operand_replay']"
    assert before[before.index(a):before.index(z)]==after[after.index(a):after.index(z)]

def test_incomplete_or_missing_gates_never_qualify():
    r=dict(context={'label':'qr_0'},status='failed')
    assert not numerical_complete(r) and not replay_valid(r)
    assert not qualify([r])['qualified']
    assert not qualify([])['qualified']

def test_real_retained_replay_validity_and_corruption_rejected():
    p=next((ROOT/'cifar_soap_saved_operand_20261008/contexts').glob('*/result.json'))
    r=json.loads(p.read_text());assert replay_valid(r)
    r['saved_operand_replay']['captures'][0]['selection']['selected_index']+=1
    assert not replay_valid(r)

def retained_fixture():
    """CPU-only assembled evidence fixture; never an actual new-run qualification."""
    old=json.loads((ROOT/'cifar_soap_wide_pilot_20261007/screen.json').read_text())
    run=old['runs'][0]
    r=json.loads(next((ROOT/'cifar_soap_saved_operand_20261008/contexts').glob('*/result.json')).read_text())
    r.update(context={'label':'qr_0'},parameter_count=old['parameter_count'],matrix_inventory=old['matrix_inventory'],
        refresh_guards=run['refresh_guards'],tracking_probes=run['tracking_probes'],observations=run['observations'],
        timing_chunks=[dict(step=h['step'],loss=h['loss'],gradient_norm=h['maximum_gradient_norm'],samples=[],host={}) for h in run['history']],
        effective_groups=run['effective_groups'],own_affinity_restored=True,initial_rng_sha256='fixture')
    o=r['observations'][-1];r['final_state']=dict(model=o['model_sha256'],optimizer=o['optimizer_sha256'],rng=o['rng_sha256'])
    r['comparison_records']=reconstruct_stream(r)
    return r

def test_complete_qualification_and_semantic_raw_corruption():
    a=retained_fixture();assert numerical_complete(a)
    b=copy.deepcopy(a);b['context']['label']='fro_1'
    c=copy.deepcopy(a);c['context']['label']='qr_2'
    assert qualify([a,b,c])['qualified']
    c['observations'][0]['optimizer_sha256']='corrupted'
    assert not qualify([a,b,c])['qualified']

def test_physical_m_failure_cannot_be_hidden_by_matching_record_stream():
    r=retained_fixture();r['tracking_probes'][0]['matrices'][0]['world_m_preservation_error']=.02
    r['comparison_records']=reconstruct_stream(r)
    assert not numerical_complete(r)
