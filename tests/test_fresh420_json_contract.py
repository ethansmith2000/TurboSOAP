import ast
import copy
import json
from pathlib import Path
import pytest
import torch
from benchmarks.fresh420_contract import canonical as old_canonical,check_record
from benchmarks.fresh420_json_contract import canonical
from benchmarks.cifar_optimizers import build_optimizer,screen_arm_config

ROOT=Path('/workspace/optimizer_replay_results/cifar_soap_fresh420_gpu3_20261008')

def test_actual_optimizer_betas_match_the_persisted_reference():
    model=torch.nn.Module();model.blocks=torch.nn.ModuleList([torch.nn.Linear(2,2)]);model.head=torch.nn.Linear(2,2)
    config=json.loads((ROOT/'config.json').read_text())
    opt=build_optimizer(model,screen_arm_config(config,config['memory_screen']['arms'][0]),'qr20')
    betas=opt.matrix.param_groups[0]['betas'];assert isinstance(betas,tuple)
    live=dict(effective_groups=[dict(betas=betas)])
    reference=json.loads(json.dumps(old_canonical('initial',live)))
    with pytest.raises(ValueError,match='betas'):check_record([reference],0,old_canonical('initial',live))
    check_record([reference],0,canonical('initial',live))
    with pytest.raises(ValueError):check_record([reference],0,canonical('initial',dict(effective_groups=[dict(betas=(.9,.99))])))

def test_retained_failed_record_is_representation_failure_and_remains_failed():
    left=json.loads((ROOT/'contexts/qr_0/result.json').read_text())
    right=json.loads((ROOT/'contexts/qr_2/result.json').read_text())
    assert right['status']=='failed' and not right['measured']
    assert left['comparison_records'][:7]==right['comparison_records']
    value=copy.deepcopy(right['comparison_records'][6]['value'])
    for group in value['effective_groups']:group['betas']=tuple(group['betas'])
    fixed=canonical('initial',value);check_record([left['comparison_records'][6]],0,fixed)
    assert right['status']=='failed'

@pytest.mark.parametrize('kind,value',[
    ('fingerprint',{'grad_before_clip':{'head.weight':{'sha256':'changed'}}}),
    ('probe',{'step':21,'diagnostic_seconds':0.,'matrices':[{'world_m_preservation_error':.02}]}),
    ('guard',{'step':21,'maximum_gram_error':.06}),
    ('initial',{'model':'changed'}),
])
def test_wire_normalization_keeps_every_selected_semantic_field(kind,value):
    assert canonical(kind,value)==json.loads(json.dumps(old_canonical(kind,value)))
    reference=canonical(kind,{'different':True})
    with pytest.raises(ValueError):check_record([reference],0,canonical(kind,value))

@pytest.mark.parametrize('value',[float('nan'),float('inf'),object()])
def test_nonfinite_and_unknown_objects_fail_closed(value):
    with pytest.raises((TypeError,ValueError)):canonical('initial',{'value':value})

def test_new_child_only_changes_canonical_import_and_outer_only_changes_child_name():
    here=Path(__file__).resolve().parents[1]/'benchmarks'
    old=(ROOT/'source/benchmarks/cifar_fresh420_child.py').read_text()
    new=(here/'cifar_fresh420_json_child.py').read_text()
    restored=new.replace('from fresh420_contract import check_record\nfrom fresh420_json_contract import canonical',
        'from fresh420_contract import canonical,check_record')
    assert ast.dump(ast.parse(restored))==ast.dump(ast.parse(old))
    outer=(here/'cifar_fresh420_json.py').read_text().replace('cifar_fresh420_json_child.py','cifar_fresh420_child.py')
    assert outer==(ROOT/'source/benchmarks/cifar_fresh420.py').read_text()
