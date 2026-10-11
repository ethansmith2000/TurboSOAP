import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from benchmarks.saved_operand_abi import classifier_abi,select_operand,OperandUnavailable
from benchmarks.saved_operand_replay import interpretation,head_execution
from benchmarks.saved_operand_contract import validate_child_change,validate_config

BASE=Path('/workspace/optimizer_replay_results/cifar_soap_cudnn_deterministic_20261008')
ROOT=Path(__file__).resolve().parents[1]


def test_actual_compiled_source_maps_operand_without_running_old_gpu():
    r=json.loads((BASE/'contexts/plan420_flush25_r0_cudnn_deterministic/result.json').read_text())
    abi=classifier_abi(r['code_after_measured'],256,1536,100)
    assert abi['saved_index']==22 and abi['saved_count']==30
    assert abi['forward_operand']=='buf36' and abi['backward_operand']=='convert_element_type_27'
    with pytest.raises(OperandUnavailable):classifier_abi(r['code_after_measured'],128,1536,100)
    bad=copy.deepcopy(r['code_after_measured']);bad[0]['source_sha256']='changed'
    with pytest.raises(OperandUnavailable,match='hash'):classifier_abi(bad,256,1536,100)


def test_real_cpu_aot_access_preserves_backward_and_parameter_gradients():
    torch.manual_seed(91);model=torch.nn.Sequential(torch.nn.LayerNorm(12),torch.nn.Linear(12,4));reference=copy.deepcopy(model)
    active=torch.compile(model,backend='aot_eager',fullgraph=True,dynamic=False);x=torch.randn(3,12)
    with torch.autocast('cpu',dtype=torch.bfloat16):out=active(x);plain=reference(x)
    node=out.grad_fn;tensors=list(node.saved_tensors)+list(getattr(node,'_tensors_no_vc_check',[]))
    index=next(i for i,t in enumerate(tensors) if t.dtype==torch.bfloat16 and list(t.shape)==[3,12])
    abi=dict(saved_count=len(tensors),saved_index=index,shape=[3,12],stride=[12,1])
    operand,metadata=select_operand(node,abi)
    out.float().sum().backward();plain.float().sum().backward()
    assert all(torch.equal(a.grad,b.grad) for a,b in zip(model.parameters(),reference.parameters()))
    assert torch.isfinite(operand).all() and operand.shape==(3,12)
    assert metadata['selected_index']==index


def test_both_channels_and_ambiguous_missing_or_wrong_index_rejected():
    node=type('CompiledFunctionBackward',(),{})();x=torch.ones(3,12,dtype=torch.bfloat16)
    node.saved_tensors=(torch.zeros(12),);node._tensors_no_vc_check=[x]
    abi=dict(saved_count=2,saved_index=1,shape=[3,12],stride=[12,1])
    operand,m=select_operand(node,abi);assert operand is x and m['no_version_check_count']==1
    bad=dict(abi,saved_index=0)
    with pytest.raises(OperandUnavailable,match='index'):select_operand(node,bad)
    node.saved_tensors=(x,)
    with pytest.raises(OperandUnavailable,match='found 2'):select_operand(node,abi)
    with pytest.raises(OperandUnavailable):select_operand(None,abi)
    node.saved_tensors=();node._tensors_no_vc_check=[]
    with pytest.raises(OperandUnavailable,match='count'):select_operand(node,abi)


def test_interpretation_requires_actual_and_warm_parity_and_available_operand():
    e=dict(output={'hash':'l'},loss=1.,grad={'head.weight':{'hash':'w'},'head.bias':{'hash':'b'}})
    cs=[dict(execution=copy.deepcopy(e),operand={'hash':'f'},selection={},selection_failure=None) for _ in range(3)]
    hs=[copy.deepcopy(head_execution(e)) for _ in range(3)]
    assert interpretation(e,e,cs,hs)['supported']
    cs[1]['execution']['grad']['head.weight']['hash']='changed'
    assert not interpretation(e,e,cs,hs)['supported']
    cs[1]['execution']=copy.deepcopy(e);cs[0]['operand']=None
    assert not interpretation(e,e,cs,hs)['checks']['operand_available']
    assert not interpretation(e,e,cs,[])['supported']


def test_primary_source_and_preceding_replays_remain_exact():
    before=(BASE/'source/benchmarks/cifar_cudnn_deterministic_child.py').read_text();after=(ROOT/'benchmarks/cifar_saved_operand_child.py').read_text()
    assert validate_child_change(before,after)
    with pytest.raises(ValueError):validate_child_change(before,after.replace('cudnn.deterministic=True','cudnn.deterministic=False'))
    with pytest.raises(ValueError):validate_child_change(before,after.replace('support.lr_at(index+1','support.lr_at(index+2'))
    c=json.loads((BASE/'config.json').read_text());c['max_wall_seconds']=900;c['execution_diagnostic']['per_process_seconds']=210
    assert validate_config(c)
    c['execution_diagnostic']['measured_steps']=100
    with pytest.raises(ValueError):validate_config(c)


def test_result_validation_rejects_wrong_selection_and_preserves_inconclusive_capture():
    from benchmarks.summarize_saved_operand import qualify
    r=json.loads((BASE/'contexts/plan420_flush25_r0_cudnn_deterministic/result.json').read_text())
    runs=[copy.deepcopy(r) for _ in range(4)];a=r['measured'][0];e=dict(output=a['output'],loss=a['loss'],grad=a['grad_before_clip'])
    abi=dict(dtype='torch.bfloat16',shape=[256,1536],stride=[1536,1],saved_index=0,saved_count=1)
    selection=dict(node_type='CompiledFunctionBackward',selected_index=0,version_checked_count=1,no_version_check_count=0,saved_metadata=[dict(index=0,dtype='torch.bfloat16',shape=[256,1536],stride=[1536,1],storage_offset=0)])
    for i,run in enumerate(runs):
        run['context']['label']=f'new{i}'
        cs=[dict(execution=copy.deepcopy(e),operand={'sha256':'operand'},selection=copy.deepcopy(selection),selection_failure=None) for _ in range(3)];hs=[copy.deepcopy(head_execution(e)) for _ in range(3)]
        run['saved_operand_replay']=dict(status='completed',captures=cs,heads=hs,abi=abi,held_operand=cs[0]['operand'],rebuilt_input=a['input'],rebuilt_target=a['target'],final_invariants_before={'state':'a'},final_invariants_after={'state':'a'},held_invariants_before={'state':'b'},held_invariants_after={'state':'b'},retained_tensor_bytes=120*2**20,capture_kernel_names_counts={'original':1},head_kernel_names_counts={'head':1},interpretation=interpretation(e,e,cs,hs))
        run['code_after_saved_operand_replay']=[]
    assert qualify(runs)['interpretation_supported']
    assert not qualify(runs[:3])['observer_valid']
    bad=copy.deepcopy(runs);bad[0]['saved_operand_replay']['captures'][0]['selection']['selected_index']=5
    assert not qualify(bad)['observer_valid']
    bad=copy.deepcopy(runs);bad[0]['saved_operand_replay']['final_invariants_after']['state']='changed'
    assert not qualify(bad)['observer_valid']
    bad=copy.deepcopy(runs)
    for run in bad:
        f=run['saved_operand_replay'];f['heads']=[];f['held_operand']=None
        for c in f['captures']:c.update(operand=None,selection=None,selection_failure='access unavailable')
        f['interpretation']=interpretation(e,e,f['captures'],[])
    result=qualify(bad);assert result['observer_valid'] and not result['interpretation_supported']
