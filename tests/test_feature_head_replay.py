import copy
import json
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from benchmarks.cifar_support.vit import ViT
from benchmarks.execution_fingerprints import gradient_records,tensor_record
from benchmarks.feature_head_replay import FeatureReturn,HeldHead,interpretation,head_execution
from benchmarks.feature_head_contract import validate_child_change,validate_config

BASE=Path('/workspace/optimizer_replay_results/cifar_soap_cudnn_deterministic_20261008')
ROOT=Path(__file__).resolve().parents[1]


def evaluate(model,call,x,target):
    model.zero_grad(set_to_none=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out=call(x);logits=out[0] if isinstance(out,tuple) else out
        loss=F.cross_entropy(logits.float(),target)
    loss.backward()
    return dict(output=tensor_record(logits),loss=float(loss.detach()),grad=gradient_records(model)),out


@pytest.mark.parametrize('zero_head',[True,False])
def test_ordered_forward_operand_and_gradients_match_independent_original(zero_head):
    torch.manual_seed(32)
    model=ViT(img_size=8,patch_size=4,embed_dim=12,depth=1,heads=3,num_classes=5)
    if not zero_head:
        with torch.no_grad():model.head.weight.normal_(0,.02)
    reference=copy.deepcopy(model);x=torch.randn(3,3,8,8);target=F.one_hot(torch.tensor([0,2,4]),5).float()
    original,_=evaluate(reference,reference,x,target)
    observed,outputs=evaluate(model,FeatureReturn(model),x,target)
    assert observed==original
    logits,cls,operand=outputs
    assert operand.dtype==torch.bfloat16 and torch.equal(operand,cls.to(torch.bfloat16))
    assert model.head is FeatureReturn(model).base.head
    isolated,_=evaluate(model,HeldHead(model.head),operand.detach().clone(),target)
    assert head_execution(isolated)==head_execution(observed)


def test_interpretation_rejects_feature_or_head_gradient_drift_separately():
    e=dict(output={'hash':'a'},loss=1.,grad={'head.weight':{'hash':'w'},'head.bias':{'hash':'b'},'upstream':{'hash':'u'}})
    f=dict(execution=e,cls_features={'hash':'f'},head_operand={'hash':'bf'})
    fs=[copy.deepcopy(f) for _ in range(3)];hs=[copy.deepcopy(head_execution(e)) for _ in range(3)]
    assert interpretation(e,e,fs,hs)['supported']
    fs[1]['execution']['grad']['head.weight']['hash']='changed'
    assert not interpretation(e,e,fs,hs)['checks']['primary_gradients_exact']
    fs[1]=copy.deepcopy(f);fs[2]['head_operand']['hash']='changed'
    assert not interpretation(e,e,fs,hs)['checks']['feature_calls_exact']
    fs[2]=copy.deepcopy(f);hs[1]['grad']['head.weight']['hash']='changed'
    assert not interpretation(e,e,fs,hs)['checks']['held_head_matches']
    assert not interpretation(e,e,fs,hs)['checks']['held_head_calls_exact']


def test_primary_source_and_prior_observers_are_unchanged():
    before=(BASE/'source/benchmarks/cifar_cudnn_deterministic_child.py').read_text()
    after=(ROOT/'benchmarks/cifar_feature_head_child.py').read_text()
    assert validate_child_change(before,after)
    with pytest.raises(ValueError):validate_child_change(before,after.replace('cudnn.deterministic=True','cudnn.deterministic=False'))
    with pytest.raises(ValueError):validate_child_change(before,after.replace('support.lr_at(index+1','support.lr_at(index+2'))
    with pytest.raises(ValueError):validate_child_change(before,after.replace("warm_record=report['initial_state_replay']['plain']","warm_record=report['initial_state_replay']['profiled']").replace('run_initial_replay(model=model,active=active','run_initial_replay(model=model,active=model'))


def test_new_limits_do_not_allow_extra_primary_work():
    c=json.loads((BASE/'config.json').read_text());c['max_wall_seconds']=900;c['execution_diagnostic']['per_process_seconds']=210
    assert validate_config(c)
    for mutate in [lambda c:c.update(max_wall_seconds=1000),lambda c:c['execution_diagnostic'].update(per_process_seconds=240),lambda c:c['execution_diagnostic'].update(measured_steps=100)]:
        bad=copy.deepcopy(c);mutate(bad)
        with pytest.raises(ValueError):validate_config(bad)


def test_qualification_keeps_inconclusive_features_and_requires_restoration():
    from benchmarks.summarize_feature_head import qualify
    r=json.loads((BASE/'contexts/plan420_flush25_r0_cudnn_deterministic/result.json').read_text())
    runs=[copy.deepcopy(r) for _ in range(4)]
    a=r['measured'][0];e=dict(output=a['output'],loss=a['loss'],grad=a['grad_before_clip'])
    feature=dict(execution=e,cls_features={'sha256':'a'},head_operand={'sha256':'b'})
    for i,run in enumerate(runs):
        run['context']['label']=f'new{i}'
        features=[copy.deepcopy(feature) for _ in range(3)];heads=[copy.deepcopy(head_execution(e)) for _ in range(3)]
        run['feature_head_replay']=dict(status='completed',features=features,heads=heads,rebuilt_input=a['input'],rebuilt_target=a['target'],final_invariants_before={'state':'a'},final_invariants_after={'state':'a'},held_invariants_before={'state':'b'},held_invariants_after={'state':'b'},retained_tensor_bytes=120*2**20,feature_kernel_names_counts={'k':1},head_kernel_names_counts={'h':1},held_operand=feature['head_operand'],interpretation=interpretation(e,e,features,heads))
        run['code_after_feature_head_replay']=[]
    assert qualify(runs)['interpretation_supported']
    assert not qualify(runs[:3])['observer_valid']
    bad=copy.deepcopy(runs);bad[0]['feature_head_replay']['final_invariants_after']['state']='changed'
    assert not qualify(bad)['observer_valid']
    bad=copy.deepcopy(runs);f=bad[0]['feature_head_replay'];f['features'][0]['execution']['grad']['head.weight']['sha256']='changed'
    f['interpretation']=interpretation(e,e,f['features'],f['heads'])
    result=qualify(bad);assert result['observer_valid'] and not result['interpretation_supported']
