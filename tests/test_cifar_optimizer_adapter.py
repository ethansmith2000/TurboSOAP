import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from benchmarks.cifar_optimizers import (apply_lr_multiplier,build_optimizer,
                                        grouped_parameters,set_policy,screen_arm_config)


def model():
    path=Path(__file__).resolve().parents[1]/'benchmarks/cifar_support/vit.py'
    spec=importlib.util.spec_from_file_location('cifar_test_vit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.ViT(img_size=32,patch_size=4,num_classes=100,embed_dim=12,depth=2,heads=3,qk_norm=False)


CONFIG=dict(lr=.0005,matrix_lr=.001,weight_decay=.05,betas=[.9,.999],
            shampoo_beta=.999,precondition_mode='all',covariance_compute_dtype='float32')


def test_vit_grouping_is_complete_and_preserves_decay_rules():
    net=model();groups=grouped_parameters(net,CONFIG)
    names=[n for g in groups for n in g['param_names']]
    assert len(names)==len(set(names))==len(list(net.parameters()))
    matrices=[n for g in groups if g['role']=='matrix' for n in g['param_names']]
    assert len(matrices)==8
    assert all(n.startswith('blocks.') and n.endswith('weight') for n in matrices)
    by_name={n:g for g in groups for n in g['param_names']}
    for n in ['head.weight','patch_embed.proj.weight']:
        assert by_name[n]['role']=='auxiliary' and by_name[n]['weight_decay']==.05
    for n in ['cls_token','pos_embed','norm.weight','blocks.0.attn.to_out.bias']:
        assert by_name[n]['role']=='auxiliary' and by_name[n]['weight_decay']==0


def test_lr_multipliers_and_branch_restore_keep_independent_group_rates():
    net=model();opt=build_optimizer(net,CONFIG)
    for _ in range(3):
        for p in net.parameters():p.grad=torch.randn_like(p)
        opt.step()
    state=copy.deepcopy(opt.state_dict())
    other=build_optimizer(net,CONFIG,'warm5');other.load_state_dict(state);set_policy(other,'warm5')
    apply_lr_multiplier(other,.2)
    for group in other.param_groups:
        assert group['lr']==(.001 if group['role']=='matrix' else .0005)*.2
    assert all(g['basis_method']=='warm' and g['precondition_frequency']==5 for g in other.matrix.param_groups)
    assert all(g['basis_method']=='qr' for g in opt.matrix.param_groups)


def test_qr_and_warm_share_the_same_cold_start_before_policy_intervention():
    a=model();b=copy.deepcopy(a)
    oa=build_optimizer(a,CONFIG,'qr10');ob=build_optimizer(b,CONFIG,'warm5')
    for _ in range(5):
        for p,q in zip(a.parameters(),b.parameters()):
            grad=torch.randn_like(p);p.grad=grad.clone();q.grad=grad.clone()
        oa.step();ob.step()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)


@pytest.mark.parametrize('covariance,beta2',[(.999,.999),(.99,.999),(.999,.99)])
def test_memory_arm_isolates_matrix_settings(covariance,beta2):
    original=copy.deepcopy(CONFIG)
    c=screen_arm_config(CONFIG,dict(matrix_lr=.0005,covariance_beta=covariance,matrix_beta2=beta2))
    opt=build_optimizer(model(),c)
    assert CONFIG==original
    for g in opt.matrix.param_groups:
        assert g['betas']==(.9,beta2) and g['shampoo_beta']==covariance
        assert g['peak_lr']==.0005
    for g in opt.auxiliary.param_groups:
        assert g['betas']==(.9,.999) and g['peak_lr']==.0005


def test_explicit_baseline_memory_preserves_updates_and_changed_matrix_beta_leaves_auxiliary_rule_intact():
    torch.manual_seed(17)
    nets=[model()]
    nets.extend([copy.deepcopy(nets[0]),copy.deepcopy(nets[0])])
    configs=[CONFIG,screen_arm_config(CONFIG,dict(matrix_lr=.001,matrix_beta2=.999,covariance_beta=.999)),
             screen_arm_config(CONFIG,dict(matrix_lr=.001,matrix_beta2=.99,covariance_beta=.999))]
    opts=[build_optimizer(net,c) for net,c in zip(nets,configs)]
    for _ in range(14):
        for params in zip(*(net.parameters() for net in nets)):
            grad=torch.randn_like(params[0])
            for p in params:p.grad=grad.clone()
        for opt in opts:opt.step()
    for p,q in zip(nets[0].parameters(),nets[1].parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)
    by_name=[dict(net.named_parameters()) for net in nets]
    for group in opts[0].auxiliary.param_groups:
        for name in group['param_names']:
            torch.testing.assert_close(by_name[0][name],by_name[2][name],atol=0,rtol=0)
    assert not torch.equal(next(iter(opts[0].matrix.state.values()))['exp_avg_sq'],
                           next(iter(opts[2].matrix.state.values()))['exp_avg_sq'])


def test_memory_arm_rejects_out_of_scope_beta():
    with pytest.raises(ValueError,match='.99'):
        screen_arm_config(CONFIG,dict(matrix_lr=.0005,matrix_beta2=.9999))
