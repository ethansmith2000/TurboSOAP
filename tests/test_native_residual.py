from pathlib import Path
import gc
import json
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import pytest
import torch
from native_residual_contract import ARMS,POLICIES,protocol,recipe,validate
from native_residual_kernel import PolicyRouter,diagnose_state,build_arm
from matched_training_runtime import streaming_digest


def test_explicit_arm_counts_and_unchanged_recipe():
    from hybrid_training_contract import BASELINE
    base=json.loads((BASELINE/'config.json').read_text());c=recipe(base);p=protocol()
    assert ARMS==('qr20','current_hybrid20','highest_residual20')
    assert p['maximum_optimizer_updates']==2020 and p['expected']['hybrid_kernel_calls']==464
    assert p['expected']['full_moment_transactions']==16 and p['expected']['diagnostic_reference_calls']==32
    assert c['width']==2048 and c['heads']==32 and c['shampoo_beta']==base['shampoo_beta']
    for key in ('betas','matrix_lr','lr','weight_decay'):
        assert c[key]==base[key]
    assert p['artifacts']['checkpoints'] is False and p['artifacts']['tensor_files'] is False
    validate(c,p,base)
    with pytest.raises(ValueError):validate(c,{**p,'probe_calls':[21]},base)


def test_full_actual_m_v_preview_is_nonmutating_and_owned_outputs_release():
    from soap_reference import ResearchSOAP
    from movement_retraction import configure_matrix
    torch.manual_seed(150);parameter=torch.nn.Parameter(torch.randn(12,8))
    opt=ResearchSOAP([parameter],basis_method='warm',precondition_frequency=20,hard_reset_interval=200,transport_precision='highest')
    configure_matrix(opt,'fro_cubic2_cap05');group=opt.param_groups[0]
    group.update(basis_lr=.5,basis_jacobi_damping=.01,param_names=['matrix'])
    for _ in range(2):parameter.grad=torch.randn_like(parameter);opt.step()
    state=opt.state[parameter];before=streaming_digest(dict(state=state,p=parameter,grad=parameter.grad));rng=torch.get_rng_state().clone()
    router=PolicyRouter(lambda *a:None,lambda:'unchanged',compiler=lambda f,**k:f)
    owners=[]
    with torch.no_grad():row=diagnose_state(opt,state,group,router,'matrix',owners)
    assert streaming_digest(dict(state=state,p=parameter,grad=parameter.grad))==before and torch.equal(rng,torch.get_rng_state())
    assert row['full_actual_momentum'] and row['full_actual_diagonal_v'] and row['all_finite'] and row['physical_m_pass']
    assert set(row['previews'])=={'qr_counterfactual',*POLICIES}
    assert row['previews']['qr_counterfactual']['inherited_adaptive_relative_to_qr']==0
    assert len(row['factors'])==2 and all(v['eager_caps']['highest_residual20']['valid'] for v in row['factors'])
    assert router.calls=={} and all(k.calls=={} for k in router.kernels.values())
    gc.collect();assert owners and all(ref() is None for ref in owners)


def test_native_meta_coverage_and_hooks_keep_original_step_and_qr_paths():
    from hybrid_training_contract import BASELINE
    from cifar_optimizer_benchmark import support
    from soap_reference import ResearchSOAP
    c=recipe(json.loads((BASELINE/'config.json').read_text()))
    with torch.device('meta'):model=support.build_model(c)
    router=PolicyRouter(lambda *a:None,lambda:'unchanged',compiler=lambda f,**k:f)
    for arm in ARMS:
        opt=build_arm(model,c,arm,router)
        assert type(opt.matrix).step is ResearchSOAP.step
        groups=opt.matrix.param_groups
        assert sum(len(g['params']) for g in groups)==4
        sizes={}
        for g in groups:
            for p in g['params']:
                for n in p.shape:sizes[str(n)]=sizes.get(str(n),0)+1
            assert g['precondition_frequency']==20 and g['transport_precision']=='highest'
            assert g['variance_policy']=='permutation'
            if arm in POLICIES:
                assert g['basis_lr']==.5 and g['basis_jacobi_damping']==.01
                assert g['basis_ns_iterations']==2 and g['basis_rotation_cap']==.5
        assert sizes=={'2048':5,'6144':1,'8192':2}
        if arm in POLICIES:assert opt.matrix._gauge_step_one is router.kernels[arm]
        else:assert opt.matrix._research_refresh.__func__ is ResearchSOAP._research_refresh


def test_hook_precision_restores_calling_context_and_does_not_count_counterfactuals():
    from hybrid_training_kernel import validate_group
    router=PolicyRouter(lambda *a:None,lambda:'unchanged',compiler=lambda f,**k:f)
    q=torch.eye(16);c=q+.01*torch.ones_like(q);before=torch.get_float32_matmul_precision()
    for name in POLICIES:
        result=router.kernels[name].evaluate(c,q)
        assert torch.get_float32_matmul_precision()==before and result.dtype==torch.float32 and torch.isfinite(result).all()
    assert router.calls=={}
