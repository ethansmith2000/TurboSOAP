from pathlib import Path
import sys,json
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import torch
from native_damped_kernel import PolicyRouter
from native_damped_diagnostic import build_arm,configure_diagnostic
from native_damped_contract import ARMS,POLICIES,protocol,recipe
from matched_training_runtime import OptimizerLease

def test_exact_anchor_compiler_path_and_matching_damped_eager_reference():
    from native_residual_named_kernel import PolicyKernel as Original
    from tracking_curvature_kernel import damped,prefix_damped
    entries=[]
    def compiler(fn,**kw):entries.append(fn);return fn
    router=PolicyRouter(lambda *a:None,lambda:'rng',compiler)
    assert len(entries)==3 and len(set(fn.__code__ for fn in entries))==3
    reference=Original('highest_residual20',router,lambda fn,**kw:fn)
    actual=router.kernels['highest_residual20']
    assert actual.eager_prefix is reference.eager_prefix and actual.eager_ret is reference.eager_ret
    assert actual.prefix is reference.prefix and actual.retraction is reference.retraction
    assert router.kernels['damped20'].eager_prefix is prefix_damped
    assert router.kernels['damped20'].whole is damped
    torch.manual_seed(158);q=torch.linalg.qr(torch.randn(16,16)).Q;h=torch.randn(16,8);c=h@h.T+torch.eye(16)
    assert torch.equal(actual.evaluate(c,q),reference.evaluate(c,q))
    assert torch.equal(router.kernels['damped20'].evaluate(c,q),router.kernels['damped20'].evaluate(c,q,eager=True))

def test_explicit_model_arms_routing_and_empty_immediate_release():
    from hybrid_training_contract import BASELINE
    from cifar_optimizer_benchmark import support
    c=recipe(json.loads((BASELINE/'config.json').read_text()))
    router=PolicyRouter(lambda *a:None,lambda:'rng',lambda fn,**kw:fn)
    with torch.device('meta'):model=support.build_model(c)
    for arm in ARMS:
        lease=OptimizerLease(model,build_arm(model,c,arm,router))
        assert lease.opt.matrix.param_groups[0]['basis_method']==('qr' if arm=='qr20' else 'warm')
        if arm in POLICIES:assert lease.opt.matrix._gauge_step_one is router.kernels[arm]
        assert lease.release()['passed']

def test_populated_all_arm_probe_history_resets_and_immediate_release(monkeypatch):
    from soap_reference import ResearchSOAP
    from movement_retraction import configure_matrix
    from cifar_optimizers import SplitOptimizer
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    for arm in ARMS:
        torch.manual_seed(158)
        model=torch.nn.Module();model.register_parameter('matrix',torch.nn.Parameter(torch.randn(12,8)))
        model.register_parameter('aux',torch.nn.Parameter(torch.randn(3)))
        router=PolicyRouter(lambda *a:None,lambda:'rng',lambda fn,**kw:fn)
        matrix=ResearchSOAP([model.matrix],basis_method='qr' if arm=='qr20' else 'warm',shampoo_beta=.99,
            precondition_frequency=20,hard_reset_interval=0 if arm=='qr20' else 200,transport_precision='highest')
        matrix.param_groups[0]['param_names']=['matrix']
        if arm in POLICIES:
            configure_matrix(matrix,'fro_cubic2_cap05');matrix.param_groups[0].update(basis_lr=.5,basis_jacobi_damping=.01)
            matrix._gauge_step_one=router.kernels[arm]
        configure_diagnostic(matrix,router,arm)
        lease=OptimizerLease(model,SplitOptimizer(matrix,torch.optim.AdamW([model.aux])))
        del matrix
        for call in range(1,421):
            model.matrix.grad=torch.randn_like(model.matrix);router.probe=call in (21,281)
            lease.opt.step()
        router.probe=False
        state=lease.opt.matrix.state[model.matrix]
        assert state['research_qr_refreshes']==(20 if arm=='qr20' else 2)
        assert state['research_warm_refreshes']==(0 if arm=='qr20' else 18)
        del state
        assert len(router.moment_probes)==2 and router.cf_calls==8
        assert len(router.probes)==(4 if arm in POLICIES else 0)
        for row in router.moment_probes:
            assert row['source_arm']==arm and row['nonmutation_pass'] and row['registered_release_pass'] and row['physical_m_pass']
        assert lease.release()['passed']

def test_cpu_compiler_layout_specializations_match_own_references_without_new_limit():
    torch._dynamo.reset();captured=[]
    def backend(gm,inputs):captured.append(gm);return gm.forward
    def compiler(fn,**kw):return torch.compile(fn,backend=backend,fullgraph=True,dynamic=False)
    router=PolicyRouter(lambda *a:None,lambda:'rng',compiler)
    limit=torch._dynamo.config.recompile_limit
    for n in (8,12,16):
        c=torch.diag(torch.linspace(1,2,n))+.01*torch.ones(n,n)
        for column in (False,True):
            q=torch.eye(n);q=q.T.contiguous().T if column else q
            for kernel in router.kernels.values():
                with torch.no_grad():assert torch.equal(kernel.evaluate(c,q),kernel.evaluate(c,q,eager=True))
    before=len(captured)
    for kernel in router.kernels.values():
        with torch.no_grad():kernel.evaluate(c,q)
    assert len(captured)==before and limit==torch._dynamo.config.recompile_limit
    from torch._dynamo.utils import counters
    assert not counters['graph_break'];torch._dynamo.reset()

def test_recipe_expected_counts_masks_and_gates_are_declared():
    p=protocol()
    assert p['arms']==list(ARMS) and set(p['precision']['policies'])==set(POLICIES)
    assert p['maximum_optimizer_updates']==2020 and p['empty_preflight_backward_calls']==3
    assert p['expected']['diagnostic_reference_calls']==32 and p['expected']['full_moment_transactions']==24
    assert p['expected']['counterfactual_basis_calls']==96
    assert p['expected']['hybrid_kernel_calls']==464
    assert p['full_moment_diagnostic']['source_arms']==list(ARMS)
    assert p['diagnostic_admission']==dict(normalized_gram=.005,absolute_gram_fro=.05,own_eager_basis=1e-4,physical_m=.01,
        policy='Both declared warm counterfactuals and QR geometry must pass; all failures retained, no relaxation/retry')
    assert p['evaluation_calls']==[] and not p['live_damped_policy']['extra_matrix_products']
