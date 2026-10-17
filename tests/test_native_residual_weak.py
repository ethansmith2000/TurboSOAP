from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import torch
from native_residual_weak_kernel import PolicyRouter,configure_diagnostics,build_arm
from native_residual_weak_contract import ARMS,recipe,protocol
from matched_training_runtime import OptimizerLease

def test_empty_native_policy_optimizers_release_immediately_without_gc():
    import json
    from hybrid_training_contract import BASELINE
    from cifar_optimizer_benchmark import support
    c=recipe(json.loads((BASELINE/'config.json').read_text()))
    with torch.device('meta'):model=support.build_model(c)
    router=PolicyRouter(lambda *a:None,lambda:'rng',compiler=lambda f,**k:f)
    for arm in ARMS:
        lease=OptimizerLease(model,build_arm(model,c,arm,router))
        report=lease.release()
        assert report['optimizer_objects_released'] and report['passed']

def test_populated_probe_original_refresh_and_immediate_release(monkeypatch):
    from soap_reference import ResearchSOAP
    from movement_retraction import configure_matrix
    from cifar_optimizers import SplitOptimizer
    torch.manual_seed(151)
    model=torch.nn.Module();model.register_parameter('matrix',torch.nn.Parameter(torch.randn(12,8)))
    model.register_parameter('aux',torch.nn.Parameter(torch.randn(3)))
    router=PolicyRouter(lambda *a:None,lambda:'rng',compiler=lambda f,**k:f)
    matrix=ResearchSOAP([model.matrix],basis_method='warm',precondition_frequency=20,hard_reset_interval=200,transport_precision='highest')
    configure_matrix(matrix,'fro_cubic2_cap05')
    matrix.param_groups[0].update(param_names=['matrix'],basis_lr=.5,basis_jacobi_damping=.01)
    matrix._gauge_step_one=router.kernels['highest_residual20']
    configure_diagnostics(matrix,router,'highest_residual20')
    lease=OptimizerLease(model,SplitOptimizer(matrix,torch.optim.AdamW([model.aux])))
    del matrix
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    for call in range(1,22):
        model.matrix.grad=torch.randn_like(model.matrix)
        router.probe=call==21
        lease.opt.step()
    router.probe=False
    assert len(router.moment_probes)==1 and len(router.probes)==2
    assert router.moment_probes[0]['nonmutation_pass'] and router.moment_probes[0]['registered_release_pass']
    assert lease.release()['passed']

def test_prior_failed_numerical_and_release_gates_are_not_loosened():
    from native_residual_contract import protocol as old
    new=protocol();previous=old()
    assert new['seed']!=previous['seed']
    for key in ('gates','bounds','precision','expected','full_moment_diagnostic','timing','artifacts'):
        assert new[key]==previous[key]
