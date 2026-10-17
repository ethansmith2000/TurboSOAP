from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
from functools import partial
import torch
from native_residual_named_kernel import PolicyRouter,build_arm
from native_residual_named_contract import ARMS,recipe,protocol
from matched_training_runtime import OptimizerLease

def test_plain_distinct_compiler_entries_and_exact_eager_policy_identity():
    from native_residual_weak_kernel import PolicyRouter as Previous
    compiled=[]
    def compiler(fn,**kwargs):
        assert not isinstance(fn,partial);compiled.append(fn.__code__);return fn
    router=PolicyRouter(lambda *a:None,lambda:'rng',compiler=compiler)
    previous=Previous(lambda *a:None,lambda:'rng',compiler=lambda f,**k:f)
    assert len(compiled)==len(set(compiled))==4
    torch.manual_seed(152);q=torch.linalg.qr(torch.randn(16,16)).Q;h=torch.randn(16,8);c=h@h.T+torch.eye(16)
    for name in router.kernels:
        assert torch.equal(router.kernels[name].evaluate(c,q,eager=True),previous.kernels[name].evaluate(c,q,eager=True))

def test_cpu_dynamo_shape_layout_specializations_do_not_share_partial_wrapper():
    from hybrid_training_kernel import precision
    torch._dynamo.reset()
    default_limit=torch._dynamo.config.recompile_limit
    captures=[]
    def backend(gm,inputs):captures.append(gm);return gm.forward
    def compiler(fn,**kwargs):return torch.compile(fn,backend=backend,fullgraph=True,dynamic=False)
    router=PolicyRouter(lambda *a:None,lambda:'rng',compiler=compiler)
    for n in (8,12,16):
        c=torch.eye(n)+.01*torch.ones(n,n)
        for column in (False,True):
            q=torch.eye(n);q=q.T.contiguous().T if column else q
            for name,kernel in router.kernels.items():
                with torch.no_grad():actual=kernel.evaluate(c,q)
                assert torch.isfinite(actual).all()
    before=len(captures)
    for name,kernel in router.kernels.items():
        with torch.no_grad():kernel.evaluate(c,q)
    assert len(captures)==before and default_limit==torch._dynamo.config.recompile_limit
    from torch._dynamo.utils import counters
    assert not counters['graph_break']
    torch._dynamo.reset()

def test_empty_immediate_release_remains_strict_for_named_kernels():
    import json
    from hybrid_training_contract import BASELINE
    from cifar_optimizer_benchmark import support
    c=recipe(json.loads((BASELINE/'config.json').read_text()))
    with torch.device('meta'):model=support.build_model(c)
    router=PolicyRouter(lambda *a:None,lambda:'rng',compiler=lambda f,**k:f)
    for arm in ARMS:assert OptimizerLease(model,build_arm(model,c,arm,router)).release()['passed']
    from native_residual_weak_contract import protocol as old
    for key in ('gates','bounds','precision','expected','full_moment_diagnostic','timing','artifacts'):assert protocol()[key]==old()[key]
