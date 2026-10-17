from pathlib import Path
import sys,json,math
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import torch
from tracking_direction_kernel import ENTRIES,power_lower_estimate,geometry,reference_from_raw,raw_direction,directional_metrics
from tracking_direction_contract import protocol,recipe,ARMS
from tracking_direction_diagnostic import Router,build_arm
from prefix_sensitivity_kernel import prefix_stages
from matched_training_runtime import OptimizerLease

def test_baseline_exact_eager_identity_and_each_direction_stage():
    from native_residual_named_kernel import highest_prefix,candidate_retraction
    torch.manual_seed(153);q=torch.linalg.qr(torch.randn(32,32)).Q;h=torch.randn(32,8);c=h@h.T+torch.eye(32)
    stage=prefix_stages(c,q,False,True)
    assert torch.equal(ENTRIES['jacobi'](c,q),candidate_retraction(highest_prefix(c,q)))
    for name in ENTRIES:
        raw=raw_direction(stage['b'],name=='energy');bound=raw.norm()/math.sqrt(2)
        assert torch.equal(raw,-raw.T)
        if name=='jacobi':assert torch.equal(raw,stage['raw'])
        assert torch.equal(ENTRIES[name](c,q),reference_from_raw(q,raw,bound,.5))

def test_power_is_lower_estimate_and_skew_fro_bound_is_upper():
    a=torch.zeros(16,16)
    for i,value in ((0,2.),(2,1.),(4,.5)):a[i,i+1]=value;a[i+1,i]=-value
    before=a.clone();estimate=float(power_lower_estimate(a));exact=float(torch.linalg.svdvals(a).max())
    assert torch.equal(a,before) and estimate<=exact*(1+1e-5) and exact<=float(a.norm()/math.sqrt(2))
    assert estimate>=.99*exact

def test_bad_geometry_rejected_without_adjusting_live_threshold():
    assert geometry(torch.eye(16))['admitted']
    assert not geometry(torch.eye(16)*2)['admitted']
    assert not geometry(torch.eye(16)*float('nan'))['admitted']
    p=protocol();assert p['tracking_diagnostic']['directions']==['jacobi','energy'] and p['tracking_diagnostic']['fixed_cap']==.5
    assert p['maximum_optimizer_updates']==690 and p['expected']['full_moment_transactions']==8
    assert p['gates']['state_gram_limit']==.05 and p['gates']['own_eager_basis_relative']==1e-4

def test_same_basis_v_control_preserves_m_but_changes_adaptive_vector():
    torch.manual_seed(154);left=torch.linalg.qr(torch.randn(8,8)).Q;right=torch.linalg.qr(torch.randn(6,6)).Q
    world=torch.randn(8,6);m=left.T@world@right;v=torch.rand(8,6)+.1
    order=torch.arange(8).flip(0);newleft=left[:,order];newm=newleft.T@world@right
    assert torch.allclose(newleft@newm@right.T,world,atol=1e-5)
    carried=newleft@(newm/v.sqrt())@right.T;permuted=newleft@(newm/v[order].sqrt())@right.T
    assert not torch.allclose(carried,permuted) and torch.allclose(permuted,left@(m/v.sqrt())@right.T,atol=1e-5)

def test_plain_compiler_entries_and_immediate_lease_release():
    from hybrid_training_contract import BASELINE
    from cifar_optimizer_benchmark import support
    entries=[]
    def compiler(fn,**kwargs):entries.append(fn.__code__);return fn
    router=Router(lambda *a:None,lambda:'rng',compiler)
    assert len(set(fn.__code__ for fn in ENTRIES.values()))==2
    c=recipe(json.loads((BASELINE/'config.json').read_text()))
    with torch.device('meta'):model=support.build_model(c)
    assert OptimizerLease(model,build_arm(model,c,ARMS[0],router)).release()['passed']

def test_cpu_dynamo_cap_entries_layouts_and_stable_repeats():
    torch._dynamo.reset();captures=[]
    def backend(gm,inputs):captures.append(gm);return gm.forward
    compiled={name:torch.compile(fn,backend=backend,fullgraph=True,dynamic=False) for name,fn in ENTRIES.items()}
    limit=torch._dynamo.config.recompile_limit
    for n in (8,12,16):
        c=torch.eye(n)+.01*torch.ones(n,n)
        for column in (False,True):
            q=torch.eye(n);q=q.T.contiguous().T if column else q
            for fn in compiled.values():
                with torch.no_grad():fn(c,q)
    before=len(captures)
    for fn in compiled.values():
        with torch.no_grad():fn(c,q)
    assert len(captures)==before and limit==torch._dynamo.config.recompile_limit
    from torch._dynamo.utils import counters
    assert not counters['graph_break'];torch._dynamo.reset()


def test_energy_derivative_independent_autograd_and_equal_budget_optimality():
    torch.manual_seed(154)
    for n in (8,16,32):
        h=torch.randn(n,n,dtype=torch.float64);b=h@h.T
        energy=raw_direction(b,True);jacobi=raw_direction(b,False)
        for raw in (energy,jacobi):
            a=raw*(.5/(raw.norm()/math.sqrt(2)).clamp_min(1e-12)).clamp(max=1)
            t=torch.zeros((),dtype=torch.float64,requires_grad=True)
            rot=torch.matrix_exp(t*a);bt=rot.T@b@rot
            off=bt-torch.diag(bt.diagonal());f=off.square().sum()
            derivative,=torch.autograd.grad(f,t)
            metrics=directional_metrics(b,a)
            assert math.isclose(float(derivative),metrics['directional_slope'],rel_tol=1e-10,abs_tol=1e-9)
            assert metrics['directional_slope']<=0
        ae=energy/energy.norm();aj=jacobi/jacobi.norm()
        e=directional_metrics(b,ae);j=directional_metrics(b,aj)
        assert math.isclose(e['cosine_to_negative_energy_gradient'],1.,rel_tol=1e-12)
        assert -e['directional_slope']>=-j['directional_slope']*(1-1e-12)
        ec=energy*(.5/(energy.norm()/math.sqrt(2))).clamp(max=1)
        assert float(ec.norm()/math.sqrt(2))<=.5*(1+1e-12)
        before=b.clone();ENTRIES['energy'](b.float(),torch.eye(n))
        assert torch.equal(b,before)
