from pathlib import Path
import sys,json,math
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import torch
from tracking_curvature_kernel import ENTRIES,generator,curvature_terms,reference_from_generator,curvature_metrics
from tracking_curvature_contract import protocol,recipe,ARMS
from tracking_curvature_diagnostic import Router,build_arm
from prefix_sensitivity_kernel import prefix_stages
from matched_training_runtime import OptimizerLease

def test_anchor_identity_and_each_direction_stage():
    from tracking_direction_kernel import ENTRIES as anchors
    torch.manual_seed(157);q=torch.linalg.qr(torch.randn(32,32)).Q;h=torch.randn(32,8);c=h@h.T+torch.eye(32)
    stage=prefix_stages(c,q,False,True);before=c.clone();oldq=q.clone()
    for name in ENTRIES:
        raw,bound,a,cap_scale,chosen=generator(stage['b'],name)
        assert torch.equal(raw,-raw.T)
        if name=='jacobi':assert torch.equal(raw,stage['raw'])
        if name in anchors:assert torch.equal(ENTRIES[name](c,q),anchors[name](c,q))
        assert torch.equal(ENTRIES[name](c,q),reference_from_generator(q,a))
    assert torch.equal(c,before) and torch.equal(q,oldq)

def test_independent_capped_formula_equals_raw_reduction_formula():
    torch.manual_seed(157)
    for n in (2,8,32):
        h=torch.randn(n,n,dtype=torch.float64);b=h@h.T
        raw,bound,a,cap_scale,chosen=generator(b,'damped')
        capped=raw*cap_scale;d=b.diagonal();gap=d[:,None]-d[None,:]
        slope=2*(b*gap*capped).sum();curvature=2*(gap*capped).square().sum()
        alpha=(-slope/curvature.clamp_min(1e-24)).clamp(0,1)
        assert torch.allclose(a,capped*alpha,rtol=1e-12,atol=1e-14)
        assert torch.equal(a,-a.T) and float(a.norm()/math.sqrt(2))<=.5*(1+1e-12)
        assert 0<=float(chosen)<=float(cap_scale) and 0<=float(alpha)<=1

def test_independent_diagonal_hessian_and_quadratic_model_minimum():
    torch.manual_seed(158)
    for n in (4,8,16):
        d=torch.rand(n,dtype=torch.float64)*10;D=torch.diag(d)
        z=torch.randn(n,n,dtype=torch.float64);a=(z-z.T)/20
        t=torch.zeros((),dtype=torch.float64,requires_grad=True)
        rot=torch.matrix_exp(t*a);b=rot.T@D@rot
        f=(b-torch.diag(b.diagonal())).square().sum()
        first,=torch.autograd.grad(f,t,create_graph=True);second,=torch.autograd.grad(first,t)
        slope,h=curvature_terms(D,a)
        assert abs(float(first.detach()))<1e-12 and float(slope)==0
        assert torch.allclose(h,second,rtol=1e-10,atol=1e-10)
        x=torch.randn(n,n,dtype=torch.float64);B=x@x.T
        raw,bound,applied,cap_scale,chosen=generator(B,'damped')
        capped=raw*cap_scale;s,h=curvature_terms(B,capped)
        alpha=float(chosen/cap_scale)
        predicted=float(s)*alpha+.5*float(h)*alpha**2
        assert predicted<=1e-10
        for trial in (0.,.25,.5,.75,1.):
            assert predicted<=float(s)*trial+.5*float(h)*trial**2+1e-9

def test_zero_degenerate_covariance_and_near_zero_are_finite():
    for b in (torch.zeros(8,8),torch.eye(8),torch.ones(8,8),torch.diag(torch.arange(8,dtype=torch.float32)),torch.eye(8)*1e-20):
        raw,bound,a,cap_scale,chosen=generator(b,'damped')
        assert all(bool(torch.isfinite(x).all()) for x in (raw,bound,a,cap_scale,chosen))
        assert torch.count_nonzero(a)==0 and float(chosen)==0
    assert torch.equal(ENTRIES['damped'](torch.zeros(8,8),torch.eye(8)),torch.eye(8))

def test_covariance_scaling_and_capped_model_alpha_invariance():
    torch.manual_seed(159);h=torch.randn(16,8);c=h@h.T+torch.eye(16);q=torch.linalg.qr(torch.randn(16,16)).Q
    original=ENTRIES['damped'](c,q)
    for k in (.03125,4.,100.):assert torch.allclose(ENTRIES['damped'](c*k,q),original,atol=2e-6,rtol=2e-6)
    B=torch.tensor([[100.,.1],[.1,1.]],dtype=torch.float64)
    raw,bound,a,cap_scale,chosen=generator(B,'damped')
    for k in (.25,3.,10.):
        raw2,bound2,a2,cap2,chosen2=generator(B*k,'damped')
        assert float(chosen)<1 and float(chosen2)<1
        assert torch.allclose(a,a2,rtol=1e-12,atol=1e-14)

def test_exact_rotation_counterexample_and_damping_remedy():
    B=torch.tensor([[100.,.1],[.1,1.]],dtype=torch.float64)
    _,_,full,_,_=generator(B,'energy');_,_,damped,_,_=generator(B,'damped')
    def f(a):
        q=torch.matrix_exp(a);b=q.T@B@q;return float((b-torch.diag(b.diagonal())).square().sum())
    before=f(torch.zeros_like(B))
    assert f(full)>before and f(damped)<before
    metrics=curvature_metrics(B,*[generator(B,'damped')[i] for i in (0,2,3,4)])
    assert 0<metrics['damping_alpha']<1 and metrics['predicted_quadratic_fractional_energy_decrease']>0

def test_gates_counts_and_diagnostic_scope():
    p=protocol();t=p['tracking_diagnostic']
    assert t['directions']==['jacobi','energy','damped'] and t['fixed_cap']==.5
    assert p['maximum_optimizer_updates']==690 and p['expected']['counterfactual_complete_calls']==432
    assert p['expected']['operator_estimate_calls']==192 and p['expected']['exact_spectral_checks']==3
    assert t['admitted_geometry']==dict(normalized_gram=.005,absolute_gram_fro=.05)
    assert p['gates']['own_eager_basis_relative']==1e-4 and t['physical_m_limit']==.01

def test_plain_compiler_entries_and_immediate_lease_release():
    from hybrid_training_contract import BASELINE
    from cifar_optimizer_benchmark import support
    router=Router(lambda *a:None,lambda:'rng',lambda fn,**kw:fn)
    assert len(set(fn.__code__ for fn in ENTRIES.values()))==3
    c=recipe(json.loads((BASELINE/'config.json').read_text()))
    with torch.device('meta'):model=support.build_model(c)
    assert OptimizerLease(model,build_arm(model,c,ARMS[0],router)).release()['passed']

def test_cpu_dynamo_entries_layouts_and_stable_repeats():
    torch._dynamo.reset();captures=[]
    def backend(gm,inputs):captures.append(gm);return gm.forward
    compiled={name:torch.compile(fn,backend=backend,fullgraph=True,dynamic=False) for name,fn in ENTRIES.items()}
    limit=torch._dynamo.config.recompile_limit
    for n in (8,12,16):
        c=torch.diag(torch.linspace(1,2,n))+.01*torch.ones(n,n)
        for column in (False,True):
            q=torch.eye(n);q=q.T.contiguous().T if column else q
            for name,fn in compiled.items():
                with torch.no_grad():assert torch.equal(fn(c,q),ENTRIES[name](c,q))
    before=len(captures)
    for fn in compiled.values():
        with torch.no_grad():fn(c,q)
    assert len(captures)==before and limit==torch._dynamo.config.recompile_limit
    from torch._dynamo.utils import counters
    assert not counters['graph_break'];torch._dynamo.reset()
