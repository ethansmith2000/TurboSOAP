"""Numerical policy and capture checks for the opt-in storage candidate."""
import copy

import pytest
import torch

from bf16_soap import CompiledSOAP, STATE_KEYS, matrix_step, qr_basis
from soap_reference import ResearchSOAP


@pytest.mark.parametrize('shape', [(5,5),(7,3),(3,7)])
def test_fp32_matches_reference_with_efficient_qr(shape):
    torch.manual_seed(31)
    a=torch.nn.Parameter(torch.randn(shape));b=torch.nn.Parameter(a.detach().clone())
    config=dict(lr=.003,betas=(.95,.95),shampoo_beta=.95,weight_decay=.07,
                precondition_frequency=2,hard_reset_interval=2)
    reference=ResearchSOAP([a],**config)
    candidate=CompiledSOAP([b],state_dtype=torch.float32,basis_dtype=torch.float32,
                           compute_dtype=torch.float32,compile_kernels=False,**config)
    for _ in range(13):
        g=torch.randn(shape);a.grad=g.clone();b.grad=g.clone()
        reference.step();candidate.step()
        torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(reference.state[a]['exp_avg_sq'],candidate.state[b]['v'][0],
                                   atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('basis_dtype',[torch.float32,torch.bfloat16])
def test_batched_independence_and_checkpoint_replay(basis_dtype):
    torch.manual_seed(45)
    params=[torch.nn.Parameter(torch.randn(5,3)) for _ in range(2)]
    config=dict(precondition_frequency=2,hard_reset_interval=6,basis_dtype=basis_dtype,
                compile_kernels=False)
    batch=CompiledSOAP(params,**config)
    separate_params=[torch.nn.Parameter(p.detach().clone()) for p in params]
    separate=[CompiledSOAP([p],**config) for p in separate_params]
    events=[]
    for _ in range(8):
        for p,q in zip(params,separate_params):
            p.grad=torch.randn_like(p);q.grad=p.grad.clone()
        batch.step();events.append(batch.last_event)
        for opt in separate:opt.step()
    assert events==['cold','ordinary','warm','ordinary','warm','ordinary','qr','ordinary']
    for i,(p,q,opt) in enumerate(zip(params,separate_params,separate)):
        torch.testing.assert_close(p,q,atol=0,rtol=0)
        for key in STATE_KEYS:
            torch.testing.assert_close(batch.state[params[0]][key][i],opt.state[q][key][0],atol=0,rtol=0)
    resumed_params=[torch.nn.Parameter(p.detach().clone()) for p in params]
    resumed=CompiledSOAP(resumed_params,**config)
    resumed.load_state_dict(copy.deepcopy(batch.state_dict()))
    for _ in range(7):
        for p,q in zip(params,resumed_params):
            p.grad=torch.randn_like(p);q.grad=p.grad.clone()
        batch.step();resumed.step()
        for p,q in zip(params,resumed_params):torch.testing.assert_close(p,q,atol=0,rtol=0)
    for key in STATE_KEYS:
        value=resumed.state[resumed_params[0]][key]
        assert value.dtype==(basis_dtype if key in ('ql','qr') else torch.bfloat16)
        assert torch.isfinite(value).all()
    assert (resumed.state[resumed_params[0]]['v']>=0).all()


@pytest.mark.parametrize('event',['ordinary','warm','qr'])
def test_fullgraph_capture_includes_parameter_writes(event):
    from functools import partial
    from bf16_soap import ordinary_group,warm_group,qr_group
    p=(torch.nn.Parameter(torch.randn(4,3)),torch.nn.Parameter(torch.randn(4,3)))
    opt=CompiledSOAP(p,compile_kernels=False)
    for value in p:value.grad=torch.randn_like(value)
    opt.step()
    state=tuple(opt.state[p[0]][key] for key in STATE_KEYS)
    graphs=[]
    def backend(graph,inputs):
        graphs.append(graph)
        return graph.forward
    fn=partial({'ordinary':ordinary_group,'warm':warm_group,'qr':qr_group}[event],
        beta1=.95,beta2=.95,cov_beta=.95,eps=1e-8,weight_decay=.05,
        compute_dtype=torch.bfloat16,basis_dtype=torch.bfloat16)
    compiled=torch.compile(fn,backend=backend,fullgraph=True,dynamic=False)
    with torch.no_grad():
        for rate in (.001,.002):
            before=p[0].clone()
            compiled(p,tuple(value.grad for value in p),state,torch.tensor(2.),torch.tensor(rate))
            assert not torch.equal(before,p[0])
    assert len(graphs)==1
    assert any('copy_' in str(node.target) for node in graphs[0].graph.nodes)


@pytest.mark.parametrize('event',['ordinary','warm','qr'])
def test_inductor_respects_bf16_rounding(event):
    torch.manual_seed(51)
    p=torch.randn(2,4,3);g=torch.randn_like(p)
    l=torch.eye(4).repeat(2,1,1).bfloat16();r=torch.eye(3).repeat(2,1,1).bfloat16()
    state=(torch.randn_like(p).bfloat16(),torch.rand_like(p).bfloat16(),
           torch.eye(4).repeat(2,1,1).bfloat16(),torch.eye(3).repeat(2,1,1).bfloat16(),l,r)
    config=dict(beta1=.95,beta2=.95,cov_beta=.95,eps=1e-8,weight_decay=.05,
                event=event,compute_dtype=torch.bfloat16,basis_dtype=torch.bfloat16)
    args=(p,g,state,torch.tensor(2.),torch.tensor(.001))
    eager=matrix_step(*args,**config)
    compiled=torch.compile(matrix_step,fullgraph=True,dynamic=False,
                           options={'emulate_precision_casts':True,'pattern_matcher':False})
    actual=compiled(*args,**config)
    # FP32 pointwise fusion can round differently at BF16 ties. Require each
    # output to agree to one BF16 unit; GPU checks use the real CUDA path.
    torch.testing.assert_close(actual[0],eager[0],atol=2e-6,rtol=2e-6)
    for a,b in zip(actual[1],eager[1]):torch.testing.assert_close(a,b,atol=.008,rtol=.008)


def test_efficient_qr_uses_same_sorted_subspace():
    torch.manual_seed(5)
    c=torch.randn(2,5,5);c=c@c.transpose(-1,-2)
    q=torch.linalg.qr(torch.randn_like(c)).Q
    result,order=qr_basis(c,q,torch.float32)
    expected=(q.transpose(-1,-2)@c@q).diagonal(dim1=-2,dim2=-1).argsort(descending=True)
    assert torch.equal(order,expected)
    z=c@q.gather(-1,order.unsqueeze(-2).expand_as(q))
    torch.testing.assert_close(result,torch.linalg.qr(z).Q,atol=3e-6,rtol=3e-6)


def test_four_shapes_three_events_fit_default_capture_cache(monkeypatch):
    real_compile=torch.compile;graphs=[]
    def backend(graph,inputs):
        graphs.append(graph)
        return graph.forward
    def capture(fn,**kwargs):
        return real_compile(fn,backend=backend,fullgraph=True,dynamic=False)
    monkeypatch.setattr(torch,'compile',capture)
    torch._dynamo.reset()
    params=[torch.nn.Parameter(torch.randn(*shape)) for shape in ((3,3),(5,3),(7,3),(3,5))]
    opt=CompiledSOAP(params,precondition_frequency=2,hard_reset_interval=4)
    for call in range(1,10):
        for p in params:p.grad=torch.randn_like(p)
        opt.param_groups[0]['lr']=.001*call
        opt.step()
    assert len(graphs)==12
    torch._dynamo.reset()
