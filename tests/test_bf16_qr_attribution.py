from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from bf16_qr_attribution import basis,preqr,next_probe
from bf16_soap import qr_basis
from soap_reference import ResearchSOAP


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('CPU check attempted CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden);torch.set_num_threads(1)
    old=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
    yield
    torch.set_float32_matmul_precision(old)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('batch',[None,2])
def test_float64_independent_oracle_order_basis_and_input_preservation(batch):
    rng=np.random.default_rng(76);d=9;n=1 if batch is None else batch
    cs=[];qs=[];answers=[]
    for _ in range(n):
        q=np.linalg.qr(rng.normal(size=(d,d)))[0];a=rng.normal(size=(d,d));c=a@a.T+np.eye(d)
        order=np.argsort(-np.diag(q.T@c@q));expected=np.linalg.qr(c@q[:,order])[0]
        cs.append(c);qs.append(q);answers.append((order,expected))
    c=torch.tensor(np.stack(cs));q=torch.tensor(np.stack(qs))
    if batch is None:c,q=c[0],q[0]
    before=(c.clone(),q.clone())
    for policy in ('original','reuse'):
        actual,order=basis(c,q,policy)
        if batch is None:actual,order=actual[None],order[None]
        for i,(eo,eq) in enumerate(answers):
            assert order[i].tolist()==eo.tolist()
            signs=np.sign((actual[i].numpy()*eq).sum(0))
            np.testing.assert_allclose(actual[i].numpy()*signs,eq,rtol=1e-12,atol=1e-12)
    assert torch.equal(c,before[0]) and torch.equal(q,before[1])
    with pytest.raises(ValueError):preqr(c,q,'other')


def test_original_is_actual_reference_refresh_and_reuse_is_production_primitive():
    torch.manual_seed(83);d=9;p=torch.nn.Parameter(torch.randn(d,3))
    opt=ResearchSOAP([p],basis_method='qr',variance_policy='permutation',transport_precision='highest')
    ql=torch.linalg.qr(torch.randn(d,d)).Q;qr=torch.linalg.qr(torch.randn(3,3)).Q
    a=torch.randn(d,d);c=a@a.T+torch.eye(d);cr=torch.diag(torch.tensor([3.,1.,2.]))
    state=dict(Q=[ql,qr],GG=[c,cr],step=100,exp_avg=torch.randn(d,3),exp_avg_sq=torch.rand(d,3))
    expected,order=basis(c,ql,'original');v=state['exp_avg_sq'].clone()
    right,ro=basis(cr,qr,'original');m=state['exp_avg'].clone()
    opt._research_refresh(state,opt.param_groups[0])
    torch.testing.assert_close(state['Q'][0],expected,atol=0,rtol=0)
    torch.testing.assert_close(state['exp_avg_sq'],v[order][:,ro],atol=0,rtol=0)
    torch.testing.assert_close(state['exp_avg'],expected.T@(ql@m@qr.T)@right,rtol=1e-5,atol=1e-6)
    batchedc=c[None].repeat(2,1,1);batchedq=ql[None].repeat(2,1,1)
    actual,ao=basis(batchedc,batchedq,'reuse');production,po=qr_basis(batchedc,batchedq,torch.float32)
    assert torch.equal(ao,po);torch.testing.assert_close(actual,production,atol=0,rtol=0)


def test_bf16_probe_variance_permutation_and_identical_policy_following_update():
    torch.manual_seed(84);d=9
    q=torch.linalg.qr(torch.randn(d,d)).Q.bfloat16().float();c=torch.diag(torch.linspace(1.,4.,d))
    new,order=basis(c,q,'reuse');m=torch.randn(1,d,32).bfloat16();v=torch.rand(1,d,32).bfloat16();g=torch.randn(1,d,32)
    before=[x.clone() for x in (c,q,m,v,g)]
    a=next_probe(c,q,new,order,m,v,g);b=next_probe(c,q,new,order,m,v,g)
    assert torch.equal(a['variance'],v[:,order])
    for k in a:assert torch.equal(a[k],b[k])
    assert bool(torch.isfinite(a['following_update']).all())
    for x,y in zip((c,q,m,v,g),before):assert torch.equal(x,y)
