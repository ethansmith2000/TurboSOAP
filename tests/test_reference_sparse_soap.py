"""Cadence-only control qualification; preserve the audited update body."""
import ast
import inspect
import textwrap

import pytest
import torch
from reference_blocked_soap import ReferenceBlockedSOAP
from reference_sparse_soap import SparseReferenceBlockedSOAP


def test_only_refresh_dispatch_changes_in_update_body():
    source = inspect.getsource(ReferenceBlockedSOAP.step)
    expected = source.replace('event=kind(self.call,self.arm)',
                              'event=self.refresh_kind(self.call)')
    assert expected != source
    assert ast.dump(ast.parse(textwrap.dedent(expected))) == ast.dump(
        ast.parse(textwrap.dedent(inspect.getsource(SparseReferenceBlockedSOAP.step))))


def test_interval_ten_reproduces_qualified_control_full_state():
    torch.manual_seed(207)
    a = [torch.nn.Parameter(torch.randn(5,7)*.01) for _ in range(2)]
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    control = ReferenceBlockedSOAP(list(zip(['a','b'],a)),block=4,maximum=8,compile_kernels=False)
    sparse = SparseReferenceBlockedSOAP(list(zip(['a','b'],b)),refresh_interval=10,
                                       block=4,maximum=8,compile_kernels=False)
    for call in range(1,112):
        for p,q in zip(a,b):
            g=torch.randn_like(p)*.03;p.grad=g;q.grad=g.clone()
        assert control.step()==sparse.step()
        for p,q in zip(a,b):torch.testing.assert_close(p,q,rtol=0,atol=0)
        for x,y in zip(control.groups,sparse.groups):
            for name in x['state']:torch.testing.assert_close(x['state'][name],y['state'][name],rtol=0,atol=0)


def test_qr50_holds_basis_but_keeps_updating_moments():
    torch.manual_seed(7)
    p=torch.nn.Parameter(torch.randn(5,7)*.01)
    opt=SparseReferenceBlockedSOAP([('weight',p)],block=4,maximum=8,compile_kernels=False)
    counts={}
    basis=None;momentum=None;covariance=None
    for call in range(1,152):
        p.grad=torch.randn_like(p)*.03
        event=opt.step();counts[event]=counts.get(event,0)+1
        expected='eigh' if call<=50 else 'qr' if call%50==0 else 'ordinary'
        assert event==expected
        state=opt.groups[0]['state']
        if 51<=call<=99:
            torch.testing.assert_close(state['lq'],basis,rtol=0,atol=0)
            assert not torch.equal(state['momentum'],momentum)
            assert not torch.equal(state['lc'],covariance)
        if call==100:assert not torch.equal(state['lq'],basis)
        basis=state['lq'].clone();momentum=state['momentum'].clone();covariance=state['lc'].clone()
    assert counts==dict(eigh=50,ordinary=99,qr=2)


@pytest.mark.parametrize('interval',[0,-1,2.5])
def test_invalid_interval_rejected(interval):
    with pytest.raises(ValueError):SparseReferenceBlockedSOAP([],refresh_interval=interval)
