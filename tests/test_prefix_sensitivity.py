from pathlib import Path
import json
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'benchmarks')]
import pytest
import torch
from common_y_kernel import warm_prefix
from prefix_sensitivity_kernel import prefix_stages, scaled_perturbation


@pytest.mark.parametrize('column',[False,True])
@pytest.mark.parametrize('defect',[0.,.0007])
def test_observer_is_original_prefix_and_cap_matches_independent_formula(column,defect):
    torch.manual_seed(148)
    q=torch.linalg.qr(torch.randn(32,32)).Q.contiguous()
    q=q*(1+defect*torch.linspace(-1,1,32))
    if column:q=q.T.contiguous().T
    h=torch.randn(32,8);c=h@h.T+torch.eye(32)
    original=q.clone();cov_before=c.clone()
    stages=prefix_stages(c,q,False,True)
    assert torch.equal(stages['candidate'],warm_prefix(c,q,False,True))
    assert torch.equal(q,original) and torch.equal(c,cov_before)
    expected_bound=stages['raw'].norm()/2**.5
    expected_cap=min(1.,.5/max(float(expected_bound),1e-12))
    assert torch.equal(stages['bound'],expected_bound)
    assert torch.equal(stages['a'],stages['raw']*expected_cap)
    assert float(stages['a'].norm()/2**.5)<=.500001
    assert torch.equal(stages['b'],stages['b'].T)


def test_common_perturbation_has_declared_norm_without_mutation():
    torch.manual_seed(149);q=torch.randn(32,32);d=torch.randn(32,32);before=q.clone()
    perturbed=scaled_perturbation(q,d,1e-3)
    assert torch.equal(q,before)
    assert abs(float((perturbed-q).norm()/q.norm())-1e-3)<1e-7


def test_observer_has_only_declared_prefix_products():
    from torch.utils._python_dispatch import TorchDispatchMode
    class Trace(TorchDispatchMode):
        def __init__(self):self.dtypes=[]
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if str(func)=='aten.mm.default':self.dtypes.append([x.dtype for x in args])
            return func(*args,**(kwargs or {}))
    trace=Trace();q=torch.eye(16);c=q+.01*torch.ones_like(q)
    with trace:prefix_stages(c,q,False,True)
    assert trace.dtypes==[[torch.float32]*2,[torch.float32]*2,[torch.bfloat16]*2]


def test_resealed_precision_or_source_tamper_rejected(tmp_path):
    from prefix_sensitivity_fixture import PROTOCOL, verify, sha
    (tmp_path/'source').mkdir();f=tmp_path/'source/kernel.py';f.write_text('frozen')
    config=tmp_path/'protocol.json';config.write_text(json.dumps(PROTOCOL))
    manifest=dict(protocol_sha256=sha(config),source_sha256={'kernel.py':sha(f)},prepared_sha256={})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest));verify(tmp_path)
    f.write_text('changed')
    with pytest.raises(AssertionError):verify(tmp_path)
    f.write_text('frozen');config.write_text(json.dumps(dict(PROTOCOL,candidate_mixed=False)))
    manifest['protocol_sha256']=sha(config);(tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):verify(tmp_path)
