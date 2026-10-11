from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks')]
import torch
import json
import pytest
from common_y_kernel import warm_prefix,retract_two
from precision_atoms_kernel import warm_stages,warm_split
from warm_compile_kernel import warm_kernel
from common_y_fixture import PROTOCOL,verify,sha


def test_prefix_and_retraction_reconstruct_current_arithmetic():
    torch.manual_seed(771)
    for n in (8,32):
        q=torch.linalg.qr(torch.randn(n,n)).Q;h=torch.randn(n,n);c=h@h.T
        for mixed in (False,True):
            y=warm_prefix(c,q,False,mixed)
            assert torch.equal(y,warm_stages(c,q,False,mixed)['candidate'])
            assert torch.equal(retract_two(y),warm_split(c,q,False,mixed))
        assert torch.equal(retract_two(warm_prefix(c,q)),warm_kernel(c,q))


def test_identical_candidate_controls_never_mutate_input():
    torch.manual_seed(872);y=torch.linalg.qr(torch.randn(32,32)).Q+.001*torch.randn(32,32)
    before=y.clone();a=retract_two(y);b=retract_two(y)
    assert torch.equal(y,before) and torch.equal(a,b) and a.dtype==torch.float32
    assert torch.isfinite(a).all()


def test_three_prefix_and_four_retraction_products():
    from torch.utils._python_dispatch import TorchDispatchMode
    class Trace(TorchDispatchMode):
        def __init__(self):self.dtypes=[]
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if str(func)=='aten.mm.default':self.dtypes.append([str(x.dtype) for x in args])
            return func(*args,**(kwargs or {}))
    q=torch.eye(16);c=q*2+.01*torch.ones_like(q)
    for mixed in (False,True):
        pre=Trace();ret=Trace()
        with pre:y=warm_prefix(c,q,False,mixed)
        with ret:retract_two(y)
        assert pre.dtypes[:2]==[['torch.float32']*2]*2
        assert pre.dtypes[2]==['torch.bfloat16' if mixed else 'torch.float32']*2
        assert ret.dtypes==[['torch.float32']*2]*4


def test_source_and_resealed_boundary_policy_rejection(tmp_path):
    (tmp_path/'source').mkdir();f=tmp_path/'source/kernel';f.write_text('frozen')
    config=tmp_path/'protocol.json';config.write_text(json.dumps(PROTOCOL))
    m=dict(protocol_sha256=sha(config),source_sha256={'kernel':sha(f)})
    (tmp_path/'manifest.json').write_text(json.dumps(m));verify(tmp_path)
    f.write_text('changed')
    with pytest.raises(AssertionError):verify(tmp_path)
    f.write_text('frozen');config.write_text(json.dumps(dict(PROTOCOL,split_endpoint_limit=.01)))
    m['protocol_sha256']=sha(config);(tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(AssertionError):verify(tmp_path)
