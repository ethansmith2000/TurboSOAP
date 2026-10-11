from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks')]
import torch
import json
import pytest
from warm_compile_kernel import warm_kernel
from precision_atoms_kernel import warm_split, warm_stages
from precision_atoms_fixture import PROTOCOL,verify,functions,sha


def test_split_endpoints_equal_frozen_policy():
    torch.manual_seed(201101)
    for n in (8,32):
        q=torch.linalg.qr(torch.randn(n,n)).Q
        h=torch.randn(n,n);c=h@h.T
        assert torch.equal(warm_split(c,q),warm_kernel(c,q))
        assert torch.equal(warm_split(c,q,True,True),warm_kernel(c,q,True))
        for cm,am in ((False,False),(True,False),(False,True),(True,True)):
            assert torch.equal(warm_stages(c,q,cm,am)['qnew'],warm_split(c,q,cm,am))


def test_precision_atoms_change_only_selected_product_inputs():
    from torch.utils._python_dispatch import TorchDispatchMode
    class Trace(TorchDispatchMode):
        def __init__(self):self.products=[]
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if str(func)=='aten.mm.default':self.products.append([str(a.dtype) for a in args])
            return func(*args,**(kwargs or {}))
    q=torch.eye(16);c=torch.eye(16)*2+.001*torch.ones(16,16)
    for cm,am in ((False,False),(True,False),(False,True),(True,True)):
        t=Trace()
        with t:out=warm_split(c,q,cm,am)
        assert len(t.products)==7 and out.dtype==torch.float32
        assert t.products[:2]==[['torch.bfloat16' if cm else 'torch.float32']*2]*2
        assert t.products[2]==['torch.bfloat16' if am else 'torch.float32']*2
        assert t.products[3:]==[['torch.float32']*2]*4


def test_stage_storage_and_inputs_are_fp32_and_unchanged():
    torch.manual_seed(789);q=torch.linalg.qr(torch.randn(16,16)).Q;c=torch.eye(16)
    old=(c.clone(),q.clone())
    stages=warm_stages(c,q,True,False)
    assert len(stages)==10 and all(v.dtype==torch.float32 for v in stages.values())
    assert torch.equal(c,old[0]) and torch.equal(q,old[1])


def test_prospective_controls_and_resealed_protocol_rejection(tmp_path):
    specs=functions()
    assert list(specs)==PROTOCOL['variants']
    assert specs['compiled_highest'][1]=='highest'
    assert specs['compiled_bf16_cov'][2] and specs['compiled_bf16_candidate'][2]
    (tmp_path/'source').mkdir();source=tmp_path/'source/original';source.write_text('frozen')
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(PROTOCOL))
    manifest=dict(protocol_sha256=sha(protocol),source_sha256={'original':sha(source)})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest));verify(tmp_path)
    source.write_text('changed')
    with pytest.raises(AssertionError):verify(tmp_path)
    source.write_text('frozen');protocol.write_text(json.dumps(dict(PROTOCOL,beta=.999)))
    manifest['protocol_sha256']=sha(protocol);(tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):verify(tmp_path)
