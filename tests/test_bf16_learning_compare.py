import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from bf16_learning_compare import CASES,make_protocol,validate_manifest
from bf16_storage_screen import build_storage_optimizer,event_kind,observation_calls,coordinate_residuals
from bf16_soap import CompiledSOAP


@pytest.fixture(autouse=True)
def precision():
    previous=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(previous)


def test_learning_schedule_and_observations():
    from collections import Counter
    from owt_restart import multiplier
    for case in CASES:
        counts=Counter(event_kind(case,c) for c in range(1,2001))
        assert counts['cold']==1 and counts['ordinary']==1800
        assert counts['qr']==(199 if case=='qr10' else 39)
        assert counts['warm']==(0 if case=='qr10' else 160)
    assert all(event_kind('bf16',c) in ('warm','qr') for c in observation_calls(2000))
    assert {501,511,1001,1011,1501,1511,1991}<=observation_calls(2000)
    assert multiplier(200)==1 and .9<multiplier(2000)<1
    manifest=dict(calls=2000,cases=CASES,schedule_horizon=10000,stock_betas=(.95,.95,.95),
                  protocol=make_protocol({},'test'))
    validate_manifest(manifest)
    with pytest.raises(AssertionError):validate_manifest({**manifest,'schedule_horizon':2000})


@pytest.mark.parametrize('case',CASES)
def test_five_builders_and_observation_does_not_mutate(case):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding(12,4)
            self.blocks=torch.nn.ModuleList([torch.nn.Linear(4,4,bias=False)])
    model=Tiny();opt=build_storage_optimizer(model,case,compile_kernels=False)
    for p in model.parameters():p.grad=torch.randn_like(p)
    opt.step()
    matrix=opt.matrix
    assert all(g['betas']==(.95,.95) and g['shampoo_beta']==.95 for g in matrix.param_groups)
    if case=='qr10':assert matrix.param_groups[0]['basis_method']=='qr'
    if case=='original':assert matrix.param_groups[0]['basis_method']=='warm'
    if case in ('bf16','bf16_fp32_basis'):
        assert isinstance(matrix,CompiledSOAP)
        assert matrix.param_groups[0]['state_dtype']==torch.bfloat16
    before={id(t):t.clone() for state in matrix.state.values() for value in state.values()
            for t in (value if isinstance(value,list) else [value]) if isinstance(t,torch.Tensor)}
    rows=coordinate_residuals(matrix)
    assert len(rows)==2 and all(0<=r['relative_coordinate_offdiagonal']<=1 for r in rows)
    for state in matrix.state.values():
        for value in state.values():
            for t in (value if isinstance(value,list) else [value]):
                if isinstance(t,torch.Tensor):torch.testing.assert_close(t,before[id(t)],atol=0,rtol=0)
