from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'benchmarks')]
from collections import Counter
import torch
from prefix_sensitivity_kernel import prefix_stages
from observer_free_prefix_fixture import make_functions, PROTOCOL


def test_compiled_measurement_never_compiles_stage_observer(monkeypatch):
    compiled=[]
    def fake_compile(fn,**kwargs):
        compiled.append(getattr(fn,'func',fn))
        assert kwargs==dict(mode='default',fullgraph=True,dynamic=False)
        return fn
    monkeypatch.setattr(torch,'compile',fake_compile)
    prefixes,stages,rets,whole=make_functions(Counter())
    assert set(stages)=={'eager_high','eager_highest'}
    assert prefix_stages not in compiled
    assert len(prefixes)==4 and len(whole)==8


def test_old_observer_gate_and_own_agreement_limits_are_preserved():
    from prefix_sensitivity_fixture import PROTOCOL as failed
    assert PROTOCOL['seed']!=failed['seed']
    assert PROTOCOL['stage_implementations']==['eager']
    for key in ('observer_candidate_limit','observer_endpoint_limit','own_eager_basis_limit','common_strict_basis_limit','limits','candidate_mixed','gram_precision'):
        assert PROTOCOL[key]==failed[key]
