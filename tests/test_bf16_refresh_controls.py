import copy
from collections import Counter
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from bf16_refresh_controls import CASES,make_protocol,validate_manifest,audit_arm
from bf16_storage_screen import build_storage_optimizer,event_kind,CONTROL_CADENCES
from owt_restart import multiplier


@pytest.mark.parametrize('case',CASES)
def test_actual_optimizer_cadence_precision_and_skipped_initialization(case):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding(12,4)
            self.blocks=torch.nn.ModuleList([torch.nn.Linear(4,4,bias=False)])
    torch.set_num_threads(1);torch.manual_seed(204)
    model=Tiny();o=build_storage_optimizer(model,case,compile_kernels=False)
    cadence=CONTROL_CADENCES[case]
    assert o.matrix.frequency==o.matrix.repair==cadence
    assert o.auxiliary.param_groups[0]['betas']==(.9,.999)
    assert all(g['betas']==(.95,.95) and g['shampoo_beta']==.95 for g in o.matrix.param_groups)
    assert all(g[k]==torch.bfloat16 for g in o.matrix.param_groups for k in ('state_dtype','basis_dtype','compute_dtype'))
    before=model.blocks[0].weight.detach().clone();events=[]
    for call in range(1,102):
        for p in model.parameters():p.grad=torch.randn_like(p)
        o.step();events.append(o.matrix.last_event)
        assert o.matrix.last_event==event_kind(case,call)
        if call==1:torch.testing.assert_close(model.blocks[0].weight,before,rtol=0,atol=0)
    assert 'warm' not in events
    assert Counter(events)['qr']==100//cadence
    assert {key[0] for key in o.matrix._functions}=={'ordinary','qr'}
    for state in o.matrix.state.values():
        assert all(state[k].dtype==torch.bfloat16 for k in ('m','v','cl','cr','ql','qr'))


def test_full_schedule_shared_cycles_and_refresh_only_observations():
    m=dict(cases=CASES,calls=2000,seed=204,schedule_horizon=10000,stock_betas=(.95,.95,.95),
        numerical_error_role='diagnostic',gpu_execution_seconds_cap=1800,reference_startup={})
    m['protocols']={c:make_protocol(c,{}) for c in CASES};validate_manifest(m)
    assert multiplier(200)==1 and .9<multiplier(2000)<1
    for case,qr in zip(CASES,(99,39)):
        counts=Counter(event_kind(case,c) for c in range(1,2001))
        assert counts==dict(cold=1,qr=qr,ordinary=1999-qr)
        p=m['protocols'][case]
        assert all(event_kind(case,c)=='qr' for c in p['diagnostic_calls'])
        assert set(p['residual_calls'])<=set(p['diagnostic_calls'])
        assert len(range(101,2001))%100==0
    assert event_kind('bf16',11)=='warm' and event_kind('bf16',51)=='qr'
    with pytest.raises(AssertionError):validate_manifest({**m,'schedule_horizon':2000})
    with pytest.raises(AssertionError):validate_manifest({**m,'gpu_execution_seconds_cap':1801})


def test_audit_accepts_finite_screen_crossing_but_rejects_capture_and_cadence_errors(tmp_path):
    case=CASES[0];expected=dict(initial_parameters_sha256='p',initial_gradient_sha256='g',initial_validation_ce=10.)
    p=make_protocol(case,expected);m=dict(protocols={case:p},reference_startup=expected,reference_matrix_state_bytes=42)
    folder=tmp_path/'runs'/case;folder.mkdir(parents=True)
    result=dict(seed=204,clean_cycle_calls=1900,initial_backward_repeat_passed=True,
        initial_parameters_sha256='p',initial_gradient_sha256='g',matrix_state_bytes=42,peak_gpu_bytes=42,
        original_geometry_guard_passed=False,precision_specific_guard_passed=False,dynamo_counters={},
        validation=[dict(call=c,ce=10. if c==0 else 4.,targets=1024*1023,
            listed_tensor_hash_preservation_checked=c in p['evaluation_hash_calls']) for c in [0,*p['validation_calls']]])
    events=[dict(event='update',call=c,kind=event_kind(case,c),loss=4.,fb_ms=1.,optimizer_ms=1.,step_ms=2.) for c in range(1,2001)]
    events += [dict(event=n,passed=True) for n in ('startup_match','initial_validation_match','final_state_finite')]
    events += [dict(event='capture_counters',call=c,stats=dict(unique_graphs=8),graph_breaks={}) for c in p['capture_calls']]
    geometry=[dict(call=c,factors=[{}]*96,momentum=[] if c==1 else [{}]*48,
        coordinate_residuals=[{}]*96 if c in p['residual_calls'] else []) for c in [1,*p['diagnostic_calls']]]
    def save():
        (folder/'result.json').write_text(json.dumps(result));(folder/'geometry.json').write_text(json.dumps(geometry))
        (folder/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n')
    save();assert audit_arm(tmp_path,case,m)['status']=='passed'
    events[-1]['stats']['unique_graphs']=12;save()
    with pytest.raises(AssertionError):audit_arm(tmp_path,case,m)
    events[-1]['stats']['unique_graphs']=8;events[20]['kind']='warm';save()
    with pytest.raises(AssertionError):audit_arm(tmp_path,case,m)
