import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from bf16_seed204_replicate import validate_manifest,normalize_reference,CASES
from bf16_full_horizon import make_protocol
from bf16_hybrid_confirm import worker_protocol,prefix_audit
from bf16_storage_screen import event_kind


def test_seed204_changes_initialization_only_and_keeps_full_bf16():
    p=make_protocol(204);old=make_protocol()
    assert {k:v for k,v in p.items() if k!='model_seed'}=={k:v for k,v in old.items() if k!='model_seed'}
    m=dict(cases=CASES,calls_per_case=dict(qr10=201,bf16=10000),seed=204,
        schedule_horizon=10000,stock_betas=(.95,.95,.95),protocol=p,
        numerical_error_role='diagnostic',reference_startup=dict(initial_parameters_sha256='204'))
    validate_manifest(m)
    assert worker_protocol('bf16',m)['model_seed']==204
    assert 'seed204' in worker_protocol('bf16',m)['startup_reference']
    assert worker_protocol('qr10',m)['validation_calls']==[50,100,200,201]
    with pytest.raises(AssertionError):validate_manifest({**m,'seed':205})


def test_old_control_normalization_preserves_raw_receipt():
    raw=dict(seed=204,calls=10000,profile='stock',arm='qr',initial_gradient=dict(hash='g'))
    before=copy.deepcopy(raw);r=normalize_reference(raw)
    assert r['initial_gradient_sha256']=='g' and raw==before
    with pytest.raises(AssertionError):normalize_reference({**raw,'profile':'previous'})


def test_diagnostic_flag_does_not_override_reference_reproduction(tmp_path):
    folder=tmp_path/'runs/qr10';folder.mkdir(parents=True)
    reference=dict(seed=204,initial_parameters_sha256='p',initial_gradient_sha256='g',
        validation=[dict(call=c,ce=10-c/100) for c in (0,50,100,200)])
    result={**reference,'clean_cycle_calls':100,'initial_backward_repeat_passed':True,
        'original_geometry_guard_passed':False,'precision_specific_guard_passed':False,'dynamo_counters':{}}
    result['validation']=[*copy.deepcopy(reference['validation']),dict(call=201,ce=8)]
    (folder/'result.json').write_text(json.dumps(result))
    events=[dict(event='update',call=c,kind=event_kind('qr10',c)) for c in range(1,202)]
    events.append(dict(event='startup_match',passed=True))
    (folder/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n')
    audit=prefix_audit(tmp_path,reference,diagnostics_advisory=True)
    assert audit['status']=='passed' and not audit['precision_specific_guard_passed']
    result['validation'][2]['ce']+=.001
    (folder/'result.json').write_text(json.dumps(result))
    with pytest.raises(AssertionError,match='Control reproduction mismatch'):
        prefix_audit(tmp_path,reference,diagnostics_advisory=True)
