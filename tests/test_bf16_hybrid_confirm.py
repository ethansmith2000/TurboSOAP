import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from bf16_hybrid_confirm import worker_protocol,check_reference_reproduction,validate_manifest,CASES
from bf16_full_horizon import make_protocol


def manifest():
    return dict(cases=CASES,calls_per_case=dict(qr10=201,bf16_fp32_basis=10000),seed=205,
        schedule_horizon=10000,stock_betas=(.95,.95,.95),protocol=make_protocol(),
        reference_startup=dict(initial_parameters_sha256='p205',initial_gradient_sha256='g205',
                               initial_loss=11.,initial_norm=1.,initial_validation_ce=11.))


def test_short_control_full_candidate_keep_one_schedule_and_matching():
    m=manifest();validate_manifest(m);old=copy.deepcopy(m)
    prefix=worker_protocol('qr10',m);full=worker_protocol('bf16_fp32_basis',m)
    assert prefix['validation_calls']==[50,100,200,201]
    assert max(prefix['diagnostic_calls'])==201
    assert prefix['evaluation_hash_calls']==[0]
    assert full['validation_calls'][-1]==10000 and full['diagnostic_calls'][-1]==9991
    assert prefix['model_seed']==full['model_seed']==205
    assert full['expected_startup']==m['reference_startup']
    assert m==old
    with pytest.raises(AssertionError):validate_manifest({**m,'seed':206})


def test_reference_reproduction_rejects_wrong_curve_before_full_training():
    r=dict(initial_parameters_sha256='p',initial_gradient_sha256='g',
           validation=[dict(call=c,ce=10-c/100) for c in (0,50,100,200,10000)])
    prefix=copy.deepcopy(r);prefix['validation']=prefix['validation'][:4]+[dict(call=201,ce=8)]
    assert check_reference_reproduction(prefix,r)==[0,50,100,200]
    prefix['validation'][2]['ce']+=1e-5
    with pytest.raises(AssertionError,match='Control reproduction mismatch'):check_reference_reproduction(prefix,r)
