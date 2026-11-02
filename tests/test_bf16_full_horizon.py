import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from bf16_full_horizon import make_protocol,worker_protocol,validate_manifest,CASES
from bf16_storage_screen import event_kind


def test_fixed_full_horizon_and_late_observations():
    p=make_protocol()
    assert p['model_seed']==205
    assert p['validation_calls'][-1]==10000
    assert p['evaluation_hash_calls']==[0,1000,5000,10000]
    assert len(p['diagnostic_calls'])==21
    assert {5001,5011,7501,7511,9951,9991}<=set(p['diagnostic_calls'])
    assert all(event_kind('bf16',c) in ('warm','qr') for c in p['diagnostic_calls'])
    m=dict(cases=CASES,calls=10000,seed=205,schedule_horizon=10000,stock_betas=(.95,.95,.95),protocol=p)
    validate_manifest(m)
    with pytest.raises(AssertionError):validate_manifest({**m,'seed':204})


def test_candidate_startup_is_paired_to_new_seed_control(tmp_path):
    manifest=dict(protocol=make_protocol())
    assert worker_protocol(tmp_path,'qr10',manifest)==manifest['protocol']
    folder=tmp_path/'runs/qr10';folder.mkdir(parents=True)
    baseline=dict(initial_parameters_sha256='new-weights',initial_gradient_sha256='new-grads',
                  validation=[dict(ce=10.9)])
    (folder/'result.json').write_text(json.dumps(baseline))
    (folder/'events.jsonl').write_text(json.dumps(dict(event='initial_gradient',loss=10.8,norm=1.2))+'\n')
    protocol=worker_protocol(tmp_path,'bf16',manifest)
    assert protocol['expected_startup']==dict(initial_parameters_sha256='new-weights',
        initial_gradient_sha256='new-grads',initial_loss=10.8,initial_norm=1.2,initial_validation_ce=10.9)
    assert 'expected_startup' not in manifest['protocol']
