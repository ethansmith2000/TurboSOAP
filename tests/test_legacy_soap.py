import hashlib
import json

import pytest
import torch

from benchmarks.legacy_soap import (SOURCE, SOURCE_SHA256, build_legacy_optimizer,
                                    effective_policy, load_legacy_class)
from benchmarks.audit_legacy_soap import METHODS, observe_calls, run_case


def test_source_is_literal_and_adapter_does_not_replace_production():
    import soap
    production = soap.SOAP
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA256
    cls=load_legacy_class()
    assert cls is not production and soap.SOAP is production
    optimizer=cls([torch.nn.Parameter(torch.zeros(3,4))])
    policy=effective_policy(optimizer)
    settings=policy['groups'][0]['settings']
    assert settings['betas']==[.95,.999]
    assert settings['basis_warmup_steps']==250
    assert settings['basis_direction_normalizer']=='col'
    assert settings['basis_direction_normalizer_beta']==.99
    assert policy==json.loads(json.dumps(policy,allow_nan=False))
    current=production([torch.nn.Parameter(torch.zeros(3,4))])
    with pytest.raises(ValueError,match='literal historical class'):
        effective_policy(current)


def test_modified_historical_source_fails_closed(tmp_path):
    p=tmp_path/'changed.py'
    p.write_bytes(SOURCE.read_bytes()+b'\n# changed\n')
    with pytest.raises(ValueError,match='hash mismatch'):
        load_legacy_class(p)


def test_actual_group_overrides_and_parameter_coverage_are_visible():
    p=torch.nn.Parameter(torch.zeros(3,4))
    optimizer=build_legacy_optimizer([dict(params=[p],lr=.0005,
                                         basis_warmup_steps=0,param_names=['matrix'])])
    group=effective_policy(optimizer)['groups'][0]
    assert group['settings']['lr']==.0005
    assert group['settings']['basis_warmup_steps']==0
    assert group['settings']['param_names']==['matrix']
    assert group['parameters'][0]['shape']==[3,4]
    optimizer.param_groups[0]['lr']=float('nan')
    with pytest.raises(ValueError):
        effective_policy(optimizer)


def test_controller_fallback_and_observer_preserve_literal_trajectory():
    row=run_case('forced_fallback',dict(
        basis_compute_dtype='float32',basis_storage_dtype='float32',
        covariance_compute_dtype='float32',basis_warmup_steps=0,
        basis_metric_frequency=1,basis_fallback_diag_err_threshold=1e-12),calls=41)
    assert row['full_state_exact_every_observed_call']
    assert row['method_counts']['_fallback_basis_one']==8
    assert len(row['warm_events'])==4


def test_literal_defaults_match_original_successful_prefix_and_cpu_failure():
    row=run_case('literal_defaults',{},calls=331,
                 expected_error='"trace" not implemented for \'BFloat16\'')
    assert row['status']=='CPU_UNSUPPORTED_DEFAULTS'
    assert row['completed_gradient_calls']==270
    assert row['failure']['call']==271
    assert row['failure']['original_exception_and_partial_state_exact']


def test_fp32_reconstruction_crosses_initial_qr_phase_and_metric_cadence():
    row=run_case('fp32_reconstruction',dict(
        basis_compute_dtype='float32',basis_storage_dtype='float32',
        covariance_compute_dtype='float32'),calls=331)
    assert row['status']=='PASS'
    assert row['method_counts']['_fallback_basis_list']==25
    assert [e['step'] for e in row['warm_events']]==list(range(260,331,10))
    assert [e['step'] for e in row['warm_events'] if e['metrics']]==[260,310]


def test_observer_restores_existing_bindings_even_on_exception():
    optimizer=build_legacy_optimizer([torch.nn.Parameter(torch.zeros(3,4))])
    original=optimizer._project
    optimizer._project=original
    with pytest.raises(RuntimeError,match='deliberate'):
        with observe_calls(optimizer,dict(call=0,warm_events=[])):
            raise RuntimeError('deliberate')
    assert optimizer._project is original
    assert all(name not in optimizer.__dict__ for name in METHODS if name!='_project')
