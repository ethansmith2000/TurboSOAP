import json
from pathlib import Path
import time

import pytest
import torch

from benchmarks.legacy_cuda_probe import check_bounds, check_completion, run_pair, verify_bundle
from benchmarks.prepare_legacy_cuda_probe import prepare


def test_new_gpu_runner_retains_literal_cpu_failure_and_exact_partial_state():
    r=run_pair([4,6],331,172,torch.device('cpu'),time.monotonic()+30,2*1024**3)
    assert r['status']=='FAILED'
    assert r['completed_gradient_calls']==270
    assert r['failure']['call']==271
    assert r['failure']['matched_original_exception']
    assert r['failure']['partial_state_exact']
    assert r['observer_restored']


def test_supported_fixture_and_completion_qualifier_reject_truncation():
    r=run_pair([4,6],331,172,torch.device('cpu'),time.monotonic()+30,2*1024**3,
               dict(basis_compute_dtype='float32',basis_storage_dtype='float32',
                    covariance_compute_dtype='float32'))
    assert r['status']=='PASS' and check_completion(r,331)
    r['completed_gradient_calls']=330
    assert not check_completion(r,331)
    r['completed_gradient_calls']=331;r['observer_restored']=False
    assert not check_completion(r,331)
    r['observer_restored']=True;r['full_state_exact_at_every_observed_call']=False
    assert not check_completion(r,331)


def test_timeout_is_fail_closed_before_additional_work():
    with pytest.raises(TimeoutError):
        check_bounds(torch.device('cpu'),time.monotonic()-1,2*1024**3)


def test_cuda_memory_limit_is_enforced_without_initializing_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda device:2049)
    with pytest.raises(MemoryError):
        check_bounds(torch.device('cuda:0'),time.monotonic()+30,2048)


def test_frozen_bundle_detects_config_and_source_mutation(tmp_path):
    b=tmp_path/'bundle';prepare(b)
    cfg,_=verify_bundle(b,require_launch=False)
    assert cfg['optimizer_overrides']=={}
    assert cfg['gradient_calls_per_shape']==331
    original=(b/'config.json').read_bytes()
    changed=json.loads(original);changed['gradient_calls_per_shape']=270
    (b/'config.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='config mismatch'):
        verify_bundle(b,require_launch=False)
    (b/'config.json').write_bytes(original)
    source=b/'source/benchmarks/legacy_cuda_probe.py';source.write_bytes(source.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='source mismatch'):
        verify_bundle(b,require_launch=False)


def test_device_manifest_is_bound_to_claim_and_base_manifest(tmp_path,monkeypatch):
    from benchmarks.legacy_cuda_probe import sha
    b=tmp_path/'bundle';m=prepare(b)
    launch=dict(config_sha256=m['config_sha256'],manifest_sha256=sha(b/'manifest.json'),
                gpu=dict(index=3,uuid='GPU-test'))
    (b/'launch_manifest.json').write_text(json.dumps(launch))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-test');monkeypatch.setenv('GPU_CLAIM_INDICES','3')
    verify_bundle(b)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-wrong')
    with pytest.raises(ValueError,match='UUID mismatch'):
        verify_bundle(b)
