"""Distinguish observed discrepancy from original safety guards and acceptance."""
import copy
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import pytest
import torch
from test_qr_reuse_profile import fixture as previous_fixture
from matched_training_runtime import streaming_digest
from reuse_observation_contract import recipe,protocol,validate,PARENT,PROFILE_COUNTS,TOTAL_COUNTS
from reuse_observation_core import run_profile

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('CPU observation attempted CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden);torch.set_num_threads(2)
    yield
    assert not torch.cuda.is_initialized()

def fixture():
    runtime,opt,events=previous_fixture();runtime.config=recipe();runtime.protocol=protocol()
    return runtime,opt,events

def test_all_three_observation_budgets_and_source_restoration():
    runtime,source,events=fixture();results=[]
    for call in (200,400,1000):
        for s in source.matrix.state.values():s['step']=call-1
        before=streaming_digest(dict(model=runtime.model.state_dict(),state=source.state_dict(),grad=[p.grad for p in runtime.model.parameters()]))
        result=run_profile(runtime,source,timed=call==1000);results.append(result)
        assert result['source_call']==call and result['qualification']['passed']
        assert before==streaming_digest(dict(model=runtime.model.state_dict(),state=source.state_dict(),grad=[p.grad for p in runtime.model.parameters()]))
        assert result['cleanup']['nonmutation_pass'] and result['cleanup']['gradient_identity_pass'] and result['cleanup']['flags_restored']
        assert result['cleanup']['registered_release']['all_registered_released'] and all(r['all_registered_released'] for r in result['cleanup']['endpoint_releases'])
    assert {k:sum(r['counts'][k] for r in results) for k in TOTAL_COUNTS}==TOTAL_COUNTS
    assert len(results[-1]['timings'])==290
    assert len(results[0]['timings'])==len(results[1]['timings'])==0

def test_legacy_point134_percent_remains_failed_without_blocking_readonly_observation(monkeypatch):
    import reuse_observation_core as core
    original=core.compare
    def discrepancy(*a):
        r=original(*a);r['actions'][0].update(relative=.0013421349,cosine=.9999989867,passed=False);r['passed']=False;return r
    monkeypatch.setattr(core,'compare',discrepancy)
    runtime,opt,events=fixture();result=run_profile(runtime,opt,timed=False);q=result['qualification']
    assert q['passed'] and not q['legacy_fidelity_pass'] and q['actions'][0]['relative']==.0013421349
    assert result['counts']['shadow_optimizer_steps']==2 and result['counts']['following_gradient_previews']==8
    assert result['cleanup']['nonmutation_pass'] and not result['timings']

@pytest.mark.parametrize('broken',['geometry','nonfinite_action'])
def test_original_geometry_and_nonfinite_still_abort_and_release(monkeypatch,broken):
    import reuse_observation_core as core
    original=core.compare
    def bad(*a):
        r=original(*a)
        if broken=='geometry':r['factors'][0]['absolute_gram_fro']=.1
        else:r['actions'][0]['relative']=float('nan')
        return r
    monkeypatch.setattr(core,'compare',bad)
    runtime,opt,events=fixture()
    with pytest.raises(RuntimeError,match='Original read-only geometry/M/finite guard failed'):run_profile(runtime,opt,timed=False)
    cleanup=events[-1][1];assert cleanup['nonmutation_pass'] and cleanup['registered_release']['all_registered_released']
    assert len(cleanup['endpoint_releases'])==2 and all(r['all_registered_released'] for r in cleanup['endpoint_releases'])

def test_timing_interruption_still_restores_and_records_all_owners():
    runtime,opt,events=fixture()
    def fail(fn):raise RuntimeError('injected observation timing interruption')
    runtime.measure_call=fail
    with pytest.raises(RuntimeError,match='injected observation timing interruption'):run_profile(runtime,opt)
    cleanup=events[-1][1];assert cleanup['nonmutation_pass'] and cleanup['registered_release']['all_registered_released']
    assert len(cleanup['endpoint_releases'])==3 and all(r['all_registered_released'] for r in cleanup['endpoint_releases'])

def test_contract_preserves_legacy_thresholds_and_declares_new_readonly_scope():
    p=protocol();validate(recipe(),p)
    assert p['profile']['observation_calls']==[200,400,1000] and p['maximum_all_optimizer_calls']==1306
    assert p['profile']['counts']==PROFILE_COUNTS and p['profile']['total_counts']==TOTAL_COUNTS
    q=p['profile']['qualification'];assert q['following_gradient_update_relative']==.001 and q['absolute_gram_fro']==.05 and q['physical_m_relative']==.01
    assert 'not hard guards' in q['adaptive_and_order_policy']
    m=p['profile']['memory'];assert sum(v for k,v in m.items() if k!='maximum_tensor_purpose_bytes')<m['maximum_tensor_purpose_bytes']
    changed=copy.deepcopy(p);changed['profile']['qualification']['following_gradient_update_relative']=.01
    with pytest.raises(ValueError,match='Undeclared'):validate(recipe(),changed)

def test_prepare_keeps231_parent_sources_and_frozen_observer_policy(tmp_path):
    import prepare_reuse_observation as prep
    ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=184,parameters=144875648,cuda_initialized=False,snapshot_purpose_bytes=2230649344,model_sha256='3'*64))
    checks=tmp_path/'checks.json';prep.write(checks,dict(status='PASS_CPU_REUSE_OBSERVATION',cuda_initialized=False,tests_passed=True))
    b=tmp_path/'bundle';m=prep.prepare(b,ref,184,PARENT/'token_cache_audit.json',checks)
    old=prep.read(PARENT/'manifest.json')['source_sha256'];assert len(old)==231 and len(m['source_sha256'])==237 and all(m['source_sha256'][n]==h for n,h in old.items())
    p=prep.read(b/'protocol.json');p['profile']['observation_calls'].append(600);prep.write(b/'protocol.json',p)
    m=prep.read(b/'manifest.json');m['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',m)
    with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
    with pytest.raises(ValueError,match='Fresh bundle'):prep.prepare(b,ref,184,PARENT/'token_cache_audit.json',checks)
