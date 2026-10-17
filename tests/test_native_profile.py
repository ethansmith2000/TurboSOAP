"""Native-only scope, complete budgets, original gates and interruption ownership."""
import copy
import json
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import pytest
import torch
from test_qr_reuse_profile import fixture as previous_fixture
from matched_training_runtime import streaming_digest
from native_profile_contract import recipe,protocol,validate,PROFILE_COUNTS,PARENT
from native_profile_core import run_profile

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('CPU native test attempted CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden);torch.set_num_threads(2)
    yield
    assert not torch.cuda.is_initialized()

def fixture():
    runtime,opt,events=previous_fixture();runtime.config=recipe();runtime.protocol=protocol()
    return runtime,opt,events

def test_native_profile_never_calls_rejected_reuse_and_restores_all_state(monkeypatch):
    import qr_reuse_profile_kernel as kernel
    import qr_reuse_profile_core as core
    def forbidden(*a,**k):raise AssertionError('Rejected product reuse invoked')
    monkeypatch.setattr(kernel,'refresh_reuse',forbidden)
    original=core.qr_basis;policies=[]
    def observed(c,q,policy):
        policies.append(policy);assert policy=='native';return original(c,q,policy)
    monkeypatch.setattr(core,'qr_basis',observed)
    runtime,opt,events=fixture()
    before=streaming_digest(dict(model=runtime.model.state_dict(),optimizer=opt.state_dict(),grad=[p.grad for p in runtime.model.parameters()]))
    result=run_profile(runtime,opt)
    assert len(policies)==80 and result['counts']==PROFILE_COUNTS
    assert len(result['qualification']['physical'])==4 and len(result['qualification']['factors'])==8 and result['qualification']['passed']
    assert set(r['policy'] for r in result['timings'] if r['scope']=='full_step')=={'ordinary','native'}
    assert len(result['timings'])==20+160+20
    assert len(result['releases'])==21+160+20
    assert result['cleanup']['nonmutation_pass'] and result['cleanup']['gradient_identity_pass'] and result['cleanup']['flags_restored']
    assert all(r['all_registered_released'] for r in result['cleanup']['endpoint_releases'])
    assert before==streaming_digest(dict(model=runtime.model.state_dict(),optimizer=opt.state_dict(),grad=[p.grad for p in runtime.model.parameters()]))

def test_native_qualification_matches_untouched_native_state():
    from native_profile_core import Profile
    from qr_reuse_profile_kernel import shadow,close_shadow
    runtime,source,events=fixture();base=copy.deepcopy(source.state_dict())
    backup=[p.clone() for p in runtime.model.parameters()]
    plain=shadow(runtime.model,runtime.config,base,'native')
    with torch.no_grad():plain.step()
    expected=streaming_digest(plain.state_dict());close_shadow(plain)
    with torch.no_grad():
        for p,v in zip(runtime.model.parameters(),backup):p.copy_(v)
    profile=Profile(runtime,source)
    original=runtime.emit
    # Capture qualification state just before original geometry validation.
    import native_profile_core as core
    old=core.common_geometry_guard;captured=[]
    def capture(opt,model):captured.append(streaming_digest(opt.state_dict()));return old(opt,model)
    core.common_geometry_guard=capture
    try:
        with torch.no_grad():result=profile.qualify()
        assert result['passed'] and captured==[expected]
    finally:
        core.common_geometry_guard=old;profile.base=profile.backup=None
        assert profile.owner.report()['all_registered_released'];profile.owner.close()

def test_interruption_persists_endpoint_releases_and_restores_source():
    runtime,source,events=fixture();before=streaming_digest(runtime.model.state_dict())
    def fail(fn):raise RuntimeError('injected native timing interruption')
    runtime.measure_call=fail
    with pytest.raises(RuntimeError,match='injected native timing interruption'):run_profile(runtime,source)
    cleanup=events[-1][1]
    assert cleanup['nonmutation_pass'] and cleanup['registered_release']['all_registered_released']
    assert len(cleanup['endpoint_releases'])==2 and all(r['all_registered_released'] for r in cleanup['endpoint_releases'])
    assert streaming_digest(runtime.model.state_dict())==before

def test_original_geometry_failure_stops_before_any_timing(monkeypatch):
    import native_profile_core as core
    runtime,source,events=fixture()
    def bad(*a):raise FloatingPointError('injected original geometry gate')
    monkeypatch.setattr(core,'common_geometry_guard',bad)
    with pytest.raises(RuntimeError,match='injected original geometry gate'):run_profile(runtime,source)
    assert not any(e=='profile_timing' for e,_ in events)
    cleanup=events[-1][1];assert cleanup['nonmutation_pass'] and cleanup['registered_release']['all_registered_released']
    assert len(cleanup['endpoint_releases'])==1 and cleanup['endpoint_releases'][0]['all_registered_released']

def test_contract_native_only_call_memory_gates_and_fresh183():
    c,p=recipe(),protocol();validate(c,p)
    assert c['seed']==183 and p['maximum_all_optimizer_calls']==1270+21
    assert p['profile']['counts']==PROFILE_COUNTS and p['profile']['policies']==['ordinary','native']
    assert p['profile']['qualification']==dict(absolute_gram_fro=.05,normalized_gram=.005,physical_m_relative=.01)
    m=p['profile']['memory'];assert sum(v for k,v in m.items() if k!='maximum_tensor_purpose_bytes')<m['maximum_tensor_purpose_bytes']
    with pytest.raises(ValueError):recipe(182)
    changed=copy.deepcopy(p);changed['profile']['policies'].append('reuse')
    with pytest.raises(ValueError,match='Undeclared'):validate(c,changed)

def test_preparation_preserves_all225_sources_and_no_bundle_overwrite(tmp_path):
    import prepare_native_profile as prep
    ref=tmp_path/'initial.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=183,parameters=144875648,cuda_initialized=False,snapshot_purpose_bytes=2230649344,model_sha256='2'*64))
    checks=tmp_path/'checks.json';prep.write(checks,dict(status='PASS_CPU_NATIVE_PROFILE',cuda_initialized=False,tests_passed=True))
    b=tmp_path/'bundle';m=prep.prepare(b,ref,183,PARENT/'token_cache_audit.json',checks)
    assert len(m['source_sha256'])==231
    old=prep.read(PARENT/'manifest.json')['source_sha256'];assert len(old)==225 and all(m['source_sha256'][n]==h for n,h in old.items())
    p=prep.read(b/'protocol.json');p['profile']['blocks']=4;prep.write(b/'protocol.json',p)
    m=prep.read(b/'manifest.json');m['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',m)
    with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
    with pytest.raises(ValueError,match='Fresh bundle'):prep.prepare(b,ref,183,PARENT/'token_cache_audit.json',checks)
