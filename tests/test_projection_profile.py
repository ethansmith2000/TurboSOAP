"""BF16 projection oracle, highest-M bypass, actual-state isolation and failure cleanup."""
from pathlib import Path
import sys,copy
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import pytest
import torch
from projection_profile_kernel import project,install
from projection_profile_core import run_profile
from projection_profile_contract import recipe,protocol,validate,PROFILE_COUNTS,PARENT
from hybrid_training_kernel import precision
from soap import SOAP
from matched_training_runtime import streaming_digest,OptimizerLease
from test_qr_reuse_profile import fixture as previous_fixture

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('CPU profile attempted CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden);torch.set_num_threads(2)
    previous=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(previous)
    assert not torch.cuda.is_initialized()

def fixture():
    r,o,e=previous_fixture();r.config=recipe();r.protocol=protocol()
    return r,o,e

@pytest.mark.parametrize('back',[False,True])
@pytest.mark.parametrize('layout',['row','column'])
def test_matches_independent_quantized_float64_stage_oracle(back,layout):
    torch.manual_seed(73);x=torch.randn(3,2)
    basis=[torch.linalg.qr(torch.randn(n,n))[0] for n in [3,2]]
    basis=[q.contiguous() if layout=='row' else q.T.contiguous().T for q in basis]
    # Quantize operands, evaluate each product in float64, round the intermediate
    # and final stage to BF16. This oracle does not call the adapter or BF16 GEMM.
    l,r=[q.to(torch.bfloat16).double() for q in basis];g=x.to(torch.bfloat16).double()
    first=((l if back else l.T)@g).to(torch.bfloat16).double()
    expected=(first@(r.T if back else r)).to(torch.bfloat16).float()
    actual=project(x,basis,back);assert actual.dtype==torch.float32
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_highest_transport_bypasses_bf16_and_stays_native(monkeypatch):
    import projection_profile_kernel as k
    r,opt,e=fixture();install(opt.matrix,'bf16');group=opt.matrix.param_groups[0];p=group['params'][0];s=opt.matrix.state[p]
    def forbidden(*a,**kw):raise AssertionError('Highest transport used BF16')
    monkeypatch.setattr(k,'project',forbidden)
    with precision('highest'):
        before=SOAP._project_back_with_basis(opt.matrix,s['exp_avg'],s['Q'],group)
        actual=opt.matrix._project_with_basis(p.grad,s['Q'],group)
        expected=SOAP._project_with_basis(opt.matrix,p.grad,s['Q'],group)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        opt.matrix._research_refresh(s,group)
        after=opt.matrix._project_back_with_basis(s['exp_avg'],s['Q'],group)
        torch.testing.assert_close(after,before,rtol=2e-5,atol=1e-7)


def test_complete_profile_all_counts_fidelity_transport_release_and_nonmutation(monkeypatch):
    import qr_reuse_profile_kernel as old
    monkeypatch.setattr(old,'refresh_reuse',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unrelated QR reuse invoked')))
    r,opt,e=fixture();before=streaming_digest(dict(model=r.model.state_dict(),state=opt.state_dict(),grad=[p.grad for p in r.model.parameters()]))
    result=run_profile(r,opt)
    assert result['counts']==PROFILE_COUNTS and len(result['timings'])==40 and len(result['releases'])==63
    q=result['qualification'];assert q['passed'] and len(q['actions'])==4 and len(q['physical'])==8 and len(q['factors'])==16
    assert q['unchanged_covariance_basis'] and {v['projection_policy'] for v in q['physical']}=={'native','bf16'}
    assert all(v['passed'] for v in q['actions']+q['physical'])
    c=result['cleanup'];assert c['nonmutation_pass'] and c['gradient_identity_pass'] and c['flags_restored'] and c['registered_release']['all_registered_released']
    assert all(v['all_registered_released'] for v in c['endpoint_releases'])
    assert before==streaming_digest(dict(model=r.model.state_dict(),state=opt.state_dict(),grad=[p.grad for p in r.model.parameters()]))


def test_timing_interruption_restores_source_and_releases_every_bank():
    r,opt,e=fixture()
    def fail(fn):raise RuntimeError('injected projection timer interruption')
    r.measure_call=fail
    with pytest.raises(RuntimeError,match='timer interruption'):run_profile(r,opt)
    c=e[-1][1];assert c['nonmutation_pass'] and c['gradient_identity_pass'] and c['flags_restored'] and c['registered_release']['all_registered_released']
    assert len(c['endpoint_releases'])==4 and all(v['all_registered_released'] for v in c['endpoint_releases'])

@pytest.mark.parametrize('failure',['geometry','adaptive'])
def test_fidelity_failures_stop_before_timing_and_release(monkeypatch,failure):
    import projection_profile_core as core
    if failure=='geometry':
        def fail(*a):raise FloatingPointError('injected original geometry failure')
        monkeypatch.setattr(core,'common_geometry_guard',fail)
    else:
        original=core.compare_updates
        def fail(*a):
            rows=original(*a);rows[0].update(relative=.02,passed=False);return rows
        monkeypatch.setattr(core,'compare_updates',fail)
    r,opt,e=fixture()
    with pytest.raises(RuntimeError,match='Bounded projection profile failed'):run_profile(r,opt)
    assert not any(n=='profile_timing' for n,v in e)
    c=e[-1][1];assert c['nonmutation_pass'] and c['registered_release']['all_registered_released'] and all(v['all_registered_released'] for v in c['endpoint_releases'])


def test_adapter_state_dtype_scope_and_lease_have_no_owner_cycle():
    r,opt,e=fixture();install(opt.matrix,'bf16')
    # This borrowed source fixture is local to the CPU test, not a live source.
    opt.step()
    for s in opt.matrix.state.values():
        assert all(v.dtype==torch.float32 for v in [s['exp_avg'],s['exp_avg_sq'],*s['GG'],*s['Q']])
    del s
    lease=OptimizerLease(r.model,opt);del opt
    assert lease.release()['passed']


def test_contract_counts_precision_new_threshold_and_ram_are_frozen():
    c,p=recipe(),protocol();validate(c,p)
    assert p['maximum_all_optimizer_calls']==1270+22 and p['profile']['counts']==PROFILE_COUNTS
    q=p['profile']['qualification'];assert q['physical_m_relative']==.01 and q['adaptive_update_relative']==.01 and q['adaptive_update_cosine']==.9999
    assert p['execution_order']==['hold200'] and p['evaluation_calls']==[]
    m=p['memory'];assert sum(v for k,v in m.items() if k!='maximum_tensor_purpose_bytes')<m['maximum_tensor_purpose_bytes']
    for name in ['adaptive_update_relative','physical_m_relative']:
        changed=copy.deepcopy(p);changed['profile']['qualification'][name]=.02
        with pytest.raises(ValueError,match='Undeclared'):validate(c,changed)


def test_prepare_preserves248sources_and_rejects_semantic_tampering(tmp_path):
    import prepare_projection_profile as prep
    ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=187,parameters=144875648,cuda_initialized=False,snapshot_purpose_bytes=2230649344,model_sha256='7'*64))
    cal=tmp_path/'cal.json';prep.write(cal,dict(status='PASS_CPU_PROJECTION_PROFILE',cuda_initialized=False,tests_passed=True))
    b=tmp_path/'bundle';m=prep.prepare(b,ref,187,PARENT/'token_cache_audit.json',cal);old=prep.read(PARENT/'manifest.json')['source_sha256']
    assert len(old)==248 and len(m['source_sha256'])==255 and all(m['source_sha256'][n]==h for n,h in old.items())
    p=prep.read(b/'protocol.json');p['profile']['qualification']['adaptive_update_relative']=.02;prep.write(b/'protocol.json',p)
    m=prep.read(b/'manifest.json');m['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',m)
    with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
    with pytest.raises(ValueError,match='Fresh bundle'):prep.prepare(b,ref,187,PARENT/'token_cache_audit.json',cal)
