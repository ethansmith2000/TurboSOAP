"""Independent QR/order/history controls and complete read-only profile/failure tests."""
from pathlib import Path
import copy
import json
import math
import sys
from types import SimpleNamespace
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import numpy as np
import pytest
import torch
from cifar_optimizers import build_optimizer
from matched_training_contract import RandomStreams
from matched_training_runtime import streaming_digest,tensor_values
from qr_reuse_profile_contract import recipe,protocol,PROFILE_COUNTS,PARENT,validate
from qr_reuse_profile_kernel import qr_basis,shadow,close_shadow,refresh_reuse
from qr_reuse_profile_core import run_profile


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('CPU profile test attempted CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden);torch.set_num_threads(2)
    yield
    assert not torch.cuda.is_initialized()


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.blocks=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(3,4,bias=False)) for _ in range(4)]);self.embedding=torch.nn.Embedding(5,3)


def fixture():
    torch.manual_seed(908);model=Tiny();c=recipe();opt=build_optimizer(model,c,'qr20')
    for p in model.parameters():p.grad=torch.arange(p.numel(),dtype=torch.float32).reshape_as(p)*.01+.03
    opt.step()
    for group in opt.matrix.param_groups:
        group.update(precondition_frequency=200,hard_reset_interval=0)
        for p in group['params']:
            s=opt.matrix.state[p];s['step']=999;s['research_qr_refreshes']=s['research_strict_transports']=4
            s['exp_avg'].copy_(torch.linspace(.01,.03,p.numel()).reshape_as(p));s['exp_avg_sq'].fill_(.15)
            s['GG']=[q@(torch.diag(torch.linspace(.8,3.,len(q)))+.025*(torch.ones_like(q)-torch.eye(len(q))))@q.T for q in s['Q']]
    for s in opt.auxiliary.state.values():s['step'].fill_(1000)
    events=[]
    def measure(fn):return fn(),1.,2.
    r=SimpleNamespace(model=model,config=c,protocol=protocol(),measure_call=measure,budget=lambda **kw:None,
        streams=RandomStreams(torch.Generator().manual_seed(41),SimpleNamespace(rng=np.random.default_rng(1),counts={})),emit=lambda e,p:events.append((e,p)))
    return r,opt,events


@pytest.mark.parametrize('dimension',[2,3,8])
@pytest.mark.parametrize('layout',['column','row'])
def test_qr_reuse_matches_independent_float64_oracle(dimension,layout):
    rng=np.random.default_rng(19+dimension);a=rng.normal(size=(dimension,dimension));q=np.linalg.qr(a)[0]
    cov=np.diag(np.linspace(1.,4.,dimension))+.02*(a@a.T)
    qt=torch.tensor(q,dtype=torch.float64)
    qt=qt.T.contiguous().T if layout=='column' else qt.contiguous()
    ct=torch.tensor(cov,dtype=torch.float64)
    expected_order=np.argsort(-np.diag(q.T@cov@q));expected=np.linalg.qr(cov@q[:,expected_order])[0]
    for policy in ['native','reuse']:
        actual,order=qr_basis(ct,qt,policy);assert order.tolist()==expected_order.tolist()
        signs=np.sign((actual.numpy()*expected).sum(0));np.testing.assert_allclose(actual.numpy()*signs,expected,rtol=1e-12,atol=1e-12)


def test_exact_ties_and_noncontiguous_input_keep_order_semantics():
    q=torch.eye(4).T;c=torch.diag(torch.tensor([1.,2.,1.,2.]))
    a,ao=qr_basis(c,q,'native');b,bo=qr_basis(c,q,'reuse')
    assert torch.equal(ao,bo);torch.testing.assert_close(a,b)
    with pytest.raises(ValueError):qr_basis(c,q,'undeclared')


def test_reuse_refresh_v_permutation_and_m_transport_independent_formulas():
    runtime,opt,_=fixture();p=opt.matrix.param_groups[0]['params'][0];s=opt.matrix.state[p];g=opt.matrix.param_groups[0]
    old=[q.clone() for q in s['Q']];cov=[x.clone() for x in s['GG']];m=s['exp_avg'].clone();v=s['exp_avg_sq'].clone()
    basis=[];orders=[]
    for c,q in zip(cov,old):
        cn,qn=c.double().numpy(),q.double().numpy();order=np.argsort(-np.diag(qn.T@cn@qn));orders.append(torch.tensor(order))
        qq=np.linalg.qr(cn@qn[:,order])[0];basis.append(torch.tensor(qq,dtype=torch.float32))
    refresh_reuse(opt.matrix,s,g)
    for actual,expected in zip(s['Q'],basis):
        signs=torch.sign((actual*expected).sum(0));torch.testing.assert_close(actual*signs,expected,rtol=2e-5,atol=1e-6)
    expected_m=s['Q'][0].T@(old[0]@m@old[1].T)@s['Q'][1]
    torch.testing.assert_close(s['exp_avg'],expected_m,rtol=2e-5,atol=1e-7)
    torch.testing.assert_close(s['exp_avg_sq'],v[orders[0]][:,orders[1]])
    assert s['research_qr_refreshes']==5 and s['research_strict_transports']==5


def test_shadow_state_never_aliases_source_or_base_and_preserves_gradients():
    runtime,source,_=fixture();base=copy.deepcopy(source.state_dict());before=streaming_digest(base)
    grad_ids=[id(p.grad) for p in runtime.model.parameters()]
    opt=shadow(runtime.model,runtime.config,base,'reuse')
    original_ptrs={t.data_ptr() for t in tensor_values(base)}|{t.data_ptr() for t in tensor_values(source.state_dict())}
    assert not original_ptrs.intersection({t.data_ptr() for t in tensor_values(opt.state_dict())})
    opt.step();assert streaming_digest(base)==before
    assert close_shadow(opt)['all_registered_released']
    assert grad_ids==[id(p.grad) for p in runtime.model.parameters()]


def test_complete_profile_exact_counts_all_owners_and_source_restoration():
    runtime,source,_=fixture();before=streaming_digest(dict(model=runtime.model.state_dict(),state=source.state_dict(),grad=[p.grad for p in runtime.model.parameters()]))
    result=run_profile(runtime,source)
    assert result['counts']==PROFILE_COUNTS and result['qualification']['passed']
    assert len(result['qualification']['physical'])==8
    assert result['cleanup']['nonmutation_pass'] and result['cleanup']['gradient_identity_pass'] and result['cleanup']['registered_release']['all_registered_released']
    assert all(r['all_registered_released'] for r in result['releases'])
    assert before==streaming_digest(dict(model=runtime.model.state_dict(),state=source.state_dict(),grad=[p.grad for p in runtime.model.parameters()]))


def test_timing_failure_restores_source_and_releases_diagnostic_banks():
    runtime,source,events=fixture();before=streaming_digest(runtime.model.state_dict())
    def fail(fn):raise RuntimeError('injected profile interruption')
    runtime.measure_call=fail
    with pytest.raises(RuntimeError,match='Bounded QR/profile failed'):run_profile(runtime,source)
    cleanup=next(p for e,p in events if e=='profile_cleanup')
    assert cleanup['nonmutation_pass'] and cleanup['registered_release']['all_registered_released']
    assert streaming_digest(runtime.model.state_dict())==before


def test_bad_retraction_geometry_is_rejected_without_gate_change(monkeypatch):
    import qr_reuse_profile_kernel as kernel
    runtime,source,events=fixture();original=kernel.qr_basis
    def bad(cov,q,policy):
        candidate,order=original(cov,q,policy)
        return (candidate*1.1 if policy=='reuse' else candidate),order
    monkeypatch.setattr(kernel,'qr_basis',bad)
    with pytest.raises(RuntimeError,match='failed prospective qualification'):run_profile(runtime,source)
    report=next(p for e,p in events if e=='qr_reuse_qualification_before_gate');assert not report['passed']
    assert events[-1][1]['nonmutation_pass'] and events[-1][1]['registered_release']['all_registered_released']


def test_contract_call_and_memory_budget_and_no_hidden_learning_grid():
    p=protocol();validate(recipe(),p)
    assert p['maximum_optimizer_updates']==25+245+1000 and p['maximum_all_optimizer_calls']==1270+32
    assert p['profile']['counts']['factor_basis_calls']+p['profile']['counts']['decomposition_calls']==240
    assert p['execution_order']==['hold200'] and p['evaluation_calls']==[]
    mem=p['profile']['memory'];assert sum(v for k,v in mem.items() if k!='maximum_tensor_purpose_bytes')<mem['maximum_tensor_purpose_bytes']
    changed=copy.deepcopy(p);changed['profile']['qualification']['following_gradient_update_relative']=.1
    with pytest.raises(ValueError,match='Undeclared'):validate(recipe(),changed)


def test_frozen_preparation_immutable_semantics_and_fresh_launch_only(tmp_path):
    import prepare_qr_reuse_profile as prep
    initial=tmp_path/'initial.json';prep.write(initial,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=182,parameters=144875648,cuda_initialized=False,snapshot_purpose_bytes=2230649344,model_sha256='1'*64))
    checks=tmp_path/'checks.json';prep.write(checks,dict(status='PASS_CPU_QR_REUSE_PROFILE',cuda_initialized=False,tests_passed=True))
    b=tmp_path/'bundle';prep.prepare(b,initial,182,PARENT/'token_cache_audit.json',checks)
    p=prep.read(b/'protocol.json');p['profile']['blocks']=4;prep.write(b/'protocol.json',p)
    m=prep.read(b/'manifest.json');m['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',m)
    with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
    with pytest.raises(ValueError,match='Fresh bundle'):prep.prepare(b,initial,182,PARENT/'token_cache_audit.json',checks)
