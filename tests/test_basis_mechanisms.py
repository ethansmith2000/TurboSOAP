"""Mechanism expectations independent of a training trajectory."""
import copy
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import audit_basis_mechanisms as audit


@pytest.fixture
def module():
    audit.block_cuda()
    return audit.load_source()


def branch_fixture(module,metrics, state=None):
    opt,g=audit.make_optimizer(module,8,'float32')
    c=torch.diag(torch.arange(1,9,dtype=torch.float32));q=torch.eye(8)
    candidate=q.clone();angle=torch.tensor(.2);co,si=angle.cos(),angle.sin()
    candidate[0,0]=co;candidate[1,1]=co;candidate[0,1]=-si;candidate[1,0]=si
    state=copy.deepcopy(state) if state is not None else dict(basis_scales=[.7],basis_direction_moments=[{}])
    def independent_candidate(C,Q,eta,normalizer,group,moment):
        module._soap_normalize_direction_by_mode(torch.ones_like(Q),normalizer,moment=moment,beta=.99)
        return candidate.clone()
    out=opt._run_stiefel_tracker(state,[c],[q],state['basis_scales'],g,independent_candidate,metrics)[0]
    return opt,state,out,q,candidate


def test_acceptance_changes_with_metrics_but_rejected_ema_is_committed(module):
    on,s_on,out_on,q,candidate=branch_fixture(module,True)
    off,s_off,out_off,_,_=branch_fixture(module,False)
    assert torch.equal(out_on,q) and torch.equal(out_off,candidate)
    assert on._basis_stats_accum['accepted_steps']==0 and on._basis_stats_accum['rejected_steps']==1
    assert abs(s_on['basis_scales'][0]-.49)<1e-12 and s_off['basis_scales']==[.7]
    assert audit.digest(s_on['basis_direction_moments'])==audit.digest(s_off['basis_direction_moments'])
    assert s_on['basis_direction_moments'][0]['col_step']==1
    assert not getattr(off,'_basis_stats_accum',None)


def test_second_rejected_candidate_uses_existing_ema_history(module):
    _,state,_,_,_=branch_fixture(module,True)
    first=state['basis_direction_moments'][0]['col'].clone()
    _,second,out,q,_=branch_fixture(module,True,state)
    assert torch.equal(out,q)
    m=second['basis_direction_moments'][0]
    assert m['col_step']==2
    assert torch.allclose(m['col'],.99*first+.01*torch.ones_like(first))
    assert state['basis_direction_moments'][0]['col_step']==1


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_declared_singular_cap_is_not_a_strict_gram_guarantee(module,dtype):
    x=1.25*torch.eye(8,dtype=dtype)
    y,record=audit.retraction_record(module,x)
    assert record['scaled_singular_max']<=1.250001
    assert record['output']['gram_error_fp32']>.05
    # Independent scalar result for the ideal boundary polynomial.
    s=1.25;p=(15*s-10*s**3+3*s**5)/8
    assert abs(record['spectral_polynomial_gram_fp64']-abs(p*p-1))<1e-5
    assert torch.isfinite(y).all()


def test_contraction_is_not_fixed_by_one_retraction(module):
    _,record=audit.retraction_record(module,.5*torch.eye(8))
    assert record['actual_scale']==1
    assert record['output']['gram_error_fp32']>.3


def test_dense_control_movement_scales_with_dimension(module):
    small=audit.dense_control(module,8,'float32');large=audit.dense_control(module,32,'float32')
    assert abs(large['rotation_frobenius']/small['rotation_frobenius']-4)<1e-5
    assert small['construction']==large['construction']


def test_pinned_loader_rejects_changed_source(tmp_path):
    p=tmp_path/'changed.py';p.write_text('raise AssertionError("must not execute")\n')
    with pytest.raises(ValueError,match='digest mismatch'):audit.load_source(p)


def test_cpu_guard_prohibits_cuda(module):
    with pytest.raises(RuntimeError,match='prohibited'):torch.cuda._lazy_init()
    assert not torch.cuda.is_initialized()
