import torch
import pytest
from benchmarks.variance_probe import diagonal_variance, VarianceProbe
from benchmarks.cifar_variance_screen import configure_variance
from benchmarks.prepare_variance_screen import arms
from benchmarks.movement_retraction import configure_matrix
from soap_reference import ResearchSOAP

def test_overlap_orientation_matches_full_diagonal_kronecker_oracle():
    gen=torch.Generator().manual_seed(716)
    left=torch.linalg.qr(torch.randn(3,3,generator=gen)).Q
    right=torch.linalg.qr(torch.randn(4,4,generator=gen)).Q
    v=torch.rand(3,4,generator=gen);old=dict(q=[torch.eye(3),torch.eye(4)],v=v)
    group=dict(basis_method='warm',hard_reset_interval=200)
    actual=diagonal_variance(old,[left,right],group,20,'overlap')
    transform=torch.kron(left.T.contiguous(),right.T.contiguous())
    exact=(transform@torch.diag(v.flatten())@transform.T).diagonal().reshape(3,4)
    torch.testing.assert_close(actual,exact,rtol=2e-6,atol=1e-7)
    torch.testing.assert_close(actual.sum(),v.sum(),rtol=2e-6,atol=1e-7)

def test_same_gradients_change_only_variance_not_basis_or_first_moment():
    gen=torch.Generator().manual_seed(918)
    ps=[torch.nn.Parameter(torch.randn(4,6,generator=gen)) for _ in range(2)]
    opts=[ResearchSOAP([p],basis_method='warm',transport_precision='highest',
                      precondition_frequency=2,hard_reset_interval=6) for p in ps]
    for opt,policy in zip(opts,['permutation','overlap']):
        configure_matrix(opt,'fro_cubic2_cap05');configure_variance(opt,policy)
    for _ in range(19):
        grad=torch.randn(4,6,generator=gen)
        for p,opt in zip(ps,opts):p.grad=grad.clone();opt.step()
        a,z=[opt.state[p] for opt,p in zip(opts,ps)]
        assert torch.equal(a['exp_avg'],z['exp_avg'])
        assert all(torch.equal(x,y) for x,y in zip(a['Q'],z['Q']))
    assert not torch.equal(a['exp_avg_sq'],z['exp_avg_sq'])
    assert bool((z['exp_avg_sq']>=0).all())

def test_permutation_reset_and_warm_carry_are_exact_without_mutation():
    v=torch.arange(6.).reshape(2,3);old=dict(q=[torch.eye(2),torch.eye(3)],
        cov=[torch.diag(torch.tensor([1.,2.])),torch.diag(torch.tensor([1.,3.,2.]))],v=v)
    g=dict(basis_method='warm',hard_reset_interval=200)
    assert diagonal_variance(old,old['q'],g,20,'permutation') is v
    expected=v.index_select(0,torch.tensor([1,0])).index_select(1,torch.tensor([1,2,0]))
    assert torch.equal(diagonal_variance(old,old['q'],g,200,'permutation'),expected)
    assert torch.equal(v,torch.arange(6.).reshape(2,3))
    with pytest.raises(ValueError):diagonal_variance(old,old['q'],g,20,'invalid')

def test_arm_plan_keeps_two_betas_and_four_exact_repeats():
    aa=arms();assert len(aa)==12 and len({x['label'] for x in aa})==12
    assert len([x for x in aa if x.get('repeat_of')])==4
    assert {x['covariance_beta'] for x in aa}=={.99,.999}
    assert all(x['matrix_beta2']==.999 for x in aa)

def test_probe_validates_actual_warm_and_qr_variance_without_state_mutation():
    from types import SimpleNamespace
    gen=torch.Generator().manual_seed(321)
    for policy in ['permutation','overlap']:
        p=torch.nn.Parameter(torch.zeros(4,6))
        opt=ResearchSOAP([dict(params=[p],param_names=['matrix'])],basis_method='warm',
            transport_precision='highest',precondition_frequency=2,hard_reset_interval=6)
        configure_matrix(opt,'fro_cubic2_cap05');configure_variance(opt,policy)
        probe=VarianceProbe(SimpleNamespace(matrix=opt),[2,4,6]);events=[]
        for _ in range(7):
            p.grad=torch.randn(4,6,generator=gen);opt.step()
            if probe.pending:
                before=opt.state[p]['exp_avg_sq'].clone()
                rows=probe.flush();assert len(rows)==1
                assert rows[0]['variance']['formula_exact'] and rows[0]['variance']['minimum']>=0
                assert torch.equal(before,opt.state[p]['exp_avg_sq'])
                events.append(rows[0]['executed_method'])
        assert events==['warm','warm','qr']
        probe.close()
