import copy
import torch

from benchmarks.cifar_tracking_probe import TrackingProbe,factor_score
from benchmarks.cifar_optimizers import SplitOptimizer
from soap_reference import ResearchSOAP


def test_diagonal_covariance_score_is_zero_for_identity_basis():
    c=torch.diag(torch.tensor([1.,2.,3.]))
    assert factor_score(c,torch.eye(3))=={'residual':0.,'gram':0.}


def test_tracking_observer_selects_ages_and_does_not_mutate_state():
    torch.manual_seed(123);p=torch.nn.Parameter(torch.randn(3,5));a=torch.nn.Parameter(torch.ones(1))
    matrix=ResearchSOAP([p],basis_method='warm',precondition_frequency=2)
    matrix.param_groups[0]['basis_ns_iterations']=6
    opt=SplitOptimizer(matrix,torch.optim.AdamW([a]));probe=TrackingProbe(opt,[2])
    for _ in range(2):
        p.grad=torch.randn_like(p);opt.step();assert not probe.pending
    p.grad=torch.randn_like(p);opt.step();assert len(probe.pending)==1
    before=copy.deepcopy(matrix.state_dict());parameter=p.detach().clone()
    precision=torch.get_float32_matmul_precision();rows=probe.flush()
    assert torch.get_float32_matmul_precision()==precision and not probe.pending
    assert len(rows)==1 and rows[0]['age']==2 and rows[0]['executed_method']=='warm'
    assert set(rows[0]['update_alternatives'])=={'qr','warm1','warm2'}
    assert rows[0]['update_alternatives']['warm1']['relative_difference_from_executed']<1e-5
    assert torch.equal(p,parameter)
    after=matrix.state_dict()
    for key in ['exp_avg','exp_avg_sq']:
        assert torch.equal(before['state'][0][key],after['state'][0][key])
    for key in ['Q','GG']:
        assert all(torch.equal(x,y) for x,y in zip(before['state'][0][key],after['state'][0][key]))
    assert before['param_groups']==after['param_groups']
    original=probe.original;probe.close();assert matrix._research_refresh==original
