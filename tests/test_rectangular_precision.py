import pytest
import torch

from benchmarks.rectangular_precision_gate import ObservedSOAP, diagnose, project, world
from soap_reference import ResearchSOAP


@pytest.mark.parametrize('mode',['all','smaller_side'])
def test_independent_matrix_oracle_matches_projection_convention(mode):
    torch.manual_seed(81); p=torch.nn.Parameter(torch.randn(3,5))
    opt=ResearchSOAP([p],precondition_mode=mode)
    p.grad=torch.randn_like(p);opt.step();q=opt.state[p]['Q'];g=opt.param_groups[0]
    x=torch.randn_like(p)
    torch.testing.assert_close(project(x,q),opt._project_with_basis(x,q,g))
    torch.testing.assert_close(world(x,q),opt._project_back_with_basis(x,q,g))
    torch.testing.assert_close(world(project(x,q),q),x)


@pytest.mark.parametrize('mode',['all','smaller_side'])
def test_world_gradient_ema_oracle_survives_qr_refreshes(mode):
    torch.manual_seed(82);p=torch.nn.Parameter(torch.randn(3,5))
    opt=ObservedSOAP([p],basis_method='qr',precondition_mode=mode,precondition_frequency=2)
    opt.pending=[];ema={p:torch.zeros_like(p)}
    for step in range(7):
        p.grad=torch.randn_like(p);opt.step()
        if step:ema[p].mul_(.9).add_(p.grad,alpha=.1)
        d=diagnose(opt,ema)
        assert d['global_world_momentum'][0]['relative_error']<2e-6
        assert all(r['transport_world_relative_error']<2e-6 for r in d['refreshes'])
        assert not opt.pending


def test_world_ema_oracle_detects_corrupted_basis_even_without_refresh():
    torch.manual_seed(83);p=torch.nn.Parameter(torch.randn(3,5))
    opt=ObservedSOAP([p],basis_method='qr');opt.pending=[]
    p.grad=torch.randn_like(p);opt.step();p.grad=torch.randn_like(p);opt.step()
    ema={p:.1*p.grad.clone()}
    assert diagnose(opt,ema)['global_world_momentum'][0]['relative_error']<2e-6
    opt.state[p]['Q'][0].mul_(1.1)
    assert diagnose(opt,ema)['global_world_momentum'][0]['relative_error']>.09


def test_shadow_diagnostics_do_not_mutate_executed_state():
    torch.manual_seed(84);p=torch.nn.Parameter(torch.randn(3,5))
    opt=ObservedSOAP([p],basis_method='warm',precondition_frequency=1);opt.pending=[]
    p.grad=torch.randn_like(p);opt.step();p.grad=torch.randn_like(p);opt.step()
    state=opt.state[p]
    saved=[t.clone() for t in [p,state['exp_avg'],state['exp_avg_sq'],*state['Q'],*state['GG']]]
    prior=torch.get_float32_matmul_precision();d=diagnose(opt,{p:.1*p.grad.clone()})
    assert torch.get_float32_matmul_precision()==prior
    for a,b in zip(saved,[p,state['exp_avg'],state['exp_avg_sq'],*state['Q'],*state['GG']]):
        assert torch.equal(a,b)
    assert d['refreshes'][0]['update_shadow_relative_error']<2e-6
