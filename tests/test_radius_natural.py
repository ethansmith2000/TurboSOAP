import torch
from benchmarks.radius_natural import POLICIES, optimizer_hash
from benchmarks.movement_retraction import warm
from benchmarks.radius_probe import policy_spec
from benchmarks.warm_retraction import retract
from soap_reference import ResearchSOAP

def test_actual_policies_match_qualified_shadow_math():
    gen=torch.Generator().manual_seed(517)
    q=torch.linalg.qr(torch.randn(12,12,generator=gen)).Q
    g=torch.randn(12,18,generator=gen); cov=g@g.T
    opt=ResearchSOAP([torch.nn.Parameter(torch.zeros_like(g))],basis_method='warm')
    group=opt.param_groups[0]; saved=(cov.clone(),q.clone())
    for actual,shadow in [('fro_cubic2_cap05','base05_cubic2'),('fro_cubic2_cap06','radius06_cubic2'),('fro_cubic3_cap06','radius06_cubic3')]:
        assert torch.equal(warm(cov,q,group,POLICIES[actual]),warm(cov,q,group,policy_spec(shadow,20)))
    assert torch.equal(cov,saved[0]) and torch.equal(q,saved[1])

def test_cubic3_reduces_rank_two_radius06_gram_error():
    a=torch.zeros(8,8);a[0,1]=.6;a[1,0]=-.6
    errors=[]
    for name in ('fro_cubic2_cap06','fro_cubic3_cap06'):
        z=retract(torch.eye(8)+a,POLICIES[name])
        errors.append(float(torch.linalg.matrix_norm(z.T@z-torch.eye(8),ord=2)))
    assert .005<errors[0]<.006 and errors[1]<.00003

def test_optimizer_hash_tracks_nested_basis_and_covariance():
    p=torch.nn.Parameter(torch.zeros(4,6)); opt=ResearchSOAP([p]);p.grad=torch.ones_like(p);opt.step()
    original=optimizer_hash(opt)
    opt.state[p]['GG'][0][0,0]+=1
    assert optimizer_hash(opt)!=original
    original=optimizer_hash(opt)
    opt.state[p]['Q'][1][0,0]+=.1
    assert optimizer_hash(opt)!=original
