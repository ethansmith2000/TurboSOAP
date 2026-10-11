import copy
import torch
from benchmarks.radius_probe import policy_spec,make_proxy,refresh,frozen_covariance_chain
from benchmarks.movement_retraction import configure_matrix
from soap_reference import ResearchSOAP


def fixture():
    torch.manual_seed(61)
    p=torch.nn.Parameter(torch.randn(4,6))
    opt=ResearchSOAP([p],basis_method='warm',transport_precision='highest',precondition_frequency=20)
    configure_matrix(opt,'fro_cubic2_cap05')
    qs=[torch.linalg.qr(torch.randn(n,n)).Q for n in [4,6]]
    cov=[]
    for n in [4,6]:
        x=torch.randn(n,n);cov.append(x@x.T+torch.eye(n))
    old=dict(q=qs,cov=cov,m=torch.randn(4,6),v=torch.rand(4,6)+.1)
    return opt,old,opt.param_groups[0]


def test_alternating_work_includes_strong_first_refresh_after_each_qr():
    for age in [20,60,180,220,420]:assert policy_spec('radius06_alternate',age).iterations==3
    for age in [40,80,240,440]:assert policy_spec('radius06_alternate',age).iterations==2
    assert sum(policy_spec('radius06_alternate',age).iterations for age in range(20,200,20))==23


def test_baseline_is_exact_and_all_shadows_preserve_incoming_state():
    opt,old,group=fixture();saved=copy.deepcopy(old)
    state=dict(Q=old['q'],GG=old['cov'],exp_avg=old['m'],exp_avg_sq=old['v'],step=20)
    ResearchSOAP._research_refresh(opt,state,group)
    result=refresh(make_proxy(opt,'base05_cubic2'),old,group,20,'base05_cubic2')
    assert torch.equal(result['m'],state['exp_avg'])
    assert all(torch.equal(a,b) for a,b in zip(result['q'],state['Q']))
    for name in ['radius06_cubic2','radius06_cubic3','radius06_alternate','qr']:
        refresh(make_proxy(opt,name),old,group,20,name)
    for key in ['m','v']:assert torch.equal(old[key],saved[key])
    for key in ['q','cov']:assert all(torch.equal(a,b) for a,b in zip(old[key],saved[key]))


def test_radius_methods_use_real_qr_reset_at_200():
    opt,old,group=fixture()
    a=refresh(make_proxy(opt,'radius06_cubic3'),old,group,200,'radius06_cubic3')
    z=refresh(make_proxy(opt,'qr'),old,group,200,'qr')
    assert torch.equal(a['m'],z['m']) and torch.equal(a['v'],z['v'])
    assert all(torch.equal(x,y) for x,y in zip(a['q'],z['q']))


def test_frozen_chain_records_accumulated_error_without_mutating_control():
    opt,old,group=fixture();saved=copy.deepcopy(old);precision=torch.get_float32_matmul_precision()
    try:result=frozen_covariance_chain(make_proxy(opt,'radius06_cubic3'),old,group,180,'radius06_cubic3')
    finally:torch.set_float32_matmul_precision(precision)
    assert len(result['world_m_errors'])==9 and result['passes']
    assert torch.equal(old['m'],saved['m'])
