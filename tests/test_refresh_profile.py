from contextlib import nullcontext
import pytest
import torch
from soap_reference import ResearchSOAP
from movement_retraction import configure_matrix
from refresh_profile_instrumentation import instrument, make_functions
from refresh_profile_contract import protocol, recipe, validate


def same(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


@pytest.mark.parametrize('shape',[(4,6),(8,8)])
@pytest.mark.parametrize('method',['qr','warm'])
def test_complete_state_and_parameter_parity_through_two_repairs(shape,method):
    gen=torch.Generator().manual_seed(44)
    initial=torch.randn(shape,generator=gen);ps=[torch.nn.Parameter(initial.clone()) for _ in range(2)]
    opts=[ResearchSOAP([p],basis_method=method,precondition_frequency=20,
        hard_reset_interval=200,transport_precision='highest') for p in ps]
    if method=='warm':
        for opt in opts:configure_matrix(opt,'fro_cubic2_cap05')
    for _ in range(421):
        grad=torch.randn(shape,generator=gen)
        for i,(p,opt) in enumerate(zip(ps,opts)):
            p.grad=grad.clone()
            with instrument(opt) if i else nullcontext():opt.step()
        same(ps[0],ps[1]);same(opts[0].state_dict(),opts[1].state_dict())
        assert all(torch.isfinite(q).all() for q in opts[1].state[ps[1]]['Q'])
    assert '_research_refresh' not in opts[1].__dict__


def test_scope_erasure_preserves_all_function_asts():
    assert len(make_functions())==6


def test_instance_patches_restore_even_on_failure():
    p=torch.nn.Parameter(torch.zeros(4,6));opt=ResearchSOAP([p],basis_method='warm')
    configure_matrix(opt,'fro_cubic2_cap05');before=dict(opt.__dict__)
    with pytest.raises(RuntimeError):
        with instrument(opt):raise RuntimeError('sentinel')
    assert opt.__dict__.keys()==before.keys()
    assert all(opt.__dict__[k] is v for k,v in before.items())


def test_protocol_rejects_precision_or_profile_window_change():
    base=dict(seed=140,width=1536,checkpoint=False)
    c=recipe(base);declared=protocol();validate(c,declared,base)
    declared['profile_calls'][0]=261
    with pytest.raises(ValueError):validate(c,declared,base)
    declared=protocol();c['basis_compute_dtype']='bfloat16'
    with pytest.raises(ValueError):validate(c,declared,base)
