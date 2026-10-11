import copy
import json
from pathlib import Path
import pytest
import torch
from benchmarks.initial_state_replay import initial_state_scope
from benchmarks.cudnn_deterministic_contract import validate_child_change
from benchmarks.cifar_optimizer_benchmark import state_digest

BASE=Path('/workspace/optimizer_replay_results/cifar_soap_benchmark_off_20261008')
ROOT=Path(__file__).resolve().parents[1]


def setup():
    model=torch.nn.Linear(3,2);initial=copy.deepcopy(model.state_dict());initial_rng=torch.get_rng_state()
    opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    model(torch.randn(2,3)).sum().backward();opt.step()
    def invariants():return dict(model=state_digest(model.state_dict()),optimizer=state_digest(opt.state_dict()),rng=state_digest(torch.get_rng_state()))
    return model,opt,initial,initial_rng,invariants


def test_replay_restores_exact_weights_optimizer_and_rng():
    model,opt,initial,rng,inv=setup();before=inv();parameter_ids=[id(p) for p in model.parameters()]
    with initial_state_scope(model,initial,rng,torch.get_rng_state,torch.set_rng_state,inv) as scope:
        assert state_digest(model.state_dict())==state_digest(initial)
        assert torch.equal(torch.get_rng_state(),rng)
        assert scope['snapshot_bytes']==sum(p.numel()*p.element_size() for p in model.parameters())
        model.zero_grad(set_to_none=True);model(torch.randn(2,3)).sum().backward()
    assert inv()==before and parameter_ids==[id(p) for p in model.parameters()]


def test_exception_still_restores_state():
    model,opt,initial,rng,inv=setup();before=inv()
    with pytest.raises(RuntimeError,match='observer error'):
        with initial_state_scope(model,initial,rng,torch.get_rng_state,torch.set_rng_state,inv):
            with torch.no_grad():model.weight.add_(10)
            torch.rand(5);raise RuntimeError('observer error')
    assert inv()==before


def test_optimizer_mutation_is_detected_without_recovery_snapshot():
    model,opt,initial,rng,inv=setup();final_model=state_digest(model.state_dict())
    with pytest.raises(AssertionError,match='after restoration'):
        with initial_state_scope(model,initial,rng,torch.get_rng_state,torch.set_rng_state,inv):
            opt.zero_grad();model(torch.ones(2,3)).sum().backward();opt.step()
    assert state_digest(model.state_dict())==final_model


def test_primary_scope_audit_rejects_second_flag_or_step_math_change():
    before=(BASE/'source/benchmarks/cifar_benchmark_off_child.py').read_text()
    after=(ROOT/'benchmarks/cifar_cudnn_deterministic_child.py').read_text()
    assert validate_child_change(before,after)
    with pytest.raises(ValueError):validate_child_change(before,after.replace('cudnn.benchmark=False','cudnn.benchmark=True'))
    with pytest.raises(ValueError):validate_child_change(before,after.replace('support.lr_at(index+1','support.lr_at(index+2'))
    with pytest.raises(ValueError):validate_child_change(before,after.replace('cudnn.deterministic=True','cudnn.deterministic=False'))


def test_qualifier_requires_restoration_but_records_cold_warm_difference_separately():
    from benchmarks.summarize_cudnn_deterministic import qualify
    import json
    base=Path('/workspace/optimizer_replay_results/cifar_soap_execution_diagnostic_20261007')
    labels=['plan420_flush10_r0','plan420_flush25_r0','plan420_flush10_r1','plan3500_flush25_r1']
    runs=[json.loads((base/'contexts'/label/'result.json').read_text()) for label in labels]
    for r in runs:
        for phase in ['backend_before_smoke','backend_after_measured']:
            r[phase]['cudnn_flags'].update(benchmark=False,deterministic=True)
        r['initial_state_replay']=dict(final_invariants_before={'state':'a'},final_invariants_after={'state':'a'},held_invariants_before={'state':'b'},held_invariants_after={'state':'b'},rebuilt_input=r['measured'][0]['input'],rebuilt_target=r['measured'][0]['target'],retained_tensor_bytes=119*2**20,actual_first_vs_plain='grad.head.weight',actual_first_vs_profiled='grad.head.weight',plain_vs_profiled=None)
    assert qualify(runs)['qualified']
    assert not qualify(runs[:3])['qualified']
    bad=copy.deepcopy(runs);bad[0]['initial_state_replay']['final_invariants_after']['state']='changed'
    assert not qualify(bad)['qualified']
    bad=copy.deepcopy(runs);bad[0]['initial_state_replay']['rebuilt_input']['sha256']='changed'
    assert not qualify(bad)['qualified']
    bad=copy.deepcopy(runs);bad[0]['initial_state_replay']['retained_tensor_bytes']=120*2**20
    assert not qualify(bad)['qualified']
