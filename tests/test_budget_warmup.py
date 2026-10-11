import copy
import random
import torch
import pytest
from benchmarks.budget_warmup import discarded_warmup
from benchmarks.cifar_optimizer_benchmark import state_digest


def fixture():
    torch.manual_seed(13)
    random.seed(17)
    model = torch.nn.Linear(3, 2)
    initial = copy.deepcopy(model.state_dict())
    rng = lambda: (torch.get_rng_state(), random.getstate())
    def restore(state):
        torch.set_rng_state(state[0])
        random.setstate(state[1])
    initial_rng = rng()
    def step(index, opt):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(5, 3)
        (model(x).square().mean()*random.random()).backward()
        opt.step()
    options = dict(steps=4, model=model, initial_model=initial,
                   initial_rng=initial_rng,
                   make_optimizer=lambda: torch.optim.Adam(model.parameters(), lr=.001),
                   train_step=step, restore_rng=restore, rng_state=rng,
                   digest=state_digest, synchronize=lambda: None,
                   validate=lambda opt: dict(finite=all(torch.isfinite(p).all().item() for p in model.parameters())),
                   counters=lambda opt: [int(v['step']) for v in opt.state.values()],
                   telemetry=lambda: {})
    return model, options


def test_discarded_updates_restore_rng_weights_and_fresh_optimizer_trajectory():
    model, options = fixture()
    initial_hash = state_digest(model.state_dict())
    control_opt = options['make_optimizer']()
    options['train_step'](0, control_opt)
    control = state_digest(model.state_dict())
    result = discarded_warmup(**options)
    assert result['optimizer_counters'] == [4, 4]
    assert result['guard']['finite']
    assert result['model_sha256'] != initial_hash
    assert result['restored_model_sha256'] == initial_hash
    opt = options['make_optimizer']()
    assert not opt.state
    options['train_step'](0, opt)
    assert state_digest(model.state_dict()) == control


def test_warmup_repeats_reproduce_disposable_optimizer_and_model_state():
    _, options = fixture()
    first = discarded_warmup(**options)
    second = discarded_warmup(**options)
    for key in ['model_sha256', 'optimizer_sha256', 'optimizer_counters',
                'restored_model_sha256', 'restored_rng_sha256']:
        assert first[key] == second[key]


def test_exception_restores_original_state_without_retaining_warm_optimizer():
    model, options = fixture()
    initial_hash = state_digest(model.state_dict())
    original_step = options['train_step']
    def fail(index, opt):
        original_step(index, opt)
        raise RuntimeError('interrupted warmup')
    options['train_step'] = fail
    with pytest.raises(RuntimeError, match='interrupted'):
        discarded_warmup(**options)
    assert state_digest(model.state_dict()) == initial_hash
    assert state_digest(options['rng_state']()) == state_digest(options['initial_rng'])


def test_warmup_rejects_zero_updates():
    _, options = fixture()
    with pytest.raises(ValueError):
        discarded_warmup(**{**options, 'steps': 0})
