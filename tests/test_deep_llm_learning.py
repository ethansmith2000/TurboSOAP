"""Full-plan boundaries and layer-observer nonmutation/failure/release checks."""
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import numpy as np
import pytest
import torch
from cifar_optimizers import build_optimizer
from deep_llm_learning_data import data_plan,input_sha,window_plan
from deep_llm_learning_observer import LayerObserver
from deep_llm_learning import compact_guard
from matched_training_runtime import streaming_digest


def test_full_plan_is_unique_and_evaluation_roles_are_distinct():
    order,panels=data_plan(300000,1000,193)
    ids=order.reshape(-1)
    assert order.shape==(35000,8) and len(set(ids))==280000
    assert np.array_equal(panels['seen_train'][1],ids[:64])
    assert not set(panels['unseen_train'][1]) & set(ids)
    assert panels['validation'][0]=='validation' and len(panels['validation'][1])==256
    same,_,windows=window_plan(300000,193)
    assert np.array_equal(order,same)
    assert len(windows)==8 and all(end-start==35000 for start,end in windows)
    assert all(windows[i][1]<=windows[i+1][0] for i in range(7))


def test_materialized_input_hash_has_one_canonical_integer_encoding():
    values=np.array([[1,2,50256]],dtype=np.int64)
    assert input_sha(values)==input_sha(torch.from_numpy(values))==input_sha(values.astype(np.int32))
    assert input_sha(values)!=input_sha(values[:,::-1])


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=torch.nn.ModuleList([torch.nn.Sequential(*[torch.nn.Linear(4,4,bias=False) for _ in range(4)]) for _ in range(12)])
        self.head=torch.nn.Linear(4,2)


def fixture():
    torch.manual_seed(193);model=Tiny()
    c=dict(matrix_lr=.0005,lr=.0005,weight_decay=.05,betas=(.9,.999),shampoo_beta=.99,
           precondition_mode='all',covariance_compute_dtype='float32',transport_precision='highest')
    opt=build_optimizer(model,c,'qr20');events=[]
    observer=LayerObserver(model,lambda event,row:events.append((event,row)),lambda:streaming_digest(torch.get_rng_state()),stages=(4,),lags=(1,2))
    return model,opt,observer,events


def advance(model,opt,observer):
    for call in range(1,5):
        for p in model.parameters():p.grad=torch.randn_like(p)
        opt.step();observer.tick(call,opt)


def test_layer_measurement_preserves_source_rng_and_releases_both_banks():
    model,opt,observer,events=fixture();advance(model,opt,observer)
    row=observer.observations[0]
    assert len(row['factors'])==24 and len(row['gradients'])==24 and len(row['adaptive'])==12
    assert {r['layer'] for r in row['factors']}=={0,5,11}
    assert row['source_unchanged'] and row['all_registered_released']
    report=observer.close()
    assert report['released_history_banks']==2 and report['live_bank_bytes']==0 and report['all_registered_released']
    assert all(int(s['step'])==3 for s in opt.matrix.state.values())


def test_observer_exception_releases_histories_and_detects_no_source_change():
    model,opt,observer,events=fixture()
    def fail(*args):raise RuntimeError('injected observation failure')
    observer.measure=fail
    with pytest.raises(RuntimeError,match='injected observation failure'):advance(model,opt,observer)
    assert observer.close()['all_registered_released'] and not observer.banks
    assert [row for event,row in events if event=='layer_observation_restoration'][-1]['source_unchanged']


def test_observer_rejects_source_mutation_without_leaving_history_banks():
    model,opt,observer,events=fixture();original=observer.measure
    def mutate(opt,call,owner):
        row=original(opt,call,owner);next(model.parameters()).add_(1.);return row
    observer.measure=mutate
    with pytest.raises(RuntimeError,match='mutated source'):advance(model,opt,observer)
    assert observer.close()['all_registered_released'] and not observer.banks


def test_compact_guard_checks_all_layers_and_rejects_wrong_source_age():
    model,opt,observer,events=fixture()
    for p in model.parameters():p.grad=torch.randn_like(p)
    opt.step();row=compact_guard(opt,model,1)
    assert len(row['layers'])==12 and all(r['factors']==8 for r in row['layers'].values())
    assert row['all_96_factor_guards_pass'] and row['all_48_counters_pass']
    next(iter(opt.matrix.state.values()))['step']=1
    with pytest.raises(RuntimeError,match='Cadence counter mismatch'):compact_guard(opt,model,1)
