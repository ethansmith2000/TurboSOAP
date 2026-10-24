"""Accumulation must preserve the intended mean gradient and single covariance update."""
import copy
import json
from pathlib import Path
import sys

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import pytest
import torch
from llm_accumulation import backward_accumulated,weighted_cost
from llm_batch_cost import recipe,verify_protocol,PARENT
from deep_llm_data import sha
from soap_reference import ResearchSOAP


def test_accumulation_matches_full_batch_gradient_and_single_global_clip():
    torch.manual_seed(201)
    a=torch.nn.Linear(3,2,bias=False).double();b=copy.deepcopy(a)
    batches=[torch.randn(2,3,dtype=torch.float64) for _ in range(16)]
    fn=lambda model,x:model(x).square().mean()
    loss,norm=backward_accumulated(a,batches,fn,count=16,clip_norm=.1)
    expected=fn(b,torch.cat(batches));expected.backward()
    expected_norm=torch.nn.utils.clip_grad_norm_(b.parameters(),.1,foreach=True)
    torch.testing.assert_close(loss,expected,atol=1e-13,rtol=1e-13)
    torch.testing.assert_close(norm,expected_norm,atol=1e-13,rtol=1e-13)
    torch.testing.assert_close(a.weight.grad,b.weight.grad,atol=1e-13,rtol=1e-13)


def test_next_global_call_clears_gradients_before_accumulating():
    model=torch.nn.Linear(2,1,bias=False)
    batches=[torch.ones(1,2),torch.full((1,2),3.)]
    fn=lambda m,x:m(x).mean()
    backward_accumulated(model,batches,fn,count=2,clip_norm=100.)
    before=model.weight.grad.clone()
    backward_accumulated(model,batches,fn,count=2,clip_norm=100.)
    torch.testing.assert_close(model.weight.grad,before)


def test_covariance_is_outer_of_global_mean_gradient_not_mean_of_micro_outers():
    model=torch.nn.Linear(2,2,bias=False)
    batches=[torch.tensor([[1.,0.]]),torch.tensor([[0.,2.]])]
    fn=lambda m,x:m(x).sum()/len(x)
    backward_accumulated(model,batches,fn,count=2,clip_norm=100.)
    grad=model.weight.grad.clone();opt=ResearchSOAP(model.parameters(),shampoo_beta=.99,transport_precision='highest')
    opt.step();state=opt.state[model.weight]
    torch.testing.assert_close(state['GG'][0],.01*(grad@grad.T))
    micro1=torch.tensor([[1.,0.],[1.,0.]]);micro2=torch.tensor([[0.,2.],[0.,2.]])
    assert not torch.allclose(state['GG'][0],.005*(micro1@micro1.T+micro2@micro2.T))
    assert int(state['step'])==0


def test_malformed_microbatch_group_is_rejected_before_backward():
    model=torch.nn.Linear(2,1);fn=lambda m,x:m(x).mean()
    with pytest.raises(ValueError,match='count'):backward_accumulated(model,[torch.ones(1,2)],fn,count=2,clip_norm=1.)
    with pytest.raises(ValueError,match='Equal'):backward_accumulated(model,[torch.ones(1,2),torch.ones(2,2)],fn,count=2,clip_norm=1.)
    assert all(p.grad is None for p in model.parameters())


def test_cost_weights_include_repair_and_require_measured_qr_samples():
    means={k:dict(samples=3,ms=v) for k,v in [('ordinary',20.),('warm',100.),('qr',600.)]}
    assert weighted_cost(means,'qr10','ms')==pytest.approx(78.)
    assert weighted_cost(means,'fast10','ms')==pytest.approx(38.)
    means['qr']['samples']=0
    with pytest.raises(ValueError,match='measured'):weighted_cost(means,'fast10','ms')


def test_recipe_changes_only_declared_batch_and_probe_metadata_and_pins_parent(tmp_path):
    parent=json.loads((PARENT/'config.json').read_text());c=recipe(parent)
    changed={k for k in c if c.get(k)!=parent.get(k)}
    assert changed=={'trial','batch_size','microbatch_size','gradient_accumulation_steps','primary_calls','disposable_calls','max_wall_seconds','transport_precision','matched_comparison'}
    assert c['batch_size']==c['microbatch_size']*c['gradient_accumulation_steps']==128
    (tmp_path/'config.json').write_text(json.dumps(c));pins={'manifest.json':sha(PARENT/'manifest.json')}
    (tmp_path/'parent_pins.json').write_text(json.dumps(pins))
    for name in ('data_audit.json','data_stat.json','initial_reference.json','backend_contract.json'):(tmp_path/name).write_bytes((PARENT/name).read_bytes())
    verify_protocol(tmp_path)
    c['microbatch_size']=4;(tmp_path/'config.json').write_text(json.dumps(c))
    with pytest.raises(RuntimeError,match='recipe'):verify_protocol(tmp_path)
    (tmp_path/'config.json').write_text(json.dumps(recipe(parent)));pins['manifest.json']='bad';(tmp_path/'parent_pins.json').write_text(json.dumps(pins))
    with pytest.raises(RuntimeError,match='parent changed'):verify_protocol(tmp_path)
