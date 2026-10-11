import torch
from benchmarks.execution_fingerprints import tensor_record,first_difference,gradient_records


def test_exact_bfloat_bytes_and_noncontiguous():
    x=torch.arange(12,dtype=torch.bfloat16).reshape(3,4).T
    assert tensor_record(x)==tensor_record(x.contiguous())
    y=x.clone();y[0,0]=1
    assert tensor_record(x)['sha256']!=tensor_record(y)['sha256']
    assert tensor_record(x)['dtype']=='torch.bfloat16'


def test_gradient_capture_is_read_only_and_detects_change():
    model=torch.nn.Linear(2,1);model(torch.ones(1,2)).sum().backward()
    before=[p.grad.clone() for p in model.parameters()]
    a=gradient_records(model)
    assert all(torch.equal(p.grad,v) for p,v in zip(model.parameters(),before))
    model.weight.grad[0,0]+=1
    b=gradient_records(model)
    assert first_difference(a,b)=='weight.sha256'


def test_first_difference_checks_missing_fields_and_ordered_steps():
    assert first_difference({'steps':[{'a':1},{'a':2}]},{'steps':[{'a':1},{'a':3}]})=='steps[1].a'
    assert first_difference({'a':1},{'b':1})=='keys'
    assert first_difference([1],[1,2])=='.length'
    assert first_difference({'a':[1,None]},{'a':[1,None]}) is None


def test_comparison_locates_output_before_later_state_changes():
    import copy
    from benchmarks.summarize_execution_diagnostic import compare,STAGES
    row={key:0 for key in STAGES};row['step']=1
    a=dict(initial_model_sha256='a',batch_plan=dict(first25_ids_sha256='b'),
           smoke_fingerprints=[row],measured=[row])
    b=copy.deepcopy(a)
    b['smoke_fingerprints'][0]['output']=1;b['smoke_fingerprints'][0]['basis']=2
    result=compare(a,b)
    assert result['phase']=='smoke_fingerprints' and result['field']=='output'
    assert compare(a,copy.deepcopy(a)) is None
