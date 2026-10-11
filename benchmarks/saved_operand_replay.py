"""Original model graph replay with an existing saved operand, then held head."""
from contextlib import nullcontext
import time
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .initial_state_replay import initial_state_scope
    from .execution_fingerprints import tensor_record,gradient_records,first_difference,loaded_code
    from .benchmark_off_contract import finite_records
    from .saved_operand_abi import classifier_abi,select_operand,OperandUnavailable
except ImportError:
    from initial_state_replay import initial_state_scope
    from execution_fingerprints import tensor_record,gradient_records,first_difference,loaded_code
    from benchmark_off_contract import finite_records
    from saved_operand_abi import classifier_abi,select_operand,OperandUnavailable


class HeldHead(nn.Module):
    def __init__(self,head):super().__init__();self.head=head
    def forward(self,operand):return self.head(operand)


def head_execution(record):
    return dict(output=record['output'],loss=record['loss'],grad={k:v for k,v in record['grad'].items() if k in ['head.weight','head.bias']})


def interpretation(actual,warm,captures,heads):
    checks=dict(complete=len(captures)==3 and len(heads)==3,
                primary_gradients_exact=all(first_difference(actual,c['execution']) is None for c in captures),
                warm_gradients_exact=all(first_difference(warm,c['execution']) is None for c in captures),
                original_capture_calls_exact=bool(captures) and all(first_difference(captures[0],c) is None for c in captures[1:]),
                operand_available=len(captures)==3 and all(c.get('operand') is not None for c in captures),
                held_head_matches=bool(captures) and len(heads)==3 and all(first_difference(head_execution(captures[0]['execution']),h) is None for h in heads),
                held_head_calls_exact=len(heads)==3 and all(first_difference(heads[0],h) is None for h in heads[1:]))
    return dict(supported=all(checks.values()),checks=checks,
                actual_vs_captures=[first_difference(actual,c['execution']) for c in captures],
                warm_vs_captures=[first_difference(warm,c['execution']) for c in captures],
                captures_vs_head=[first_difference(head_execution(captures[0]['execution']),h) for h in heads] if captures else [])


def run_saved_operand_replay(*,model,active,initial,initial_rng,get_rng,restore_rng,make_batch,first_record,warm_record,invariants,progress):
    started=time.monotonic();result=dict(status='running',captures=[],heads=[],scope='Original compiled model; saved operand reference retained until backward finishes; no optimizer step')
    with initial_state_scope(model,initial,initial_rng,get_rng,restore_rng,invariants) as scope:
        x,target=make_batch();inp=tensor_record(x);tar=tensor_record(target)
        assert inp==first_record['input'] and tar==first_record['target'],'Rebuilt initial input/target differs'
        restore_rng(scope['current_rng']);held_before=invariants()
        result.update(rebuilt_input=inp,rebuilt_target=tar,held_invariants_before=held_before,final_invariants_before=scope['before'])
        try:abi=classifier_abi(loaded_code(),x.shape[0],model.embed_dim,model.num_classes);result['abi']=abi
        except OperandUnavailable as exc:abi=None;result['abi_failure']=str(exc)
        def capture_call(profile=False,retain=False):
            model.zero_grad(set_to_none=True)
            window=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if profile else nullcontext()
            with window as prof:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=active(x);loss=F.cross_entropy(logits.float(),target)
                # Metadata/Python references only here. No tensor hashes, clones,
                # new graph outputs, hooks, retain_graph, or tensor math before backward.
                operand=None;selection=None;failure=result.get('abi_failure')
                if abi is not None:
                    try:operand,selection=select_operand(logits.grad_fn,abi)
                    except OperandUnavailable as exc:failure=str(exc)
                loss.backward();torch.cuda.synchronize()
            record=dict(execution=dict(output=tensor_record(logits),loss=float(loss.detach()),grad=gradient_records(model)),
                        operand=tensor_record(operand) if operand is not None else None,selection=selection,selection_failure=failure)
            assert finite_records(record),'Nonfinite saved-operand diagnostic'
            held=operand.detach().clone() if retain and operand is not None else None
            kernels={}
            if profile:
                for event in prof.events():
                    if str(event.device_type)=='DeviceType.CUDA':kernels[event.name]=kernels.get(event.name,0)+1
                assert kernels,'Missing original replay CUDA provenance'
            return record,held,kernels
        first,held,_=capture_call(retain=True);result['captures'].append(first)
        retained=scope['snapshot_bytes']+x.numel()*x.element_size()+target.numel()*target.element_size()+(held.numel()*held.element_size() if held is not None else 0)
        assert retained<128*2**20,'Retained RAM cap exceeded'
        result.update(retained_tensor_bytes=retained,snapshot_bytes=scope['snapshot_bytes'],held_operand=tensor_record(held) if held is not None else None);progress(result)
        for profile in [False,True]:
            record,_,kernels=capture_call(profile);result['captures'].append(record)
            if profile:result['capture_kernel_names_counts']=kernels
            progress(result)
        if held is not None:
            head=torch.compile(HeldHead(model.head),mode='default',fullgraph=True,dynamic=False)
            for profile in [False,False,True]:
                model.zero_grad(set_to_none=True)
                window=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if profile else nullcontext()
                with window as prof:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=head(held);loss=F.cross_entropy(logits.float(),target)
                    loss.backward();torch.cuda.synchronize()
                record=head_execution(dict(output=tensor_record(logits),loss=float(loss.detach()),grad=gradient_records(model)))
                assert finite_records(record);result['heads'].append(record)
                if profile:
                    kernels={}
                    for event in prof.events():
                        if str(event.device_type)=='DeviceType.CUDA':kernels[event.name]=kernels.get(event.name,0)+1
                    assert kernels;result['head_kernel_names_counts']=kernels
                progress(result)
            assert tensor_record(held)==result['held_operand'],'Held operand mutated'
            del head,logits,loss
        actual=dict(output=first_record['output'],loss=first_record['loss'],grad=first_record['grad_before_clip'])
        result['interpretation']=interpretation(actual,warm_record,result['captures'],result['heads'])
        result['held_invariants_after']=invariants();assert result['held_invariants_after']==held_before
        del x,target,held
    result.update(status='completed',final_invariants_after=invariants(),seconds=time.monotonic()-started)
    assert result['final_invariants_after']==result['final_invariants_before'];progress(result)
    return result
