"""Post-trajectory initial-weight replay with exact model/optimizer/RNG rollback."""
import copy
from contextlib import contextmanager
import time
import torch
from torch.nn import functional as F
try:
    from .execution_fingerprints import tensor_record,gradient_records,first_difference
except ImportError:
    from execution_fingerprints import tensor_record,gradient_records,first_difference


@contextmanager
def initial_state_scope(model,initial,initial_rng,get_rng,restore_rng,invariants):
    before=invariants();current_rng=get_rng();current=copy.deepcopy(model.state_dict())
    snapshot_bytes=sum(t.numel()*t.element_size() for t in current.values())
    try:
        model.load_state_dict(initial);restore_rng(initial_rng)
        yield dict(before=before,current_rng=current_rng,snapshot_bytes=snapshot_bytes)
    finally:
        model.load_state_dict(current);restore_rng(current_rng)
        assert invariants()==before,'Initial replay mutated model/optimizer/RNG after restoration'


def run_initial_replay(*,model,active,initial,initial_rng,get_rng,restore_rng,
                       make_batch,first_record,invariants):
    started=time.monotonic()
    with initial_state_scope(model,initial,initial_rng,get_rng,restore_rng,invariants) as scope:
        x,target=make_batch()
        input_record=tensor_record(x);target_record=tensor_record(target)
        assert input_record==first_record['input'] and target_record==first_record['target'],'Rebuilt first input/target differs'
        retained_bytes=scope['snapshot_bytes']+x.numel()*x.element_size()+target.numel()*target.element_size()
        assert retained_bytes<120*2**20,'Declared replay retained-tensor RAM limit exceeded'
        restore_rng(scope['current_rng'])
        held_before=invariants()
        def backward_only():
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=active(x);loss=F.cross_entropy(logits.float(),target)
            loss.backward();torch.cuda.synchronize()
            return dict(output=tensor_record(logits),loss=float(loss.detach()),grad=gradient_records(model))
        plain=backward_only()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=active(x);loss=F.cross_entropy(logits.float(),target)
            loss.backward();torch.cuda.synchronize()
        profiled=dict(output=tensor_record(logits),loss=float(loss.detach()),grad=gradient_records(model))
        held_after=invariants();assert held_before==held_after,'Held initial-state replay mutated model/optimizer/RNG'
        kernels={}
        for event in prof.events():
            if str(event.device_type)=='DeviceType.CUDA':kernels[event.name]=kernels.get(event.name,0)+1
        assert kernels,'No initial-state runtime CUDA provenance'
        actual=dict(output=first_record['output'],loss=first_record['loss'],grad=first_record['grad_before_clip'])
        result=dict(scope='initial weights and regenerated exact first batch, after primary trajectory; no optimizer step',
            rebuilt_input=input_record,rebuilt_target=target_record,retained_tensor_bytes=retained_bytes,
            snapshot_bytes=scope['snapshot_bytes'],plain=plain,profiled=profiled,
            actual_first_vs_plain=first_difference(actual,plain),actual_first_vs_profiled=first_difference(actual,profiled),
            plain_vs_profiled=first_difference(plain,profiled),kernel_names_counts=kernels,
            held_invariants_before=held_before,held_invariants_after=held_after,
            final_invariants_before=scope['before'])
        del prof,plain,profiled,logits,loss,x,target
    result.update(final_invariants_after=invariants(),seconds=time.monotonic()-started)
    assert result['final_invariants_before']==result['final_invariants_after']
    return result
