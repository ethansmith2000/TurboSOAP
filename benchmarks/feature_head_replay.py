"""Separate post-trajectory feature-return graph and held-feature head replay."""
from contextlib import nullcontext
import time
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .initial_state_replay import initial_state_scope
    from .execution_fingerprints import tensor_record,gradient_records,first_difference
    from .benchmark_off_contract import finite_records
except ImportError:
    from initial_state_replay import initial_state_scope
    from execution_fingerprints import tensor_record,gradient_records,first_difference
    from benchmark_off_contract import finite_records


class FeatureReturn(nn.Module):
    """Same ordered ViT forward; explicit autocast operand is also returned."""
    def __init__(self,model):
        super().__init__();self.base=model
        assert not model.gradient_checkpointing,'Frozen diagnostic excludes checkpointing'

    def forward(self,x):
        m=self.base;batch=x.shape[0]
        x=m.patch_embed(x)
        x=torch.cat([m.cls_token.expand(batch,-1,-1),x],dim=1)
        x=x+m.pos_embed;x=m.pos_drop(x)
        for block in m.blocks:x=block(x)
        x=m.norm(x);features=x[:,0]
        operand=features.to(torch.bfloat16)
        return m.head(operand),features,operand


class HeldHead(nn.Module):
    def __init__(self,head):super().__init__();self.head=head
    def forward(self,operand):return self.head(operand)


def head_execution(record):
    return dict(output=record['output'],loss=record['loss'],
                grad={k:v for k,v in record['grad'].items() if k in ['head.weight','head.bias']})


def interpretation(actual,warm,features,heads):
    primary=[first_difference(actual,f['execution']) for f in features]
    warmed=[first_difference(warm,f['execution']) for f in features]
    feature_repeat=[first_difference(features[0],f) for f in features[1:]]
    head_matches=[first_difference(head_execution(features[0]['execution']),h) for h in heads]
    head_repeat=[first_difference(heads[0],h) for h in heads[1:]]
    checks=dict(primary_gradients_exact=all(x is None for x in primary),
                warm_gradients_exact=all(x is None for x in warmed),
                feature_calls_exact=all(x is None for x in feature_repeat),
                held_head_matches=all(x is None for x in head_matches),
                held_head_calls_exact=all(x is None for x in head_repeat))
    return dict(supported=all(checks.values()),checks=checks,actual_vs_features=primary,
                warm_vs_features=warmed,feature_repeat=feature_repeat,
                features_vs_held_head=head_matches,head_repeat=head_repeat)


def run_feature_head_replay(*,model,initial,initial_rng,get_rng,restore_rng,
                            make_batch,first_record,warm_record,invariants,progress):
    started=time.monotonic()
    result=dict(status='running',scope='Post-trajectory initial weights; separate feature-return and held-head compiled graphs; no optimizer step',
                features=[],heads=[])
    with initial_state_scope(model,initial,initial_rng,get_rng,restore_rng,invariants) as scope:
        x,target=make_batch();input_record=tensor_record(x);target_record=tensor_record(target)
        assert input_record==first_record['input'] and target_record==first_record['target'],'Rebuilt first input/target differs'
        restore_rng(scope['current_rng']);held_before=invariants()
        result.update(rebuilt_input=input_record,rebuilt_target=target_record,
                      final_invariants_before=scope['before'],held_invariants_before=held_before)
        active=torch.compile(FeatureReturn(model),mode='default',fullgraph=True,dynamic=False)
        def feature_call(profile=False,retain=False):
            model.zero_grad(set_to_none=True)
            window=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if profile else nullcontext()
            with window as prof:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,features,operand=active(x);loss=F.cross_entropy(logits.float(),target)
                loss.backward();torch.cuda.synchronize()
            record=dict(execution=dict(output=tensor_record(logits),loss=float(loss.detach()),grad=gradient_records(model)),
                        cls_features=tensor_record(features),head_operand=tensor_record(operand))
            kernels={}
            if profile:
                for event in prof.events():
                    if str(event.device_type)=='DeviceType.CUDA':kernels[event.name]=kernels.get(event.name,0)+1
                assert kernels,'No feature runtime CUDA provenance'
            assert finite_records(record),'Nonfinite feature diagnostic'
            held=operand.detach().clone() if retain else None
            return record,held,kernels
        first,held,_=feature_call(retain=True);result['features'].append(first)
        retained=scope['snapshot_bytes']+x.numel()*x.element_size()+target.numel()*target.element_size()+held.numel()*held.element_size()
        assert retained<128*2**20,'Declared purpose-specific retained RAM cap exceeded'
        result.update(retained_tensor_bytes=retained,snapshot_bytes=scope['snapshot_bytes'],held_operand=tensor_record(held));progress(result)
        second,_,_=feature_call();result['features'].append(second);progress(result)
        profiled,_,kernels=feature_call(profile=True);result['features'].append(profiled)
        result['feature_kernel_names_counts']=kernels;progress(result)
        head=torch.compile(HeldHead(model.head),mode='default',fullgraph=True,dynamic=False)
        def head_call(profile=False):
            model.zero_grad(set_to_none=True)
            window=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if profile else nullcontext()
            with window as prof:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=head(held);loss=F.cross_entropy(logits.float(),target)
                loss.backward();torch.cuda.synchronize()
            record=dict(output=tensor_record(logits),loss=float(loss.detach()),
                        grad={k:v for k,v in gradient_records(model).items() if k in ['head.weight','head.bias']})
            kernels={}
            if profile:
                for event in prof.events():
                    if str(event.device_type)=='DeviceType.CUDA':kernels[event.name]=kernels.get(event.name,0)+1
                assert kernels,'No held-head runtime CUDA provenance'
            assert finite_records(record),'Nonfinite held-head diagnostic'
            return record,kernels
        for profiled in [False,False,True]:
            record,kernels=head_call(profiled);result['heads'].append(record)
            if profiled:result['head_kernel_names_counts']=kernels
            progress(result)
        actual=dict(output=first_record['output'],loss=first_record['loss'],grad=first_record['grad_before_clip'])
        result['interpretation']=interpretation(actual,warm_record,result['features'],result['heads'])
        result['held_invariants_after']=invariants()
        assert result['held_invariants_after']==held_before,'Feature/head replay mutated held model/optimizer/RNG'
        assert tensor_record(held)==result['held_operand'],'Held operand mutated'
        del active,head,x,target,held
    result.update(status='completed',final_invariants_after=invariants(),seconds=time.monotonic()-started)
    assert result['final_invariants_after']==result['final_invariants_before']
    progress(result)
    return result
