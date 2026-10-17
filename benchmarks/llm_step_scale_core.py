"""Fixed-C/Q read-only sweep with exact state/flags/RNG/gradient release checks."""
import gc,time
import torch
from llm_sequence_averaging_core import fp32_math_control
from llm_step_scale_atoms import factor_sweeps
from llm_math_confirmation import endpoint,compiler_stats
from matched_training_runtime import streaming_digest,failure_record
from tensor_ownership import TensorOwnership

def diagnose(runtime,opt):
    started=time.monotonic();spec=runtime.protocol['step_scale'];before=endpoint(runtime,opt)
    grad_before=streaming_digest([p.grad for p in runtime.model.parameters()]);ids=[id(p.grad) for p in runtime.model.parameters()]
    modules=list(runtime.model.modules());flags=[m.training for m in modules];rng=runtime.streams.capture();compiler=compiler_stats()
    owner=TensorOwnership();sentinel=torch.empty(0);owner.watch(sentinel,'diagnostic.lifetime_sentinel');del sentinel
    purpose=sum(p.numel()*p.element_size() for p in runtime.model.parameters())+30*max(v['dimension']**2*4 for v in spec['factors'])+12*2**20
    error=None;rows=None
    runtime.step_counts=dict(sweep_candidates=0,pure_eager_references=0,qr_controls=0)
    try:
        if purpose>spec['memory']['maximum_tensor_purpose_bytes']:raise MemoryError('Step-scale named tensor-purpose cap exceeded')
        runtime.model.eval()
        with torch.no_grad(),fp32_math_control():rows=factor_sweeps(runtime,opt,owner)
    except BaseException as exc:
        error=failure_record(exc);exc.__traceback__=None
    finally:
        runtime.streams.restore(rng)
        for module,flag in zip(modules,flags):module.training=flag
        runtime.step_factor=None;gc.collect();release=owner.report();owner.close()
        after=endpoint(runtime,opt);grad_after=streaming_digest([p.grad for p in runtime.model.parameters()])
        nonmutation=before==after and grad_before==grad_after and ids==[id(p.grad) for p in runtime.model.parameters()] and flags==[m.training for m in modules] and compiler==compiler_stats()
        cleanup=dict(before=before,after=after,gradient_before_sha256=grad_before,gradient_after_sha256=grad_after,
            state_rng_flags_gradients_and_identity_pass=nonmutation,registered_release=release,completed_basis_counts=dict(runtime.step_counts),seconds=time.monotonic()-started)
        runtime.emit('step_scale_diagnostic_cleanup',cleanup)
    if not release['all_registered_released'] or not nonmutation:raise RuntimeError('Step-scale nonmutation or registered release failed')
    if error is not None:raise RuntimeError('Step-scale diagnostic failed: '+str(error))
    directions=[d for r in rows for d in r['directions']];candidates=[c for d in directions for c in d['candidates']]
    if len(rows)!=8 or len(directions)!=16 or len(candidates)!=64:raise RuntimeError('Declared step-scale factor/direction/candidate counts differ')
    if runtime.step_counts!=dict(sweep_candidates=64,pure_eager_references=16,qr_controls=8):raise RuntimeError('Observed basis call counts differ')
    result=dict(factors=rows,direction_observations=16,sweep_candidates=64,pure_eager_references=16,qr_controls=8,complete_basis_calls=88,
        rejected_candidates=sum(not c['admitted'] for c in candidates),optimizer_updates=0,backward_calls=0,
        tensor_purpose_bytes=purpose,cleanup=cleanup,passed=True)
    runtime.budget(resources=True);return result
