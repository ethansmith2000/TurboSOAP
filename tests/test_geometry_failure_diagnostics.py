"""CPU-only observer invariants and abort ownership, using original tiny runtime."""
import ast
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from benchmarks import geometry_failure_diagnostics as diag
from benchmarks import matched_training_diagnostic_runtime as new
from benchmarks import matched_training_runtime as old
from benchmarks.matched_training_contract import build_arm, digest

# Reuse the pinned tiny CPU adapter; this never executes its test functions.
spec=importlib.util.spec_from_file_location('_original_cpu_adapter',Path(__file__).with_name('test_matched_training_runtime.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
CPU=fixture.CPU;CONFIG=fixture.CONFIG;PROTOCOL=fixture.PROTOCOL


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail(*args,**kwargs):raise AssertionError('CPU diagnostic attempted CUDA initialization')
    monkeypatch.setattr(torch.cuda,'_lazy_init',fail)
    yield
    assert not torch.cuda.is_initialized()


def make_factor(q):
    runtime=CPU();opt=build_arm(runtime.model,CONFIG,'may_literal')
    p=opt.matrix.param_groups[0]['params'][0]
    opt.matrix.state[p].update(Q=[q,[],None],step=270,precondition_frequency=10,
        basis_scales=[.1,.2],basis_direction_moments=[{'col':torch.arange(4).float()}],
        controller={'eta':.01})
    return runtime,opt,p


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_factor_details_read_only_and_same_original_guard_math(dtype):
    q=torch.eye(4,dtype=dtype);q[0,0]=1.001 if dtype==torch.float32 else 1.0078125
    runtime,opt,p=make_factor(q)
    before=digest(dict(model=runtime.model.state_dict(),state=opt.state_dict(),rng=runtime.streams.capture()))
    torch.set_float32_matmul_precision('high'); reference=old.check_state(runtime.model,opt)
    result=diag.check_state(runtime.model,opt,external_call=271,phase='after_refresh')
    assert {k:result[k] for k in reference}==reference
    row=result['factor_diagnostics'][0]
    assert row['parameter_name']=='blocks.0.weight' and row['factor_axis']==0
    assert row['shape']==[4,4] and row['dtype']==str(dtype) and row['external_call']==271
    assert row['internal_step']==270 and not row['branch']['metrics_enabled']
    assert not row['branch']['pipeline_fallback_eligible']
    assert row['columns']['worst_column']==0 and row['columns']['count']==4
    assert row['existing_controller_normalizer']['controller']['eta']==.01
    assert row['existing_controller_normalizer']['basis_scales']==[.1,.2]
    assert row['controller_settings']['basis_step_controller']=='correction_ratio'
    json.dumps(result,allow_nan=False)
    assert before==digest(dict(model=runtime.model.state_dict(),state=opt.state_dict(),rng=runtime.streams.capture()))
    assert torch.get_float32_matmul_precision()=='high'


@pytest.mark.parametrize('scale,expected_finite',[(2.,True),(1e30,False)])
def test_failure_details_finite_or_overflow_without_nan_json_and_precision_restored(scale,expected_finite):
    runtime,opt,_=make_factor(torch.eye(4)*scale)
    before=digest(opt.state_dict());torch.set_float32_matmul_precision('medium')
    with pytest.raises(FloatingPointError):old.check_state(runtime.model,opt)
    with pytest.raises(diag.GeometryFailure) as caught:
        diag.check_state(runtime.model,opt,external_call=271)
    detail=caught.value.details;row=detail['failing_factor']
    assert row['finite_input'] and row['finite_gram']==expected_finite
    assert detail['threshold']==.05
    if expected_finite:assert row['normalized_gram_error']==3.
    else:assert row['normalized_gram_error']=={'nonfinite':'+inf'}
    json.dumps(new.failure_record(caught.value),allow_nan=False)
    assert before==digest(opt.state_dict())
    assert torch.get_float32_matmul_precision()=='medium'


def test_default_gate_is_strict_at_exact_threshold():
    runtime,opt,_=make_factor(torch.eye(4))
    # Identity has exactly zero error; strict < means zero threshold rejects.
    for fn in [old.check_state,diag.check_state]:
        with pytest.raises(FloatingPointError):fn(runtime.model,opt,maximum_gram_error=0.)
    assert diag.check_state.__wrapped__.__defaults__==(.05,)


@pytest.mark.parametrize('position',['scalar','state_tensor','model_tensor','cache_tensor'])
def test_nonfinite_guard_order_and_same_failure_as_original(position):
    runtime,opt,p=make_factor(torch.eye(4))
    if position=='scalar':opt.matrix.state[p]['controller']['eta']=float('nan')
    elif position=='state_tensor':opt.matrix.state[p]['basis_direction_moments'][0]['col'][0]=float('nan')
    elif position=='model_tensor':runtime.model.scale.fill_(float('nan'))
    else:opt.matrix._basis_weight_cache={'x':torch.tensor(float('nan'))}
    errors=[]
    for fn in [old.check_state,diag.check_state]:
        with pytest.raises(FloatingPointError) as caught:fn(runtime.model,opt)
        errors.append(str(caught.value))
    assert errors[0]==errors[1]


@pytest.mark.parametrize('internal,metrics,force,pipeline',[(250,False,True,False),(260,True,False,True),(270,False,False,False),(280,False,False,False),(310,True,False,True)])
def test_literal_source_branch_eligibility_without_invoking_tracker(internal,metrics,force,pipeline):
    runtime,opt,p=make_factor(torch.eye(4));state=opt.matrix.state[p];state['step']=internal
    group=opt.matrix.param_groups[0]
    result=diag.refresh_branch(state,group)
    assert result['metrics_enabled']==metrics and result['force_warmup_fallback']==force
    assert result['pipeline_fallback_eligible']==pipeline
    if internal>250:
        assert opt.matrix._should_run_basis_metrics(state,group)==metrics
    state['step']-=1
    assert diag.refresh_branch(state,group,next_call=True)==result


def test_pre_refresh_observer_does_not_enforce_post_refresh_gate_or_mutate():
    runtime,opt,_=make_factor(torch.eye(4)*2)
    before=digest(dict(state=opt.state_dict(),rng=runtime.streams.capture(),model=runtime.model.state_dict()))
    row=diag.refresh_snapshot(runtime.model,opt,271)
    assert not row['threshold_enforced'] and row['factors'][0]['normalized_gram_error']==3.
    assert before==digest(dict(state=opt.state_dict(),rng=runtime.streams.capture(),model=runtime.model.state_dict()))


def test_observers_preserve_full_orchestration_outputs_gradients_rng_and_endpoints(monkeypatch):
    reference=CPU();baseline=old.run_shared(reference,PROTOCOL)
    observed=CPU();monkeypatch.setattr(new,'BRACKET_CALLS',(1,2,3))
    outcome=new.run_shared(observed,PROTOCOL)
    assert outcome['status']==baseline['status']=='completed'
    assert reference.updates==observed.updates
    for event in ('preflight_probe','actual_first_preupdate'):
        assert [r for e,r in reference.events if e==event]==[r for e,r in observed.events if e==event]
    assert [old.quality_view(r) for r in baseline['runs']]==[new.quality_view(r) for r in outcome['runs']]
    assert all(r['release']['passed'] for r in outcome['runs'])
    assert outcome['snapshot_release']['all_registered_released']
    assert any(e=='before_refresh_diagnostics' for e,_ in observed.events)
    assert outcome['abort_audit'] is None


class AuditCPU(CPU):
    def __init__(self,fault=None,audit_fault=None):super().__init__(fault);self.audit_fault=audit_fault;self.audit_calls=[]
    def capture_abort_resources(self):
        self.audit_calls.append('resources')
        if self.audit_fault=='resources':raise RuntimeError('Injected audit resource failure')
        return dict(phase='abort',allocated_peak_bytes=1234,purpose_bytes=self.snapshot.budget_bytes)
    def capture_abort_backend(self):
        self.audit_calls.append('backend')
        if self.audit_fault=='backend':raise RuntimeError('Injected audit backend failure')
        return dict(phase='abort',completed_after_measured_gate=False)
    def capture_abort_code(self):
        self.audit_calls.append('code')
        if self.audit_fault=='code':raise RuntimeError('Injected audit code failure')
        return dict(phase='abort',cpu_test_only=True)


@pytest.mark.parametrize('audit_fault',[None,'resources','backend','code'])
def test_failure_remains_original_and_all_abort_components_collected_and_release_passes(audit_fault):
    runtime=AuditCPU('optimizer',audit_fault);result=new.run_shared(runtime,PROTOCOL)
    assert result['status']=='failed' and not result['quality_qualified']
    arms=[r for event,r in runtime.events if event=='arm_result']
    assert 'Injected optimizer failure' in arms[-1]['failure']['error']
    assert runtime.audit_calls==['resources','backend','code']
    audit=result['abort_audit'];assert not audit['completed_after_measured_gate']
    for key,value in audit['components'].items():assert value['status']==('failed' if key==audit_fault else 'captured')
    assert runtime.snapshot.closed and runtime.closed and result['snapshot_release']['all_registered_released']
    assert all(r['release']['passed'] for r in arms)
    json.dumps(result,allow_nan=False)


def test_geometry_exception_does_not_hold_optimizer_or_storage_after_failure(monkeypatch):
    real_check=new.check_state
    def inject(model,opt,*a,**kw):
        if kw.get('phase')=='after_refresh' and type(opt.matrix).__name__=='SOAP' and kw.get('external_call')==1:
            for state in opt.matrix.state.values():
                for q in state['Q']:
                    if torch.is_tensor(q):q.mul_(2)
        return real_check(model,opt,*a,**kw)
    monkeypatch.setattr(new,'check_state',inject)
    runtime=AuditCPU();result=new.run_shared(runtime,PROTOCOL)
    arm=[r for event,r in runtime.events if event=='arm_result'][-1]
    assert arm['arm']=='may_literal' and arm['steps']==1 and 'details' in arm['failure']
    assert arm['release']['passed'] and result['snapshot_release']['all_registered_released']
    assert runtime.closed and result['status']=='failed'


def test_gpu_abort_adapter_does_not_initialize_cuda_and_retains_compact_fields(tmp_path):
    # Import frozen-compatible entry, but never run main or initialize().
    sys.path[:0]=[str(Path(__file__).parents[1]/'benchmarks'),str(Path(__file__).parents[1]/'benchmarks/cifar_support')]
    from benchmarks.cifar_geometry_diagnostic import GPURuntime
    runtime=GPURuntime.__new__(GPURuntime);runtime.bundle=tmp_path;runtime.started=0.;runtime.snapshot=None
    (tmp_path/'config.json').write_text('{}')
    row=runtime.capture_abort_resources();assert row['allocated_peak_bytes'] is None and not row['cuda_initialized']
    assert row['compact_evidence_bytes']==2 and row['snapshot_purpose_budget_bytes'] is None
    assert runtime.capture_abort_backend()['status']=='not_initialized'


def test_original_forward_backward_nodes_and_policy_helpers_unchanged():
    def defs(p):return {n.name:ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    root=Path(__file__).parents[1]/'benchmarks';a=defs(root/'matched_training_runtime.py');b=defs(root/'matched_training_diagnostic_runtime.py')
    changed={'check_state','failure_record','train_arm','disposable_smoke','run_shared'}
    assert all(a[n]==b[n] for n in a if n not in changed)
    # Calls defining actual model forward/backward/clip/optimizer math unchanged.
    def calls(p):
        names={'cross_entropy','backward','clip_grad_norm_','step','lr_at','apply_lr_multiplier'}
        return [ast.dump(n) for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.Call) and
            ((isinstance(n.func,ast.Attribute) and n.func.attr in names) or (isinstance(n.func,ast.Name) and n.func.id in names))]
    assert calls(root/'matched_training_runtime.py')==calls(root/'matched_training_diagnostic_runtime.py')


def test_abort_audit_persistence_failure_cannot_replace_trigger(tmp_path, monkeypatch):
    sys.path[:0]=[str(Path(__file__).parents[1]/'benchmarks'),str(Path(__file__).parents[1]/'benchmarks/cifar_support')]
    from benchmarks import cifar_geometry_diagnostic as entry
    runtime=AuditCPU()
    def fail(*args,**kwargs):raise OSError('Injected audit disk-write failure')
    monkeypatch.setattr(entry,'write',fail)
    original=RuntimeError('original trigger')
    try:
        raise original
    except RuntimeError as exc:
        row=entry.retain_abort_audit(runtime,tmp_path)
        assert exc is original
    assert 'Injected audit disk-write failure' in row['persistence_failure']
    assert all(c['status']=='captured' for c in row['components'].values())


def test_mask_cleanup_exception_preserves_original_failure_and_snapshot_release():
    class BadClose(AuditCPU):
        def close(self):
            super().close()
            raise RuntimeError('Injected mask audit failure')
    runtime=BadClose('optimizer');result=new.run_shared(runtime,PROTOCOL)
    assert result['status']=='failed' and result['snapshot_release']['all_registered_released']
    assert 'may_literal' in result['failure']['error']
    assert 'Injected mask audit failure' in result['placement_release']['failure']['error']
    assert runtime.snapshot.closed
    arm=[r for e,r in runtime.events if e=='arm_result'][-1]
    assert 'Injected optimizer failure' in arm['failure']['error'] and arm['release']['passed']


def test_dense_orthogonal_bf16_reductions_and_all_factor_fingerprint_parity():
    torch.manual_seed(71)
    q=torch.linalg.qr(torch.randn(4,4)).Q.bfloat16()
    runtime,opt,_=make_factor(q)
    # Existing gradients and flags also remain identical under standalone observer.
    runtime.model(torch.ones(3,4)).sum().backward()
    before=digest(dict(state=opt.state_dict(),rng=runtime.streams.capture(),
        gradients={n:p.grad for n,p in runtime.model.named_parameters()},flags=[m.training for m in runtime.model.modules()]))
    original=old.check_state(runtime.model,opt)
    result=diag.check_state(runtime.model,opt)
    assert {k:result[k] for k in original}==original
    diag.refresh_snapshot(runtime.model,opt,271)
    after=digest(dict(state=opt.state_dict(),rng=runtime.streams.capture(),
        gradients={n:p.grad for n,p in runtime.model.named_parameters()},flags=[m.training for m in runtime.model.modules()]))
    assert before==after


def test_successful_refresh_retains_existing_tracker_aggregate_stats_without_alias():
    runtime,opt,_=make_factor(torch.eye(4))
    opt.matrix.latest_basis_stats={'accepted_steps':1.,'correction_ratio':torch.tensor(.95)}
    opt.matrix._basis_stats_accum={'diag_err':[.2,.3]}
    before=digest(opt.state_dict())
    result=diag.check_state(runtime.model,opt,external_call=261,phase='after_refresh')
    evidence=result['existing_algorithm_diagnostics']
    assert evidence['latest_basis_stats']['accepted_steps']==1.
    assert evidence['latest_basis_stats']['correction_ratio']['scalar']==float(torch.tensor(.95))
    assert evidence['_basis_stats_accum']['diag_err']==[.2,.3]
    assert before==digest(opt.state_dict())
    json.dumps(result,allow_nan=False)
    lease=new.OptimizerLease(runtime.model,opt);del opt
    assert lease.release()['passed']  # returned scalar stats cannot retain tensor aliases
