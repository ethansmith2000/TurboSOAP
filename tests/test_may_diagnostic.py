"""Bounded-driver CPU checks. Tiny success fixture uses QR stage, not GPU proof."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
from benchmarks import may_diagnostic_runtime as short
from benchmarks import matched_training_diagnostic_runtime as core
from benchmarks.matched_training_contract import build_arm, digest
from benchmarks import prepare_may_diagnostic as prep

spec=importlib.util.spec_from_file_location('_short_cpu_adapter',Path(__file__).with_name('test_geometry_failure_diagnostics.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)

@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()
    def fail(*a,**kw):raise AssertionError('Short CPU preparation attempted CUDA initialization')
    monkeypatch.setattr(torch.cuda,'_lazy_init',fail)
    yield
    assert not torch.cuda.is_initialized()

class CPU(fixture.AuditCPU):
    def __init__(self,fault=None,audit_fault=None):
        super().__init__(fault,audit_fault);self.last_arm_result={};self.preflight_probes=0;self.short_completed=False
        self.schedule_steps=35000;self.warmup_steps=875
    def emit(self,event,payload):
        if event=='arm_result':self.last_arm_result=json.loads(json.dumps(payload,allow_nan=False))
        if event=='preflight_probe':self.preflight_probes+=1
        super().emit(event,payload)
    def after_short_diagnostic(self):
        self.short_completed=True
        self.emit('backend_after_short_diagnostic',dict(phase='short_completion',cpu_test_only=True,
            full_comparison_completed=False,completed_after_measured_gate=False))
        self.emit('resource_peak',dict(phase='short_completion',allocated_peak_bytes=1234,cpu_test_only=True))
    def after_measured(self):raise AssertionError('Old full-comparison backend hook must not run')

def cpu_factory(model,config,arm):
    opt=build_arm(model,config,arm)
    # CPU orchestration success fixture avoids qualifying the BF16 tracked kernel.
    # Production default factory remains the exact pinned build_arm, unmodified.
    if arm=='may_literal':
        for g in opt.matrix.param_groups:g['basis_stage1_method']='qr'
    return opt


def test_success_exact_counts_only_may_no_full_comparison_or_quality():
    r=CPU();p=prep.protocol();out=short.run_may(r,p,factory=cpu_factory)
    assert out['status']=='completed',out['failure']
    assert out['diagnostic_completed'] and not out['quality_qualified'] and not out['timing_qualified']
    assert not out['full_comparison_completed'] and not out['qr_repeat_completed']
    assert out['smoke_calls_completed']==25 and out['may_calls_completed']==291 and out['maximum_optimizer_calls']==316
    assert r.preflight_probes==4 and len(r.updates)==291 and all(arm=='may_literal' for arm,_ in r.updates)
    assert r.short_completed and r.snapshot.closed and r.closed and out['snapshot_release']['all_registered_released']
    assert len(out['runs'])==1 and out['runs'][0]['steps']==291 and out['runs'][0]['release']['passed']
    assert [o['step'] for o in out['runs'][0]['observations']]==[0,100]
    assert [v['external_call'] for e,v in r.events if e=='before_refresh_diagnostics']==[251,261,271,281]
    assert r.schedule_steps==35000 and r.warmup_steps==875
    assert out['abort_audit'] is None
    json.dumps(out,allow_nan=False)


def test_injected271_geometry_stop_specific_trigger_abort_and_release(monkeypatch):
    original=core.check_state
    def inject(model,opt,*a,**kw):
        if kw.get('phase')=='after_refresh' and kw.get('external_call')==271:
            for state in opt.matrix.state.values():
                for q in state['Q']:
                    if torch.is_tensor(q):q.mul_(2)
        return original(model,opt,*a,**kw)
    monkeypatch.setattr(core,'check_state',inject)
    r=CPU();out=short.run_may(r,prep.protocol(),factory=cpu_factory)
    assert out['status']=='failed' and not out['diagnostic_completed']
    assert out['may_calls_completed']==271 and out['smoke_calls_completed']==25
    assert len(r.updates)==271 and not r.short_completed
    factor=out['failure']['details']['failing_factor'];assert factor['external_call']==271 and factor['internal_step']==270
    assert factor['normalized_gram_error']>=.05
    assert out['failed_arm']['release']['passed'] and out['snapshot_release']['all_registered_released']
    assert out['abort_audit']['components']['resources']['value']['allocated_peak_bytes']==1234
    assert not out['abort_audit']['completed_after_measured_gate']
    assert not any(e=='backend_after_short_diagnostic' for e,_ in r.events)
    assert [v['external_call'] for e,v in r.events if e=='before_refresh_diagnostics']==[251,261,271]
    assert r.snapshot.closed and r.closed
    json.dumps(out,allow_nan=False)


@pytest.mark.parametrize('fault',['preflight_input','actual_first_input'])
def test_exact_preflight_actual_first_failure_prevents_may_updates(fault):
    r=CPU(fault);out=short.run_may(r,prep.protocol(),factory=cpu_factory)
    assert out['status']=='failed' and not r.updates and out['may_calls_completed']==0
    assert out['snapshot_release']['all_registered_released'] and r.snapshot.closed
    assert not r.short_completed


@pytest.mark.parametrize('fault',['resources','backend','code'])
def test_audit_failure_preserves_primary_optimizer_failure(fault):
    r=CPU('optimizer',fault);out=short.run_may(r,prep.protocol(),factory=cpu_factory)
    assert 'Injected optimizer failure' in out['failure']['error']
    assert out['abort_audit']['components'][fault]['status']=='failed'
    assert r.audit_calls==['resources','backend','code']
    assert out['failed_arm']['release']['passed'] and out['snapshot_release']['all_registered_released']


def test_prod_factory_literal_policy_and_preflight_are_original():
    r=CPU();opt=short.build_arm(r.model,r.config,'may_literal');g=opt.matrix.param_groups[0]
    assert g['basis_stage1_method']=='tangent' and g['basis_storage_dtype']=='bfloat16'
    assert g['precondition_frequency']==10 and g['basis_warmup_steps']==250 and g['basis_metric_frequency']==5
    assert g['basis_direction_normalizer']=='col' and g['basis_direction_normalizer_beta']==.99
    assert g['shampoo_beta']==.99 and g['lr']==.0005 and g['betas']==(.9,.999)
    assert short.preflight is core.preflight and short.disposable_smoke is core.disposable_smoke and short.train_arm is core.train_arm


def test_readonly_observer_orchestration_parity_at291(monkeypatch):
    a=CPU();expected=short.run_may(a,prep.protocol(),factory=cpu_factory)
    monkeypatch.setattr(core,'BRACKET_CALLS',())
    b=CPU();without=short.run_may(b,prep.protocol(),factory=cpu_factory)
    assert expected['status']==without['status']=='completed'
    assert core.quality_view(expected['runs'][0])==core.quality_view(without['runs'][0])
    for event in ('preflight_probe','actual_first_preupdate'):
        assert [v for e,v in a.events if e==event]==[v for e,v in b.events if e==event]


@pytest.fixture
def bundle(tmp_path):
    b=tmp_path/'short';prep.prepare(b);return b


def test_freeze_lineage_recipe_bounds_sources_and_no_launch(bundle):
    m=prep.verify_bundle(bundle);p=json.loads((bundle/'protocol.json').read_text());base=prep.read(prep.BASE/'protocol.json')
    assert len(m['source_sha256'])==30
    for n,h in prep.read(prep.ARCHIVE/'manifest.json')['source_sha256'].items():assert m['source_sha256'][n]==h
    for n in ['config.json','split.json','data_prepared.json','environment_contract.json']:
        assert (bundle/n).read_bytes()==(prep.BASE/n).read_bytes()
    assert p['schedule']==base['schedule'] and p['process']==base['process'] and p['guards']==base['guards']
    assert p['steps_per_arm']==3500 and p['may_calls']==291 and p['maximum_optimizer_updates']==316
    assert p['bounds']['internal_wall_seconds']==240 and p['bounds']['external_watchdog_seconds']==300
    assert p['bounds']['maximum_peak_allocated_bytes']==24*2**30 and p['bounds']['maximum_initial_snapshot_purpose_cpu_bytes']==256*2**20
    assert p['observation_steps']==[0,100] and p['arms']==['may_literal'] and len(p['preflight_arms'])==4
    assert m['device'] is None and not m['gpu_selected']
    assert not any((bundle/f).exists() for f in ['launch_manifest.json','execution_status.json','events.jsonl','run.log'])
    with pytest.raises(ValueError,match='New unexecuted'):prep.prepare(bundle)
    assert 'autostart=false' in (bundle/'supervisor_template.conf').read_text()


@pytest.mark.parametrize('file',['protocol.json','config.json','split.json','supervisor_template.conf','source/benchmarks/geometry_failure_diagnostics.py'])
def test_corrupt_member_rejected_before_gpu(bundle,file):
    p=bundle/file;p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError):prep.verify_bundle(bundle)


def test_short_completed_backend_is_labeled_without_full_gate_claim(bundle,monkeypatch):
    sys.path[:0]=[str(bundle/'source/benchmarks'),str(bundle/'source/benchmarks/cifar_support')]
    import benchmarks.cifar_may_diagnostic as entry
    r=entry.GPURuntime.__new__(entry.GPURuntime);events=[]
    monkeypatch.setattr(entry,'backend_record',lambda:{'cpu_test_only':True})
    monkeypatch.setattr(entry,'verify_backend_launch',lambda b,v,p:dict(passed=True,reference_phase=p))
    r.bundle=bundle;r.emit=lambda e,p:events.append((e,p));r.retain_code=lambda phase:None
    r.capture_abort_resources=lambda:dict(phase='abort',allocated_peak_bytes=1234)
    r.after_short_diagnostic()
    event,row=events[0];assert event=='backend_after_short_diagnostic' and row['phase']=='short_completion'
    assert row['historical_reference_phase']=='backend_after_measured'
    assert not row['full_comparison_completed'] and not row['completed_after_measured_gate']
    assert events[1][1]['phase']=='short_completion'


def test_original_initializer_and_batchplan_retained_by_inheritance(bundle):
    sys.path[:0]=[str(bundle/'source/benchmarks'),str(bundle/'source/benchmarks/cifar_support')]
    import benchmarks.cifar_may_diagnostic as entry
    assert entry.GPURuntime.initialize is entry.OriginalRuntime.initialize
    assert entry.GPURuntime.make_batch is entry.OriginalRuntime.make_batch
    assert entry.GPURuntime.before_smoke is entry.OriginalRuntime.before_smoke
    assert entry.GPURuntime.after_smoke is entry.OriginalRuntime.after_smoke
    assert prep.protocol()['steps_per_arm']==3500


def test_frozen_entry_missing_launch_fails_cpu_then_rejects_second_attempt(bundle):
    script=bundle/'source/benchmarks/cifar_may_diagnostic.py'
    env=dict(__import__('os').environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    a=subprocess.run(['/venv/main/bin/python','-I','-B',str(script),'--bundle',str(bundle)],env=env,capture_output=True,text=True)
    assert a.returncode!=0 and 'launch_manifest.json' in a.stderr
    state=prep.read(bundle/'execution_status.json');assert state['status']=='failed'
    before=prep.sha(bundle/'execution_status.json')
    b=subprocess.run(['/venv/main/bin/python','-I','-B',str(script),'--bundle',str(bundle)],env=env,capture_output=True,text=True)
    assert b.returncode!=0 and 'One attempt only' in b.stderr
    assert prep.sha(bundle/'execution_status.json')==before
    assert not (bundle/'events.jsonl').exists()


def test_bundle_immutable_imports_do_not_initialize_cuda(bundle):
    code='''import sys,json
from pathlib import Path
b=Path(sys.argv[1]);sys.path[:0]=[str(b/'source'),str(b/'source/benchmarks'),str(b/'source/benchmarks/cifar_support')]
import torch
assert not torch.cuda.is_initialized()
def fail(*a,**k):raise AssertionError('Frozen short import initialized CUDA')
torch.cuda._lazy_init=fail
import cifar_may_diagnostic as entry
import prepare_may_diagnostic as prep
prep.verify_bundle(b)
assert entry.GPURuntime.initialize is entry.OriginalRuntime.initialize
assert not torch.cuda.is_initialized()
print('PASS_ISOLATED_FROZEN_IMPORT_NO_CUDA')
'''
    env=dict(__import__('os').environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    r=subprocess.run(['/venv/main/bin/python','-I','-B','-c',code,str(bundle)],env=env,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert 'PASS_ISOLATED_FROZEN_IMPORT_NO_CUDA' in r.stdout


def test_short_completion_backend_failure_preserves_trigger_and_cleans_up():
    class BackendFail(CPU):
        def after_short_diagnostic(self):raise RuntimeError('Injected short-completion backend failure')
    r=BackendFail();out=short.run_may(r,prep.protocol(),factory=cpu_factory)
    assert out['status']=='failed' and out['may_calls_completed']==291
    assert 'Injected short-completion backend failure' in out['failure']['error']
    assert out['runs'][0]['release']['passed'] and out['snapshot_release']['all_registered_released']
    assert r.snapshot.closed and r.closed and not out['full_comparison_completed']
    assert digest(r.model.state_dict())==r.initial_model


def test_real_inherited_wall_and_allocated_caps_use_short_protocol(tmp_path,monkeypatch):
    import time
    sys.path[:0]=[str(Path(__file__).parents[1]/'benchmarks'),str(Path(__file__).parents[1]/'benchmarks/cifar_support')]
    import benchmarks.cifar_may_diagnostic as entry
    r=entry.GPURuntime.__new__(entry.GPURuntime);r.protocol=prep.protocol();r.bundle=tmp_path
    r.started=time.monotonic()-241
    with pytest.raises(RuntimeError,match='wall bound'):r.budget()
    r.started=time.monotonic()
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:24*2**30)
    with pytest.raises(MemoryError,match='Allocated'):r.budget(resources=True)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    r.protocol['bounds']['maximum_evidence_bytes']=2
    (tmp_path/'evidence.json').write_text('{}')
    with pytest.raises(OSError,match='Compact evidence'):r.budget(resources=True)
