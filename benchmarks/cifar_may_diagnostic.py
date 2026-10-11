"""Frozen, one-attempt short May entry. No complete comparison or quality claim."""
import argparse
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE),str(HERE/'cifar_support')]
import torch
from cifar_geometry_diagnostic import GPURuntime as OriginalRuntime, write, retain_abort_audit
from execution_fingerprints import backend_record
from launch_environment_contract import verify_environment_launch
from launch_environment_wire_contract import verify_backend_launch
from may_diagnostic_runtime import run_may, failure_record
from prepare_may_diagnostic import verify_bundle


class GPURuntime(OriginalRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.last_arm_result={};self.preflight_probes=0

    def emit(self,event,payload):
        if event=='arm_result':self.last_arm_result=payload
        if event=='preflight_probe':self.preflight_probes+=1
        super().emit(event,payload)

    def after_short_diagnostic(self):
        record=backend_record()
        check=verify_backend_launch(self.bundle,record,'backend_after_measured')
        self.emit('backend_after_short_diagnostic',dict(phase='short_completion',record=record,check=check,
            historical_reference_phase='backend_after_measured',full_comparison_completed=False,
            qr_repeat_completed=False,completed_after_measured_gate=False))
        self.retain_code('short_completion')
        resources=self.capture_abort_resources();resources['phase']='short_completion'
        self.emit('resource_peak',resources)


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    b=parser.parse_args().bundle.resolve()
    if HERE.parent!=b/'source':raise RuntimeError('Execute only frozen short source')
    if (b/'execution_status.json').exists() or (b/'events.jsonl').exists():
        raise RuntimeError('One attempt only; existing evidence cannot be overwritten')
    write(b/'execution_status.json',dict(status='initializing',attempt=1))
    runtime=None;result=None
    try:
        verify_bundle(b);verify_environment_launch(b)
        if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one lifetime-claimed GPU required')
        runtime=GPURuntime(b,started=started);runtime.initialize()
        result=run_may(runtime,runtime.protocol)
        write(b/'summary.json',result)
        if result['status']!='completed':raise RuntimeError('Short May diagnostic stopped; no retry')
    except BaseException as exc:
        primary=failure_record(exc)
        try:write(b/'execution_status.json',dict(status='failed',**primary))
        except BaseException as persistence_error:
            print(json.dumps(dict(status='failed',primary=primary,status_persistence_error=repr(persistence_error))),flush=True)
        if runtime is not None:
            retain_abort_audit(runtime,b,result.get('abort_audit') if result else None)
            if runtime.snapshot is not None and not runtime.snapshot.closed:
                cleanup=[]
                for name,action in [('gradients',lambda:runtime.model.zero_grad(set_to_none=True)),
                    ('snapshot',runtime.snapshot.close),('placement',runtime.close)]:
                    try:action()
                    except BaseException as error:cleanup.append(dict(stage=name,error=repr(error)))
                if cleanup:print('initialization_cleanup '+json.dumps(cleanup),flush=True)
        raise


if __name__=='__main__':main()
