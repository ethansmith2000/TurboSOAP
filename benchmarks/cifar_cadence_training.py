"""Frozen three-arm GPU entry; CPU preparation/import never executes main."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE / 'cifar_support')]
import torch
from cifar_matched_training import GPURuntime, write
from cadence_training_contract import protocol as declared_protocol, recipe
from cadence_training_runtime import run_cadence
from execution_fingerprints import backend_record
from launch_environment_contract import verify_environment_launch
from prepare_cadence_training import verify


class CadenceRuntime(GPURuntime):
    def capture_abort_resources(self):
        return dict(allocated_bytes=torch.cuda.memory_allocated(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            snapshot_purpose_budget_bytes=self.snapshot.budget_bytes if self.snapshot else None)

    def capture_abort_backend(self):
        return dict(record=backend_record(), completed_after_measured_gate=False)

    def capture_abort_code(self):
        self.retain_code('abort')
        return dict(path=str(self.bundle / 'code_abort.json'))


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser(); p.add_argument('--bundle', type=Path, required=True)
    b = p.parse_args().bundle.resolve()
    if HERE.parent != b / 'source':
        raise RuntimeError('Only execute frozen source')
    if (b / 'execution_status.json').exists() or (b / 'events.jsonl').exists():
        raise RuntimeError('One attempt only; do not overwrite evidence')
    runtime = None
    try:
        verify(b)
        if json.loads((b / 'protocol.json').read_text()) != declared_protocol():
            raise ValueError('Prospective cadence contract differs from frozen implementation')
        old = json.loads((Path(json.loads((b / 'manifest.json').read_text())['baseline_bundle']) / 'config.json').read_text())
        if json.loads((b / 'config.json').read_text()) != recipe(old):
            raise ValueError('Prospective recipe differs from frozen implementation')
        verify_environment_launch(b)
        write(b / 'execution_status.json', dict(status='initializing', attempt=1))
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one lifetime-claimed GPU required')
        runtime = CadenceRuntime(b, started=started)
        runtime.initialize()
        report = run_cadence(runtime, runtime.protocol)
        write(b / 'summary.json', report)
        if report['status'] != 'completed':
            raise RuntimeError('Cadence screen failed; no retry')
    except BaseException as exc:
        write(b / 'execution_status.json', dict(status='failed', error=repr(exc)))
        if runtime is not None:
            try:
                runtime.retain_code('failure')
            except BaseException as audit_error:
                write(b / 'code_retention_failure.json', dict(error=repr(audit_error)))
            if runtime.snapshot is not None and not runtime.snapshot.closed:
                runtime.model.zero_grad(set_to_none=True)
                runtime.snapshot.close()
                runtime.close()
        raise


if __name__ == '__main__':
    main()
