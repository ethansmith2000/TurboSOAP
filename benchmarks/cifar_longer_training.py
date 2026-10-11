"""One frozen longer pair attempt; native trainer and optimizer remain unchanged."""
import argparse
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE / 'cifar_support')]
import torch
from cifar_cadence_training import CadenceRuntime
from cifar_matched_training import write
from launch_environment_contract import verify_environment_launch
from prepare_longer_training import verify
from longer_training_contract import run_longer


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(); parser.add_argument('--bundle', type=Path, required=True)
    b = parser.parse_args().bundle.resolve()
    if HERE.parent != b / 'source':
        raise RuntimeError('Only execute fresh frozen source')
    if (b / 'execution_status.json').exists() or (b / 'events.jsonl').exists():
        raise RuntimeError('One attempt only; never overwrite evidence')
    runtime = None
    try:
        verify(b)
        verify_environment_launch(b)
        write(b / 'execution_status.json', dict(status='initializing', attempt=1, seed=140, horizon=10000))
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one lifetime-claimed GPU required')
        runtime = CadenceRuntime(b, started=started)
        runtime.initialize()
        report = run_longer(runtime, runtime.protocol)
        write(b / 'summary.json', report)
        if report['status'] != 'completed':
            raise RuntimeError('Longer pair failed; no retry')
    except BaseException as exc:
        write(b / 'execution_status.json', dict(status='failed', error=repr(exc)))
        if runtime is not None:
            try:
                runtime.retain_code('failure')
            except BaseException as audit_error:
                write(b / 'code_retention_failure.json', dict(error=repr(audit_error)))
            if runtime.snapshot is not None and not runtime.snapshot.closed:
                runtime.model.zero_grad(set_to_none=True)
                runtime.snapshot.close(); runtime.close()
        raise


if __name__ == '__main__':
    main()
