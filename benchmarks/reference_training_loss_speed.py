"""One-attempt configurable entry for the unchanged208 execution profiler."""
import argparse
import json
import os
from pathlib import Path
from reference_training_speed import SpeedRun,write,HERE


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args()
    assert HERE.parent==args.bundle.resolve()/'source','frozen entry only'
    assert not (args.bundle/'execution_status.json').exists(),'one attempt only'
    trial=json.loads((args.bundle/'config.json').read_text())['trial']
    write(args.bundle/'execution_status.json',dict(status='starting',trial=trial,pid=os.getpid()))
    try:
        SpeedRun(args.bundle).run()
        write(args.bundle/'execution_status.json',dict(status='completed',trial=trial,pid=os.getpid()))
    except BaseException as exc:
        import traceback
        write(args.bundle/'execution_status.json',dict(status='failed',trial=trial,pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc()))
        raise
