"""Sequential bounded fresh processes under one exclusive outer GPU claim."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from normalized_diagnostic_contract import verify_launch
from launch_environment_contract import verify_environment_launch


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);a=p.parse_args()
    b=a.bundle.resolve();assert os.environ.get('GPU_CLAIM_INDICES')
    # NORMALIZED_OUTER_LAUNCH_START
    verify_environment_launch(b)
    verify_launch(b)
    # NORMALIZED_OUTER_LAUNCH_END
    c=json.loads((b/'config.json').read_text());d=c['execution_diagnostic'];started=time.monotonic()
    status=dict(status='running',runs=[])
    def save():(b/'execution_status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    try:
        for context in d['contexts']:
            remaining=c['max_wall_seconds']-(time.monotonic()-started)
            assert remaining>0,'Overall wall budget exhausted'
            label=context['label'];print('START '+label,flush=True)
            with (b/(label+'.log')).open('x') as log:
                proc=subprocess.Popen([sys.executable,'-u',str(b/'source/benchmarks/cifar_normalized_env_wire_child.py'),
                                       '--bundle',str(b),'--context',label],stdout=log,stderr=subprocess.STDOUT)
                try:code=proc.wait(timeout=min(remaining,d['per_process_seconds']))
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    raise RuntimeError('Fresh-process timeout; no retry')
            status['runs'].append(dict(label=label,exit_code=code));save()
            if code:raise RuntimeError('Child failed: '+label)
            print('FINISH '+label,flush=True)
        status.update(status='completed',all_in_seconds=time.monotonic()-started);save()
    except BaseException as exc:
        status.update(status='failed',failure=repr(exc),all_in_seconds=time.monotonic()-started);save();raise


if __name__=='__main__':main()
