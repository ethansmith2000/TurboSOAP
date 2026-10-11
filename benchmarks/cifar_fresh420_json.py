"""Sequential bounded fresh processes under one exclusive outer GPU claim."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);a=p.parse_args()
    b=a.bundle.resolve();assert os.environ.get('GPU_CLAIM_INDICES')
    c=json.loads((b/'config.json').read_text());d=c['fresh420'];contexts=c['memory_screen']['arms'];started=time.monotonic()
    status=dict(status='running',runs=[])
    def save():(b/'execution_status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    try:
        for context in contexts:
            remaining=c['max_wall_seconds']-(time.monotonic()-started)
            assert remaining>0,'Overall wall budget exhausted'
            label=context['label'];print('START '+label,flush=True)
            with (b/(label+'.log')).open('x') as log:
                proc=subprocess.Popen([sys.executable,'-u',str(b/'source/benchmarks/cifar_fresh420_json_child.py'),
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
