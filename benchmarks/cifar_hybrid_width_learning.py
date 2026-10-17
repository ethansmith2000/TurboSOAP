"""Fresh within-width2048 matched3500 learning, original trainer/optimizer and testedcap."""
import argparse
from contextlib import contextmanager
import json,os,time
from pathlib import Path
import torch
import cifar_matched_training as trainer
from cifar_hybrid_learning import run_pilot
from cifar_hybrid_width import Runtime as WidthRuntime
from execution_fingerprints import backend_record
from hybrid_width_learning_contract import SNAPSHOT_CAP
from prepare_hybrid_width_learning import verify,verify_launch,verify_backend

class Runtime(WidthRuntime):
 def before_smoke(self):
  record=backend_record();verify_backend(self.bundle,record)
  self.emit('backend_before_smoke',dict(record=record,passed=True))
 def after_measured(self):
  record=backend_record();verify_backend(self.bundle,record)
  self.emit('backend_after_measured',dict(record=record,passed=True))
  self.retain_code('after_measured');self.budget(resources=True)

def main():
 started=time.monotonic();parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
 b=parser.parse_args().bundle.resolve();runtime=None
 if Path(__file__).resolve().parent.parent!=b/'source':raise RuntimeError('Frozen entry only')
 if (b/'execution_status.json').exists() or (b/'events.jsonl').exists():raise RuntimeError('One attempt only')
 try:
  verify(b);verify_launch(b)
  trainer.write(b/'execution_status.json',dict(status='initializing',attempt=1,pid=os.getpid()))
  if torch.cuda.device_count()!=1:raise RuntimeError('One lifetime-claimed GPU required')
  runtime=Runtime(b,started=started);runtime.initialize();result=run_pilot(runtime)
  result.update(all_in_seconds=time.monotonic()-started,paired_fixed_observations=True,width2048_learning=True)
  verify(b);trainer.write(b/'summary.json',result);runtime.emit('finished',dict(status='completed'))
 except BaseException as exc:
  trainer.write(b/'execution_status.json',dict(status='failed',error=repr(exc),all_in_seconds=time.monotonic()-started))
  if runtime is not None:
   if runtime.snapshot is not None and not runtime.snapshot.closed:
    runtime.model.zero_grad(set_to_none=True);runtime.snapshot.close()
   trainer.write(b/'failure_cleanup.json',runtime.close())
  raise

if __name__=='__main__':main()
