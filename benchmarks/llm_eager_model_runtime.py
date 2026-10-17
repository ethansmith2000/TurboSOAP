"""Common eager model reference; retain original compiled optimizer kernels."""
import torch,json,os
from llm_feasibility_runtime import Runtime as OriginalRuntime,FlatOutput,build_model,captured_backward
from llm_preserved_casts_runtime import global_guard
from execution_fingerprints import backend_record
class Runtime(OriginalRuntime):
 def emit(self,event,payload):
  super().emit(event,payload)
  if event in ('started','preflight_passed','finished'):
   p=self.bundle/'execution_status.json';status=json.loads(p.read_text());status['pid']=os.getpid();p.write_text(json.dumps(status,indent=2)+'\n')
 def initialize(self):
  global_guard();super().initialize();self.active=FlatOutput(self.model)
  self.emit('model_execution_contract',self.execution_contract())
 def execution_contract(self):
  global_guard()
  if type(self.active) is not FlatOutput or self.active.model is not self.model:raise RuntimeError('Common eager model execution contract changed')
  return dict(model='common eager reference',compiled_model=False,global_emulate_precision_casts=False,optimizer_compile_options_unchanged=True,passed=True)
 def budget(self,resources=False):
  global_guard();return super().budget(resources=resources)
 def before_smoke(self):
  from prepare_llm_eager_model import verify_backend
  record=backend_record();verify_backend(self.bundle,record);self.emit('backend_before_smoke',dict(record=record,passed=True,execution_contract=self.execution_contract()))
 def after_measured(self):
  from prepare_llm_eager_model import verify_backend
  record=backend_record();verify_backend(self.bundle,record);self.emit('backend_after_measured',dict(record=record,passed=True,execution_contract=self.execution_contract()))
  self.retain_code('after_measured');self.budget(resources=True)
