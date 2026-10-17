"""Per-model compile options without changing optimizer compilation/arithmetic."""
import torch
import json,os
from torch._inductor import config as inductor
from llm_feasibility_runtime import Runtime as OriginalRuntime,FlatOutput,build_model,captured_backward
from llm_preserved_casts_contract import MODEL_OPTIONS
from execution_fingerprints import backend_record

def global_guard():
 if inductor.emulate_precision_casts is not False:raise RuntimeError('Global precision casts would change SOAP arithmetic')
def compiler_options(active):
 # Pinned torch2.11 fullgraph public getter returns None; inspect actual wrapper.
 backend=active.dynamo_ctx.callback
 for _ in range(8):
  if isinstance(backend,torch._TorchCompileInductorWrapper):return dict(backend.config)
  backend=getattr(backend,'_torchdynamo_orig_backend',None)
 raise RuntimeError('Cannot audit actual model compiler backend options')
def compile_model(model):
 global_guard()
 active=torch.compile(FlatOutput(model),options=MODEL_OPTIONS.copy(),fullgraph=True,dynamic=False)
 if compiler_options(active)['emulate_precision_casts'] is not True:raise RuntimeError('Missing per-model numerical option')
 global_guard();return active
class Runtime(OriginalRuntime):
 def emit(self,event,payload):
  super().emit(event,payload)
  if event in ('started','preflight_passed','finished'):
   p=self.bundle/'execution_status.json';status=json.loads(p.read_text());status['pid']=os.getpid();p.write_text(json.dumps(status,indent=2)+'\n')
 def initialize(self):
  global_guard();super().initialize();self.active=compile_model(self.model)
  self.emit('model_compiler_isolation',self.compiler_isolation())
 def compiler_isolation(self):
  global_guard();model_option=compiler_options(self.active)['emulate_precision_casts']
  if model_option is not True:raise RuntimeError('Model precision-cast option changed')
  return dict(global_emulate_precision_casts=False,model_emulate_precision_casts=model_option,optimizer_compile_options_unchanged=True,passed=True)
 def budget(self,resources=False):
  global_guard();return super().budget(resources=resources)
 def before_smoke(self):
  from prepare_llm_preserved_casts import verify_backend
  record=backend_record();verify_backend(self.bundle,record);self.emit('backend_before_smoke',dict(record=record,passed=True,compiler_isolation=self.compiler_isolation()))
 def after_measured(self):
  from prepare_llm_preserved_casts import verify_backend
  record=backend_record();verify_backend(self.bundle,record);self.emit('backend_after_measured',dict(record=record,passed=True,compiler_isolation=self.compiler_isolation()))
  self.retain_code('after_measured');self.budget(resources=True)
