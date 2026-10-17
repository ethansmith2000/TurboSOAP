"""Unchanged original model/evaluator; named read-only kernels and exact bounds."""
import weakref,torch,shutil
from llm_jacobi2_learning_runtime import Runtime as PreviousRuntime,FlatOutput,build_model,captured_backward
from execution_fingerprints import backend_record
class Runtime(PreviousRuntime):
 def initialize(self):
  super().initialize()
  from llm_cubic3_kernel import Kernels
  from llm_cubic3_contract import COUNTS,OBS
  self.cubic3_kernel=Kernels();self.cubic3_counts={k:0 for k in COUNTS};self.cubic3_observations={k:0 for k in OBS};self.cubic3_rows=[]
  self.kernel.cubic3_runtime_ref=weakref.ref(self)
 def budget(self,resources=False):
  super().budget(resources=False)
  if resources:
   b=self.protocol['bounds'];cache=self.bundle/'compile_cache'
   peak=torch.cuda.max_memory_allocated()
   compact=sum(p.stat().st_size for p in self.bundle.rglob('*') if p.is_file() and cache not in p.parents)
   cached=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
   if peak>=b['maximum_peak_allocated_bytes'] or compact>=b['maximum_evidence_bytes'] or cached>=b['maximum_compiler_cache_bytes'] or shutil.disk_usage(self.bundle).free<b['minimum_free_disk_bytes']:raise RuntimeError('Original allocatedGPU/compact/cache/free-disk bound exceeded')
 def before_smoke(self):
  from prepare_llm_cubic3 import verify_backend
  record=backend_record();verify_backend(self.bundle,record);self.emit('backend_before_smoke',dict(record=record,passed=True,execution_contract=self.execution_contract()))
 def after_measured(self):
  from prepare_llm_cubic3 import verify_backend
  record=backend_record();verify_backend(self.bundle,record);self.emit('backend_after_measured',dict(record=record,passed=True,execution_contract=self.execution_contract()))
  self.retain_code('after_measured');self.budget(resources=True)
