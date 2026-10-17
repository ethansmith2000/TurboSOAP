"""Reuse qualified model, token evaluator and compiled SOAP; new frozen launch gates."""
from llm_math_learning_runtime import Runtime as PreviousRuntime,FlatOutput,build_model,captured_backward
from execution_fingerprints import backend_record
class Runtime(PreviousRuntime):
 def initialize(self):
  super().initialize()
  from llm_jacobi2_learning_kernel import PolicyRouter
  from matched_training_runtime import streaming_digest
  self.kernel=PolicyRouter(self.emit,lambda:streaming_digest(self.streams.capture()))

 def before_smoke(self):
  from prepare_llm_jacobi2_learning import verify_backend
  record=backend_record();verify_backend(self.bundle,record);self.emit('backend_before_smoke',dict(record=record,passed=True,execution_contract=self.execution_contract()))
 def after_measured(self):
  from prepare_llm_jacobi2_learning import verify_backend
  record=backend_record();verify_backend(self.bundle,record);self.emit('backend_after_measured',dict(record=record,passed=True,execution_contract=self.execution_contract()))
  self.retain_code('after_measured');self.budget(resources=True)
