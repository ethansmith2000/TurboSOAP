"""Scoped common math-only model attention; original compiled SOAP unchanged."""
from contextlib import contextmanager
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from llm_eager_model_runtime import Runtime as PreviousRuntime,FlatOutput,build_model,captured_backward
from execution_fingerprints import backend_record
class Runtime(PreviousRuntime):
 @contextmanager
 def autocast(self):
  with sdpa_kernel([SDPBackend.MATH]),super().autocast():
   if not torch.backends.cuda.math_sdp_enabled() or any((torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled(),torch.backends.cuda.cudnn_sdp_enabled())):raise RuntimeError('Common math-only model attention contract failed')
   yield
 def execution_contract(self):
  row=super().execution_contract();row['attention']='Scoped SDPA math-only in all model forwards; outside flags restored';return row
 def before_smoke(self):
  from prepare_llm_math_attention import verify_backend
  record=backend_record();verify_backend(self.bundle,record);self.emit('backend_before_smoke',dict(record=record,passed=True,execution_contract=self.execution_contract()))
 def after_measured(self):
  from prepare_llm_math_attention import verify_backend
  record=backend_record();verify_backend(self.bundle,record);self.emit('backend_after_measured',dict(record=record,passed=True,execution_contract=self.execution_contract()))
  self.retain_code('after_measured');self.budget(resources=True)
