"""Instance-only live prototype; original step/covariance/M/V paths untouched."""
from collections import Counter
import torch
from native_damped_kernel import LiveKernel as NativeKernel
from native_residual_named_kernel import PolicyKernel as OriginalKernel,candidate_retraction
from llm_jacobi2_kernel import prefix_jacobi2
from llm_jacobi2_learning_contract import POLICIES,DIAGNOSTIC_COUNTS

class LiveKernel(NativeKernel):
 def __init__(self,name,router,compiler=torch.compile):
  if name not in POLICIES:raise ValueError('Undeclared live Jacobi policy')
  if name=='highest_residual20':super().__init__(name,router,compiler)
  else:
   self.name=name;self.router=router;self.mode='highest';self.calls=Counter()
   self.eager_prefix=prefix_jacobi2;self.eager_ret=candidate_retraction
   self.prefix=compiler(self.eager_prefix,mode='default',fullgraph=True,dynamic=False)
   self.retraction=compiler(self.eager_ret,mode='default',fullgraph=True,dynamic=False)
 def evaluate(self,cov,q,eager=False):return OriginalKernel.evaluate(self,cov,q,eager)
 def __call__(self,cov,q,group):
  if self.router.probe:self.router.diagnostic_counts['own_actual_references']+=1
  return super().__call__(cov,q,group)

class PolicyRouter:
 def __init__(self,emit,rng_digest,compiler=torch.compile):
  self.emit=emit;self.rng_digest=rng_digest;self.calls=Counter();self.probes=[];self.moment_probes=[];self.probe=False
  self.kernels={n:LiveKernel(n,self,compiler) for n in POLICIES};self.cf_calls=0;self.covariance_samples={}
  self.warmup_counts=dict(compiled_basis=0,eager_references=0)
  self.diagnostic_counts={n:0 for n in DIAGNOSTIC_COUNTS}
 def evaluate_cf(self,name,cov,q):
  self.cf_calls+=1;self.diagnostic_counts['compiled_counterfactuals']+=1
  if self.cf_calls>144:raise RuntimeError('Frozen Jacobi2 counterfactual call bound exceeded')
  return self.kernels[name].evaluate(cov,q)
