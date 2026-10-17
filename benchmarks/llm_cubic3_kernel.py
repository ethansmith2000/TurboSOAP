"""Named cubic3 retraction; immutable doubled generator and original cube2."""
import torch
from native_residual_named_kernel import highest_prefix,candidate_retraction
from llm_jacobi2_kernel import prefix_jacobi2
from hybrid_training_kernel import precision

def retract_three(y):
 for _ in range(3):
  residual=y.T@y;residual.diagonal().sub_(1.)
  correction=(y.bfloat16()@residual.bfloat16()).float()
  y=y-.5*correction
 return y

class Kernels:
 def __init__(self,compiler=torch.compile):
  self.prefixes={n:compiler(f,mode='default',fullgraph=True,dynamic=False) for n,f in [('highest_residual20',highest_prefix),('jacobi2',prefix_jacobi2)]}
  self.two=compiler(candidate_retraction,mode='default',fullgraph=True,dynamic=False)
  self.three=compiler(retract_three,mode='default',fullgraph=True,dynamic=False)
 def evaluate(self,name,cov,q,eager=False):
  if name not in ('highest_residual20','jacobi2','jacobi2_cubic3'):raise ValueError('Unknown cubic3control policy')
  key='highest_residual20' if name=='highest_residual20' else 'jacobi2'
  with precision('highest'):
   f=highest_prefix if key=='highest_residual20' else prefix_jacobi2
   y=f(cov,q) if eager else self.prefixes[key](cov,q)
   ret=retract_three if name=='jacobi2_cubic3' else candidate_retraction
   return ret(y) if eager else (self.three if name=='jacobi2_cubic3' else self.two)(y)
