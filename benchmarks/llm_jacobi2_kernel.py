"""Pure compile endpoints; separate prototype, untouched production helpers."""
import torch
from native_residual_named_kernel import highest_prefix,candidate_retraction
from tracking_curvature_kernel import generator,damped
from hybrid_training_kernel import precision

def prefix_jacobi2(cov,q):
 scale=(cov.trace().abs()/len(q)).clamp_min(1e-12)
 c=cov.float()/scale
 b=q.float().T@c@q.float();b=.5*(b+b.T)
 a=generator(b,'jacobi')[2]*2.
 return q.float()+(q.bfloat16()@a.bfloat16()).float()

class Kernels:
 def __init__(self,compiler=torch.compile):
  self.prefixes={n:compiler(f,mode='default',fullgraph=True,dynamic=False) for n,f in [('highest_residual20',highest_prefix),('jacobi2',prefix_jacobi2)]}
  self.ret=compiler(candidate_retraction,mode='default',fullgraph=True,dynamic=False)
  self.damped=compiler(damped,mode='default',fullgraph=True,dynamic=False)
 def evaluate(self,name,cov,q,eager=False):
  with precision('highest'):
   if name=='damped20':return damped(cov,q) if eager else self.damped(cov,q)
   if name not in self.prefixes:raise ValueError('Unknown prototype policy')
   f=highest_prefix if name=='highest_residual20' else prefix_jacobi2
   y=f(cov,q) if eager else self.prefixes[name](cov,q)
   return candidate_retraction(y) if eager else self.ret(y)
