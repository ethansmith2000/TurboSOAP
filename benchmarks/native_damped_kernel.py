"""Keep Jacobi's existing compiled path; give damped live updates exact references."""
from collections import Counter
import torch
from native_residual_named_kernel import PolicyKernel as OriginalKernel,candidate_retraction
from hybrid_training_kernel import precision
from tracking_curvature_kernel import damped,prefix_damped,geometry
from native_damped_contract import POLICIES

class LiveKernel(OriginalKernel):
    def __init__(self,name,router,compiler=torch.compile):
        if name not in POLICIES:raise ValueError('Unknown live damped pilot policy')
        if name=='highest_residual20':
            super().__init__(name,router,compiler)
        else:
            self.name=name;self.router=router;self.mode='highest';self.calls=Counter()
            self.eager_prefix=prefix_damped;self.eager_ret=candidate_retraction
            self.whole=compiler(damped,mode='default',fullgraph=True,dynamic=False)

    def evaluate(self,cov,q,eager=False):
        if self.name=='highest_residual20':return super().evaluate(cov,q,eager)
        with precision('highest'):
            return self.eager_ret(self.eager_prefix(cov,q)) if eager else self.whole(cov,q)

    def __call__(self,cov,q,group):
        new=super().__call__(cov,q,group)
        if self.router.probe:
            with precision('highest'):g=geometry(new)
            row=self.router.probes[-1];row.update(absolute_gram_fro=g['absolute_gram_fro'],absolute_geometry_pass=g['admitted'])
            row['passed']&=g['admitted'];self.router.emit('actual_live_geometry_before_gate',dict(policy=self.name,dimension=len(q),**g))
            if not row['passed']:raise FloatingPointError('Actual live-policy absolute geometry failed')
        return new

class PolicyRouter:
    def __init__(self,emit,rng_digest,compiler=torch.compile):
        self.emit=emit;self.rng_digest=rng_digest;self.calls=Counter();self.probes=[];self.moment_probes=[];self.probe=False
        self.kernels={a:LiveKernel(a,self,compiler) for a in POLICIES};self.cf_calls=0

    def evaluate_cf(self,name,cov,q):
        self.cf_calls+=1
        if self.cf_calls>120:raise RuntimeError('Declared native CF call bound')
        return self.kernels[name].evaluate(cov,q)
