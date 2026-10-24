"""Sparse-QR context control; qualified update body with cadence dispatch only."""
import math
import torch
from reference_blocked_soap import ReferenceBlockedSOAP, precision


class SparseReferenceBlockedSOAP(ReferenceBlockedSOAP):
    def __init__(self, named_parameters, arm='sparse', refresh_interval=50, **kwargs):
        if arm != 'sparse' or not isinstance(refresh_interval,int) or refresh_interval<=0:
            raise ValueError('Require sparse arm and a positive integer QR interval')
        super().__init__(named_parameters,arm='qr',**kwargs)
        self.arm='sparse'
        self.refresh_interval=refresh_interval

    def refresh_kind(self,call):
        if call<=50:return 'eigh'
        return 'ordinary' if call%self.refresh_interval else 'qr'

    @torch.no_grad()
    def step(self):
        self.call+=1;self.step_tensor.add_(1);event=self.refresh_kind(self.call)
        with precision('highest'):
            for group in self.groups:
                s=group['state'];meta=group['meta'];params=group['params']
                gradients=torch.stack([p.grad for p in params]);pstack=torch.stack(params)
                s['momentum'],s['lc'],s['rc']=self.covariance(gradients,s['momentum'],s['lc'],s['rc'],meta)
                if event!='ordinary':
                    for side,axis in (('l',-2),('r',-1)):
                        if not meta[axis]:continue
                        cov,q=s[side+'c'],s[side+'q']
                        nu=s['nu'] if side=='l' else s['nu'].transpose(-1,-2)
                        eps=torch.full((len(cov),),1e-10,device=cov.device)
                        if event=='eigh':
                            _,vectors=torch.linalg.eigh(cov+eps[:,None,None]*torch.eye(cov.shape[-1],device=cov.device))
                            q=vectors.flip(-1)
                        elif event=='qr':q,nu=self.qr(cov,q,nu,eps)
                        else:
                            with precision('high'):q,nu=self.warm(cov,q,nu,eps)
                        s[side+'q']=q
                        s['nu']=nu if side=='l' else nu.transpose(-1,-2)
                delta,s['nu'],s['vector']=self.projected(gradients,s['momentum'],s['nu'],s['lq'],s['rq'],s['vector'],pstack,self.step_tensor,meta,group['embedding'],5 if self.call<=50 else 1,self.horizon,self.warmup)
                torch._foreach_add_(params,list(delta.unbind(0)))
        return event
