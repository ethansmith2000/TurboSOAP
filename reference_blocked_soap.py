"""Base-width blocked, normalized SOAP port, isolated from old optimizers.

Only the pinned recipe's base width is implemented: shape/epsilon/LR relative
scales are all one. This is not a general implementation of its width scaling.
"""
from collections import defaultdict
from contextlib import contextmanager
import math
import torch
from torch.nn import functional as F
from batched_soap_refresh import batched_qr_refresh, batched_warm_refresh


@contextmanager
def precision(mode):
    old=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(mode)
    try:yield
    finally:torch.set_float32_matmul_precision(old)


def metadata(m,n,block=512,maximum=10000):
    left,right=m<=maximum,n<=maximum
    nr=math.ceil(m/block) if left else 1; nc=math.ceil(n/block) if right else 1
    r=block if nr>1 else m; c=block if nc>1 else n
    return (m,n,nr,nc,r,c,left,right)


def tiles(x,meta):
    m,n,nr,nc,r,c,_,_=meta
    x=F.pad(x,(0,nc*c-n,0,nr*r-m))
    return x.reshape(-1,nr,r,nc,c).permute(0,1,3,2,4).reshape(-1,r,c)


def merge(x,meta,batch):
    m,n,nr,nc,r,c,_,_=meta
    return x.reshape(batch,nr,nc,r,c).permute(0,1,3,2,4).reshape(batch,nr*r,nc*c)[:,:m,:n]


def covariance(g,momentum,left_cov,right_cov,meta):
    moment=.9*momentum+.1*g
    t=tiles(g,meta)
    if meta[-2]:left_cov=.999*left_cov+.001*(t@t.transpose(-1,-2))
    if meta[-1]:right_cov=.999*right_cov+.001*(t.transpose(-1,-2)@t)
    return moment,left_cov,right_cov


def projected_update(g,momentum,nu,lq,rq,vector,p,step,meta,embedding,power_steps,horizon,warmup):
    g=tiles(g,meta); m=tiles(momentum,meta)
    if meta[-2]:g=lq.transpose(-1,-2)@g; m=lq.transpose(-1,-2)@m
    if meta[-1]:g=g@rq; m=m@rq
    nu=.999*nu+.001*g.square()
    u=m/(nu.sqrt()+1e-8)
    if meta[-1]:u=u@rq.transpose(-1,-2)
    if meta[-2]:u=lq@u
    u=merge(u,meta,p.shape[0])*(torch.sqrt(1-.999**step)/(1-.9**step))
    v=vector
    for _ in range(power_steps):
        y=(u@v.unsqueeze(-1)).squeeze(-1)
        sigma=torch.linalg.vector_norm(y,dim=-1,keepdim=True)
        z=(u.transpose(-1,-2)@y.unsqueeze(-1)).squeeze(-1)
        candidate=z/(torch.linalg.vector_norm(z,dim=-1,keepdim=True)+1e-10)
        v=torch.where(torch.linalg.vector_norm(candidate,dim=-1,keepdim=True)>.5,candidate,v)
    if embedding:u=u/(u.square().mean(dim=(-2,-1),keepdim=True).sqrt()+1e-10)
    else:u=u*(math.sqrt(meta[1]/meta[0])/(sigma.unsqueeze(-1)+1e-10))
    t=step-1
    schedule=torch.where(t<warmup,t/warmup,(1-(t-warmup)/(horizon-warmup)).clamp_min(0))
    delta=-schedule*(.032*u+.0002*p)
    return delta,nu,v


def kind(call,arm):
    if call<=50:return 'eigh'
    if call%10:return 'ordinary'
    return 'qr' if arm=='qr' or call%50==0 else 'warm'


class ReferenceBlockedSOAP:
    def __init__(self,named_parameters,arm='qr',block=512,maximum=10000,compile_kernels=True,
                 seed=203,horizon=28332,warmup=177):
        assert arm in ('qr','warm'); self.arm=arm;self.call=0;self.horizon=horizon;self.warmup=warmup
        grouped=defaultdict(list)
        for name,p in named_parameters:
            assert p.ndim==2 and p.dtype==torch.float32
            grouped[(*p.shape,name=='embed')].append((name,p))
        self.groups=[]
        for (m,n,embedding),items in grouped.items():
            params=[p for _,p in items]; device=params[0].device; meta=metadata(m,n,block,maximum)
            nr,nc,r,c,left,right=meta[2:]; b=len(params); count=b*nr*nc
            gen=torch.Generator(device=device).manual_seed(seed+len(self.groups))
            v=torch.randn(b,n,device=device,generator=gen);v=v/(v.norm(dim=-1,keepdim=True)+1e-10)
            zero=torch.empty(0,device=device)
            state=dict(momentum=torch.zeros(b,m,n,device=device),nu=torch.zeros(count,r,c,device=device),
                       lc=torch.zeros(count,r,r,device=device) if left else zero,
                       rc=torch.zeros(count,c,c,device=device) if right else zero,
                       lq=torch.eye(r,device=device).expand(count,-1,-1).clone() if left else zero,
                       rq=torch.eye(c,device=device).expand(count,-1,-1).clone() if right else zero,
                       vector=v)
            self.groups.append(dict(names=[name for name,_ in items],params=params,meta=meta,embedding=embedding,state=state))
        device=self.groups[0]['params'][0].device
        self.step_tensor=torch.zeros((),device=device)
        compiler=(lambda fn:torch.compile(fn,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True})) if compile_kernels else (lambda fn:fn)
        self.covariance=compiler(covariance);self.projected=compiler(projected_update)
        self.qr=compiler(batched_qr_refresh);self.warm=compiler(batched_warm_refresh)

    @torch.no_grad()
    def step(self):
        self.call+=1;self.step_tensor.add_(1);event=kind(self.call,self.arm)
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
