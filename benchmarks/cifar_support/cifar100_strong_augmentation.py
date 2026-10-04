"""GPU crop/flip, torchvision RandAugment, then Mixup or area-correct CutMix.

RandAugment samples a policy per minibatch (torchvision batched semantics).
Crop/flip remain per-image. Mixed targets receive label smoothing exactly once.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import v2, InterpolationMode


def crop_flip_uint8(images, offsets, flips):
    x=F.pad(images.float(),(4,4,4,4),mode='reflect')
    b=torch.arange(len(images),device=images.device)[:,None,None,None]
    c=torch.arange(3,device=images.device)[None,:,None,None]
    rows=offsets[:,0,None]+torch.arange(32,device=images.device)[None,:]
    cols=offsets[:,1,None]+torch.arange(32,device=images.device)[None,:]
    x=x[b,c,rows[:,None,:,None],cols[:,None,None,:]]
    return torch.where(flips[:,None,None,None],x.flip(-1),x).to(torch.uint8)


def mixed_targets(labels, permutation, lam, classes=100, smoothing=.1):
    targets=F.one_hot(labels,classes).float()*(1-smoothing)+smoothing/classes
    return targets*lam+targets[permutation]*(1-lam)


def mix_batch(x, labels, permutation, lam, box=None, smoothing=.1):
    if box is None:
        mixed=x*lam+x[permutation]*(1-lam)
    else:
        y0,y1,x0,x1=box
        h,w=x.shape[-2:]
        if not (0<=y0<=y1<=h and 0<=x0<=x1<=w): raise ValueError('CutMix box')
        lam=1-(y1-y0)*(x1-x0)/(h*w)
        mixed=x.clone()
        mixed[:,:,y0:y1,x0:x1]=x[permutation,:,y0:y1,x0:x1]
    return mixed,mixed_targets(labels,permutation,lam,smoothing=smoothing),float(lam)


class StrongAugmentation:
    def __init__(self,c,mean,std):
        self.mean,self.std=mean,std
        self.rng=np.random.default_rng(c['seed']+3001)
        self.smoothing=c['label_smoothing']
        self.mixup_alpha=c['mixup_alpha']; self.cutmix_alpha=c['cutmix_alpha']
        self.ra=v2.RandAugment(num_ops=2,magnitude=c['randaugment_magnitude'],
                              interpolation=InterpolationMode.BILINEAR,
                              fill=[round(float(v)*255) for v in mean.flatten().cpu()])
        self.crop=torch.compile(crop_flip_uint8,mode='default',fullgraph=True,dynamic=False)
        self.counts={'mixup':0,'cutmix':0}

    def __call__(self,raw,labels,offsets,flips):
        raw=self.ra(self.crop(raw,offsets,flips))
        x=(raw.float()/255-self.mean)/self.std
        permutation=torch.tensor(self.rng.permutation(len(labels)),device=labels.device)
        if self.rng.random()<.5:
            kind='mixup'; lam=float(self.rng.beta(self.mixup_alpha,self.mixup_alpha)); box=None
        else:
            kind='cutmix'; lam=float(self.rng.beta(self.cutmix_alpha,self.cutmix_alpha))
            ratio=math.sqrt(1-lam); h,w=x.shape[-2:]
            ch,cw=int(h*ratio),int(w*ratio)
            cy,cx=int(self.rng.integers(h)),int(self.rng.integers(w))
            box=(max(0,cy-ch//2),min(h,cy+(ch+1)//2),
                 max(0,cx-cw//2),min(w,cx+(cw+1)//2))
        mixed,targets,effective_lam=mix_batch(x,labels,permutation,lam,box,self.smoothing)
        self.counts[kind]+=1
        return mixed,targets,dict(kind=kind,effective_lam=effective_lam)
