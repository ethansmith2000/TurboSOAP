"""PyTorch base-width adaptation of the pinned Qiu et al. Llama recipe.

Canonical [input, output] kernel layout matches the source optimizer. Native
PyTorch SDPA replaces JAX attention; seeds are framework-specific. No checkpoint
or training launcher is part of this module.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def rms(x):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)+1e-6)).to(x.dtype)


def rope(x, cos, sin):
    a,b=x.float().chunk(2,dim=-1)
    return torch.cat((a*cos-b*sin,b*cos+a*sin),dim=-1).to(x.dtype)


class Block(nn.Module):
    def __init__(self, width, expansion, head_dim, device=None):
        super().__init__(); self.head_dim=head_dim; self.heads=width//head_dim
        for name in ('query','key','value'):
            self.register_parameter(name, nn.Parameter(torch.randn(width,width,device=device)*width**-.5))
        self.output=nn.Parameter(torch.zeros(width,width,device=device))
        self.fc1=nn.Parameter(torch.randn(width,expansion*width,device=device)*width**-.5)
        self.fc3=nn.Parameter(torch.randn(width,expansion*width,device=device)*width**-.5)
        self.fc2=nn.Parameter(torch.zeros(expansion*width,width,device=device))

    def forward(self,x,cos,sin):
        z=rms(x); shape=(*x.shape[:2],self.heads,self.head_dim)
        q=rope(rms((z@self.query).reshape(shape)),cos,sin)
        k=rope(rms((z@self.key).reshape(shape)),cos,sin)
        v=(z@self.value).reshape(shape)
        out=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True)
        x=x+out.transpose(1,2).reshape_as(x)@self.output
        z=rms(x)
        return x+(F.silu(z@self.fc1)*(z@self.fc3))@self.fc2


class ReferenceLlama(nn.Module):
    def __init__(self,width=512,depth=32,expansion=4,head_dim=64,vocab=50257,sequence=1024,
                 recompute=True,device=None):
        super().__init__(); self.recompute=recompute
        self.embed=nn.Parameter(.1*torch.randn(vocab,width,device=device))
        self.blocks=nn.ModuleList([Block(width,expansion,head_dim,device) for _ in range(depth)])
        self.readout=nn.Parameter(torch.zeros(width,vocab,device=device))
        angles=torch.arange(sequence,device=device)[:,None]/(10000**(2*torch.arange(head_dim//2,device=device)/head_dim))[None,:]
        self.register_buffer('cos',angles.cos()[None,:,None,:],persistent=False)
        self.register_buffer('sin',angles.sin()[None,:,None,:],persistent=False)

    def forward(self,tokens):
        x=F.embedding(tokens,self.embed)
        if tokens.is_cuda:x=x.bfloat16()
        cos,sin=self.cos[:,:tokens.shape[1]],self.sin[:,:tokens.shape[1]]
        for block in self.blocks:
            x=checkpoint(block,x,cos,sin,use_reentrant=False,preserve_rng_state=False) if self.recompute else block(x,cos,sin)
        return rms(x)@self.readout
