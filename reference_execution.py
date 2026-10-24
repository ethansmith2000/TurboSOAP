"""Execution-only alternatives for the qualified reference Llama math."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from reference_llama import ReferenceLlama, rms


def execute_block(block, x, cos, sin):
    return block(x, cos, sin)


def token_loss(logits, labels):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))


class ExecutionLlama(ReferenceLlama):
    def configure_execution(self, compiled=False, checkpoint_stride=1, backend='inductor'):
        if checkpoint_stride not in (0,1,2):
            raise ValueError('Checkpoint stride must be zero, one or two')
        self.checkpoint_stride=checkpoint_stride
        options={'emulate_precision_casts':True} if backend=='inductor' else None
        if not hasattr(self,'compiled_block'):
            self.compiled_block=torch.compile(execute_block,backend=backend,fullgraph=True,
                                              dynamic=False,options=options)
        self.execute=self.compiled_block if compiled else execute_block

    def forward(self,tokens):
        x=F.embedding(tokens,self.embed)
        if tokens.is_cuda:x=x.bfloat16()
        cos,sin=self.cos[:,:tokens.shape[1]],self.sin[:,:tokens.shape[1]]
        for i,block in enumerate(self.blocks):
            if self.recompute and self.checkpoint_stride and i%self.checkpoint_stride==0:
                x=checkpoint(self.execute,block,x,cos,sin,use_reentrant=False,preserve_rng_state=False)
            else:x=self.execute(block,x,cos,sin)
        return rms(x)@self.readout
