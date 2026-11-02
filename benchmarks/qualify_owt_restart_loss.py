"""Bounded same-weight, real-batch eager/compiled CE gradient attribution."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

BUNDLE=Path('/workspace/optimizer_replay_results/soap_owt_restart_20261026_v4')
sys.path[:0]=[str(BUNDLE/'source'),str(BUNDLE/'source/benchmarks')]
from owt_restart import Transformer, token_loss, tensor_digest


def main():
    start=time.monotonic()
    assert os.environ.get('GPU_CLAIM_INDICES') and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.manual_seed(204)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    model=Transformer(dim=768,depth=12,heads=12,ff_mult=2.,ff_hidden_dim=1536,
        vocab_size=50257,max_seq_len=1024,tie_embeddings=True,use_rope=True,
        qk_norm=True,bias=False,dropout=0.,fused_qkv=True,fused_swiglu=True,
        gradient_checkpointing=False).cuda()
    initial=tensor_digest(model.named_parameters())
    cohort=np.load(BUNDLE/'train.npy',mmap_mode='r')
    tokens=torch.from_numpy(np.array(cohort[:32],dtype=np.int64)).cuda()
    reference=None;losses=[];rows=[]
    for path in ('eager','compiled'):
        loss_fn=token_loss if path=='eager' else torch.compile(token_loss,fullgraph=True,
                            dynamic=False,options={'emulate_precision_casts':True})
        model.zero_grad(set_to_none=True)
        loss_value=0.
        for t in tokens.split(8):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):
                loss=loss_fn(model(t[:,:-1]),t[:,1:])/4
            loss.backward();loss_value+=float(loss.detach())
        losses.append(loss_value)
        if path=='eager':
            reference={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
        else:
            squared=norm_squared=dot=compiled_squared=0.
            for n,p in model.named_parameters():
                actual=p.grad.detach().cpu();expected=reference[n]
                delta=float((actual.double()-expected.double()).square().sum())
                a2=float(actual.double().square().sum());e2=float(expected.double().square().sum())
                product=float((actual.double()*expected.double()).sum())
                squared+=delta;norm_squared+=e2;compiled_squared+=a2;dot+=product
                rows.append(dict(parameter=n,gradient_relative=(delta/max(e2,1e-30))**.5))
            relative=(squared/norm_squared)**.5
            cosine=dot/(norm_squared*compiled_squared)**.5
    assert initial==tensor_digest(model.named_parameters())
    passed=relative<=.02 and cosine>=.999 and abs(losses[1]-losses[0])<=.002
    result=dict(status='passed' if passed else 'failed',eager_ce=losses[0],compiled_ce=losses[1],
        gradient_relative=relative,gradient_cosine=cosine,parameter_gradients=rows,
        initial_parameters_sha256=initial,model_unchanged=True,elapsed_seconds=time.monotonic()-start,
        criteria=dict(maximum_gradient_relative=.02,minimum_cosine=.999,maximum_ce_difference=.002),
        scope='Numerical execution screen at one initial real batch; not a learning equivalence guarantee',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (BUNDLE/'loss_execution_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='parameter_gradients'}),flush=True)
    assert passed


if __name__=='__main__':main()
