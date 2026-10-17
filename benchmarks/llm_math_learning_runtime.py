"""Token-weighted fixed LLM evaluation on the qualified math-attention runtime."""
import math
import torch
from torch.nn import functional as F
from llm_math_attention_runtime import Runtime as PreviousRuntime,FlatOutput,build_model,captured_backward
from llm_math_learning_contract import evaluation_ids,ids_sha
class Runtime(PreviousRuntime):
 def initialize(self):
  super().initialize();p=self.protocol['llm_evaluation']
  if len(self.train_dataset)!=p['train_blocks'] or len(self.validation_dataset)!=p['validation_blocks']:raise ValueError('Pinned evaluation block coverage changed')
  ids=evaluation_ids(self.config['seed'],len(self.train_dataset),len(self.validation_dataset))
  if any(ids_sha(v)!=p['panels'][n]['ids_sha256'] or len(v)!=p['panels'][n]['blocks'] for n,v in ids.items()):raise ValueError('Fixed evaluation panel changed')
  self.ids={n:torch.tensor(v,dtype=torch.int64) for n,v in ids.items()}
  self.emit('evaluation_plan',dict(protocol=p,ids_sha256={n:ids_sha(v) for n,v in ids.items()},rng_untouched=True))
 @torch.no_grad()
 def evaluate(self):
  p=self.protocol['llm_evaluation'];seq=self.config['sequence_length'];device=next(self.model.parameters()).device
  metrics={}
  for name,ids in self.ids.items():
   dataset=self.validation_dataset if name=='validation' else self.train_dataset
   total_ce=0.;correct=0;tokens=0
   for start in range(0,len(ids),p['batch_size']):
    self.budget();subset=ids[start:start+p['batch_size']]
    offsets=subset[:,None]*seq+torch.arange(seq+1,dtype=torch.int64)[None,:]
    blocks=dataset.tokens[offsets].to(device=device,dtype=torch.int64)
    x=blocks[:,:-1].contiguous();target=blocks[:,1:].contiguous().reshape(-1)
    with self.autocast():logits=self.active(x)
    ce=float(F.cross_entropy(logits.float(),target,reduction='sum'))
    if not math.isfinite(ce) or ce<0:raise FloatingPointError('Nonfinite token evaluation CE')
    total_ce+=ce;correct+=int((logits.argmax(dim=-1)==target).sum());tokens+=target.numel()
    del blocks,x,target,logits,offsets,subset
   if not tokens or tokens!=len(ids)*seq:raise RuntimeError('Evaluation valid-token coverage changed')
   metrics[name]=dict(ce=total_ce/tokens,accuracy=correct/tokens,examples=len(ids),valid_tokens=tokens)
  return metrics
 def before_smoke(self):
  from prepare_llm_math_learning import verify_backend
  from execution_fingerprints import backend_record
  record=backend_record();verify_backend(self.bundle,record);self.emit('backend_before_smoke',dict(record=record,passed=True,execution_contract=self.execution_contract()))
 def after_measured(self):
  from prepare_llm_math_learning import verify_backend
  from execution_fingerprints import backend_record
  record=backend_record();verify_backend(self.bundle,record);self.emit('backend_after_measured',dict(record=record,passed=True,execution_contract=self.execution_contract()))
  self.retain_code('after_measured');self.budget(resources=True)
