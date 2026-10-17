"""Independent token-metric oracle and state-restoration failure controls."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import sys,copy,random
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import numpy as np
import pytest,torch
from torch.nn import functional as F
from train_llm import PackedTokenDataset
from llm_math_learning_contract import recipe,protocol,evaluation_ids,ids_sha,SEED,PARAMETERS,SNAPSHOT_PURPOSE
from llm_math_learning_runtime import Runtime
from llm_math_learning import evaluate_transaction,endpoint
from matched_training_contract import RandomStreams
from matched_training_runtime import streaming_digest

def test_semantic_recipe_counters_panel_and_original_gates():
 from llm_math_attention_contract import recipe as prior,protocol as pp
 c=prior();c['seed']=SEED;c['matched_comparison']=recipe()['matched_comparison'];assert recipe()==c
 p=protocol();old=pp()
 assert all(p['gates'][k]==v for k,v in old['gates'].items() if k!='no_speed_or_learning_qualification')
 assert p['model_parity']==old['model_parity'] and p['compiler_isolation']==old['compiler_isolation']
 assert p['expected']['hybrid_kernel_calls']==2688 and p['expected']['policy_calls_by_dimension']['damped20']=={'2048':840,'4096':168,'6144':168,'8192':168}
 assert p['expected']['primary']['damped20']['research_warm_refreshes']==157
 assert p['expected']['primary']['damped20']['research_qr_refreshes']==17
 assert p['expected']['primary']['qr20']['research_qr_refreshes']==174
 state=random.getstate();panels=evaluation_ids();assert random.getstate()==state
 assert len(set(panels['train_probe']))==256 and panels['validation']==list(range(511))
 assert p['llm_evaluation']['panels']['validation']['valid_tokens']==261632
 assert all(ids_sha(ids)==p['llm_evaluation']['panels'][name]['ids_sha256'] for name,ids in panels.items())
 assert p['llm_data']['training_tokens_per_primary_arm']==28672000

class FixedLogits(torch.nn.Module):
 def __init__(self):
  super().__init__();self.table=torch.nn.Embedding(4,4)
  with torch.no_grad():self.table.weight.copy_(torch.tensor([[1.,5.,0.,-1.],[4.,0.,0.,0.],[-3.,-2.,-1.,0.],[0.,0.,0.,0.]]))
  self.seen=[]
 def forward(self,x):
  assert not torch.is_grad_enabled();self.seen.append(len(x));return self.table(x).reshape(-1,4)

def runtime_fixture():
 r=object.__new__(Runtime);r.model=FixedLogits();r.active=r.model;r.config=dict(sequence_length=3)
 r.ids=dict(validation=torch.tensor([0,1,3]),train_probe=torch.tensor([0,2]))
 r.protocol=dict(llm_evaluation=dict(batch_size=2,panels={n:dict(valid_tokens=len(ids)*3) for n,ids in r.ids.items()}))
 r.train_dataset=r.validation_dataset=PackedTokenDataset(torch.tensor([0,1,0,2,3,3,1,0,2,0,0,1,2]),3)
 r.autocast=nullcontext;r.budget=lambda **kw:None
 r.streams=RandomStreams(torch.Generator().manual_seed(13),SimpleNamespace(rng=np.random.default_rng(14),counts={}))
 r.model.table.eval();r.model.train();r.model.table.eval() # Heterogeneous flags deliberately.
 r.model.table.weight.grad=torch.ones_like(r.model.table.weight)
 opt=SimpleNamespace(matrix=SimpleNamespace(),auxiliary=SimpleNamespace(),state_dict=lambda:dict(moment=torch.ones(2)))
 return r,opt

def test_token_metrics_match_independent_per_target_oracle_and_uneven_batch():
 r,_=runtime_fixture();before=streaming_digest(r.model.state_dict());g=streaming_digest([p.grad for p in r.model.parameters()])
 result=r.evaluate()
 for name,ids in r.ids.items():
  losses=[];correct=[]
  for idx in ids.tolist():
   x,y=r.validation_dataset[idx]
   for token,target in zip(x.tolist(),y.tolist()):
    scores=r.model.table.weight.detach()[token].double()
    losses.append(float(torch.logsumexp(scores,0)-scores[target]));correct.append(int(scores.argmax())==target)
  assert result[name]['ce']==pytest.approx(sum(losses)/len(losses),abs=3e-7)
  assert result[name]['accuracy']==sum(correct)/len(correct) and result[name]['valid_tokens']==len(losses)
  assert result[name]['examples']==len(ids)
 assert r.model.seen==[2,1,2] and before==streaming_digest(r.model.state_dict()) and g==streaming_digest([p.grad for p in r.model.parameters()])
 # With three blocks split2+1, a mean of batch means is a different answer.
 ids=r.ids['validation'];means=[]
 for subset in (ids[:2],ids[2:]):
  pairs=[r.validation_dataset[int(i)] for i in subset]
  x=torch.stack([a for a,b in pairs]);y=torch.stack([b for a,b in pairs]).reshape(-1)
  means.append(float(F.cross_entropy(r.model.table(x).detach().reshape(-1,4),y)))
 assert abs(sum(means)/2-result['validation']['ce'])>.01

@pytest.mark.parametrize('fails',[False,True])
def test_transaction_restores_global_private_rng_heterogeneous_flags_gradients_on_success_and_failure(fails):
 r,opt=runtime_fixture();original=r.evaluate;flags=[m.training for m in r.model.modules()];before=endpoint(r,opt);grads=streaming_digest([p.grad for p in r.model.parameters()])
 def evaluate():
  random.random();np.random.random();torch.rand(3);torch.rand(2,generator=r.streams.crop);r.streams.augmentation.rng.random();r.model.table.train()
  if fails:raise ValueError('deliberate evaluation failure')
  return original()
 r.evaluate=evaluate
 if fails:
  with pytest.raises(ValueError,match='deliberate'):evaluate_transaction(r,opt)
 else:
  row=evaluate_transaction(r,opt);assert row['restoration_passed'] and row['token_weighted']
 assert endpoint(r,opt)==before and [m.training for m in r.model.modules()]==flags
 assert streaming_digest([p.grad for p in r.model.parameters()])==grads

def test_transaction_detects_gradient_mutation_even_when_evaluator_throws():
 r,opt=runtime_fixture()
 def broken():
  r.model.table.weight.grad.add_(1);raise ValueError('broken')
 r.evaluate=broken
 with pytest.raises(RuntimeError,match='mutated'):evaluate_transaction(r,opt)

def test_preparation_rejects_resealed_token_metric_policy_change(tmp_path,monkeypatch):
 import prepare_llm_math_learning as prep
 parent=tmp_path/'parent';parent.mkdir();(parent/'source').mkdir();prep.write(parent/'manifest.json',dict(source_sha256={}));prep.write(parent/'execution_status.json',dict(status='completed'));monkeypatch.setattr(prep,'PARENTS',(parent,))
 qual=tmp_path/'qual';qual.mkdir();prep.write(qual/'execution_status.json',dict(status='completed'));prep.write(qual/'summary.json',dict(policies=[dict(policy='math_attention',all_exact=True)]));monkeypatch.setattr(prep,'QUALIFICATION',qual)
 cache=tmp_path/'cache';cache.write_bytes(b'pinned');monkeypatch.setattr(prep,'CACHE',cache)
 audit=tmp_path/'audit.json';prep.write(audit,dict(status='PASS_CPU_TOKEN_CACHE_AUDIT',cuda_initialized=False,path=str(cache),file_sha256=prep.sha(cache)))
 ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=SEED,parameters=PARAMETERS,cuda_initialized=False,snapshot_purpose_bytes=SNAPSHOT_PURPOSE))
 b=tmp_path/'fresh';m=prep.prepare(b,ref,data_audit=audit);assert len(m['source_sha256'])==5
 p=prep.read(b/'protocol.json');p['llm_evaluation']['metrics']='mean of batch means';prep.write(b/'protocol.json',p)
 m['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',m)
 with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
