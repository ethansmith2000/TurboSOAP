"""Confirm only declared seed/order/panel changes and inherited evaluation safeguards."""
from pathlib import Path
import sys,copy,random
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support'),str(Path(__file__).resolve().parent)]
import numpy as np
import pytest,torch
from llm_math_confirmation_contract import recipe,protocol,evaluation_ids,SEED,PARAMETERS,SNAPSHOT_PURPOSE
from llm_math_confirmation_runtime import Runtime
from llm_math_confirmation import evaluate_transaction,endpoint
from matched_training_runtime import streaming_digest
from test_llm_math_learning import runtime_fixture

def test_independent_recipe_only_declared_seed_order_scope_panels_differ():
 from llm_math_learning_contract import recipe as prior,protocol as pp
 old=prior();now=recipe();assert now['seed']==172
 assert {k:v for k,v in now.items() if k not in ('seed','matched_comparison')}=={k:v for k,v in old.items() if k not in ('seed','matched_comparison')}
 assert now['matched_comparison']['execution_order']==['qr20','highest_residual20','damped20']
 assert now['matched_comparison']['steps_per_arm']==old['matched_comparison']['steps_per_arm']==3500
 p=protocol();before=pp()
 allowed={'seed','execution_order','implementation','llm_evaluation','bound_rationale','decision'}
 assert {k:v for k,v in p.items() if k not in allowed}=={k:v for k,v in before.items() if k not in allowed}
 assert p['gates']==before['gates'] and p['expected']==before['expected'] and p['bounds']==before['bounds'] and p['precision']==before['precision']
 a=p['llm_evaluation'];b=before['llm_evaluation'];assert {k:v for k,v in a.items() if k!='panels'}=={k:v for k,v in b.items() if k!='panels'}
 assert a['panels']['validation']==b['panels']['validation'] and a['panels']['train_probe']['ids_sha256']!=b['panels']['train_probe']['ids_sha256']
 assert evaluation_ids()['train_probe']!=evaluation_ids(171)['train_probe']

@pytest.mark.parametrize('fails',[False,True])
def test_new_frozen_entry_reuses_weighted_eval_and_restores_rng_gradients_flags_on_failure(fails):
 base,opt=runtime_fixture();r=object.__new__(Runtime);r.__dict__.update(base.__dict__)
 flags=[m.training for m in r.model.modules()];before=endpoint(r,opt);grads=streaming_digest([p.grad for p in r.model.parameters()])
 def evaluate():
  random.random();np.random.random();torch.rand(2);torch.rand(2,generator=r.streams.crop);r.streams.augmentation.rng.random();r.model.table.train()
  if fails:raise ValueError('confirmation failure control')
  return Runtime.evaluate(r)
 r.evaluate=evaluate
 if fails:
  with pytest.raises(ValueError,match='confirmation failure'):evaluate_transaction(r,opt)
 else:
  row=evaluate_transaction(r,opt);assert row['token_weighted'] and row['restoration_passed'] and row['metrics']['validation']['valid_tokens']==9
 assert endpoint(r,opt)==before and [m.training for m in r.model.modules()]==flags and streaming_digest([p.grad for p in r.model.parameters()])==grads

def test_resealed_confirmation_semantic_change_rejected(tmp_path,monkeypatch):
 import prepare_llm_math_confirmation as prep
 parent=tmp_path/'parent';parent.mkdir();(parent/'source').mkdir();prep.write(parent/'manifest.json',dict(source_sha256={}));prep.write(parent/'execution_status.json',dict(status='completed'));monkeypatch.setattr(prep,'PARENTS',(parent,))
 qual=tmp_path/'qual';qual.mkdir();prep.write(qual/'execution_status.json',dict(status='completed'));prep.write(qual/'summary.json',dict(policies=[dict(policy='math_attention',all_exact=True)]));monkeypatch.setattr(prep,'QUALIFICATION',qual)
 cache=tmp_path/'cache';cache.write_bytes(b'pinned');monkeypatch.setattr(prep,'CACHE',cache)
 audit=tmp_path/'audit.json';prep.write(audit,dict(status='PASS_CPU_TOKEN_CACHE_AUDIT',cuda_initialized=False,path=str(cache),file_sha256=prep.sha(cache)))
 ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=SEED,parameters=PARAMETERS,cuda_initialized=False,snapshot_purpose_bytes=SNAPSHOT_PURPOSE))
 b=tmp_path/'fresh';m=prep.prepare(b,ref,data_audit=audit);assert len(m['source_sha256'])==5
 p=prep.read(b/'protocol.json');p['gates']['physical_m_reconstruction_limit']=.02;prep.write(b/'protocol.json',p);m['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',m)
 with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
