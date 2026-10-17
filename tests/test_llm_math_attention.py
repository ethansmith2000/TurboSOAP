from pathlib import Path
import sys,copy
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import torch,pytest
from llm_math_attention_contract import recipe,protocol,SEED,PARAMETERS,SNAPSHOT_PURPOSE
from llm_math_attention_runtime import Runtime,build_model,FlatOutput

def flags():return tuple(fn() for fn in (torch.backends.cuda.flash_sdp_enabled,torch.backends.cuda.mem_efficient_sdp_enabled,torch.backends.cuda.math_sdp_enabled,torch.backends.cuda.cudnn_sdp_enabled))
def test_math_context_restores_flags_and_exact_repeated_loss_gradient_outside_context():
 c=recipe();c.update(dim=16,heads=4,ff_hidden_dim=32,sequence_length=8,batch_size=2,vocab_size=32);torch.manual_seed(SEED)
 model=build_model(c);runtime=object.__new__(Runtime);runtime.model=model;runtime.active=FlatOutput(model)
 before=flags();reference=None;x=torch.arange(16).reshape(2,8)%32;y=(x+1).reshape(-1)%32
 for _ in range(3):
  model.zero_grad(set_to_none=True)
  with runtime.autocast():
   assert flags()==(False,False,True,False);loss=torch.nn.functional.cross_entropy(runtime.active(x),y)
  assert flags()==before
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
  gradients={n:p.grad.clone() for n,p in model.named_parameters()}
  if reference is not None:assert all(torch.equal(gradients[n],reference[n]) for n in reference)
  reference=gradients
 assert runtime.execution_contract()['compiled_model'] is False and flags()==before

def test_recipe_optimizer_and_exact_gates_unchanged():
 from llm_eager_model_contract import recipe as previous,protocol as pp
 old=previous();old['seed']=SEED;old['attention_backend']=recipe()['attention_backend'];assert recipe()==old
 assert protocol()['gates']==pp()['gates'] and protocol()['model_parity']==pp()['model_parity']

def test_preparation_rejects_resealed_attention_policy_change(tmp_path,monkeypatch):
 import prepare_llm_math_attention as prep
 parent=tmp_path/'parent';parent.mkdir();(parent/'source').mkdir();prep.write(parent/'manifest.json',dict(source_sha256={}));prep.write(parent/'execution_status.json',dict(status='failed'));monkeypatch.setattr(prep,'PARENTS',(parent,))
 qual=tmp_path/'qual';qual.mkdir();prep.write(qual/'execution_status.json',dict(status='completed'));prep.write(qual/'summary.json',dict(policies=[dict(policy='math_attention',all_exact=True)]));monkeypatch.setattr(prep,'QUALIFICATION',qual)
 cache=tmp_path/'cache';cache.write_bytes(b'fixeddata');monkeypatch.setattr(prep,'CACHE',cache)
 audit=tmp_path/'audit.json';prep.write(audit,dict(status='PASS_CPU_TOKEN_CACHE_AUDIT',cuda_initialized=False,path=str(cache),file_sha256=prep.sha(cache)))
 ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=SEED,parameters=PARAMETERS,cuda_initialized=False,snapshot_purpose_bytes=SNAPSHOT_PURPOSE))
 b=tmp_path/'fresh';m=prep.prepare(b,ref,data_audit=audit);assert len(m['source_sha256'])==5
 c=recipe();c['attention_backend']='Flash';prep.write(b/'config.json',c);altered=copy.deepcopy(m);altered['artifacts']['config.json']=prep.sha(b/'config.json');prep.write(b/'manifest.json',altered)
 with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
