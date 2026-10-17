from pathlib import Path
import sys,copy
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import torch,pytest
from torch._inductor import config as inductor
from llm_eager_model_contract import recipe,protocol,SEED,PARAMETERS,SNAPSHOT_PURPOSE
from llm_eager_model_runtime import Runtime,FlatOutput,build_model
from native_damped_kernel import PolicyRouter

def test_eager_model_manual_mean_loss_gradient_and_no_model_compile(monkeypatch):
 c=recipe();c.update(dim=16,heads=4,ff_hidden_dim=32,sequence_length=8,batch_size=2,vocab_size=32);torch.manual_seed(SEED)
 model=build_model(c);active=FlatOutput(model);x=torch.arange(16).reshape(2,8)%32;y=(x+1)%32
 from torch._dynamo.utils import counters
 before={k:dict(v) for k,v in counters.items() if v}
 actual=torch.nn.functional.cross_entropy(active(x),y.reshape(-1));actual.backward();grads={n:p.grad.clone() for n,p in model.named_parameters()};model.zero_grad(set_to_none=True)
 expected=model(x,y);expected.backward();assert torch.equal(actual,expected)
 assert all(torch.equal(p.grad,grads[n]) for n,p in model.named_parameters())
 assert {k:dict(v) for k,v in counters.items() if v}==before and inductor.emulate_precision_casts is False
 runtime=object.__new__(Runtime);runtime.model=model;runtime.active=active;assert runtime.execution_contract()['passed']
 runtime.active=torch.nn.Identity()
 with pytest.raises(RuntimeError,match='execution contract'):runtime.execution_contract()

def test_original_optimizer_backend_options_remain_unmodified(monkeypatch):
 import torch._inductor.compile_fx as backend
 torch._dynamo.reset();records=[]
 def capture(gm,inputs,**kw):records.append(dict(kw.get('config_patches') or {}));return gm.forward
 monkeypatch.setattr(backend,'compile_fx',capture)
 router=PolicyRouter(lambda *a:None,lambda:'rng');cov=torch.diag(torch.linspace(1,2,8))+.01*torch.ones(8,8);q=torch.eye(8)
 for kernel in router.kernels.values():torch.testing.assert_close(kernel.evaluate(cov,q),kernel.evaluate(cov,q,eager=True))
 assert records and all(not r.get('emulate_precision_casts',False) for r in records) and inductor.emulate_precision_casts is False;torch._dynamo.reset()

def test_eager_recipe_gates_and_semantic_resealing_rejected(tmp_path,monkeypatch):
 from llm_feasibility_contract import protocol as oldprotocol
 assert protocol()['gates']==oldprotocol()['gates']
 assert protocol()['model_parity']['logits_fro_relative']==oldprotocol()['model_parity']['logits_fro_relative']
 import prepare_llm_eager_model as prep
 parent=tmp_path/'parent';parent.mkdir();(parent/'source').mkdir();prep.write(parent/'manifest.json',dict(source_sha256={}));prep.write(parent/'execution_status.json',dict(status='failed'));monkeypatch.setattr(prep,'PARENTS',(parent,))
 qual=tmp_path/'qual';qual.mkdir();prep.write(qual/'execution_status.json',dict(status='completed'));prep.write(qual/'summary.json',dict(model_policies=[dict(policy='compiled_preserve_casts',candidate_pass=True)]));monkeypatch.setattr(prep,'QUALIFICATION',qual)
 cache=tmp_path/'cache';cache.write_bytes(b'fixeddata');monkeypatch.setattr(prep,'CACHE',cache)
 audit=tmp_path/'audit.json';prep.write(audit,dict(status='PASS_CPU_TOKEN_CACHE_AUDIT',cuda_initialized=False,path=str(cache),file_sha256=prep.sha(cache)))
 ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=SEED,parameters=PARAMETERS,cuda_initialized=False,snapshot_purpose_bytes=SNAPSHOT_PURPOSE))
 b=tmp_path/'fresh';m=prep.prepare(b,ref,data_audit=audit);assert len(m['source_sha256'])==5
 c=recipe();c['batch_size']=8;prep.write(b/'config.json',c);altered=copy.deepcopy(m);altered['artifacts']['config.json']=prep.sha(b/'config.json');prep.write(b/'manifest.json',altered)
 with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
