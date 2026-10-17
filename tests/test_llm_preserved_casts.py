from pathlib import Path
import sys,json,copy
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import torch,pytest
from torch._inductor import config as inductor
from llm_preserved_casts_contract import recipe,protocol,SEED,ORDER,PARAMETERS,SNAPSHOT_PURPOSE
from llm_preserved_casts_runtime import compile_model,compiler_options,global_guard,build_model,FlatOutput
from native_damped_kernel import PolicyRouter

def tiny():
 c=recipe();c.update(dim=16,heads=4,ff_hidden_dim=32,sequence_length=8,batch_size=2,vocab_size=32);return c

def test_model_compile_options_do_not_propagate_to_optimizer_compiler(monkeypatch):
 import torch._inductor.compile_fx as backend
 torch._dynamo.reset();records=[]
 def capture(gm,inputs,**kw):records.append(dict(kw.get('config_patches') or {}));return gm.forward
 monkeypatch.setattr(backend,'compile_fx',capture)
 torch.manual_seed(SEED);model=build_model(tiny());active=compile_model(model)
 x=torch.arange(16).reshape(2,8)%32;actual=active(x);expected=FlatOutput(model)(x)
 torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
 assert len(records)==1 and records[0]['emulate_precision_casts'] is True
 assert inductor.emulate_precision_casts is False
 router=PolicyRouter(lambda *a:None,lambda:'rng')
 cov=torch.diag(torch.linspace(1,2,8))+.01*torch.ones(8,8);q=torch.eye(8)
 for kernel in router.kernels.values():kernel.evaluate(cov,q)
 assert len(records)>1 and all(not r.get('emulate_precision_casts',False) for r in records[1:])
 global_guard();torch._dynamo.reset()

def test_real_cpu_inductor_mean_loss_gradients_and_configuration_isolation():
 torch._dynamo.reset();torch.manual_seed(SEED);model=build_model(tiny());x=torch.arange(16).reshape(2,8)%32;y=(x+1).reshape(-1)%32
 expected=torch.nn.functional.cross_entropy(FlatOutput(model)(x),y);expected.backward();grads={n:p.grad.clone() for n,p in model.named_parameters()};model.zero_grad(set_to_none=True)
 active=compile_model(model);actual=torch.nn.functional.cross_entropy(active(x),y);actual.backward()
 torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
 for n,p in model.named_parameters():torch.testing.assert_close(p.grad,grads[n],atol=1e-5,rtol=1e-4)
 assert model.token_embedding.weight is model.lm_head.weight and compiler_options(active)['emulate_precision_casts'] is True
 global_guard();torch._dynamo.reset()

def test_fresh_recipe_only_model_option_seed_order_and_semantic_tamper(tmp_path,monkeypatch):
 from llm_feasibility_contract import recipe as oldrecipe,protocol as oldprotocol
 c=recipe();before=oldrecipe();before['seed']=SEED;before['model_compile_options']={'emulate_precision_casts':True};before['matched_comparison']=c['matched_comparison']
 assert c==before and c['matched_comparison']['execution_order']==list(ORDER)
 assert protocol()['gates']==oldprotocol()['gates'] and protocol()['model_parity']==oldprotocol()['model_parity']
 import prepare_llm_preserved_casts as prep
 parent=tmp_path/'parent';parent.mkdir();(parent/'source').mkdir();prep.write(parent/'manifest.json',dict(source_sha256={}));prep.write(parent/'execution_status.json',dict(status='failed'));monkeypatch.setattr(prep,'PARENTS',(parent,))
 qualification=tmp_path/'qualification';qualification.mkdir();prep.write(qualification/'execution_status.json',dict(status='completed'));prep.write(qualification/'summary.json',dict(model_policies=[dict(policy='compiled_preserve_casts',candidate_pass=True)]));monkeypatch.setattr(prep,'QUALIFICATION',qualification)
 cache=tmp_path/'cache';cache.write_bytes(b'fixeddata');monkeypatch.setattr(prep,'CACHE',cache)
 audit=tmp_path/'audit.json';prep.write(audit,dict(status='PASS_CPU_TOKEN_CACHE_AUDIT',cuda_initialized=False,path=str(cache),file_sha256=prep.sha(cache)))
 ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=SEED,parameters=PARAMETERS,cuda_initialized=False,snapshot_purpose_bytes=SNAPSHOT_PURPOSE))
 b=tmp_path/'fresh';m=prep.prepare(b,ref,data_audit=audit);assert len(m['source_sha256'])==5
 c['model_compile_options']['emulate_precision_casts']=False;prep.write(b/'config.json',c);altered=copy.deepcopy(m);altered['artifacts']['config.json']=prep.sha(b/'config.json');prep.write(b/'manifest.json',altered)
 with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
