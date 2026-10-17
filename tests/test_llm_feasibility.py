from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import sys,json,copy
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import pytest,numpy as np,torch
from llm_feasibility_contract import recipe,protocol,ARMS,POLICIES,PARAMETERS,DIMENSIONS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE
from llm_feasibility_runtime import build_model,FlatOutput,captured_backward
from native_damped_kernel import PolicyRouter
from llm_feasibility_diagnostic import build_arm
from matched_training_contract import RandomStreams,SharedInitialState
from matched_training_runtime import streaming_digest,OptimizerLease
from cifar_optimizers import grouped_parameters

def tiny():
    c=recipe();c.update(dim=16,heads=4,ff_hidden_dim=32,sequence_length=8,batch_size=2,vocab_size=32)
    return c

def test_actual_llm_shape_ties_rope_snapshot_resources_and_recipe():
    c=recipe();p=protocol()
    with torch.device('meta'):model=build_model(c)
    groups=grouped_parameters(model,c)
    shapes=[tuple(v.shape) for g in groups if g['role']=='matrix' for v in g['params']]
    assert shapes==[(6144,2048),(2048,2048),(8192,2048),(2048,4096)]
    dims={}
    for s in shapes:
        for n in s:dims[str(n)]=dims.get(str(n),0)+1
    assert dims==DIMENSIONS and sum(v.numel() for v in model.parameters())==PARAMETERS
    assert model.token_embedding.weight is model.lm_head.weight
    assert not any(m._non_persistent_buffers_set for m in model.modules())
    sizes=[v.numel()*v.element_size() for v in model.state_dict().values()]
    assert sum(sizes)==SNAPSHOT_BYTES and sum(sizes)+3*max(sizes)+4*2**20==SNAPSHOT_PURPOSE
    assert SNAPSHOT_PURPOSE<3*2**30
    assert p['llm_data']['token_batch']==8192 and p['llm_data']['training_tokens_per_primary_arm']==3440640
    assert p['expected']['hybrid_calls_by_dimension']=={'2048':290,'4096':58,'6144':58,'8192':58}
    assert p['expected']['full_moment_transactions']==24 and p['expected']['counterfactual_basis_calls']==96

def test_mean_token_loss_streamed_preflight_and_exact_tied_reset():
    torch.manual_seed(162);c=tiny();model=build_model(c)
    streams=RandomStreams(torch.Generator().manual_seed(163),SimpleNamespace(rng=np.random.default_rng(164),counts={}))
    snapshot=SharedInitialState(model,streams)
    x=torch.arange(16).reshape(2,8)%32;y=(x+1)%32;ids=torch.tensor([0,1])
    row,loss,norm=captured_backward(model,FlatOutput(model),lambda:(ids,x,y.reshape(-1)),streams,clip_norm=1.,autocast=nullcontext)
    digest_grad=streaming_digest([p.grad for p in model.parameters()]);flags=[m.training for m in model.modules()]
    snapshot.reset();expected=model(x,y);expected.backward();expected_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
    assert torch.equal(loss,expected.detach()) and torch.equal(norm,expected_norm.detach())
    assert streaming_digest([p.grad for p in model.parameters()])==digest_grad
    assert row['output']['shape']==[16,32] and row['target']['shape']==[16]
    with torch.no_grad():model.token_embedding.weight.add_(1.)
    snapshot.reset();assert model.lm_head.weight is model.token_embedding.weight and [m.training for m in model.modules()]==flags
    snapshot.close()

def test_populated_tiny_llm_all_arms_moment_samples_nonmutation_release(monkeypatch):
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda:0);monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    c=tiny()
    for arm in ARMS:
        torch.manual_seed(162);model=build_model(c)
        router=PolicyRouter(lambda *a:None,lambda:'rng',lambda fn,**kw:fn);router.covariance_samples={}
        lease=OptimizerLease(model,build_arm(model,c,arm,router))
        for call in range(1,421):
            for param in model.parameters():param.grad=torch.randn_like(param)
            router.probe=call in (21,281);lease.opt.step()
        router.probe=False
        assert len(router.moment_probes)==8 and router.cf_calls==32
        assert len(router.probes)==(16 if arm in POLICIES else 0)
        assert len(router.covariance_samples)==8
        for row in router.moment_probes:
            assert row['source_arm']==arm and row['nonmutation_pass'] and row['registered_release_pass'] and row['physical_m_pass']
            samples=row['fixed_coordinate_covariance_samples']
            assert len(samples)==2
            assert all(z['relative_sample_drift'] is None for z in samples) if row['source_step']==20 else all(z['previous_source_step']==20 and z['relative_sample_drift'] is not None for z in samples)
        router.covariance_samples.clear();assert lease.release()['passed']

def test_four_dimensions_two_layouts_default_compiler_limit_and_own_references():
    torch._dynamo.reset();captured=[]
    def backend(gm,inputs):captured.append(gm);return gm.forward
    def compiler(fn,**kw):return torch.compile(fn,backend=backend,fullgraph=True,dynamic=False)
    router=PolicyRouter(lambda *a:None,lambda:'rng',compiler);limit=torch._dynamo.config.recompile_limit
    for n in (8,12,16,20):
        cov=torch.diag(torch.linspace(1,2,n))+.01*torch.ones(n,n)
        for column in (False,True):
            q=torch.eye(n);q=q.T.contiguous().T if column else q
            for kernel in router.kernels.values():
                with torch.no_grad():assert torch.equal(kernel.evaluate(cov,q),kernel.evaluate(cov,q,eager=True))
    before=len(captured)
    for kernel in router.kernels.values():
        with torch.no_grad():kernel.evaluate(cov,q)
    assert len(captured)==before and torch._dynamo.config.recompile_limit==limit
    from torch._dynamo.utils import counters
    assert not counters['graph_break'];torch._dynamo.reset()

def test_fresh_llm_preparation_pins_cache_and_rejects_semantic_drift(tmp_path,monkeypatch):
    import prepare_llm_feasibility as prep
    parent=tmp_path/'parent';parent.mkdir();(parent/'source').mkdir();prep.write(parent/'manifest.json',dict(source_sha256={}))
    prep.write(parent/'execution_status.json',dict(status='completed'));monkeypatch.setattr(prep,'PARENTS',(parent,))
    cache=tmp_path/'cache';cache.write_bytes(b'fixed-data');monkeypatch.setattr(prep,'CACHE',cache)
    audit=tmp_path/'audit.json';prep.write(audit,dict(status='PASS_CPU_TOKEN_CACHE_AUDIT',cuda_initialized=False,path=str(cache),file_sha256=prep.sha(cache)))
    ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=162,parameters=PARAMETERS,cuda_initialized=False,snapshot_purpose_bytes=SNAPSHOT_PURPOSE))
    b=tmp_path/'fresh';m=prep.prepare(b,ref,data_audit=audit);assert len(m['source_sha256'])==8
    cfg=prep.read(b/'config.json');cfg['batch_size']=4;prep.write(b/'config.json',cfg)
    altered=copy.deepcopy(m);altered['artifacts']['config.json']=prep.sha(b/'config.json');prep.write(b/'manifest.json',altered)
    with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
