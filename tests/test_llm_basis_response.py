"""Independent covariance/QR/movement controls and actual tiny-model diagnostics."""
from pathlib import Path
from types import SimpleNamespace
import math,sys,copy
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import numpy as np
import pytest,torch
from soap_reference import ResearchSOAP
from cifar_optimizers import grouped_parameters
from train_llm import PackedTokenDataset
from llm_feasibility_runtime import build_model,FlatOutput
from matched_training_contract import RandomStreams
from matched_training_runtime import streaming_digest
from llm_math_confirmation import endpoint
from llm_basis_response_contract import recipe,protocol,pools,ids_sha,SEED,BASIS_EXTRA_BYTES,CPU_STATS_BYTES,FACTORS,BASIS_GATES
from llm_basis_response_core import diagnose
from llm_basis_response_atoms import covariance_update,basis_step,aligned_delta,offdiag_norm,geometry,full_clip_gram_relative
from tensor_ownership import TensorOwnership
from hybrid_training_kernel import precision

def fixture(fail=False,active_clip=False):
    c=recipe();c.update(dim=16,heads=4,ff_hidden_dim=32,sequence_length=8,batch_size=16,vocab_size=32)
    if active_clip:c['clip_norm']=1e-4
    torch.manual_seed(SEED);model=build_model(c)
    r=SimpleNamespace(model=model,active=FlatOutput(model),config=c,protocol=protocol())
    r.protocol['sequence_diagnostic']['pools']=[dict(ids=v,ids_sha256=ids_sha(v)) for v in pools(SEED,64)]
    r.train_dataset=PackedTokenDataset(torch.randint(0,32,(64*8+1,),generator=torch.Generator().manual_seed(22)),8)
    for p in model.parameters():p.grad=torch.ones_like(p)
    for i,m in enumerate(model.modules()):m.training=bool(i%2)
    r.streams=RandomStreams(torch.Generator().manual_seed(23),SimpleNamespace(rng=np.random.default_rng(24),counts={}))
    groups=[g for g in grouped_parameters(model,c) if g['role']=='matrix']
    states={};names={};gen=torch.Generator().manual_seed(25)
    for group in groups:
        group.update(shampoo_beta=.99,betas=(.9,.999),merge_dims=False)
        for name,p in zip(group['param_names'],group['params']):
            covs=[];qs=[]
            for dim in p.shape:
                a=torch.randn(dim,dim,generator=gen);covs.append(a@a.T+torch.diag(torch.linspace(.5,2.,dim)))
                qs.append(torch.linalg.qr(torch.randn(dim,dim,generator=gen)).Q.contiguous())
            states[p]=dict(GG=covs,Q=qs,step=999);names[name]=p
    for spec in r.protocol['basis_response']['factors']:spec['dimension']=names[spec['parameter']].shape[spec['axis']]
    opt=SimpleNamespace(matrix=SimpleNamespace(param_groups=groups,state=states),auxiliary=SimpleNamespace(),
        state_dict=lambda:dict(matrix={name:states[p] for name,p in names.items()}))
    r.events=[];r.emit=lambda event,payload:r.events.append((event,payload));r.calls=0
    def budget(**kwargs):
        r.calls+=1
        if fail and r.calls>8:raise ValueError('deliberate basis response failure')
    r.budget=budget
    return r,opt

def independent_controls():
    owner=TensorOwnership()
    g=torch.tensor([[2.,-1.,.5],[.3,1.,-2.],[3.,.2,-.7],[.4,.9,1.2]])
    c=torch.tensor([[2.,.1,.2,0.],[.1,3.,.3,.2],[.2,.3,4.,.1],[0.,.2,.1,5.]])
    gram=g@g.T;got=covariance_update(c,gram,.9,owner)
    oracle=np.array([[.9*float(c[i,j])+.1*sum(float(g[i,k])*float(g[j,k]) for k in range(3)) for j in range(4)] for i in range(4)])
    assert np.allclose(got.numpy(),oracle,rtol=1e-6,atol=1e-6)
    q=torch.linalg.qr(torch.tensor([[1.,.2,0.,.1],[.4,1.,.3,0.],[0.,.2,1.,.5],[.2,0.,.1,1.]])).Q
    rot=np.asarray(q.double().numpy()).T@oracle@np.asarray(q.double().numpy());np.fill_diagonal(rot,0)
    assert offdiag_norm(q,got,owner)==pytest.approx(np.linalg.norm(rot),rel=2e-6,abs=1e-6)
    perturb=q.clone();perturb[0,0]+=.01
    direct=perturb.double().numpy().T@perturb.double().numpy()-np.eye(4)
    assert geometry(perturb,owner)['absolute_gram_fro']==pytest.approx(np.linalg.norm(direct),rel=1e-5,abs=1e-6)
    order=torch.tensor([2,0,3,1]);candidate=q[:,order]*torch.tensor([-1.,1.,-1.,1.])
    assert torch.count_nonzero(aligned_delta(candidate,q,order,owner))==0
    clipped=(g*.125)@(g*.125).T
    assert full_clip_gram_relative(gram,clipped,.125,owner)<1e-12
    assert full_clip_gram_relative(gram,clipped*1.1,.125,owner)>.08
    # Exact original QR sorting, QR product, and V permutation semantics.
    param=torch.nn.Parameter(torch.randn(4,3,generator=torch.Generator().manual_seed(42)))
    opt=ResearchSOAP([param],basis_method='qr',transport_precision='highest',precondition_frequency=20)
    qright=torch.eye(3);right=g.T@g+torch.eye(3)
    state=dict(Q=[q.clone(),qright],GG=[got.clone(),right],exp_avg=torch.zeros_like(param),exp_avg_sq=torch.arange(1,13,dtype=torch.float32).reshape(4,3),step=999)
    left_expected,lo=basis_step('qr20',got,q,owner);right_expected,ro=basis_step('qr20',right,qright,owner)
    v_expected=state['exp_avg_sq'].index_select(0,lo).index_select(1,ro)
    with precision('high'):opt._research_refresh(state,opt.param_groups[0])
    assert torch.equal(state['Q'][0],left_expected) and torch.equal(state['Q'][1],right_expected) and torch.equal(state['exp_avg_sq'],v_expected)
    owner.close()
    return True

def test_independent_covariance_original_qr_variance_permutation_geometry_and_gauge_alignment():
    assert independent_controls()

@pytest.mark.parametrize('active_clip',[False,True])
def test_tiny_actual_model_complete_response_counts_and_exact_nonmutation_release(active_clip):
    r,opt=fixture(active_clip=active_clip);before=endpoint(r,opt);g=streaming_digest([p.grad for p in r.model.parameters()]);ids=[id(p.grad) for p in r.model.parameters()];flags=[m.training for m in r.model.modules()]
    result=diagnose(r,opt)
    assert result['passed'] and result['optimizer_updates']==0 and result['backward_calls']==68 and result['basis_policy_calls']==204
    assert len(result['groups'])==8 and len(result['references'])==4 and len(result['inherited_basis_responses'])==4
    assert len(result['cross_pool_sketch_statistics'])==104 and all(v['pools']==4 for v in result['cross_pool_sketch_statistics'])
    for row in result['groups']:
        assert len(row['basis_responses'])==8
        for factor in row['basis_responses']:
            assert len(factor['policies'])==3 and all(v['passed'] for v in factor['policies'])
            if factor['mode']=='clipped':assert factor['clip_gram_relative']<=BASIS_GATES['clip_gram_relative']
    if active_clip:assert all(v['global_clip_coefficient']<1 for v in result['groups'])
    assert result['cleanup']['registered_release']['all_registered_released'] and result['cleanup']['cpu_statistics_released']
    assert endpoint(r,opt)==before and g==streaming_digest([p.grad for p in r.model.parameters()]) and ids==[id(p.grad) for p in r.model.parameters()] and flags==[m.training for m in r.model.modules()]

def test_basis_failure_restores_state_rng_flags_gradients_and_all_registered_buffers():
    r,opt=fixture(fail=True);before=endpoint(r,opt);g=streaming_digest([p.grad for p in r.model.parameters()]);flags=[m.training for m in r.model.modules()];ids=[id(p.grad) for p in r.model.parameters()]
    with pytest.raises(RuntimeError,match='deliberate basis response failure'):diagnose(r,opt)
    cleanup=next(v for e,v in r.events if e=='sequence_diagnostic_cleanup')
    assert cleanup['registered_release']['all_registered_released'] and cleanup['cpu_statistics_released'] and cleanup['state_rng_flags_gradients_and_identity_pass']
    assert endpoint(r,opt)==before and g==streaming_digest([p.grad for p in r.model.parameters()]) and flags==[m.training for m in r.model.modules()] and ids==[id(p.grad) for p in r.model.parameters()]

def test_frozen_recipe_factor_coverage_counts_pools_bounds_and_purpose():
    from llm_sequence_averaging_contract import recipe as prior,protocol as pp,pools as old_pools
    c=recipe();old=prior();assert {k:v for k,v in c.items() if k not in ('seed','matched_comparison')}=={k:v for k,v in old.items() if k not in ('seed','matched_comparison')}
    p=protocol();assert p['gates']==pp()['gates'] and p['bounds']==pp()['bounds'] and p['precision']==pp()['precision'] and p['model_parity']==pp()['model_parity']
    assert p['maximum_optimizer_updates']==1270 and p['basis_response']['maximum_policy_calls']==204
    assert p['sequence_diagnostic']['sizes']==[1,16] and pools()==old_pools(SEED) and len({i for v in pools() for i in v})==64
    assert BASIS_EXTRA_BYTES==9569273344<9*2**30 and CPU_STATS_BYTES==6815744<8*2**20
    with torch.device('meta'):model=build_model(c)
    named=dict(model.named_parameters());assert all(named[n].shape[axis]==d for n,axis,d in FACTORS)

def test_preparation_rejects_missing_or_resealed_bad_basis_calibration_and_protocol(tmp_path,monkeypatch):
    import prepare_llm_basis_response as prep
    parent=tmp_path/'parent';parent.mkdir();(parent/'source').mkdir();prep.write(parent/'manifest.json',dict(source_sha256={}));prep.write(parent/'execution_status.json',dict(status='completed'));monkeypatch.setattr(prep,'PARENTS',(parent,))
    qual=tmp_path/'qual';qual.mkdir();prep.write(qual/'execution_status.json',dict(status='completed'));prep.write(qual/'summary.json',dict(policies=[dict(policy='math_attention',all_exact=True)]));monkeypatch.setattr(prep,'QUALIFICATION',qual)
    cache=tmp_path/'cache';cache.write_bytes(b'pinned');monkeypatch.setattr(prep,'CACHE',cache)
    audit=tmp_path/'audit.json';prep.write(audit,dict(status='PASS_CPU_TOKEN_CACHE_AUDIT',cuda_initialized=False,path=str(cache),file_sha256=prep.sha(cache)))
    ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=SEED,parameters=144875648,cuda_initialized=False,snapshot_purpose_bytes=2230649344))
    b=tmp_path/'fresh'
    with pytest.raises(ValueError,match='calibration'):prep.prepare(b,ref,data_audit=audit)
    cal=tmp_path/'cal.json';calibration=dict(status='PASS_CPU_BASIS_RESPONSE_CALIBRATION',seed=SEED,cuda_initialized=False,gates=protocol()['sequence_diagnostic']['gates'],diagnostic_backward_calls=68,groups=8,active_clip_control_pass=True,basis_gates=BASIS_GATES,basis_policy_calls=204,cross_pool_statistics=104,independent_covariance_qr_metric_controls_pass=True,registered_release_pass=True,state_and_gradient_identity_pass=True,cpu_statistics_released=True)
    prep.write(cal,dict(calibration,seed=173))
    with pytest.raises(ValueError,match='calibration'):prep.prepare(b,ref,data_audit=audit,calibration=cal)
    assert not b.exists()
    prep.write(cal,calibration)
    m=prep.prepare(b,ref,data_audit=audit,calibration=cal);assert len(m['source_sha256'])==7
    declared=prep.read(b/'protocol.json');declared['basis_response']['gates']['clip_gram_relative']=.1;prep.write(b/'protocol.json',declared);m['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',m)
    with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
