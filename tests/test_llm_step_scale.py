"""Independent generators/rotation/retraction and full tiny fixed-state failure paths."""
from pathlib import Path
from types import SimpleNamespace
import sys,math
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support'),str(Path(__file__).resolve().parent)]
import numpy as np
import pytest,torch
from test_llm_basis_response import fixture as old_fixture
from llm_step_scale_contract import recipe,protocol,SEED,STEP_GATES,EXTRA_BYTES,FACTORS
from llm_step_scale_core import diagnose
from llm_step_scale_atoms import staged_retraction,admitted,metrics
from llm_basis_response_atoms import offdiag_norm
from tracking_curvature_kernel import generator
from tensor_ownership import TensorOwnership
from matched_training_runtime import streaming_digest
from llm_math_confirmation import endpoint
from cifar_optimizers import grouped_parameters
from llm_feasibility_runtime import build_model,FlatOutput

def fixture(fail=False):
    old,opt=old_fixture();c=recipe();c.update(dim=16,heads=4,ff_hidden_dim=32,sequence_length=8,batch_size=16,vocab_size=32)
    torch.manual_seed(SEED);model=build_model(c)
    for i,m in enumerate(model.modules()):m.training=bool(i%2)
    for p in model.parameters():p.grad=torch.ones_like(p)
    previous={name:opt.matrix.state[p] for g in opt.matrix.param_groups for name,p in zip(g['param_names'],g['params'])}
    groups=[g for g in grouped_parameters(model,c) if g['role']=='matrix'];states={};named={}
    for g in groups:
        g.update(shampoo_beta=.99,betas=(.9,.999),merge_dims=False)
        for name,p in zip(g['param_names'],g['params']):states[p]=previous[name];named[name]=p
    opt=SimpleNamespace(matrix=SimpleNamespace(param_groups=groups,state=states),auxiliary=SimpleNamespace(),state_dict=lambda:dict(matrix={n:states[p] for n,p in named.items()}))
    old.model=model;old.active=FlatOutput(model);old.config=c;old.protocol=protocol();old.calls=0;old.events=[]
    old.emit=lambda event,payload:old.events.append((event,payload))
    for spec in old.protocol['step_scale']['factors']:spec['dimension']=named[spec['parameter']].shape[spec['axis']]
    def budget(**kwargs):
        old.calls+=1
        if fail and old.calls>5:raise ValueError('deliberate mid-sweep failure')
    old.budget=budget
    return old,opt

def independent_controls():
    b=torch.tensor([[3.,.4,-.2],[.4,1.,.1],[-.2,.1,2.]])
    for name in ('jacobi','damped'):
        raw,bound,a,cap,scale=generator(b,name);bn=b.double().numpy();diag=np.diag(bn);tau=max(abs(diag))*.01
        omega=np.array([[-bn[i,j]*(diag[i]-diag[j])/(max((diag[i]-diag[j])**2+tau**2,1e-24) if name=='jacobi' else 1.) for j in range(3)] for i in range(3)])
        oracle=.25*(omega-omega.T);norm=np.linalg.norm(oracle)/math.sqrt(2);cap_expected=min(1.,.5/max(norm,1e-12))
        slope=sum(2*bn[i,j]*(diag[i]-diag[j])*oracle[i,j] for i in range(3) for j in range(3))
        curvature=sum(2*((diag[i]-diag[j])*oracle[i,j])**2 for i in range(3) for j in range(3))
        chosen=cap_expected if name=='jacobi' else min(cap_expected,max(0.,min(1.,-slope/max(curvature,1e-24))))
        assert np.allclose(raw.numpy(),oracle,rtol=1e-6,atol=1e-7) and float(bound)==pytest.approx(norm,rel=1e-6)
        assert float(cap)==pytest.approx(cap_expected,rel=1e-6) and float(scale)==pytest.approx(chosen,rel=1e-6)
        assert np.allclose(a.numpy(),oracle*chosen,rtol=1e-6,atol=1e-7)
        # Independent central derivative along an exact exponential rotation.
        bb=b.double();aa=raw.double();eps=1e-5
        def objective(t):
            q=torch.matrix_exp(t*aa);rot=q.T@bb@q;rot.diagonal().zero_();return float(rot.square().sum())
        derivative=(objective(eps)-objective(-eps))/(2*eps)
        assert derivative==pytest.approx(slope,rel=1e-5,abs=1e-8)
    owner=TensorOwnership()
    # Independent dense numpy scalar contraction for the two mixed corrections.
    y=torch.tensor([[1.,-.25],[.25,1.]])
    expected=y.double().numpy();first_expected=None
    for iteration in range(2):
        residual=expected.T@expected-np.eye(2)
        left=torch.from_numpy(expected.astype(np.float32)).bfloat16().float().numpy()
        right=torch.from_numpy(residual.astype(np.float32)).bfloat16().float().numpy()
        correction=np.array([[sum(float(left[i,k])*float(right[k,j]) for k in range(2)) for j in range(2)] for i in range(2)])
        correction=torch.from_numpy(correction.astype(np.float32)).bfloat16().float().numpy()
        expected=expected-.5*correction
        if iteration==0:first_expected=expected.copy()
    final,first=staged_retraction(y,owner)
    assert np.allclose(first.numpy(),first_expected,rtol=0,atol=1e-7) and np.allclose(final.numpy(),expected,rtol=0,atol=1e-7)
    exact_polar=np.array([[1.,-.25],[.25,1.]])/math.sqrt(1+.25**2)
    assert np.linalg.norm(final.numpy()-exact_polar)<1e-4
    q=torch.eye(2);cov=torch.tensor([[3.,.2],[.2,1.]])
    unsafe=torch.tensor([[1.,-2.],[2.,1.]])
    rejected,_=staged_retraction(unsafe,owner);row=metrics(rejected,cov,q,offdiag_norm(q,cov,owner),owner)
    assert row['finite'] and not admitted(row,STEP_GATES)
    owner.close();return True

def test_independent_generators_exact_rotation_derivative_mixed_retraction_and_rejected_proposal():assert independent_controls()

def test_tiny_all_factor_sweep_exact_references_counts_state_and_buffer_release():
    r,opt=fixture();before=endpoint(r,opt);grad=streaming_digest([p.grad for p in r.model.parameters()]);ids=[id(p.grad) for p in r.model.parameters()]
    result=diagnose(r,opt)
    assert result['passed'] and result['complete_basis_calls']==88 and result['sweep_candidates']==64 and result['backward_calls']==result['optimizer_updates']==0
    assert len(result['factors'])==8 and result['direction_observations']==16
    for factor in result['factors']:
        assert factor['identity']['tracking_ratio']==pytest.approx(1) and factor['qr']['finite']
        for direction in factor['directions']:
            assert direction['reference']['passed'] and {v['multiplier'] for v in direction['candidates']}=={.5,1.,2.,4.}
            assert next(v for v in direction['candidates'] if v['multiplier']==1.)['admitted']
    assert result['cleanup']['registered_release']['all_registered_released'] and result['cleanup']['state_rng_flags_gradients_and_identity_pass']
    assert endpoint(r,opt)==before and grad==streaming_digest([p.grad for p in r.model.parameters()]) and ids==[id(p.grad) for p in r.model.parameters()]

def test_failure_with_live_factor_generator_and_candidate_aliases_restores_and_releases():
    r,opt=fixture(fail=True);before=endpoint(r,opt);grad=streaming_digest([p.grad for p in r.model.parameters()]);ids=[id(p.grad) for p in r.model.parameters()]
    with pytest.raises(RuntimeError,match='deliberate mid-sweep failure'):diagnose(r,opt)
    c=next(v for event,v in r.events if event=='step_scale_diagnostic_cleanup')
    assert c['registered_release']['all_registered_released'] and c['state_rng_flags_gradients_and_identity_pass']
    assert before==endpoint(r,opt) and grad==streaming_digest([p.grad for p in r.model.parameters()]) and ids==[id(p.grad) for p in r.model.parameters()]

def test_contract_all_factor_shapes_counts_memory_original_recipe_and_gates():
    from llm_basis_response_contract import recipe as old,protocol as pp
    c=recipe();prev=old();assert {k:v for k,v in c.items() if k not in ('seed','matched_comparison')}=={k:v for k,v in prev.items() if k not in ('seed','matched_comparison')}
    p=protocol();assert all(p[k]==pp()[k] for k in ('gates','bounds','precision','model_parity'))
    assert 'sequence_diagnostic' not in p and 'basis_response' not in p and p['step_scale']['maximum_complete_basis_calls']==88
    assert p['maximum_optimizer_updates']==1270 and EXTRA_BYTES==8645149184<9*2**30
    with torch.device('meta'):model=build_model(c)
    named=dict(model.named_parameters());assert len(FACTORS)==8 and all(named[n].shape[axis]==dim for n,axis,dim in FACTORS)

def test_preparation_rejects_wrong_seed_and_resealed_changed_observer_gate(tmp_path,monkeypatch):
    import prepare_llm_step_scale as prep
    parent=tmp_path/'parent';parent.mkdir();(parent/'source').mkdir();prep.write(parent/'manifest.json',dict(source_sha256={}));prep.write(parent/'execution_status.json',dict(status='completed'));monkeypatch.setattr(prep,'PARENTS',(parent,))
    qual=tmp_path/'qual';qual.mkdir();prep.write(qual/'execution_status.json',dict(status='completed'));prep.write(qual/'summary.json',dict(policies=[dict(policy='math_attention',all_exact=True)]));monkeypatch.setattr(prep,'QUALIFICATION',qual)
    cache=tmp_path/'cache';cache.write_bytes(b'pinned');monkeypatch.setattr(prep,'CACHE',cache)
    audit=tmp_path/'audit.json';prep.write(audit,dict(status='PASS_CPU_TOKEN_CACHE_AUDIT',cuda_initialized=False,path=str(cache),file_sha256=prep.sha(cache)))
    ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=SEED,parameters=144875648,cuda_initialized=False,snapshot_purpose_bytes=2230649344))
    cal=tmp_path/'cal.json';record=dict(status='PASS_CPU_STEP_SCALE_CALIBRATION',seed=SEED,cuda_initialized=False,gates=STEP_GATES,diagnostic_backward_calls=0,complete_basis_calls=88,sweep_candidates=64,observed_basis_counts=dict(sweep_candidates=64,pure_eager_references=16,qr_controls=8),all_unmodified_references_pass=True,independent_generator_rotation_retraction_controls_pass=True,registered_release_pass=True,state_and_gradient_identity_pass=True)
    prep.write(cal,dict(record,seed=174));b=tmp_path/'fresh'
    with pytest.raises(ValueError,match='calibration'):prep.prepare(b,ref,data_audit=audit,calibration=cal)
    assert not b.exists();prep.write(cal,record)
    m=prep.prepare(b,ref,data_audit=audit,calibration=cal);assert len(m['source_sha256'])==7
    declared=prep.read(b/'protocol.json');declared['step_scale']['gates']['prefix_relative']=.01;prep.write(b/'protocol.json',declared);m['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',m)
    with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
