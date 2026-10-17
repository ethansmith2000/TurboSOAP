"""Independent aggregation, clipping, covariance contraction and failure-path controls."""
from pathlib import Path
from types import SimpleNamespace
import sys,copy,math
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import numpy as np
import pytest,torch
from soap import SOAP
from train_llm import PackedTokenDataset
from cifar_optimizers import grouped_parameters
from llm_feasibility_runtime import build_model,FlatOutput
from matched_training_contract import RandomStreams
from matched_training_runtime import streaming_digest
from llm_math_confirmation import endpoint
from llm_sequence_averaging_core import diagnose,CrossPoolStats
from llm_sequence_averaging_contract import recipe,protocol,pools,ids_sha,EXTRA_BYTES,SEED,PARAMETERS,SNAPSHOT_PURPOSE

def fixture(fail=False):
 c=recipe();c.update(dim=16,heads=4,ff_hidden_dim=32,sequence_length=8,batch_size=16,vocab_size=32)
 torch.manual_seed(SEED);model=build_model(c)
 r=SimpleNamespace(model=model,active=FlatOutput(model),config=c,protocol=protocol())
 ids=pools(SEED,64);r.protocol['sequence_diagnostic']['pools']=[dict(ids=v,ids_sha256=ids_sha(v)) for v in ids]
 r.train_dataset=PackedTokenDataset(torch.randint(0,32,(64*8+1,),generator=torch.Generator().manual_seed(22)),8)
 for p in model.parameters():p.grad=torch.ones_like(p)
 for i,m in enumerate(model.modules()):m.training=bool(i%2)
 r.streams=RandomStreams(torch.Generator().manual_seed(23),SimpleNamespace(rng=np.random.default_rng(24),counts={}))
 r.events=[];r.emit=lambda event,payload:r.events.append((event,payload));r.calls=0
 def budget(**kwargs):
  r.calls+=1
  if fail and r.calls>5:raise ValueError('deliberate diagnostic failure')
 r.budget=budget
 groups=[g for g in grouped_parameters(model,c) if g['role']=='matrix']
 opt=SimpleNamespace(matrix=SimpleNamespace(param_groups=groups),auxiliary=SimpleNamespace(),state_dict=lambda:dict(moment=torch.ones(2)))
 return r,opt

def test_tiny_actual_model_mean_gradients_clipping_exact_state_and_registered_release():
 r,opt=fixture();before=endpoint(r,opt);g=streaming_digest([p.grad for p in r.model.parameters()]);ids=[id(p.grad) for p in r.model.parameters()];flags=[m.training for m in r.model.modules()]
 result=diagnose(r,opt)
 assert result['passed'] and result['optimizer_updates']==0 and result['backward_calls']==68
 assert len(result['groups'])==20 and len(result['references'])==4 and all(v['passed'] for v in result['references'])
 assert all(v['raw_gradient']['passed'] and v['clipped_gradient']['passed'] for v in result['references'])
 assert all(v['pools']==4 for v in result['cross_pool_sketch_statistics']) and len(result['cross_pool_sketch_statistics'])==100
 assert result['cleanup']['registered_release']['all_registered_released'] and result['cleanup']['cpu_statistics_released']
 assert endpoint(r,opt)==before and g==streaming_digest([p.grad for p in r.model.parameters()]) and ids==[id(p.grad) for p in r.model.parameters()] and flags==[m.training for m in r.model.modules()]


def test_active_global_clip_matches_direct_reference_without_touching_training_gradients():
 r,opt=fixture();r.config['clip_norm']=1e-4;before=streaming_digest([p.grad for p in r.model.parameters()]);result=diagnose(r,opt)
 assert all(v['global_clip_coefficient']<1 for v in result['groups'])
 assert all(v['clipped_gradient']['passed'] for v in result['references']) and before==streaming_digest([p.grad for p in r.model.parameters()])


def test_failure_restores_all_state_flags_rng_gradient_identity_and_releases_buffers():
 r,opt=fixture(True);before=endpoint(r,opt);g=streaming_digest([p.grad for p in r.model.parameters()]);flags=[m.training for m in r.model.modules()];ids=[id(p.grad) for p in r.model.parameters()]
 with pytest.raises(RuntimeError,match='deliberate diagnostic failure'):diagnose(r,opt)
 cleanup=next(v for e,v in r.events if e=='sequence_diagnostic_cleanup')
 assert cleanup['registered_release']['all_registered_released'] and cleanup['cpu_statistics_released'] and cleanup['state_rng_flags_gradients_and_identity_pass']
 assert endpoint(r,opt)==before and g==streaming_digest([p.grad for p in r.model.parameters()]) and flags==[m.training for m in r.model.modules()] and ids==[id(p.grad) for p in r.model.parameters()]


def test_original_soap_unnormalized_two_sided_covariance_matches_independent_loops_and_mean_covariance_distinction():
 g=torch.tensor([[2.,-1.,.5],[.3,1.,-2.],[3.,.2,-.7],[.4,.9,1.2]])
 left=np.array([[sum(float(g[i,k])*float(g[j,k]) for k in range(3)) for j in range(4)] for i in range(4)])
 right=np.array([[sum(float(g[k,i])*float(g[k,j]) for k in range(4)) for j in range(3)] for i in range(3)])
 state=dict(GG=[torch.ones(4,4),torch.ones(3,3)]);obj=object.__new__(SOAP)
 obj._accumulate_covariance(g,state,dict(shampoo_beta=.9,betas=(.9,.999),merge_dims=False,covariance_compute_dtype='float32'))
 assert np.allclose(state['GG'][0].numpy(),.9+.1*left,rtol=1e-6,atol=1e-6) and np.allclose(state['GG'][1].numpy(),.9+.1*right,rtol=1e-6,atol=1e-6)
 mean=(g-g)/2;mean_cov=mean@mean.T;average_outer=(g@g.T+(-g)@(-g).T)/2
 assert mean_cov.count_nonzero()==0 and float(average_outer.norm())>1
 # Clip-after-average and average-after-individual-clips are distinct estimators.
 a=torch.tensor([10.,0.]);b=torch.tensor([0.,1.]);raw=(a+b)/2;proper=raw/raw.norm();wrong=(a/a.norm()+b)/2
 assert float((proper-wrong).norm())>.4


def test_cross_pool_online_statistics_match_centered_double_oracle():
 s=CrossPoolStats(8*2**20);xs=[np.array([[1.,x],[x*x,-x]]) for x in (1.,2.,4.,8.)]
 for v in xs:s.add((16,'matrix','gradient'),v)
 row=s.finish()[0];mean=np.mean(xs,axis=0);var=sum(np.square(v-mean).sum() for v in xs)/3
 assert row['sample_variation_trace']==pytest.approx(var) and row['mean_sample_energy']==pytest.approx(float(np.square(mean).sum())) and s.close()


def test_contract_sampling_gates_recipe_and_memory_shape_purpose():
 from llm_math_confirmation_contract import recipe as prior,protocol as pp
 c=recipe();old=prior();assert {k:v for k,v in c.items() if k not in ('seed','matched_comparison')}=={k:v for k,v in old.items() if k not in ('seed','matched_comparison')}
 p=protocol();assert p['gates']==pp()['gates'] and p['bounds']==pp()['bounds'] and p['precision']==pp()['precision'] and p['model_parity']==pp()['model_parity']
 assert p['maximum_optimizer_updates']==1270 and p['steps_per_arm']==1000 and p['evaluation_calls']==[1000] and p['probe_calls']==[]
 assert len(set(i for pool in pools() for i in pool))==64 and len(pools())==4
 assert p['sequence_diagnostic']['memory']['conservative_tensor_purpose_bytes']==EXTRA_BYTES==2053080576 and EXTRA_BYTES<2*2**30
 with torch.device('meta'):model=build_model(c)
 pb=sum(v.numel()*v.element_size() for v in model.parameters());mat=[v for g in grouped_parameters(model,c) if g['role']=='matrix' for v in g['params']]
 assert 3*pb+sum(v.numel()*v.element_size() for v in mat)+2*max(v.numel()*v.element_size() for v in mat)+12*2**20==EXTRA_BYTES


def test_preparation_requires_calibration_and_rejects_resealed_aggregation_gate_change(tmp_path,monkeypatch):
 import prepare_llm_sequence_averaging as prep
 parent=tmp_path/'parent';parent.mkdir();(parent/'source').mkdir();prep.write(parent/'manifest.json',dict(source_sha256={}));prep.write(parent/'execution_status.json',dict(status='completed'));monkeypatch.setattr(prep,'PARENTS',(parent,))
 qual=tmp_path/'qual';qual.mkdir();prep.write(qual/'execution_status.json',dict(status='completed'));prep.write(qual/'summary.json',dict(policies=[dict(policy='math_attention',all_exact=True)]));monkeypatch.setattr(prep,'QUALIFICATION',qual)
 cache=tmp_path/'cache';cache.write_bytes(b'pinned');monkeypatch.setattr(prep,'CACHE',cache)
 audit=tmp_path/'audit.json';prep.write(audit,dict(status='PASS_CPU_TOKEN_CACHE_AUDIT',cuda_initialized=False,path=str(cache),file_sha256=prep.sha(cache)))
 ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=SEED,parameters=PARAMETERS,cuda_initialized=False,snapshot_purpose_bytes=SNAPSHOT_PURPOSE))
 b=tmp_path/'fresh'
 with pytest.raises(ValueError,match='calibration'):prep.prepare(b,ref,data_audit=audit)
 assert not b.exists()
 cal=tmp_path/'cal.json';prep.write(cal,dict(status='PASS_CPU_SEQUENCE_CALIBRATION',cuda_initialized=False,gates=protocol()['sequence_diagnostic']['gates'],diagnostic_backward_calls=68,groups=20,active_clip_control_pass=True))
 m=prep.prepare(b,ref,data_audit=audit,calibration=cal);assert len(m['source_sha256'])==6
 p=prep.read(b/'protocol.json');p['sequence_diagnostic']['gates']['gradient_relative']=.01;prep.write(b/'protocol.json',p);m['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',m)
 with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
