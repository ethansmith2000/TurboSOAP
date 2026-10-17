"""Independent chosen-A/BF16/physical-M controls and whole transaction failures."""
from pathlib import Path
from types import SimpleNamespace
import sys,math,copy
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support'),str(Path(__file__).resolve().parent)]
import numpy as np
import pytest,torch
from test_llm_step_scale import fixture as old_fixture,independent_controls as old_controls
from llm_jacobi2_contract import protocol,recipe,SEED,GATES,COUNTS,PURPOSE,calibration_contract,OBSERVATIONS
from llm_jacobi2_core import diagnose,warmup,previews,reference_metrics
from llm_jacobi2_kernel import Kernels,prefix_jacobi2
from tracking_curvature_kernel import generator
from soap_reference import ResearchSOAP
from matched_training_runtime import streaming_digest
from llm_math_confirmation import endpoint
from tensor_ownership import TensorOwnership

def fixture(fail=False,safe=True):
 r,opt=old_fixture();r.protocol=protocol();r.config['seed']=SEED
 for spec in r.protocol['jacobi2']['factors']:
  p=next(p for g in opt.matrix.param_groups for n,p in zip(g['param_names'],g['params']) if n==spec['parameter'])
  spec['dimension']=p.shape[spec['axis']]
 gen=torch.Generator().manual_seed(SEED)
 for p,state in opt.matrix.state.items():
  if safe:
   state['GG']=[q @ (torch.diag(torch.linspace(.5,4.,len(q)))+.002*(torch.ones_like(q)-torch.eye(len(q)))) @ q.T for q in state['Q']]
  state['exp_avg']=torch.randn(p.shape,generator=gen)*.1
  state['exp_avg_sq']=torch.rand(p.shape,generator=gen)+.1
 for g in opt.matrix.param_groups:g.update(eps=1e-8,merge_dims=False)
 helper=ResearchSOAP([torch.nn.Parameter(torch.ones(2,2))])
 opt.matrix._project_with_basis=helper._project_with_basis;opt.matrix._project_back_with_basis=helper._project_back_with_basis
 r.qualification_kernels=Kernels(compiler=lambda f,**kw:f);r.qualification_counts={};r.calls=0
 def budget(**kwargs):
  r.calls+=1
  if fail and r.calls>8:raise ValueError('deliberate prototype partial allocation failure')
 r.budget=budget
 return r,opt

def independent_controls():
 assert old_controls()
 q=torch.eye(3)
 for off in (.05,20.):
  cov=torch.tensor([[3.,off,-off*.5],[off,1.,off*.25],[-off*.5,off*.25,2.]])
  c=cov/(cov.trace().abs()/3);b=c.numpy();diag=np.diag(b);tau=max(abs(diag))*.01
  raw=np.array([[-.5*b[i,j]*(diag[i]-diag[j])/max((diag[i]-diag[j])**2+tau**2,1e-24) for j in range(3)] for i in range(3)],dtype=np.float32)
  bound=np.linalg.norm(raw)/math.sqrt(2);scale=min(1.,.5/max(bound,1e-12));applied=torch.from_numpy(raw*scale*2)
  assert (bound>.5)==(off==20.)
  # Independent scalar contraction after explicit BF16 operand/output quantization.
  left=q.bfloat16().float().numpy();right=applied.bfloat16().float().numpy()
  prod=np.array([[sum(float(left[i,k])*float(right[k,j]) for k in range(3)) for j in range(3)] for i in range(3)],dtype=np.float32)
  expected=q+torch.from_numpy(prod).bfloat16().float()
  assert torch.allclose(prefix_jacobi2(cov,q),expected,rtol=0,atol=1e-7)
  assert np.allclose(generator(torch.from_numpy(b),'jacobi')[2].numpy()*2,applied.numpy(),rtol=2e-6,atol=1e-7)
 # Physical M two-sided identity and exact QR diagonal V permutation oracle.
 r,opt=fixture();owner=TensorOwnership();group=opt.matrix.param_groups[0];state=next(iter(opt.matrix.state.values()))
 shape=state['exp_avg'].shape;old=[torch.eye(d) for d in shape];orders=[torch.arange(d-1,-1,-1) for d in shape]
 state=dict(state,Q=old);banks={n:[q[:,o] for q,o in zip(old,orders)] for n in ('qr20','highest_residual20','damped20','jacobi2')}
 rows=previews(r,opt.matrix,state,group,banks,orders,owner,'control')
 assert all(v['physical_m_admitted'] for v in rows.values())
 expected=state['exp_avg']/(state['exp_avg_sq'].sqrt()+group['eps'])
 assert rows['qr_permutationV']['adaptive_norm']==pytest.approx(float(expected.norm()),rel=1e-6)
 assert rows['qr_permutationV']['adaptive_relative_to_qr_permutation']==0 and rows['qr_carryV']['adaptive_relative_to_qr_permutation']>0
 # Deliberate geometry defect must fail physical-M gate, no gate relaxation.
 banks['jacobi2']=[2*q for q in banks['jacobi2']]
 bad=previews(r,opt.matrix,state,group,banks,orders,owner,'rejected')['jacobi2']
 assert not bad['physical_m_admitted'] and bad['world_m_reconstruction_relative']>1
 owner.close();return True

def test_independent_capped_uncapped_before_bf16_mixed_retraction_fullM_and_V_controls():assert independent_controls()

def test_complete_tiny_warmup_and_qualification_counts_nonmutation_and_release():
 r,opt=fixture();before=endpoint(r,opt);grad=streaming_digest([p.grad for p in r.model.parameters()]);ids=[id(p.grad) for p in r.model.parameters()]
 w=warmup(r,opt);d=diagnose(r,opt)
 assert w['passed'] and d['passed'] and len(w['rows'])==48 and len(d['rows'])==4
 assert r.qualification_observations==OBSERVATIONS
 assert r.qualification_counts==COUNTS and w['complete_basis_calls']+d['complete_basis_calls']==320
 assert endpoint(r,opt)==before and grad==streaming_digest([p.grad for p in r.model.parameters()]) and ids==[id(p.grad) for p in r.model.parameters()]
 assert all(x['cleanup']['registered_release']['all_registered_released'] and x['cleanup']['state_rng_flags_gradients_and_identity_pass'] for x in (w,d))
 assert d['cleanup']['compiler_unchanged'] and d['optimizer_updates']==d['backward_calls']==0

def test_partial_allocation_failure_releases_and_restores_exact_state():
 r,opt=fixture(fail=True);before=endpoint(r,opt)
 with pytest.raises(RuntimeError,match='deliberate prototype partial allocation failure'):warmup(r,opt)
 cleanup=next(v for e,v in r.events if e=='prototype_transaction_cleanup')
 assert cleanup['registered_release']['all_registered_released'] and cleanup['state_rng_flags_gradients_and_identity_pass'] and endpoint(r,opt)==before

def test_compiled_reference_rejection_preserves_original_gate_and_failure_evidence():
 r,opt=fixture();evaluate=r.qualification_kernels.evaluate
 def broken(name,cov,q,eager=False):
  value=evaluate(name,cov,q,eager)
  return value*1.1 if not eager and name=='jacobi2' else value
 r.qualification_kernels.evaluate=broken
 with pytest.raises(RuntimeError,match='Warmup own-eager fidelity failed'):warmup(r,opt)
 row=next(v for e,v in r.events if e=='prototype_warmup_before_gate' and v['policy']=='jacobi2')
 assert not row['reference']['passed'] and row['reference']['basis_relative']>GATES['reference_basis_relative']
 assert next(v for e,v in r.events if e=='prototype_transaction_cleanup')['registered_release']['all_registered_released']

def test_contract_recipe_bounds_and_semantic_gate_rejection():
 from llm_step_scale_contract import recipe as old_recipe,protocol as old_protocol
 c=recipe();old=old_recipe();assert {k:v for k,v in c.items() if k not in ('seed','matched_comparison')}=={k:v for k,v in old.items() if k not in ('seed','matched_comparison')}
 p=protocol();oldp=old_protocol();assert all(p[k]==oldp[k] for k in ('bounds','gates','model_parity','precision'))
 assert sum(COUNTS.values())==320 and PURPOSE<12*2**30 and p['maximum_optimizer_updates']==1270
 from llm_jacobi2_contract import validate
 broken=copy.deepcopy(p);broken['jacobi2']['gates']['physical_m_relative']=.1
 with pytest.raises(ValueError,match='Undeclared'):validate(c,broken)
 assert calibration_contract()['counts']==COUNTS

def test_unsafe_random_tiny_geometry_and_M_rejection_keeps_all_four_matrices():
 r,opt=fixture(safe=False);before=endpoint(r,opt);warmup(r,opt)
 with pytest.raises(RuntimeError,match='all observations retained'):diagnose(r,opt)
 full=next(v for e,v in r.events if e=='prototype_all_observations_before_gate' and not v['warming'])
 assert len(full['rows'])==4 and full['counts']==COUNTS and not all(v['passed'] for v in full['rows'])
 assert endpoint(r,opt)==before
 cleanup=[v for e,v in r.events if e=='prototype_transaction_cleanup'][-1]
 assert cleanup['registered_release']['all_registered_released'] and cleanup['state_rng_flags_gradients_and_identity_pass']

def test_compile_pure_endpoint_default_fullgraph_dynamic_and_unchanged_default_limit():
 calls=[]
 def compiler(f,**kwargs):calls.append((f.__name__,kwargs));return f
 kernels=Kernels(compiler)
 assert len(calls)==4 and all(v==dict(mode='default',fullgraph=True,dynamic=False) for _,v in calls)
 for name in ('highest_residual20','damped20','jacobi2'):
  out=kernels.evaluate(name,torch.tensor([[2.,.01],[.01,1.]]),torch.eye(2))
  assert torch.is_tensor(out) and out.shape==(2,2)
