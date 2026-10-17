"""Independent mixed retraction recurrence, strong rotations and whole controls."""
from pathlib import Path
from types import SimpleNamespace
import sys,copy,math,weakref
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support'),str(Path(__file__).resolve().parent)]
import numpy as np
import torch,pytest
from test_llm_jacobi2_learning import fixture as old_fixture
from test_llm_jacobi2 import independent_controls as prior_controls
from llm_cubic3_kernel import Kernels,retract_three
from native_residual_named_kernel import candidate_retraction
from llm_cubic3_contract import recipe,protocol,SEED,GATES,COUNTS,OBS,PURPOSE,calibration_contract
from llm_cubic3_runtime import Runtime
from llm_cubic3_diagnostic import build_arm,warmup
from matched_training_runtime import streaming_digest
from llm_math_confirmation import endpoint

def fixture():
 old,opt=old_fixture();r=object.__new__(Runtime);r.__dict__.update(old.__dict__);r.config=recipe();r.config.update(dim=16,heads=4,ff_hidden_dim=32,sequence_length=8,batch_size=16,vocab_size=32);r.protocol=protocol()
 r.cubic3_kernel=Kernels(compiler=lambda f,**kw:f);r.cubic3_counts={k:0 for k in COUNTS};r.cubic3_observations={k:0 for k in OBS};r.cubic3_rows=[]
 r.kernel.cubic3_runtime_ref=weakref.ref(r);r.kernel.emit=r.emit;r.kernel.rng_digest=lambda:streaming_digest(r.streams.capture());r.budget=lambda **kw:None
 for s in r.protocol['cubic3']['factors']:s['dimension']=dict(r.model.named_parameters())[s['parameter']].shape[s['axis']]
 fresh=build_arm(r.model,r.config,'qr20',r.kernel)
 for p,state in opt.matrix.state.items():fresh.matrix.state[p]=copy.deepcopy(state)
 return r,fresh

def independent_controls():
 assert prior_controls()
 for sigma in (.15,.85,1.):
  y=np.array([[1.,-sigma],[sigma,1.]],dtype=np.float32);got=retract_three(torch.from_numpy(y));expected=y.copy()
  for iteration in range(3):
   residual=expected.T@expected-np.eye(2,dtype=np.float32)
   left=torch.from_numpy(expected).bfloat16().float().numpy();right=torch.from_numpy(residual).bfloat16().float().numpy()
   correction=np.array([[sum(float(left[i,k])*float(right[k,j]) for k in range(2)) for j in range(2)] for i in range(2)],dtype=np.float32)
   expected=expected-.5*torch.from_numpy(correction).bfloat16().float().numpy()
  assert np.allclose(got.numpy(),expected,rtol=0,atol=1e-7)
  # Independent exact residual recurrence in FP64, not shared mixed helper.
  z=np.array([[1.,-sigma],[sigma,1.]],dtype=np.float64)
  for _ in range(3):
   e=z.T@z-np.eye(2);z=z@(np.eye(2)-.5*e)
   actual=z.T@z-np.eye(2);oracle=-.75*(e@e)+.25*(e@e@e)
   assert np.allclose(actual,oracle,rtol=1e-10,atol=1e-14)
  assert np.linalg.norm(got.numpy()-z)<.003
  two=candidate_retraction(torch.from_numpy(y));two_geom=float((two.T@two-torch.eye(2)).norm());three_geom=float((got.T@got-torch.eye(2)).norm())
  m=torch.tensor([[.1,.3],[.2,.7]]);world=got@(got.T@m@got)@got.T;pe=float((world-m).norm()/m.norm())
  if sigma==.85:assert two_geom>.05 and three_geom<.05 and pe<.01
  if sigma==1.:assert three_geom>.05 or pe>.01  # cubic3 is not a universal guarantee atcap1.
 return True

def full_control():
 r,opt=fixture();before=endpoint(r,opt);w=warmup(r,opt);assert w['passed'] and endpoint(r,opt)==before
 for call in (21,1021,3481):
  for group in opt.matrix.param_groups:
   for p in group['params']:
    state=opt.matrix.state[p];state['step']=call-1
    state['GG']=[q @ (torch.diag(torch.linspace(.5,4.,len(q)))+.002*(torch.ones_like(q)-torch.eye(len(q)))) @ q.T for q in state['Q']]
    r.kernel.probe=True;opt.matrix._research_refresh(state,group);r.kernel.probe=False
 assert r.cubic3_counts==COUNTS and r.cubic3_observations==OBS and len(r.cubic3_rows)==12 and all(x['passed'] for x in r.cubic3_rows)
 assert all(x['nonmutation_pass'] and x['registered_release_pass'] and x['compiler_unchanged'] and x['gradient_identity_pass'] for x in r.cubic3_rows)
 assert r.kernel.calls=={} and r.kernel.cf_calls==0
 return True

def test_independent_mixed_cubic3_exact_recurrence_concentrated_motion_and_rejections():assert independent_controls()
def test_full_same_state_all_factor_M_V_counts_nonmutation_defaultcompiler_release():assert full_control()

def test_known_bad_cubic2_shadow_retained_without_gating_proposed_control():
 r,opt=fixture()
 # A real unsafe tiny state exercises unchanged cubic2 rejection and cubic3 admission.
 from test_llm_jacobi2 import fixture as unsafe
 rr,oo=unsafe(safe=False)
 states={n:oo.matrix.state[p] for g in oo.matrix.param_groups for n,p in zip(g['param_names'],g['params'])}
 for g in opt.matrix.param_groups:
  for n,p in zip(g['param_names'],g['params']):opt.matrix.state[p]=copy.deepcopy(states[n])
 group=opt.matrix.param_groups[0];p=group['params'][0];opt.matrix.state[p]['step']=20;r.kernel.probe=True
 opt.matrix._research_refresh(opt.matrix.state[p],group)
 row=r.cubic3_rows[-1];assert not row['factors'][1]['policies']['jacobi2']['admitted'] and row['factors'][1]['policies']['jacobi2_cubic3']['admitted']
 assert row['passed'] and row['registered_release_pass']

def test_partial_allocation_CF_failure_restores_and_releases():
 r,opt=fixture();group=opt.matrix.param_groups[0];p=group['params'][0];state=opt.matrix.state[p];state['step']=20;before=streaming_digest(dict(state=state,p=p,grad=p.grad));evaluate=r.cubic3_kernel.evaluate
 def fails(name,cov,q,eager=False):
  if r.cubic3_counts['diagnostic_compiled']>=4:raise ValueError('deliberate cubic3 partial allocation failure')
  return evaluate(name,cov,q,eager)
 r.cubic3_kernel.evaluate=fails;r.kernel.probe=True
 with pytest.raises(RuntimeError,match='deliberate cubic3 partial allocation failure'):opt.matrix._research_refresh(state,group)
 cleanup=next(v for e,v in r.events if e=='cubic3_moment_cleanup');assert cleanup['nonmutation_pass'] and cleanup['registered_release_pass'] and cleanup['gradient_identity_pass']
 assert streaming_digest(dict(state=state,p=p,grad=p.grad))==before

def test_failed_proposed_admission_observes_all_four_safe_source_matrix_transactions():
 r,opt=fixture();original=r.cubic3_kernel.evaluate
 def bad(name,cov,q,eager=False):return q*1.004 if name=='jacobi2_cubic3' else original(name,cov,q,eager)
 r.cubic3_kernel.evaluate=bad;r.kernel.probe=True
 for group in opt.matrix.param_groups:
  for p in group['params']:
   opt.matrix.state[p]['step']=20;opt.matrix._research_refresh(opt.matrix.state[p],group)
 assert len(r.cubic3_rows)==4 and all(not x['passed'] for x in r.cubic3_rows) and all(x['registered_release_pass'] and x['nonmutation_pass'] for x in r.cubic3_rows)

@pytest.mark.parametrize('fails',[False,True])
def test_new_entry_evaluation_restoration_weighted_panels(fails):
 from test_llm_math_learning import runtime_fixture
 from llm_cubic3_qualification import evaluate_transaction
 base,opt=runtime_fixture();r=object.__new__(Runtime);r.__dict__.update(base.__dict__);before=endpoint(r,opt);flags=[m.training for m in r.model.modules()]
 def evaluate():
  torch.rand(2);r.model.table.train()
  if fails:raise ValueError('cubic3 eval failure')
  return Runtime.evaluate(r)
 r.evaluate=evaluate
 if fails:
  with pytest.raises(ValueError,match='cubic3 eval failure'):evaluate_transaction(r,opt)
 else:assert evaluate_transaction(r,opt)['token_weighted']
 assert endpoint(r,opt)==before and flags==[m.training for m in r.model.modules()]

def test_contract_original_recipe_bounds_gates_and_frozen_failed_parent_provenance(tmp_path,monkeypatch):
 from llm_jacobi2_learning_contract import recipe as old,protocol as op
 c=recipe();oc=old();assert {k:v for k,v in c.items() if k not in ('seed','matched_comparison')}=={k:v for k,v in oc.items() if k not in ('seed','matched_comparison')}
 p=protocol();assert p['gates']==op()['gates'] and p['bounds']==op()['bounds'] and p['maximum_optimizer_updates']==3770 and PURPOSE<12*2**30 and sum(COUNTS.values())==600
 import prepare_llm_cubic3 as prep
 parent=tmp_path/'parent';parent.mkdir();(parent/'source').mkdir();prep.write(parent/'manifest.json',dict(source_sha256={}));prep.write(parent/'execution_status.json',dict(status='failed'));prep.write(parent/'qr20_primary.json',dict(calls=20));prep.write(parent/'cpu_calibration.json',dict(status='PASS_CPU_JACOBI2_LIVE_CALIBRATION'));monkeypatch.setattr(prep,'PARENTS',(parent,))
 qual=tmp_path/'qual';qual.mkdir();prep.write(qual/'execution_status.json',dict(status='completed'));prep.write(qual/'summary.json',dict(policies=[dict(policy='math_attention',all_exact=True)]));monkeypatch.setattr(prep,'QUALIFICATION',qual)
 cache=tmp_path/'cache';cache.write_bytes(b'pinned');monkeypatch.setattr(prep,'CACHE',cache)
 audit=tmp_path/'audit.json';prep.write(audit,dict(status='PASS_CPU_TOKEN_CACHE_AUDIT',cuda_initialized=False,path=str(cache),file_sha256=prep.sha(cache)))
 ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=SEED,parameters=144875648,cuda_initialized=False,snapshot_purpose_bytes=2230649344));cal=tmp_path/'cal.json';prep.write(cal,calibration_contract())
 b=tmp_path/'fresh';m=prep.prepare(b,ref,data_audit=audit,calibration=cal);changed=prep.read(b/'protocol.json');changed['cubic3']['gates']['physical_m_relative']=.1;prep.write(b/'protocol.json',changed);m['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',m)
 with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
