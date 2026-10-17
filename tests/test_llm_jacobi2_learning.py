"""Actual instance hooks, cadence/transport and scoped read-only failure controls."""
from pathlib import Path
from types import SimpleNamespace
import sys,copy,math
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support'),str(Path(__file__).resolve().parent)]
import pytest,torch,numpy as np
from test_llm_jacobi2 import fixture as old_fixture,independent_controls
from llm_jacobi2_learning_contract import recipe,protocol,SEED,ARMS,POLICIES,PROBES,WARMUP_COUNTS,DIAGNOSTIC_COUNTS,PURPOSE,GATES,calibration_contract
from llm_jacobi2_learning_kernel import PolicyRouter
from llm_jacobi2_learning_diagnostic import build_arm,warmup
from matched_training_runtime import streaming_digest
from llm_math_confirmation import endpoint
from cadence_training_contract import validate_counters
from soap_reference import ResearchSOAP
from movement_retraction import configure_matrix
from hybrid_training_kernel import configure_hybrid,precision
from llm_basis_response_atoms import basis_step
from tensor_ownership import TensorOwnership

def fixture(arm='qr20'):
 r,old=old_fixture();r.config=recipe();r.config.update(dim=16,heads=4,ff_hidden_dim=32,sequence_length=8,batch_size=16,vocab_size=32);r.protocol=protocol()
 r.events=[];r.emit=lambda event,payload:r.events.append((event,payload));r.kernel=PolicyRouter(r.emit,lambda:streaming_digest(r.streams.capture()),compiler=lambda f,**kw:f)
 r.calls=0;r.budget=lambda **kw:None
 opt=build_arm(r.model,r.config,arm,r.kernel)
 for p,s in old.matrix.state.items():opt.matrix.state[p]=copy.deepcopy(s)
 named=dict(r.model.named_parameters())
 for s in r.protocol['jacobi2_live']['factors']:s['dimension']=named[s['parameter']].shape[s['axis']]
 return r,opt

def full_control():
 r,opt=fixture();w=warmup(r,opt);assert w['passed'] and r.kernel.warmup_counts==WARMUP_COUNTS
 for arm in ARMS:
  opt=build_arm(r.model,r.config,arm,r.kernel)
  source_r,source_opt=old_fixture()
  source={n:source_opt.matrix.state[p] for g in source_opt.matrix.param_groups for n,p in zip(g['param_names'],g['params'])}
  for group in opt.matrix.param_groups:
   for name,p in zip(group['param_names'],group['params']):opt.matrix.state[p]=copy.deepcopy(source[name])
  for call in PROBES:
   for group in opt.matrix.param_groups:
    for p in group['params']:
     state=opt.matrix.state[p];state['step']=call-1
     # Independent well-conditioned covariance expressed in the ACTUAL current basis.
     state['GG']=[q @ (torch.diag(torch.linspace(.5,4.,len(q)))+.002*(torch.ones_like(q)-torch.eye(len(q)))) @ q.T for q in state['Q']]
     r.kernel.probe=True;opt.matrix._research_refresh(state,group);r.kernel.probe=False
 assert len(r.kernel.probes)==48 and len(r.kernel.moment_probes)==36 and r.kernel.cf_calls==144
 assert r.kernel.diagnostic_counts==DIAGNOSTIC_COUNTS
 assert all(row['passed'] and row['nonmutation_pass'] and row['registered_release_pass'] and row['gradient_identity_pass'] and row['compiler_unchanged'] for row in r.kernel.moment_probes)
 assert sum(v.nbytes for _,v in r.kernel.covariance_samples.values())<=393216
 r.kernel.covariance_samples.clear();return True

def test_full_preprimary_warmup_and_all_source_probe_counts_release_flags_and_fullM():assert full_control()

def test_independent_capped_uncapped_bf16_retraction_and_physical_reference_controls():assert independent_controls()

def test_real_warm_transport_carryV_and_QR_permutation_exact_without_step_path_changes():
 router=PolicyRouter(lambda *a:None,lambda:'rng',compiler=lambda f,**kw:f)
 for arm in ARMS:
  p=torch.nn.Parameter(torch.arange(6,dtype=torch.float32).reshape(2,3)/10)
  opt=ResearchSOAP([p],basis_method='qr' if arm=='qr20' else 'warm',precondition_frequency=20,hard_reset_interval=0 if arm=='qr20' else 200,transport_precision='highest')
  if arm!='qr20':configure_matrix(opt,'fro_cubic2_cap05');configure_hybrid(opt,router.kernels[arm])
  group=opt.param_groups[0];old=[torch.eye(2),torch.eye(3)]
  covs=[torch.tensor([[3.,.01],[.01,1.]]),torch.tensor([[3.,.01,.005],[.01,2.,.003],[.005,.003,1.]])]
  m=torch.tensor([[.1,.2,.3],[.4,.5,.6]]);v=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
  state=dict(Q=old,GG=covs,exp_avg=m.clone(),exp_avg_sq=v.clone(),step=20)
  with precision('highest'):world=old[0]@m@old[1].T
  basis=[];orders=[];owner=TensorOwnership()
  for cov,q in zip(covs,old):
   if arm=='qr20':qq,order=basis_step('qr20',cov,q,owner);orders.append(order)
   else:qq=router.kernels[arm].evaluate(cov,q,eager=True)
   basis.append(qq)
  with precision('highest'):expected=basis[0].T@world@basis[1]
  with precision('high'):opt._research_refresh(state,group)
  assert torch.equal(state['exp_avg'],expected) and all(torch.equal(a,b) for a,b in zip(state['Q'],basis))
  wanted=v.index_select(0,orders[0]).index_select(1,orders[1]) if arm=='qr20' else v
  assert torch.equal(state['exp_avg_sq'],wanted)
  assert state['research_strict_transports']==1 and state['research_qr_refreshes']==int(arm=='qr20') and state['research_warm_refreshes']==int(arm!='qr20')
  assert float((state['Q'][0]@state['exp_avg']@state['Q'][1].T-world).norm()/world.norm())<.01
  owner.close()

def test_actual245_native_cadence_repair_counters_per_policy_and_initial_reset():
 from llm_feasibility_runtime import build_model
 from matched_training_contract import RandomStreams,SharedInitialState
 torch.manual_seed(SEED);c=recipe();c.update(dim=8,heads=2,ff_hidden_dim=16,sequence_length=8,batch_size=16,vocab_size=32)
 model=build_model(c);streams=RandomStreams(torch.Generator().manual_seed(1),SimpleNamespace(rng=np.random.default_rng(2),counts={}))
 snapshot=SharedInitialState(model,streams,extra_workspace_bytes=4096,ram_cap=2**26)
 scalar_prefix={}
 for arm in ARMS:
  snapshot.reset();router=PolicyRouter(lambda *a:None,lambda:streaming_digest(streams.capture()),compiler=lambda f,**kw:f);opt=build_arm(model,c,arm,router)
  for call in range(1,246):
   for group in opt.matrix.param_groups:
    for p in group['params']:
     grad=torch.zeros_like(p);idx=torch.arange(min(p.shape));grad[idx,idx]=torch.linspace(.01,.04,len(idx));p.grad=grad
   for group in opt.auxiliary.param_groups:
    for p in group['params']:p.grad=torch.zeros_like(p)
   opt.step()
  guard=validate_counters(opt,245);assert guard['parameter_count']==4
  assert sum(router.calls.values())==(88 if arm!='qr20' else 0)
  for group in opt.matrix.param_groups:
   for p in group['params']:
    state=opt.matrix.state[p];assert state['research_qr_refreshes']==(12 if arm=='qr20' else 1) and state['research_warm_refreshes']==(0 if arm=='qr20' else 11)
  model.zero_grad(set_to_none=True);del opt
 snapshot.reset();assert streaming_digest(model.state_dict())==snapshot.initial_digest;snapshot.close()

def test_partial_counterfactual_failure_restores_state_and_releases_live_banks():
 r,opt=fixture('jacobi2');group=opt.matrix.param_groups[0];p=group['params'][0];state=opt.matrix.state[p];state['step']=20
 before=streaming_digest(dict(state=state,p=p,grad=p.grad));evaluate=r.kernel.evaluate_cf
 def fails(name,cov,q):
  if r.kernel.cf_calls>=3:raise ValueError('deliberate live CF partial allocation failure')
  return evaluate(name,cov,q)
 r.kernel.evaluate_cf=fails;r.kernel.probe=True
 with pytest.raises(RuntimeError,match='deliberate live CF partial allocation failure'):opt.matrix._research_refresh(state,group)
 cleanup=next(v for e,v in r.events if e=='jacobi2_moment_cleanup')
 assert cleanup['registered_release_pass'] and cleanup['nonmutation_pass'] and cleanup['gradient_identity_pass']
 assert streaming_digest(dict(state=state,p=p,grad=p.grad))==before

def test_live_own_eager_rejection_and_original_absolute_geometry_gate():
 r,opt=fixture('jacobi2');kernel=r.kernel.kernels['jacobi2'];original=kernel.prefix
 kernel.prefix=lambda c,q:original(c,q)*1.1;r.kernel.probe=True
 group=opt.matrix.param_groups[0];p=group['params'][0];state=opt.matrix.state[p]
 with pytest.raises(FloatingPointError,match='parity failed'):kernel(state['GG'][0],state['Q'][0],group)
 assert not r.kernel.probes[-1]['passed'] and r.kernel.probes[-1]['basis_relative']>GATES['reference_basis_relative']
 # Explicit absolute-only rejection: normalizedGram~.004 passes .005, absolute~.064 fails .05.
 kernel.evaluate=lambda cov,q,eager=False:q*1.002
 with pytest.raises(FloatingPointError,match='absolute geometry failed'):kernel(torch.eye(256),torch.eye(256),group)
 assert r.kernel.probes[-1]['gram']<.005 and r.kernel.probes[-1]['absolute_gram_fro']>.05

@pytest.mark.parametrize('fails',[False,True])
def test_new_evaluator_exact_flags_RNG_grad_restore_and_weighted_panels(fails):
 from test_llm_math_learning import runtime_fixture
 from llm_jacobi2_learning_runtime import Runtime
 from llm_jacobi2_learning import evaluate_transaction
 base,opt=runtime_fixture();r=object.__new__(Runtime);r.__dict__.update(base.__dict__);flags=[m.training for m in r.model.modules()];before=endpoint(r,opt);grad=streaming_digest([p.grad for p in r.model.parameters()])
 def evaluate():
  torch.rand(2);r.model.table.train()
  if fails:raise ValueError('live eval failure control')
  return Runtime.evaluate(r)
 r.evaluate=evaluate
 if fails:
  with pytest.raises(ValueError,match='live eval failure control'):evaluate_transaction(r,opt)
 else:
  row=evaluate_transaction(r,opt);assert row['restoration_passed'] and row['token_weighted'] and row['metrics']['validation']['valid_tokens']==9
 assert endpoint(r,opt)==before and flags==[m.training for m in r.model.modules()] and grad==streaming_digest([p.grad for p in r.model.parameters()])

def test_contract_original_recipe_bounds_gates_and_resealed_M_change_rejection(tmp_path,monkeypatch):
 from llm_math_confirmation_contract import recipe as old,protocol as op
 c=recipe();oc=old();assert {k:v for k,v in c.items() if k not in ('seed','matched_comparison')}=={k:v for k,v in oc.items() if k not in ('seed','matched_comparison')}
 pp=protocol();assert pp['gates']==op()['gates'] and pp['bounds']==op()['bounds'] and pp['maximum_optimizer_updates']==11260 and PURPOSE<12*2**30
 assert pp['expected']['hybrid_kernel_calls']==2688 and pp['expected']['counterfactual_basis_calls']==144 and pp['expected']['full_moment_transactions']==36
 import prepare_llm_jacobi2_learning as prep
 parent=tmp_path/'parent';parent.mkdir();(parent/'source').mkdir();prep.write(parent/'manifest.json',dict(source_sha256={}));prep.write(parent/'execution_status.json',dict(status='completed'));monkeypatch.setattr(prep,'PARENTS',(parent,))
 qual=tmp_path/'qual';qual.mkdir();prep.write(qual/'execution_status.json',dict(status='completed'));prep.write(qual/'summary.json',dict(policies=[dict(policy='math_attention',all_exact=True)]));monkeypatch.setattr(prep,'QUALIFICATION',qual)
 cache=tmp_path/'cache';cache.write_bytes(b'pinned');monkeypatch.setattr(prep,'CACHE',cache)
 audit=tmp_path/'audit.json';prep.write(audit,dict(status='PASS_CPU_TOKEN_CACHE_AUDIT',cuda_initialized=False,path=str(cache),file_sha256=prep.sha(cache)))
 ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=SEED,parameters=144875648,cuda_initialized=False,snapshot_purpose_bytes=2230649344))
 cal=tmp_path/'cal.json';prep.write(cal,calibration_contract());b=tmp_path/'fresh';m=prep.prepare(b,ref,data_audit=audit,calibration=cal)
 broken=prep.read(b/'protocol.json');broken['jacobi2_live']['gates']['physical_m_relative']=.1;prep.write(b/'protocol.json',broken);m['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',m)
 with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
