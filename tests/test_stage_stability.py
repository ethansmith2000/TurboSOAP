"""Stage metrics, history lifetime, epoch boundaries and unchanged native training."""
from pathlib import Path
import sys,copy,json
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import pytest,numpy as np,torch
from stage_stability_contract import recipe,protocol,validate,PARENT,STAGES,LAGS,counts
from stage_stability_data import repeated_plan
from stage_stability_kernel import Recorder,covariance_metrics,gradient_metrics,residual,adaptive_metrics,build_arm
from tensor_ownership import TensorOwnership
from matched_training_runtime import streaming_digest
from qr_reuse_profile_kernel import following_update
from test_projection_learning import runtime as parent_runtime
import stage_stability as runner

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
 monkeypatch.setattr(torch.cuda,'_lazy_init',lambda *a,**k:(_ for _ in ()).throw(AssertionError('CUDA in CPU checks')))
 previous=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('high');torch.set_num_threads(2)
 yield
 torch.set_float32_matmul_precision(previous);assert not torch.cuda.is_initialized()

def runtime(tmp_path):
 r,e=parent_runtime(tmp_path);coverage=r.protocol['coverage'];r.config=recipe();r.protocol=protocol();r.protocol['coverage']=coverage;r.protocol['evaluation_calls']=[]
 return r,e

def advance(opt,start,stop):
 p=opt.matrix.param_groups[0]['params'][0]
 for call in range(start,stop+1):
  p.grad=torch.sin(torch.arange(p.numel(),dtype=p.dtype).reshape_as(p)+.03*call)+.0001*call
  opt.step()


def test_repeated_plan_full_epochs_tail_and_training_rng_isolation():
 np.random.seed(14);before=np.random.get_state();a,epochs=repeated_plan(190,13,4,8)
 assert a.shape==(8,4) and [v['batches'] for v in epochs]==[3,3,2]
 assert [(v['first_call'],v['last_call']) for v in epochs]==[(1,3),(4,6),(7,8)]
 assert all(v['dropped_incomplete_tail_blocks']==1 for v in epochs)
 assert len(set(a[:3].flat))==12 and len(set(a[3:6].flat))==12
 b,_=repeated_plan(190,13,4,8);assert np.array_equal(a,b)
 after=np.random.get_state();assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]


def test_first_pilot_prefix_exact_and_long_plan_boundaries():
 a,epochs=repeated_plan(190,78124,16,35000)
 original=np.random.default_rng(2191).permutation(78124)[:56000].reshape(3500,16)
 assert np.array_equal(a[:3500],original)
 assert len(epochs)==8 and epochs[0]['last_call']==4882 and epochs[-1]['last_call']==35000
 assert len(np.unique(a[:4882]))==78112 and a.min()>=0 and a.max()<78124

@pytest.mark.parametrize('args',[(190,0,4,2),(190,3,4,2),(190,8,4,0)])
def test_invalid_plan_rejected(args):
 with pytest.raises(ValueError):repeated_plan(*args)


def test_scale_shape_residual_and_gradient_metrics_have_independent_oracles():
 owner=TensorOwnership();a=torch.tensor([[3.,1.],[1.,2.]]);b=a/2
 c=covariance_metrics(a,b,owner);assert c['relative']==pytest.approx(1.) and c['trace_ratio']==pytest.approx(2.) and c['trace_normalized_relative']==pytest.approx(0.)
 value=residual(a,torch.eye(2),owner);assert value==pytest.approx((2/15)**.5,rel=1e-6)
 g=gradient_metrics(torch.tensor([1.,0.]),torch.tensor([0.,2.]),owner)
 assert g['cosine']==0 and g['norm_ratio']==.5 and g['relative']==pytest.approx(5**.5/2)
 assert owner.report()['all_registered_released'];owner.close()


def test_readonly_probe_nonmutation_all_metrics_and_history_release(tmp_path):
 r,e=runtime(tmp_path);opt=build_arm(r.model,r.config,'qr20',r.kernel);rec=Recorder(r,stages=(60,),lags=(2,10))
 try:
  advance(opt,1,50);rec.observe(50,opt);advance(opt,51,58);rec.observe(58,opt);advance(opt,59,60)
  before=rec.fingerprint(opt);row=rec.observe(60,opt)
  assert before==rec.fingerprint(opt) and row['cleanup']['source_unchanged']
  assert len(row['factors'])==2 and len(row['gradients'])==2 and len(row['adaptive'])==1
  assert set(row['factors'][0]['lags'])=={'2','10'} and row['factors'][0]['lags']['10']['capture_call']==50
  assert row['cleanup']['workspace_release']['all_registered_released'] and all(v['all_registered_released'] for v in row['cleanup']['history_releases'])
  assert not rec.banks and rec.close()['all_registered_released']
 finally:r.snapshot.close()

@pytest.mark.parametrize('failure',['metric','source_mutation'])
def test_probe_failure_releases_history_and_reports_source_guard(tmp_path,monkeypatch,failure):
 import stage_stability_kernel as k
 r,e=runtime(tmp_path);opt=build_arm(r.model,r.config,'qr20',r.kernel);rec=Recorder(r,stages=(60,),lags=(2,10))
 try:
  advance(opt,1,50);rec.observe(50,opt);advance(opt,51,58);rec.observe(58,opt);advance(opt,59,60)
  original=k.factor_metrics
  def fail(*a,**kw):
   if failure=='metric':raise RuntimeError('injected stage interruption')
   a[0].mul_(1.01);return original(*a,**kw)
  monkeypatch.setattr(k,'factor_metrics',fail)
  with pytest.raises(RuntimeError,match='Stage'):rec.observe(60,opt)
  assert rec.close()['all_registered_released'] and not rec.banks
  cleanup=next(v for n,v in reversed(e) if n=='stage_observation_cleanup')
  assert cleanup['workspace_release']['all_registered_released'] and all(v['all_registered_released'] for v in cleanup['history_releases'])
  assert cleanup['source_unchanged']==(failure=='metric')
 finally:r.snapshot.close()


def test_early_failure_closes_retained_partial_history(tmp_path):
 r,e=runtime(tmp_path);opt=build_arm(r.model,r.config,'qr20',r.kernel);rec=Recorder(r,stages=(60,),lags=(2,10))
 try:
  advance(opt,1,50);rec.observe(50,opt);assert len(rec.banks)==1
  assert rec.close()['all_registered_released'] and not rec.banks
 finally:r.snapshot.close()


def test_candidate_screen_exceedance_is_retained_without_aborting_source(tmp_path,monkeypatch):
 import stage_stability_kernel as k
 r,e=runtime(tmp_path);opt=build_arm(r.model,r.config,'qr20',r.kernel);rec=Recorder(r,stages=(60,),lags=(2,10))
 try:
  advance(opt,1,50);rec.observe(50,opt);advance(opt,51,58);rec.observe(58,opt);advance(opt,59,60)
  monkeypatch.setattr(k,'adaptive_metrics',lambda *a:dict(relative=.02,cosine=.999,passes_187_screen=False))
  row=rec.observe(60,opt);assert not row['adaptive'][0]['passes_187_screen'] and row['cleanup']['source_unchanged'] and rec.close()['all_registered_released']
 finally:r.snapshot.close()


def test_runner_interruption_releases_retained_history_and_source_lease(tmp_path,monkeypatch):
 r,e=runtime(tmp_path);original=Recorder.observe
 def interrupted(self,call,opt):
  if call==30:raise RuntimeError('injected with history retained')
  return original(self,call,opt)
 monkeypatch.setattr(Recorder,'observe',interrupted)
 monkeypatch.setattr(runner,'Recorder',lambda rt:Recorder(rt,stages=(40,),lags=(2,20)))
 try:
  expected=runner.preflight(r)
  with pytest.raises(RuntimeError,match='Trajectory failed'):runner.trajectory(r,'qr20','primary',245,expected)
  row=json.loads((tmp_path/'qr20_primary.json').read_text())
  assert row['completed_optimizer_step_calls']==30 and row['history_release']['released_history_banks']==1 and row['history_release']['all_registered_released'] and row['release']['passed']
  assert streaming_digest(r.model.state_dict())==r.snapshot.initial_digest
 finally:r.snapshot.close()


def test_partial_capture_failure_releases_already_copied_arrays(tmp_path,monkeypatch):
 import stage_stability_kernel as k
 from matched_training_runtime import failure_record
 r,e=runtime(tmp_path);opt=build_arm(r.model,r.config,'qr20',r.kernel);rec=Recorder(r,stages=(60,),lags=(2,10));original=k.TensorOwnership.watch;seen=[]
 def interrupted(self,tensor,label):
  original(self,tensor,label);seen.append(label)
  if len(seen)==2:raise RuntimeError('injected partial capture')
 try:
  advance(opt,1,50);monkeypatch.setattr(k.TensorOwnership,'watch',interrupted)
  try:rec.observe(50,opt)
  except RuntimeError as exc:failure_record(exc)
  else:raise AssertionError('Partial capture injection did not fire')
  assert rec.close()['all_registered_released'] and not rec.banks
 finally:r.snapshot.close()


def test_native_trajectory_stays_exact_with_all_observers_and_releases(tmp_path,monkeypatch):
 r,e=runtime(tmp_path)
 monkeypatch.setattr(runner,'Recorder',lambda rt:Recorder(rt,stages=(40,60,80,100,120,140,160),lags=(2,20)))
 try:
  expected=runner.preflight(r);d=runner.trajectory(r,'qr20','disposable',245,expected);r._stability_disposable={'qr20':d};p=runner.trajectory(r,'qr20','primary',245,expected)
  assert d['prefixes']['245']==p['prefixes']['245'] and [(v['loss'],v['gradient_norm']) for v in d['timings']]==[(v['loss'],v['gradient_norm']) for v in p['timings']]
  assert len(p['stage_observations'])==7 and p['history_release']['released_history_banks']==14 and p['history_release']['all_registered_released'] and p['release']['passed']
  assert p['final_counters']['parameters'][0]['actual']==counts(245,'qr20')
  assert any(n=='long_prefix_passed' and v['exact_scalar_and_state_prefix'] for n,v in e)
 finally:r.snapshot.close()


def test_extended_physical_m_samples_include_late_native_refresh(tmp_path):
 from stage_stability_kernel import qualify_actual_m
 r,e=runtime(tmp_path);opt=build_arm(r.model,r.config,'qr20',r.kernel)
 try:
  advance(opt,1,2);state=next(iter(opt.matrix.state.values()));state['step']=30000
  qualify_actual_m(opt.matrix,r.kernel,'qr20','primary');opt.matrix._research_refresh(state,opt.matrix.param_groups[0])
  assert r.kernel.moments[-1]['age']==30000 and r.kernel.moments[-1]['passed']
 finally:r.snapshot.close()


def test_long_source_prefix_mismatch_aborts_at245_and_releases(tmp_path,monkeypatch):
 r,e=runtime(tmp_path)
 monkeypatch.setattr(runner,'Recorder',lambda rt:Recorder(rt,stages=(40,60,80,100,120,140,160),lags=(2,20)))
 try:
  expected=runner.preflight(r);d=runner.trajectory(r,'qr20','disposable',245,expected)
  forged=copy.deepcopy(d);forged['prefixes']['245']['model_sha256']='invalid'
  r._stability_disposable={'qr20':forged}
  with pytest.raises(RuntimeError,match='Trajectory failed'):runner.trajectory(r,'qr20','primary',35000,expected)
  row=json.loads((tmp_path/'qr20_primary.json').read_text())
  assert row['calls']==245 and 'Early long-source exact245prefix failed' in row['failure']['error']
  assert row['history_release']['all_registered_released'] and row['release']['passed']
 finally:r.snapshot.close()


def test_contract_full_schedule_common_refresh_phase_counts_and_memory():
 c,p=recipe(),protocol();validate(c,p)
 assert p['maximum_optimizer_updates']==35270 and p['expected']['actual_m_transactions']==36
 assert p['schedule']==dict(total=35000,lr_warmup=875) and all(t%20==0 for t in STAGES)
 assert p['llm_data']['training_tokens_per_primary_arm']==286720000 and p['stage_stability']['counts']['extra_optimizer_steps']==0
 assert sum(v for k,v in p['memory'].items() if type(v) is int and k!='maximum_tensor_purpose_bytes')<p['memory']['maximum_tensor_purpose_bytes']
 changed=copy.deepcopy(p);changed['stage_stability']['lags']=[40,400]
 with pytest.raises(ValueError,match='Undeclared'):validate(c,changed)


def test_prepare_preserves266sources_and_rejects_protocol_tampering(tmp_path):
 import prepare_stage_stability as prep
 ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=190,parameters=144875648,cuda_initialized=False,snapshot_purpose_bytes=2230649344,model_sha256='9'*64))
 cal=tmp_path/'cal.json';prep.write(cal,dict(status='PASS_CPU_STAGE_STABILITY',tests_passed=True,cuda_initialized=False))
 b=tmp_path/'bundle';m=prep.prepare(b,ref,190,PARENT/'token_cache_audit.json',cal)
 old=prep.read(PARENT/'manifest.json')['source_sha256'];assert len(old)==266 and len(m['source_sha256'])==274 and all(m['source_sha256'][n]==h for n,h in old.items())
 p=prep.read(b/'protocol.json');p['stage_stability']['stages']=[1000];prep.write(b/'protocol.json',p);m=prep.read(b/'manifest.json');m['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',m)
 with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
 with pytest.raises(ValueError,match='Fresh bundle'):prep.prepare(b,ref,190,PARENT/'token_cache_audit.json',cal)
