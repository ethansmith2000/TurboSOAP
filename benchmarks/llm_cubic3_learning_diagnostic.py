"""Read-only two-sided actual-history diagnostic and pre-primary layout warmup."""
import gc,math,weakref,time,hashlib
import numpy as np
import torch
from hybrid_training_kernel import precision,configure_hybrid
from cadence_training_contract import build_arm as baseline
from llm_cubic3_learning_contract import ARMS,POLICIES,GATES,PURPOSE
from llm_jacobi2_core import reference_metrics,numerical_backend,limits
from llm_sequence_averaging_core import fp32_math_control
from llm_math_confirmation import endpoint,compiler_stats
from llm_step_scale_atoms import metrics,admitted,relative,rotated_covariance,candidate_from_generator
from llm_basis_response_atoms import basis_step,offdiag_norm,geometry
from tracking_curvature_kernel import generator
from matched_training_runtime import streaming_digest,failure_record
from tensor_ownership import TensorOwnership

def increment(router,key):router.diagnostic_counts[key]+=1

def previews(matrix,state,group,banks,orders,owner,router,parameter):
 def own(t,label):owner.watch(t,label);return t
 def project(back,*args):
  increment(router,'full_m_projection_calls')
  return (matrix._project_back_with_basis if back else matrix._project_with_basis)(*args)
 world=own(project(True,state['exp_avg'],state['Q'],group),'physical.worldM')
 v=state['exp_avg_sq'];qv=v
 for axis,order in enumerate(orders):qv=own(qv.index_select(axis,order),'physical.qrV')
 projected=own(project(False,world,banks['qr20'],group),'physical.qrM')
 ref=own(project(True,projected/(qv.sqrt()+group['eps']),banks['qr20'],group),'physical.qrAdaptive');del projected
 rows={};current=None
 for name,basis,variance in [('qr_permutationV',banks['qr20'],qv),('qr_carryV',banks['qr20'],v)]+[(n,banks[n],v) for n in POLICIES]:
  projected=own(project(False,world,basis,group),'physical.projectedM')
  reconstructed=own(project(True,projected,basis,group),'physical.reconstructedM')
  adaptive=own(project(True,projected/(variance.sqrt()+group['eps']),basis,group),'physical.adaptive')
  pe=relative(reconstructed,world)
  row=dict(world_m_reconstruction_relative=pe,adaptive_relative_to_qr_permutation=relative(adaptive,ref),
   adaptive_cosine_to_qr_permutation=float((adaptive*ref).sum()/(adaptive.norm()*ref.norm()).clamp_min(1e-24)),adaptive_norm=float(adaptive.norm()),
   physical_m_admitted=math.isfinite(pe) and pe<=GATES['physical_m_relative'],variance_policy='QR permutation' if name=='qr_permutationV' else 'unchanged carry')
  if name=='highest_residual20':current=own(adaptive.clone(),'physical.J1adaptive')
  if current is not None:row['adaptive_relative_to_jacobi1']=relative(adaptive,current)
  row['finite']=all(math.isfinite(v) for v in row.values() if isinstance(v,float));rows[name]=row;increment(router,'full_m_previews')
  router.emit('cubic3_live_adaptive_before_gate',dict(parameter=parameter,source_step=int(state['step']),policy=name,**row))
  del projected,reconstructed,adaptive
 return rows

def factor(matrix,state,group,router,parameter,axis,owner):
 cov=state['GG'][axis];q=state['Q'][axis];before=offdiag_norm(q,cov,owner)
 identity=metrics(q,cov,q,before,owner);bank={}
 qr,order=basis_step('qr20',cov,q,owner);increment(router,'qr_controls');bank['qr20']=qr
 policies={'qr20':dict(metrics=metrics(qr,cov,q,before,owner,order))};policies['qr20']['admitted']=admitted(policies['qr20']['metrics'],GATES)
 for name in POLICIES:
  actual=router.evaluate_cf(name,cov,q);owner.watch(actual,'CF.compiled')
  ref=router.kernels[name].evaluate(cov,q,eager=True);owner.watch(ref,'CF.eager');increment(router,'eager_counterfactuals')
  comparison=reference_metrics(actual,ref,q,GATES);mm=metrics(actual,cov,q,before,owner)
  row=dict(metrics=mm,reference=comparison,admitted=comparison['passed'] and admitted(mm,GATES))
  if name=='jacobi2_cubic3':
   b,_=rotated_covariance(cov,q,owner);raw,bound,a,cap,scale=generator(b,'jacobi');y,first,overlay=candidate_from_generator(q,a,2.,owner)
   increment(router,'scale2_overlay_references')
   residual=overlay.T@overlay;residual.diagonal().sub_(1.);correction=(overlay.bfloat16()@residual.bfloat16()).float();third=overlay-.5*correction
   owner.watch(residual,'overlay.third_Gram');owner.watch(correction,'overlay.third_correction');owner.watch(third,'overlay.third_output');increment(router,'third_correction_overlay_references');err=relative(ref,third)
   del residual,correction,third
   row.update(scale2_overlay_relative=err,raw_bound=float(bound),original_chosen_bound=float(a.norm()/math.sqrt(2)),applied_bound=float(2*a.norm()/math.sqrt(2)),cap_active=float(bound)>.5)
   row['admitted'] &= math.isfinite(err) and err<=GATES['scale2_overlay_relative']
   del b,raw,bound,a,cap,scale,y,first,overlay
  router.emit('cubic3_live_CF_before_gate',dict(parameter=parameter,axis=axis,dimension=len(q),source_step=int(state['step']),policy=name,**row))
  policies[name]=row;bank[name]=actual;del actual,ref
 return bank,order,dict(axis=axis,dimension=len(q),before_covariance_offdiag=before,identity=identity,policies=policies,passed=admitted(identity,GATES) and all(p['admitted'] for p in policies.values()))

def diagnose(matrix,state,group,router,parameter,owner,arm):
 banks={n:[] for n in ARMS};orders=[];factors=[]
 with torch.no_grad(),fp32_math_control():
  for axis in range(2):
   bank,order,row=factor(matrix,state,group,router,parameter,axis,owner)
   for n in ARMS:banks[n].append(bank[n])
   orders.append(order);factors.append(row);del bank,order
  view=previews(matrix,state,group,banks,orders,owner,router,parameter)
 row=dict(parameter=parameter,source_step=int(state['step']),source_arm=arm,shape=list(state['exp_avg'].shape),factors=factors,previews=view,
  fixed_coordinate_covariance_samples=[],full_actual_momentum=True,full_actual_diagonal_v=True,adaptive_preview_not_future_step=True,
  all_finite=all(p['finite'] for p in view.values()),physical_m_pass=all(p['physical_m_admitted'] for p in view.values()),passed=all(f['passed'] for f in factors) and all(p['physical_m_admitted'] and p['finite'] for p in view.values()))
 increment(router,'full_moment_transactions');router.emit('cubic3_live_full_matrix_before_gate',row)
 return row

def configure_diagnostic(matrix,router,arm):
 original=type(matrix)._research_refresh;ref=weakref.ref(matrix)
 def refresh(state,group):
  self=ref()
  if self is None:raise RuntimeError('Missing live optimizer owner')
  if router.probe:
   name,p=next((n,p) for n,p in zip(group['param_names'],group['params']) if self.state[p] is state)
   grad_identity=id(p.grad);before=streaming_digest(dict(state=state,parameter=p,gradient=p.grad));rng=router.rng_digest();backend=numerical_backend();compiler=compiler_stats();lim=limits();owner=TensorOwnership();error=None;row=None
   try:row=diagnose(self,state,group,router,name,owner,arm)
   except BaseException as exc:error=failure_record(exc);exc.__traceback__=None
   finally:
    gc.collect();release=owner.report();owner.close();after=streaming_digest(dict(state=state,parameter=p,gradient=p.grad))
    cleanup=dict(parameter=name,source_arm=arm,source_step=int(state['step']),before_sha256=before,after_sha256=after,rng_before=rng,rng_after=router.rng_digest(),
     nonmutation_pass=before==after and rng==router.rng_digest() and backend==numerical_backend() and grad_identity==id(p.grad),gradient_identity_pass=grad_identity==id(p.grad),registered_release=release,registered_release_pass=release['all_registered_released'],compiler_unchanged=compiler==compiler_stats(),compiler_limits=limits())
    router.emit('cubic3_live_moment_cleanup',cleanup)
   if error is not None:raise RuntimeError('Actual-history read-only diagnostic failed: '+str(error))
   row.update(cleanup);router.moment_probes.append(row);router.emit('full_moment_probe_before_gate',row)
   if not (row['passed'] and cleanup['nonmutation_pass'] and cleanup['registered_release_pass'] and cleanup['compiler_unchanged'] and lim==limits()==dict(recompile_limit=8,accumulated_recompile_limit=256)):raise FloatingPointError('Jacobi2 actual-history original fidelity/geometry/fullM/release/compiler admission failed')
  return original(self,state,group)
 matrix._research_refresh=refresh

def build_arm(model,config,arm,router):
 if arm not in ARMS:raise ValueError('Undeclared live arm')
 opt=baseline(model,config,'qr20' if arm=='qr20' else 'warm20')
 if arm in POLICIES:configure_hybrid(opt.matrix,router.kernels[arm])
 configure_diagnostic(opt.matrix,router,arm);return opt

def warmup(runtime,opt):
 started=time.monotonic();spec=runtime.protocol['cubic3_live'];before=endpoint(runtime,opt);rng=runtime.streams.capture();grad=streaming_digest([p.grad for p in runtime.model.parameters()]);ids=[id(p.grad) for p in runtime.model.parameters()]
 modules=list(runtime.model.modules());flags=[m.training for m in modules];backend=numerical_backend();lim=limits();owner=TensorOwnership();rows=[];error=None
 try:
  if lim!=spec['compiler_limits'] or PURPOSE>spec['memory']['maximum_tensor_purpose_bytes']:raise RuntimeError('Declared warmup limit or purpose exceeded')
  runtime.model.eval()
  named={n:(p,g) for g in opt.matrix.param_groups for n,p in zip(g['param_names'],g['params'])}
  with torch.no_grad(),fp32_math_control():
   for factor_spec in spec['factors']:
    p,group=named[factor_spec['parameter']];state=opt.matrix.state[p];cov=state['GG'][factor_spec['axis']];old=state['Q'][factor_spec['axis']]
    if old.shape!=(factor_spec['dimension'],factor_spec['dimension']):raise ValueError('Declared warmup actual factor shape differs')
    for policy in POLICIES:
     q=old
     for layout in ('qr_output','canonical_warm_output'):
      runtime.budget(resources=True)
      actual=runtime.kernel.kernels[policy].evaluate(cov,q);owner.watch(actual,'warmup.compiled');runtime.kernel.warmup_counts['compiled_basis']+=1
      eager=runtime.kernel.kernels[policy].evaluate(cov,q,eager=True);owner.watch(eager,'warmup.eager');runtime.kernel.warmup_counts['eager_references']+=1
      rr=reference_metrics(actual,eager,q,GATES)
      row=dict(**factor_spec,policy=policy,layout=layout,input_stride=list(q.stride()),output_stride=list(actual.stride()),reference=rr,geometry_observational_only=geometry(actual,owner));runtime.emit('cubic3_live_warmup_before_gate',row);rows.append(row)
      if not rr['passed']:raise FloatingPointError('Live Jacobi2 warmup own-eager fidelity failed')
      if tuple(actual.stride()) not in ((1,len(q)),(len(q),1)):raise ValueError('Unexpected live warm output layout')
      q=actual.contiguous();owner.watch(q,'warmup.row_major');del eager
     del actual,q
 except BaseException as exc:error=failure_record(exc);exc.__traceback__=None
 finally:
  runtime.streams.restore(rng)
  for m,f in zip(modules,flags):m.training=f
  gc.collect();release=owner.report();owner.close();after=endpoint(runtime,opt)
  cleanup=dict(before=before,after=after,nonmutation_pass=before==after and grad==streaming_digest([p.grad for p in runtime.model.parameters()]) and ids==[id(p.grad) for p in runtime.model.parameters()] and flags==[m.training for m in modules] and backend==numerical_backend() and lim==limits(),registered_release=release,compiler_limits=limits(),counts=dict(runtime.kernel.warmup_counts),seconds=time.monotonic()-started)
  runtime.emit('cubic3_live_warmup_cleanup',cleanup)
 if not cleanup['nonmutation_pass'] or not release['all_registered_released']:raise RuntimeError('Live warmup restoration or registered release failed')
 if error is not None:raise RuntimeError('Live warmup failed: '+str(error))
 if runtime.kernel.warmup_counts!=spec['warmup_counts']:raise RuntimeError('Declared live warmup count differs')
 return dict(rows=rows,cleanup=cleanup,passed=True)
