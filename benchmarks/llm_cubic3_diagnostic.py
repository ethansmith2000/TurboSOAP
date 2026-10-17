"""Read-only two-sided actual-history diagnostic and pre-primary layout warmup."""
import gc,math,weakref,time,hashlib
import numpy as np
import torch
from hybrid_training_kernel import precision,configure_hybrid
from cadence_training_contract import build_arm as baseline
from llm_cubic3_contract import ARMS,GATES,PURPOSE,NAMES,MANDATORY,COUNTS,OBS
POLICIES=NAMES[1:]
from llm_jacobi2_core import reference_metrics,numerical_backend,limits
from llm_sequence_averaging_core import fp32_math_control
from llm_math_confirmation import endpoint,compiler_stats
from llm_step_scale_atoms import metrics,admitted,relative,rotated_covariance,candidate_from_generator
from llm_basis_response_atoms import basis_step,offdiag_norm,geometry
from tracking_curvature_kernel import generator
from matched_training_runtime import streaming_digest,failure_record
from tensor_ownership import TensorOwnership

def increment(runtime,key):
 target=runtime.cubic3_counts if key in COUNTS else runtime.cubic3_observations
 target[key]+=1

def measured_metrics(runtime,*args,**kwargs):
 increment(runtime,'factor_metrics');return metrics(*args,**kwargs)

def timed_basis(runtime,name,cov,q,owner):
 if q.is_cuda:
  marks=runtime.optimizer_events();runtime.synchronize();marks[0].record();started=time.perf_counter()
 else:marks=None;started=time.perf_counter()
 if name=='qr20':
  with precision('high'):
   order=torch.argsort((q.T@cov@q).diagonal(),descending=True);out,_triangular=torch.linalg.qr(cov@q[:,order])
 else:out=runtime.cubic3_kernel.evaluate(name,cov,q)
 if marks is not None:
  marks[1].record();runtime.synchronize();event_ms=marks[0].elapsed_time(marks[1])
 else:event_ms=None
 host_ms=1000*(time.perf_counter()-started);owner.watch(out,'timing.output');increment(runtime,'timing_whole_basis');del out
 return dict(optimizer_event_ms=event_ms,host_ms=host_ms)

def previews(matrix,state,group,banks,orders,owner,runtime,parameter):
 def own(t,label):owner.watch(t,label);return t
 def project(back,*args):
  increment(runtime,'full_m_projection_calls')
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
  row['finite']=all(math.isfinite(v) for v in row.values() if isinstance(v,float));rows[name]=row;increment(runtime,'full_m_previews')
  runtime.emit('jacobi2_adaptive_before_gate',dict(parameter=parameter,source_step=int(state['step']),policy=name,**row))
  del projected,reconstructed,adaptive
 return rows

def factor(matrix,state,group,runtime,parameter,axis,owner):
 cov=state['GG'][axis];q=state['Q'][axis];increment(runtime,'before_offdiag');before=offdiag_norm(q,cov,owner)
 identity=measured_metrics(runtime,q,cov,q,before,owner);bank={}
 qr,order=basis_step('qr20',cov,q,owner);increment(runtime,'diagnostic_qr');bank['qr20']=qr
 policies={'qr20':dict(metrics=measured_metrics(runtime,qr,cov,q,before,owner,order))};policies['qr20']['admitted']=admitted(policies['qr20']['metrics'],GATES)
 for name in POLICIES:
  actual=runtime.cubic3_kernel.evaluate(name,cov,q);increment(runtime,'diagnostic_compiled');owner.watch(actual,'CF.compiled')
  ref=runtime.cubic3_kernel.evaluate(name,cov,q,eager=True);owner.watch(ref,'CF.eager');increment(runtime,'diagnostic_eager')
  increment(runtime,'own_comparisons');comparison=reference_metrics(actual,ref,q,GATES);mm=measured_metrics(runtime,actual,cov,q,before,owner)
  row=dict(metrics=mm,reference=comparison,admitted=comparison['passed'] and admitted(mm,GATES))
  if name in ('jacobi2','jacobi2_cubic3'):
   increment(runtime,'rotated_covariance_and_generator_calls');b,_=rotated_covariance(cov,q,owner);raw,bound,a,cap,scale=generator(b,'jacobi')
   if name=='jacobi2':
    y,first,overlay2=candidate_from_generator(q,a,2.,owner);increment(runtime,'scale2_cubic2_overlays');overlay=overlay2
   else:
    residual=overlay2.T@overlay2;residual.diagonal().sub_(1.);correction=(overlay2.bfloat16()@residual.bfloat16()).float()
    overlay=overlay2-.5*correction;owner.watch(residual,'overlay.third_Gram');owner.watch(correction,'overlay.third_correction');owner.watch(overlay,'overlay.third_output');increment(runtime,'third_correction_overlays');del residual,correction
   err=relative(ref,overlay);row.update(scale2_overlay_relative=err,raw_bound=float(bound),applied_bound=float(2*a.norm()/math.sqrt(2)),cap_active=float(bound)>.5)
   row['admitted'] &= math.isfinite(err) and err<=GATES['scale2_overlay_relative']
   del b,raw,bound,a,cap,scale,overlay
  runtime.emit('cubic3_candidate_before_gate',dict(parameter=parameter,axis=axis,dimension=len(q),source_step=int(state['step']),policy=name,**row))
  policies[name]=row;bank[name]=actual;del actual,ref
 del y,first,overlay2
 timings=[]
 for repeat in range(runtime.protocol['cubic3']['timing_repeats']):
  for offset in range(4):
   name=NAMES[(repeat+offset)%4];runtime.budget(resources=True)
   timings.append(dict(policy=name,repeat=repeat,order=offset,**timed_basis(runtime,name,cov,q,owner)))
 mandatory=all(policies[n]['admitted'] for n in MANDATORY)
 refs=all(policies[n]['reference']['passed'] and math.isfinite(policies[n].get('scale2_overlay_relative',0.)) and policies[n].get('scale2_overlay_relative',0.)<=GATES['scale2_overlay_relative'] for n in POLICIES)
 return bank,order,dict(axis=axis,dimension=len(q),before_covariance_offdiag=before,identity=identity,policies=policies,timings=timings,passed=admitted(identity,GATES) and mandatory and refs)


def diagnose(matrix,state,group,runtime,parameter,owner,arm):
 banks={n:[] for n in NAMES};orders=[];factors=[]
 with torch.no_grad(),fp32_math_control():
  for axis in range(2):
   bank,order,row=factor(matrix,state,group,runtime,parameter,axis,owner)
   for n in NAMES:banks[n].append(bank[n])
   orders.append(order);factors.append(row);del bank,order
  view=previews(matrix,state,group,banks,orders,owner,runtime,parameter)
 row=dict(parameter=parameter,source_step=int(state['step']),source_arm=arm,shape=list(state['exp_avg'].shape),factors=factors,previews=view,
  full_actual_momentum=True,full_actual_diagonal_v=True,adaptive_preview_not_future_step=True,
  all_finite=all(p['finite'] for p in view.values()),physical_m_pass=all(view[n]['physical_m_admitted'] for n in ('qr_permutationV','qr_carryV','highest_residual20','jacobi2_cubic3')),passed=all(f['passed'] for f in factors) and all(p['finite'] for p in view.values()) and all(view[n]['physical_m_admitted'] and view[n]['finite'] for n in ('qr_permutationV','qr_carryV','highest_residual20','jacobi2_cubic3')))
 increment(runtime,'matrix_transactions');runtime.emit('cubic3_full_matrix_before_gate',row)
 return row

def configure_diagnostic(matrix,runtime):
 original=type(matrix)._research_refresh;matrix_ref=weakref.ref(matrix);runtime_ref=weakref.ref(runtime)
 def refresh(state,group):
  self=matrix_ref();r=runtime_ref()
  if self is None or r is None:raise RuntimeError('Missing cubic3 source owner')
  if r.kernel.probe:
   name,p=next((n,p) for n,p in zip(group['param_names'],group['params']) if self.state[p] is state)
   gradid=id(p.grad);before=streaming_digest(dict(state=state,parameter=p,gradient=p.grad));rng=r.kernel.rng_digest();backend=numerical_backend();compiler=compiler_stats();lim=limits();owner=TensorOwnership();error=None;row=None
   try:row=diagnose(self,state,group,r,name,owner,'qr20')
   except BaseException as exc:error=failure_record(exc);exc.__traceback__=None
   finally:
    gc.collect();release=owner.report();owner.close();after=streaming_digest(dict(state=state,parameter=p,gradient=p.grad))
    cleanup=dict(parameter=name,source_step=int(state['step']),before_sha256=before,after_sha256=after,nonmutation_pass=before==after and rng==r.kernel.rng_digest() and backend==numerical_backend() and gradid==id(p.grad),gradient_identity_pass=gradid==id(p.grad),registered_release=release,registered_release_pass=release['all_registered_released'],compiler_unchanged=compiler==compiler_stats(),compiler_limits=limits())
    r.emit('cubic3_moment_cleanup',cleanup)
   if error is not None:raise RuntimeError('Cubic3read-only diagnostic failed: '+str(error))
   row.update(cleanup);r.cubic3_rows.append(row)
   if not (cleanup['nonmutation_pass'] and cleanup['registered_release_pass'] and cleanup['compiler_unchanged'] and lim==limits()==dict(recompile_limit=8,accumulated_recompile_limit=256)):raise RuntimeError('Cubic3 read-only safety/compiler/release gate failed')
   r.budget(resources=True)
  return original(self,state,group)
 matrix._research_refresh=refresh

def build_arm(model,config,arm,router):
 if arm not in ARMS:raise ValueError('Undeclared original empty preflight arm')
 if arm!='qr20':
  from llm_jacobi2_learning_diagnostic import build_arm as prior
  return prior(model,config,arm,router)
 opt=baseline(model,config,'qr20');r=router.cubic3_runtime_ref();configure_diagnostic(opt.matrix,r);return opt

def warmup(runtime,opt):
 started=time.monotonic();spec=runtime.protocol['cubic3'];before=endpoint(runtime,opt);rng=runtime.streams.capture();grad=streaming_digest([p.grad for p in runtime.model.parameters()]);ids=[id(p.grad) for p in runtime.model.parameters()]
 modules=list(runtime.model.modules());flags=[m.training for m in modules];backend=numerical_backend();lim=limits();owner=TensorOwnership();rows=[];error=None
 try:
  if lim!={k:spec['compile'][k] for k in ('recompile_limit','accumulated_recompile_limit')} or PURPOSE>spec['memory']['maximum_tensor_purpose_bytes']:raise RuntimeError('Declared warmup limit or purpose exceeded')
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
      actual=runtime.cubic3_kernel.evaluate(policy,cov,q);owner.watch(actual,'warmup.compiled');increment(runtime,'warmup_compiled')
      eager=runtime.cubic3_kernel.evaluate(policy,cov,q,eager=True);owner.watch(eager,'warmup.eager');increment(runtime,'warmup_eager')
      increment(runtime,'warmup_comparisons');increment(runtime,'warmup_geometry');rr=reference_metrics(actual,eager,q,GATES)
      row=dict(**factor_spec,policy=policy,layout=layout,input_stride=list(q.stride()),output_stride=list(actual.stride()),reference=rr,geometry_observational_only=geometry(actual,owner));runtime.emit('cubic3_warmup_before_gate',row);rows.append(row)
      if not rr['passed']:raise FloatingPointError('Live Jacobi2 warmup own-eager fidelity failed')
      if tuple(actual.stride()) not in ((1,len(q)),(len(q),1)):raise ValueError('Unexpected live warm output layout')
      q=actual.contiguous();owner.watch(q,'warmup.row_major');del eager
     del actual,q
 except BaseException as exc:error=failure_record(exc);exc.__traceback__=None
 finally:
  runtime.streams.restore(rng)
  for m,f in zip(modules,flags):m.training=f
  gc.collect();release=owner.report();owner.close();after=endpoint(runtime,opt)
  cleanup=dict(before=before,after=after,nonmutation_pass=before==after and grad==streaming_digest([p.grad for p in runtime.model.parameters()]) and ids==[id(p.grad) for p in runtime.model.parameters()] and flags==[m.training for m in modules] and backend==numerical_backend() and lim==limits(),registered_release=release,compiler_limits=limits(),counts=dict(runtime.cubic3_counts),seconds=time.monotonic()-started)
  runtime.emit('cubic3_warmup_cleanup',cleanup)
 if not cleanup['nonmutation_pass'] or not release['all_registered_released']:raise RuntimeError('Live warmup restoration or registered release failed')
 if error is not None:raise RuntimeError('Live warmup failed: '+str(error))
 if runtime.cubic3_counts['warmup_compiled']!=48 or runtime.cubic3_counts['warmup_eager']!=48:raise RuntimeError('Declared cubic3 warmup count differs')
 return dict(rows=rows,cleanup=cleanup,passed=True)
