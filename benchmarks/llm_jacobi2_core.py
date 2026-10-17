"""Read-only streamed compiled basis/cost/full-M/inherited-V transactions."""
import gc,math,time
import torch
from llm_jacobi2_contract import NAMES,COUNTS,PURPOSE,OBSERVATIONS
from llm_sequence_averaging_core import fp32_math_control
from llm_step_scale_atoms import rotated_covariance,candidate_from_generator,metrics,relative,admitted
from llm_basis_response_atoms import basis_step,offdiag_norm,geometry
from tracking_curvature_kernel import generator
from hybrid_training_kernel import precision
from llm_math_confirmation import endpoint,compiler_stats
from matched_training_runtime import streaming_digest,failure_record
from tensor_ownership import TensorOwnership
from execution_fingerprints import backend_record

def observe(runtime,name):
 if not hasattr(runtime,'qualification_observations'):runtime.qualification_observations={}
 runtime.qualification_observations[name]=runtime.qualification_observations.get(name,0)+1

def measured_metrics(runtime,*args,**kwargs):
 observe(runtime,'factor_metrics');return metrics(*args,**kwargs)

def numerical_backend():return {k:v for k,v in backend_record().items() if k not in ('environment','loaded_libraries')}
def limits():return dict(recompile_limit=torch._dynamo.config.recompile_limit,accumulated_recompile_limit=torch._dynamo.config.accumulated_recompile_limit)

def count(runtime,name):runtime.qualification_counts[name]=runtime.qualification_counts.get(name,0)+1

def reference_metrics(actual,reference,q,gates):
 delta=actual-q;target=reference-q
 dn=float(delta.norm());tn=float(target.norm())
 row=dict(basis_relative=relative(actual,reference),movement_relative=float((delta-target).norm())/max(tn,1e-24),
  movement_cosine=float((delta*target).sum())/max(dn*tn,1e-24) if dn>0 and tn>0 else (1. if dn==tn==0 else 0.))
 row['passed']=all(math.isfinite(v) for v in row.values()) and row['basis_relative']<=gates['reference_basis_relative'] and row['movement_relative']<=gates['movement_relative'] and row['movement_cosine']>=gates['movement_cosine']
 return row

def named_states(runtime,opt):
 named={n:(p,g) for g in opt.matrix.param_groups for n,p in zip(g['param_names'],g['params'])}
 specs=runtime.protocol['jacobi2']['factors']
 for name in dict.fromkeys(s['parameter'] for s in specs):
  p,g=named[name];state=opt.matrix.state[p];selected=[s for s in specs if s['parameter']==name]
  if [s['axis'] for s in selected]!=[0,1] or g.get('merge_dims',False):raise ValueError('Two-sided unmerged coverage required')
  for s in selected:
   q=state['Q'][s['axis']];cov=state['GG'][s['axis']]
   if cov.dtype!=torch.float32 or q.dtype!=torch.float32 or cov.shape!=q.shape or q.shape!=(s['dimension'],s['dimension']):raise ValueError('Actual factor differs')
  yield name,state,g,selected

def warm_matrix(runtime,name,state,group,specs,owner):
 rows=[];gates=runtime.protocol['jacobi2']['gates']
 for spec in specs:
  axis=spec['axis'];cov=state['GG'][axis];old=state['Q'][axis]
  for policy in NAMES[1:]:
   q=old
   for layout in ('qr_output','warm_output'):
    runtime.budget(resources=True)
    actual=runtime.qualification_kernels.evaluate(policy,cov,q);owner.watch(actual,'warmup.compiled_output');count(runtime,'warmup_compiled')
    ref=runtime.qualification_kernels.evaluate(policy,cov,q,eager=True);owner.watch(ref,'warmup.eager_output');count(runtime,'warmup_eager_references')
    observe(runtime,'own_eager_comparisons');comparison=reference_metrics(actual,ref,q,gates)
    observe(runtime,'warmup_geometry')
    row=dict(**spec,policy=policy,layout=layout,covariance_stride=list(cov.stride()),basis_stride=list(q.stride()),output_stride=list(actual.stride()),reference=comparison,geometry_observational_only=geometry(actual,owner))
    runtime.emit('prototype_warmup_before_gate',row);rows.append(row)
    if not comparison['passed']:raise FloatingPointError('Warmup own-eager fidelity failed')
    if tuple(actual.stride()) not in ((1,len(q)),(len(q),1)):raise ValueError('Unexpected warm output layout')
    q=actual.contiguous();owner.watch(q,'warmup.canonical_row_major');del ref
   del actual,q
 return rows

def timed_basis(runtime,policy,cov,q,owner):
 # No metrics/transport/watch registrations inside event interval.
 if q.is_cuda:
  marks=runtime.optimizer_events();runtime.synchronize();marks[0].record();start=time.perf_counter()
 else:marks=None;start=time.perf_counter()
 if policy=='qr20':
  with precision('high'):
   order=torch.argsort((q.T@cov@q).diagonal(),descending=True)
   output,_triangular=torch.linalg.qr(cov@q[:,order])
 else:output=runtime.qualification_kernels.evaluate(policy,cov,q)
 if marks is not None:
  marks[1].record();runtime.synchronize();event_ms=marks[0].elapsed_time(marks[1])
 else:event_ms=None
 host_ms=1000*(time.perf_counter()-start)
 owner.watch(output,'timing.output');count(runtime,'timing_complete_basis')
 del output
 return dict(optimizer_event_ms=event_ms,host_ms=host_ms)

def factor_bank(runtime,spec,state,owner):
 gates=runtime.protocol['jacobi2']['gates'];axis=spec['axis'];cov=state['GG'][axis];q=state['Q'][axis]
 observe(runtime,'before_offdiag');before=offdiag_norm(q,cov,owner);identity=measured_metrics(runtime,q,cov,q,before,owner)
 qr,order=basis_step('qr20',cov,q,owner);count(runtime,'qualification_qr')
 bank={'qr20':qr};results={'qr20':dict(metrics=measured_metrics(runtime,qr,cov,q,before,owner,order))}
 results['qr20']['admitted']=admitted(results['qr20']['metrics'],gates)
 for policy in NAMES[1:]:
  runtime.budget(resources=True)
  actual=runtime.qualification_kernels.evaluate(policy,cov,q);owner.watch(actual,'qualification.compiled');count(runtime,'qualification_compiled')
  ref=runtime.qualification_kernels.evaluate(policy,cov,q,eager=True);owner.watch(ref,'qualification.eager');count(runtime,'qualification_eager_references')
  observe(runtime,'own_eager_comparisons');comparison=reference_metrics(actual,ref,q,gates);m=measured_metrics(runtime,actual,cov,q,before,owner)
  row=dict(reference=comparison,metrics=m,admitted=comparison['passed'] and admitted(m,gates))
  if policy=='jacobi2':
   b,_=rotated_covariance(cov,q,owner);raw,bound,a,cap,scale=generator(b,'jacobi')
   y,first,overlay=candidate_from_generator(q,a,2.,owner);count(runtime,'scale2_overlay_references')
   overlay_error=relative(ref,overlay);row.update(scale2_overlay_relative=overlay_error,raw_bound=float(bound),original_chosen_bound=float(a.norm()/math.sqrt(2)),applied_bound=float(2*a.norm()/math.sqrt(2)),cap_active=float(bound)>.5)
   row['admitted'] &= math.isfinite(overlay_error) and overlay_error<=gates['scale2_overlay_relative']
   del b,raw,bound,a,cap,scale,y,first,overlay
  runtime.emit('compiled_candidate_before_admission',dict(**spec,policy=policy,source_step=int(state['step']),**row))
  bank[policy]=actual;results[policy]=row;del ref,actual
 timings=[]
 for repeat in range(runtime.protocol['jacobi2']['timing_repeats']):
  for offset in range(4):
   policy=NAMES[(repeat+offset)%4];runtime.budget(resources=True)
   timings.append(dict(policy=policy,repeat=repeat,order=offset,**timed_basis(runtime,policy,cov,q,owner)))
 row=dict(**spec,source_step=int(state['step']),before_covariance_offdiag=before,identity=identity,policies=results,timings=timings)
 row['passed']=admitted(identity,gates) and all(r['admitted'] for r in results.values())
 runtime.emit('qualified_factor_before_gate',row)
 return bank,order,row

def previews(runtime,matrix,state,group,banks,orders,owner,parameter):
 def own(t,label):owner.watch(t,label);return t
 def project(back,*args):
  observe(runtime,'full_m_projection_calls')
  return (matrix._project_back_with_basis if back else matrix._project_with_basis)(*args)
 gates=runtime.protocol['jacobi2']['gates'];m=state['exp_avg'];v=state['exp_avg_sq']
 world=own(project(True,m,state['Q'],group),'moment.world')
 qv=v
 for axis,order in enumerate(orders):qv=own(qv.index_select(axis,order),'moment.qr_permuted_v')
 qm=own(project(False,world,banks['qr20'],group),'moment.qr_projected')
 qr_adaptive=own(project(True,qm/(qv.sqrt()+group['eps']),banks['qr20'],group),'moment.qr_adaptive')
 del qm
 out={};current=None
 for policy,basis,variance in [('qr_permutationV',banks['qr20'],qv),('qr_carryV',banks['qr20'],v)]+[(n,banks[n],v) for n in NAMES[1:]]:
  projected=own(project(False,world,basis,group),'moment.projected')
  reconstructed=own(project(True,projected,basis,group),'moment.reconstructed')
  adaptive=own(project(True,projected/(variance.sqrt()+group['eps']),basis,group),'moment.adaptive')
  pe=relative(reconstructed,world)
  row=dict(world_m_reconstruction_relative=pe,adaptive_relative_to_qr_permutation=relative(adaptive,qr_adaptive),
   adaptive_cosine_to_qr_permutation=float((adaptive*qr_adaptive).sum()/(adaptive.norm()*qr_adaptive.norm()).clamp_min(1e-24)),adaptive_norm=float(adaptive.norm()),
   physical_m_admitted=math.isfinite(pe) and pe<=gates['physical_m_relative'],variance_policy='QR permutation' if policy=='qr_permutationV' else 'unchanged carry')
  if policy=='highest_residual20':current=own(adaptive.clone(),'moment.jacobi1_adaptive')
  if current is not None:row['adaptive_relative_to_jacobi1']=relative(adaptive,current)
  row['finite']=all(math.isfinite(x) for x in row.values() if isinstance(x,float))
  observe(runtime,'full_m_previews');out[policy]=row;runtime.emit('physical_m_adaptive_before_gate',dict(parameter=parameter,policy=policy,source_step=int(state['step']),**row))
  del projected,reconstructed,adaptive
 return out

def qualify_matrix(runtime,opt,name,state,group,specs,owner):
 banks={n:[] for n in NAMES};orders=[];factors=[]
 for spec in specs:
  bank,order,row=factor_bank(runtime,spec,state,owner)
  for n in NAMES:banks[n].append(bank[n])
  orders.append(order);factors.append(row)
  del bank,order
 preview=previews(runtime,opt.matrix,state,group,banks,orders,owner,name)
 return dict(parameter=name,source_step=int(state['step']),shape=list(state['exp_avg'].shape),factors=factors,previews=preview,
  full_actual_momentum=True,full_actual_diagonal_v=True,adaptive_preview_not_future_step=True,
  passed=all(f['passed'] for f in factors) and all(p['physical_m_admitted'] and p['finite'] for p in preview.values()))

def transaction(runtime,opt,warming=False):
 started=time.monotonic();spec=runtime.protocol['jacobi2'];before=endpoint(runtime,opt)
 grad=streaming_digest([p.grad for p in runtime.model.parameters()]);ids=[id(p.grad) for p in runtime.model.parameters()]
 modules=list(runtime.model.modules());flags=[m.training for m in modules];rng=runtime.streams.capture();compiler=compiler_stats()
 backend=numerical_backend();compiler_limits=limits()
 if compiler_limits!={k:spec['compile'][k] for k in ('recompile_limit','accumulated_recompile_limit')}:raise ValueError('Default compiler limits changed')
 owner=TensorOwnership();sentinel=torch.empty(0);owner.watch(sentinel,'prototype.transaction_sentinel');del sentinel
 rows=[];error=None
 try:
  if PURPOSE>spec['memory']['maximum_tensor_purpose_bytes']:raise MemoryError('Declared streamed tensor purpose exceeded')
  runtime.model.eval()
  with torch.no_grad(),fp32_math_control():
   for name,state,group,selected in named_states(runtime,opt):
    runtime.budget(resources=True)
    row=warm_matrix(runtime,name,state,group,selected,owner) if warming else qualify_matrix(runtime,opt,name,state,group,selected,owner)
    rows.extend(row) if warming else rows.append(row)
    runtime.emit('prototype_matrix_completed',dict(parameter=name,warming=warming,passed=True if warming else row['passed']))
    del row
    gc.collect()
  runtime.emit('prototype_all_observations_before_gate',dict(warming=warming,rows=rows,counts=dict(runtime.qualification_counts),observation_counts=dict(runtime.qualification_observations)))
  expected={k:v for k,v in COUNTS.items() if warming or not k.startswith('warmup')}
  if warming:expected={k:v for k,v in COUNTS.items() if k.startswith('warmup')}
  observed={k:v for k,v in runtime.qualification_counts.items() if k in expected}
  if observed!=expected:raise RuntimeError('Declared observed complete basis counts differ')
  if not warming and not all(r['passed'] for r in rows):raise FloatingPointError('Compiled/fidelity/geometry/full physicalM admission failed; all observations retained')
 except BaseException as exc:
  error=failure_record(exc);exc.__traceback__=None
 finally:
  runtime.streams.restore(rng)
  for module,flag in zip(modules,flags):module.training=flag
  gc.collect();release=owner.report();owner.close();after=endpoint(runtime,opt)
  nonmutation=before==after and grad==streaming_digest([p.grad for p in runtime.model.parameters()]) and ids==[id(p.grad) for p in runtime.model.parameters()] and flags==[m.training for m in modules]
  nonmutation &= backend==numerical_backend() and compiler_limits==limits()
  stable_compiler=compiler==compiler_stats()
  cleanup=dict(before=before,after=after,state_rng_flags_gradients_and_identity_pass=nonmutation,registered_release=release,compiler_before=compiler,compiler_after=compiler_stats(),compiler_unchanged=stable_compiler,counts=dict(runtime.qualification_counts),observation_counts=dict(getattr(runtime,'qualification_observations',{})),compiler_limits=compiler_limits,numerical_flags_restored=backend==numerical_backend(),seconds=time.monotonic()-started)
  runtime.emit('prototype_transaction_cleanup',dict(warming=warming,**cleanup))
 if not release['all_registered_released'] or not nonmutation or (not warming and not stable_compiler):raise RuntimeError('Prototype transaction safety/registered release/compiler gate failed')
 if error is not None:raise RuntimeError('Prototype qualification failed: '+str(error))
 if not warming and runtime.qualification_observations!=OBSERVATIONS:raise RuntimeError('Declared scalar/projection observation counts differ')
 runtime.budget(resources=True)
 return dict(rows=rows,warming=warming,passed=True,cleanup=cleanup,tensor_purpose_bytes=PURPOSE,complete_basis_calls=96 if warming else 224,optimizer_updates=0,backward_calls=0)

def warmup(runtime,opt):return transaction(runtime,opt,True)
def diagnose(runtime,opt):return transaction(runtime,opt,False)
