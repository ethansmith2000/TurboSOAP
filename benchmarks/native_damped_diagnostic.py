"""Read-only common actual C,Q,M,V step-bound and same-basis V controls."""
import gc,math,weakref
import torch
from hybrid_training_kernel import precision,configure_hybrid
from native_residual_named_kernel import relative
from native_damped_kernel import PolicyRouter
from prefix_sensitivity_kernel import prefix_stages
from tracking_curvature_kernel import reference_from_generator,geometry,generator,directional_metrics,curvature_metrics
from native_damped_contract import ARMS,POLICIES
from matched_training_runtime import streaming_digest

def diagnose(matrix,state,group,router,parameter,owners):
    def own(x):owners.append(weakref.ref(x));return x
    old=state['Q'];covs=state['GG'];m=state['exp_avg'];v=state['exp_avg_sq']
    bases={name:[] for name in POLICIES};admitted={name:True for name in POLICIES};qr=[];orders=[];rows=[]
    with precision('highest'):
        for axis,(cov,q) in enumerate(zip(covs,old)):
            stage=prefix_stages(cov,q,False,True)
            pure=router.kernels[ARMS[0]].eager_prefix(cov,q)
            observer_error=relative(stage['candidate'],pure)
            router.emit('tracking_observer_before_gate',dict(parameter=parameter,axis=axis,observer_relative=observer_error))
            if observer_error>1e-6:raise FloatingPointError('Tracking eager observer invalid')
            raw,bound=stage['raw'],stage['bound'];baseline_a=stage['a']
            input_gram=geometry(q);cov_norm=float(cov.norm())
            before=q.T@cov@q;before.diagonal().zero_();before_norm=float(before.norm())
            diag=stage['b'].diagonal();tau=float(diag.abs().amax().clamp_min(1e-12)*.01)
            indices=torch.linspace(0,len(q)-1,64,device=q.device).long();sample=diag[indices]
            gaps=(sample[:,None]-sample[None,:]).abs().flatten()
            with precision('high'):
                order=torch.argsort((q.T@cov@q).diagonal(),descending=True)
                qref,_=torch.linalg.qr(cov@q[:,order])
            qr.append(own(qref));orders.append(order)
            qgeom=geometry(qref)
            if not qgeom['admitted']:raise FloatingPointError('QR reference geometry failed')
            qtrack=qref.T@cov@qref;qtrack.diagonal().zero_()
            results=dict(qr=dict(geometry=qgeom,covariance_offdiag=float(qtrack.norm()),
                normalized_covariance_offdiag=float(qtrack.norm())/max(cov_norm,1e-24),
                tracking_ratio=float(qtrack.norm())/max(before_norm,1e-24)))
            for name in POLICIES:
                cap=.5
                direction='jacobi' if name=='highest_residual20' else 'damped'
                direction_raw,direction_bound,a,cap_scale,chosen_scale=generator(stage['b'],direction)
                candidate=q.float()+(q.bfloat16()@a.bfloat16()).float()
                pure_direction=router.kernels[name].eager_prefix(cov,q)
                direction_observer_error=relative(candidate,pure_direction)
                router.emit('direction_eager_observer_before_gate',dict(parameter=parameter,axis=axis,policy=name,relative=direction_observer_error))
                if direction_observer_error>1e-6:raise FloatingPointError('Direction eager observer invalid')
                if name=='highest_residual20' and not torch.equal(direction_raw,raw):raise FloatingPointError('Jacobi reference generator changed')
                actual=own(router.evaluate_cf(name,cov,q))
                reference=reference_from_generator(q,a)
                mm=geometry(actual);fidelity=relative(actual,reference) if mm['finite'] else None
                if fidelity is not None and not math.isfinite(fidelity):fidelity=None
                valid=mm['admitted'] and fidelity is not None and fidelity<=1e-4
                admitted[name]&=valid;bases[name].append(actual)
                result=dict(cap=cap,geometry=mm,own_eager_basis_relative=fidelity,admitted=valid,
                    raw_bound=float(direction_bound),raw_norm=float(direction_raw.norm()),applied_bound=float(a.norm()/math.sqrt(2)),cap_active=float(direction_bound)>cap,
                    eager_observer_relative=direction_observer_error,**directional_metrics(stage['b'],a),
                    **curvature_metrics(stage['b'],direction_raw,a,cap_scale,chosen_scale),
                    skew_fro_operator_upper=float(a.norm()/math.sqrt(2))*(1+1e-6),
                    rejected_reason=None if valid else 'geometry or own-eager fidelity admission failed')
                if mm['finite']:
                    tracking=actual.T@cov@actual;tracking.diagonal().zero_()
                    result.update(covariance_offdiag=float(tracking.norm()),normalized_covariance_offdiag=float(tracking.norm())/max(cov_norm,1e-24),
                        tracking_ratio=float(tracking.norm())/max(before_norm,1e-24));del tracking
                router.emit('direction_candidate_before_admission',dict(parameter=parameter,axis=axis,dimension=len(q),source_step=int(state['step']),policy=name,**result))
                results[name]=result;del reference
                del direction_raw,direction_bound,a,candidate,pure_direction,cap_scale,chosen_scale
            rows.append(dict(axis=axis,dimension=len(q),input_geometry=input_gram,covariance_norm=cov_norm,
                before_covariance_offdiag=before_norm,before_normalized_covariance_offdiag=before_norm/max(cov_norm,1e-24),
                normalized_covariance_scale=float(stage['scale']),raw_norm=float(raw.norm()),raw_bound=float(bound),
                damping_tau=tau,sampled_diagonal_gap_median=float(gaps.median()),sampled_diagonal_gap_max=float(gaps.max()),
                stage_attribution='validated eager-only',observer_relative=observer_error,
                policies=results))
            del stage,pure,raw,bound,baseline_a,actual,qref,qtrack,before,diag,sample,gaps
        world=own(matrix._project_back_with_basis(m,old,group))
        qv=v
        for axis,order in enumerate(orders):qv=qv.index_select(axis,order)
        qm=matrix._project_with_basis(world,qr,group)
        qr_adaptive=own(matrix._project_back_with_basis(qm/(qv.sqrt()+group['eps']),qr,group))
        previews={};adaptive_current=None
        for name,basis,variance in [('qr_permutationV',qr,qv),('qr_carryV',qr,v)]+[(name,bases[name],v) for name in POLICIES if admitted[name]]:
            projected=matrix._project_with_basis(world,basis,group)
            reconstructed=matrix._project_back_with_basis(projected,basis,group)
            adaptive=own(matrix._project_back_with_basis(projected/(variance.sqrt()+group['eps']),basis,group))
            physical_error=relative(reconstructed,world)
            previews[name]=dict(world_m_reconstruction_relative=physical_error,
                adaptive_relative_to_qr_permutation=relative(adaptive,qr_adaptive),
                adaptive_cosine_to_qr_permutation=float((adaptive*qr_adaptive).sum()/(adaptive.norm()*qr_adaptive.norm()).clamp_min(1e-24)),
                adaptive_norm=float(adaptive.norm()),physical_m_admitted=physical_error<=.01)
            if name=='highest_residual20':adaptive_current=own(adaptive.clone())
            if name in POLICIES and adaptive_current is not None:previews[name]['adaptive_relative_to_jacobi']=relative(adaptive,adaptive_current)
            router.emit('tracking_adaptive_before_gate',dict(parameter=parameter,source_step=int(state['step']),policy=name,**previews[name]))
            del projected,reconstructed,adaptive
    if not all(admitted.values()):raise FloatingPointError('Declared warm counterfactual admission failed')
    if not all(p['physical_m_admitted'] for p in previews.values()):raise FloatingPointError('Declared full physicalM admission failed')
    return dict(parameter=parameter,source_step=int(state['step']),source_arm='set_by_source_hook',shape=list(m.shape),factors=rows,
        admitted_warm_policies=admitted,previews=previews,full_actual_momentum=True,full_actual_diagonal_v=True,
        adaptive_preview_not_future_step=True,
        all_finite=all(math.isfinite(x) for value in previews.values() for x in value.values() if isinstance(x,float)),
        physical_m_pass=all(p['physical_m_admitted'] for p in previews.values()))

def configure_diagnostic(matrix,router,arm):
    original=type(matrix)._research_refresh;matrix_ref=weakref.ref(matrix)
    def refresh(state,group):
        self=matrix_ref()
        if self is None:raise RuntimeError('Missing live optimizer owner')
        if router.probe:
            name,p=next((name,p) for name,p in zip(group['param_names'],group['params']) if self.state[p] is state)
            before=streaming_digest(dict(state=state,parameter=p,gradient=p.grad));rng=router.rng_digest();owners=[]
            allocated=torch.cuda.memory_allocated()
            row=diagnose(self,state,group,router,name,owners);row['source_arm']=arm
            gc.collect();alive=sum(ref() is not None for ref in owners)
            after=streaming_digest(dict(state=state,parameter=p,gradient=p.grad));rng_after=router.rng_digest()
            row.update(before_sha256=before,after_sha256=after,rng_before=rng,rng_after=rng_after,
                nonmutation_pass=before==after and rng==rng_after,registered_owners=len(owners),alive_after_release=alive,
                registered_release_pass=alive==0,conservative_peak_extra_bytes=max(0,torch.cuda.max_memory_allocated()-allocated))
            router.moment_probes.append(row);router.emit('full_moment_probe_before_gate',row)
            if not(row['nonmutation_pass'] and row['registered_release_pass'] and row['all_finite'] and row['physical_m_pass'] and row['conservative_peak_extra_bytes']<=16*2**30):
                raise FloatingPointError('Tracking transaction safety/release gate failed')
        return original(self,state,group)
    matrix._research_refresh=refresh

def build_arm(model,config,arm,router):
    from cadence_training_contract import build_arm as baseline
    if arm not in ARMS:raise ValueError('Only declared fixed live source arm')
    opt=baseline(model,config,'qr20' if arm=='qr20' else 'warm20')
    if arm in POLICIES:configure_hybrid(opt.matrix,router.kernels[arm])
    configure_diagnostic(opt.matrix,router,arm)
    return opt
