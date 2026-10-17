"""Eager stage-neutral references and retained rejected scalar-sweep proposals."""
import math
import torch
from native_residual_named_kernel import highest_prefix,candidate_retraction
from tracking_curvature_kernel import prefix_damped,generator
from llm_basis_response_atoms import basis_step,aligned_delta,offdiag_norm,geometry,selected_states

def rotated_covariance(cov,q,owner):
    scale=(cov.trace().abs()/len(q)).clamp_min(1e-12)
    c=cov.float()/scale;owner.watch(c,'direction.normalized_covariance')
    b=q.float().T@c@q.float();b=.5*(b+b.T);owner.watch(b,'direction.rotated_covariance')
    return b,float(scale)

def staged_retraction(y,owner):
    first=None
    for iteration in range(2):
        residual=y.T@y;residual.diagonal().sub_(1.)
        correction=(y.bfloat16()@residual.bfloat16()).float()
        owner.watch(residual,'retraction.residual');owner.watch(correction,'retraction.correction')
        y=y-.5*correction;owner.watch(y,'retraction.output')
        if iteration==0:first=y
    return y,first

def candidate_from_generator(q,a,multiplier,owner):
    applied=a*multiplier;owner.watch(applied,'direction.scaled_generator')
    y=q.float()+(q.bfloat16()@applied.bfloat16()).float();owner.watch(y,'sweep.candidate')
    final,first=staged_retraction(y,owner)
    return y,first,final

def relative(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-24))

def metrics(candidate,cov,q,before,owner,order=None):
    if not bool(torch.isfinite(candidate).all()):
        return dict(finite=False,geometry=None,covariance_offdiag=None,tracking_ratio=None,movement_norm=None,movement_relative_to_basis=None)
    g=geometry(candidate,owner);after=offdiag_norm(candidate,cov,owner);delta=aligned_delta(candidate,q,order,owner)
    movement=float(delta.norm());ratio=after/max(before,1e-24)
    finite=all(math.isfinite(v) for v in (after,movement,ratio,g['absolute_gram_fro'],g['normalized_gram']))
    if not finite:return dict(finite=False,geometry=None,covariance_offdiag=None,tracking_ratio=None,movement_norm=None,movement_relative_to_basis=None)
    return dict(finite=True,geometry=g,covariance_offdiag=after,tracking_ratio=ratio,movement_norm=movement,movement_relative_to_basis=movement/math.sqrt(len(q)))

def admitted(row,gates):
    return row['finite'] and row['geometry']['normalized_gram']<=gates['normalized_gram'] and row['geometry']['absolute_gram_fro']<=gates['absolute_gram_fro']

def direction_sweep(runtime,policy,cov,q,b,before,owner):
    gates=runtime.protocol['step_scale']['gates'];name='jacobi' if policy=='highest_residual20' else 'damped'
    raw,bound,a,cap_scale,chosen_scale=generator(b,name);owner.watch(raw,'direction.raw');owner.watch(a,'direction.chosen')
    info=dict(policy=policy,raw_bound=float(bound),raw_norm=float(raw.norm()),cap_scale=float(cap_scale),chosen_scale=float(chosen_scale),
              damping_alpha=float(chosen_scale/cap_scale.clamp_min(1e-24)),applied_bound=float(a.norm()/math.sqrt(2)),
              cap_active=float(bound)>.5,model_limited=bool(chosen_scale<cap_scale))
    pure_y=highest_prefix(cov,q) if name=='jacobi' else prefix_damped(cov,q);owner.watch(pure_y,'reference.pure_candidate')
    pure_final=candidate_retraction(pure_y);owner.watch(pure_final,'reference.pure_final')
    runtime.step_counts['pure_eager_references']+=1
    reference_metrics=metrics(pure_final,cov,q,before,owner)
    y,first,final=candidate_from_generator(q,a,1.,owner)
    runtime.step_counts['sweep_candidates']+=1
    delta=final-q;target=pure_final-q;owner.watch(delta,'reference.actual_movement');owner.watch(target,'reference.expected_movement')
    dn=float(delta.norm());tn=float(target.norm())
    movement_relative=float((delta-target).norm())/max(tn,1e-24)
    cosine=float((delta*target).sum())/max(dn*tn,1e-24) if dn>0 and tn>0 else (1. if dn==tn==0 else 0.)
    reference=dict(prefix_relative=relative(y,pure_y),basis_relative=relative(final,pure_final),movement_relative=movement_relative,movement_cosine=cosine,
                   pure_metrics=reference_metrics)
    reference['passed']=all(math.isfinite(reference[k]) for k in ('prefix_relative','basis_relative','movement_relative','movement_cosine')) and reference['prefix_relative']<=gates['prefix_relative'] and reference['basis_relative']<=gates['reference_basis_relative'] and movement_relative<=gates['movement_relative'] and cosine>=gates['movement_cosine'] and admitted(reference_metrics,gates)
    runtime.emit('step_scale_reference_before_gate',dict(current_factor=runtime.step_factor,**info,**reference))
    if not reference['passed']:raise FloatingPointError('Step-scale eager stage/reference validity gate failed')
    del pure_y,pure_final,delta,target
    rows=[]
    for multiplier in runtime.protocol['step_scale']['execution_order']:
        runtime.budget(resources=True)
        if multiplier!=1.:
            y,first,final=candidate_from_generator(q,a,multiplier,owner)
            runtime.step_counts['sweep_candidates']+=1
        row=dict(multiplier=multiplier,applied_bound=info['applied_bound']*multiplier,
            candidate=metrics(y,cov,q,before,owner),first=metrics(first,cov,q,before,owner),final=metrics(final,cov,q,before,owner))
        row['admitted']=admitted(row['final'],gates)
        row['rejected_reason']=None if row['admitted'] else 'Nonfinite or unchanged normalized/absolute Gram admission failure'
        runtime.emit('step_scale_candidate_before_gate',dict(current_factor=runtime.step_factor,policy=policy,**row))
        if multiplier==1. and not row['admitted']:raise FloatingPointError('Unmodified1x warm geometry gate failed')
        rows.append(row);del y,first,final
    return dict(**info,reference=reference,candidates=rows)

def factor_sweeps(runtime,opt,owner):
    rows=[]
    # Reuse the unchanged state selector; only the new all-factor list differs.
    selector=runtime.protocol['step_scale']
    proxy=type('FactorSelector',(),{})();proxy.protocol={'basis_response':selector}
    for spec,state,cov,q,beta in selected_states(proxy,opt):
        runtime.step_factor=dict(spec);runtime.budget(resources=True)
        before=offdiag_norm(q,cov,owner);input_geometry=geometry(q,owner)
        identity=metrics(q,cov,q,before,owner)
        if not admitted(identity,selector['gates']):raise FloatingPointError('Inherited basis geometry gate failed')
        qr,order=basis_step('qr20',cov,q,owner);qr_metrics=metrics(qr,cov,q,before,owner,order)
        runtime.step_counts['qr_controls']+=1
        runtime.emit('step_scale_qr_before_gate',dict(**spec,metrics=qr_metrics))
        if not admitted(qr_metrics,selector['gates']):raise FloatingPointError('QR control geometry gate failed')
        del qr,order
        b,scale=rotated_covariance(cov,q,owner)
        directions=[direction_sweep(runtime,name,cov,q,b,before,owner) for name in selector['policies']]
        rows.append(dict(**spec,source_step=state['step'],covariance_norm=float(cov.norm()),covariance_scale=scale,
            before_covariance_offdiag=before,input_geometry=input_geometry,identity=identity,qr=qr_metrics,directions=directions))
        del b
    return rows
