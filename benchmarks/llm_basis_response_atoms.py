"""Read-only full-factor responses using existing eager native policy arithmetic."""
import math
import torch
from hybrid_training_kernel import precision
from native_residual_named_kernel import highest_prefix,candidate_retraction
from tracking_curvature_kernel import prefix_damped
from llm_sequence_averaging_core import indices,scalar_array

def basis_step(policy,cov,q,owner):
    if policy=='qr20':
        with precision('high'):
            score=q.T@cov@q;owner.watch(score,'qr.sort_scores')
            order=torch.argsort(score.diagonal(),descending=True)
            owner.watch(order,'qr.order')
            sorted_q=q.index_select(1,order);owner.watch(sorted_q,'qr.sorted_basis')
            product=cov@sorted_q;owner.watch(product,'qr.power_product')
            candidate,triangular=torch.linalg.qr(product)
            owner.watch(candidate,'qr.candidate');owner.watch(triangular,'qr.triangular')
        return candidate,order
    if policy not in ('highest_residual20','damped20'):raise ValueError('Unknown basis response policy')
    with precision('highest'):
        y=highest_prefix(cov,q) if policy=='highest_residual20' else prefix_damped(cov,q)
        owner.watch(y,'warm.candidate');candidate=candidate_retraction(y);owner.watch(candidate,'warm.retracted')
    return candidate,None

def aligned_delta(candidate,q,order,owner):
    aligned=candidate.clone() if order is None else candidate.index_select(1,torch.argsort(order))
    owner.watch(aligned,'movement.unpermuted_basis')
    overlap=(aligned*q).sum(dim=0)
    signs=torch.where(overlap<0,-torch.ones_like(overlap),torch.ones_like(overlap))
    owner.watch(overlap,'movement.column_overlap');owner.watch(signs,'movement.column_sign')
    aligned.mul_(signs)
    delta=aligned-q;owner.watch(delta,'movement.aligned_delta')
    return delta

def offdiag_norm(q,cov,owner):
    rotated=q.T@cov@q;owner.watch(rotated,'tracking.rotated_covariance')
    rotated.diagonal().zero_()
    return float(rotated.norm())

def geometry(q,owner):
    gram=q.T@q;owner.watch(gram,'geometry.gram')
    gram.diagonal().sub_(1.)
    absolute=float(gram.norm())
    return dict(absolute_gram_fro=absolute,normalized_gram=absolute/math.sqrt(len(q)))

def covariance_update(cov,gram,beta,owner):
    prospective=cov.clone();owner.watch(prospective,'prospective.covariance')
    prospective.lerp_(gram,1-beta)
    return prospective

def full_clip_gram_relative(raw,clipped,coefficient,owner):
    a=raw.reshape(-1);b=clipped.reshape(-1);error=energy=0.
    for start in range(0,a.numel(),2**18):
        x=a[start:start+2**18].double()*coefficient**2;y=b[start:start+2**18].double();delta=x-y
        owner.watch(x,'clip_gram.expected_chunk');owner.watch(y,'clip_gram.actual_chunk');owner.watch(delta,'clip_gram.delta_chunk')
        error+=float(delta.square().sum());energy+=float(y.square().sum())
        del x,y,delta
    return math.sqrt(error/max(energy,1e-30))

def measure_policy(runtime,policy,cov,q,before,stats,key,owner,pool=None):
    runtime.budget(resources=True)
    candidate,order=basis_step(policy,cov,q,owner)
    g=geometry(candidate,owner);after=offdiag_norm(candidate,cov,owner)
    delta=aligned_delta(candidate,q,order,owner)
    movement=float(delta.norm());ri=indices(len(q),q.device)
    tile=delta.index_select(0,ri).index_select(1,ri);owner.watch(tile,'movement.physical_sample')
    sample=tile.cpu().double().numpy().copy()
    if stats is not None:stats.add((*key,policy),sample)
    row=dict(policy=policy,geometry=g,covariance_offdiag=after,tracking_ratio=after/max(before,1e-24),
        normalized_covariance_offdiag=after/max(float(cov.norm()),1e-24),movement_norm=movement,
        movement_relative_to_basis=movement/math.sqrt(len(q)),aligned_movement_sample=scalar_array(sample))
    gates=runtime.protocol['basis_response']['gates']
    row['passed']=all(math.isfinite(x) for x in (after,movement,g['absolute_gram_fro'],g['normalized_gram'],row['tracking_ratio'])) and g['absolute_gram_fro']<=gates['absolute_gram_fro'] and g['normalized_gram']<=gates['normalized_gram']
    runtime.emit('basis_response_policy_before_gate',dict(pool=pool,key=list(key),**row))
    if not row['passed']:raise FloatingPointError('Basis-response finite/geometry gate failed')
    return row

def selected_states(runtime,opt):
    named={name:(p,group) for group in opt.matrix.param_groups for name,p in zip(group['param_names'],group['params'])}
    for spec in runtime.protocol['basis_response']['factors']:
        parameter=spec['parameter'];axis=spec['axis'];p,group=named[parameter];state=opt.matrix.state[p]
        cov=state['GG'][axis];q=state['Q'][axis]
        if cov.dtype!=torch.float32 or q.dtype!=torch.float32 or cov.shape!=q.shape or len(q)!=spec['dimension'] or group.get('merge_dims',False):raise ValueError('Frozen factor/state shape or dtype differs')
        beta=group['shampoo_beta'] if group['shampoo_beta']>=0 else group['betas'][1]
        yield spec,state,cov,q,beta

def inherited_responses(runtime,opt,owner):
    rows=[]
    for spec,state,cov,q,beta in selected_states(runtime,opt):
        before=offdiag_norm(q,cov,owner);g=geometry(q,owner)
        if g['normalized_gram']>runtime.protocol['basis_response']['gates']['normalized_gram'] or g['absolute_gram_fro']>runtime.protocol['basis_response']['gates']['absolute_gram_fro']:raise FloatingPointError('Inherited basis geometry gate failed')
        policies=[measure_policy(runtime,name,cov,q,before,None,(0,spec['parameter'],'inherited',spec['axis']),owner) for name in runtime.protocol['basis_response']['policies']]
        rows.append(dict(**spec,source_step=state['step'],before_covariance_offdiag=before,covariance_norm=float(cov.norm()),input_geometry=g,policies=policies))
    return rows

def prospective_responses(runtime,opt,sums,n,coefficient,stats,pool,owner):
    rows=[]
    for spec,state,cov,q,beta in selected_states(runtime,opt):
        name=spec['parameter'];axis=spec['axis'];mean=sums[name]/n;owner.watch(mean,'gram.mean_gradient')
        raw=mean@mean.T if axis==0 else mean.T@mean;owner.watch(raw,'gram.raw')
        for mode in runtime.protocol['basis_response']['modes']:
            if mode=='raw':gram=raw;relative=None
            else:
                clipped=mean*coefficient;owner.watch(clipped,'gram.explicitly_clipped_gradient')
                gram=clipped@clipped.T if axis==0 else clipped.T@clipped;owner.watch(gram,'gram.clipped')
                relative=full_clip_gram_relative(raw,gram,coefficient,owner)
                runtime.emit('full_clip_gram_before_gate',dict(pool=pool,sequences=n,**spec,relative=relative))
                if not math.isfinite(relative) or relative>runtime.protocol['basis_response']['gates']['clip_gram_relative']:raise FloatingPointError('Explicit clipping Gram semantic check failed')
                del clipped
            prospective=covariance_update(cov,gram,beta,owner)
            ri=indices(len(q),q.device);tile=prospective.index_select(0,ri).index_select(1,ri);owner.watch(tile,'prospective.physical_sample')
            sample=tile.cpu().double().numpy().copy();stats.add((n,name,'prospective_covariance',axis,mode),sample)
            before=offdiag_norm(q,prospective,owner)
            difference=prospective-cov;owner.watch(difference,'prospective.inherited_change')
            covnorm=float(cov.norm());gramnorm=float(gram.norm())
            row=dict(pool=pool,sequences=n,mode=mode,**spec,source_step=state['step'],beta=beta,
                inherited_covariance_norm=covnorm,incoming_gram_norm=gramnorm,
                weighted_incoming_over_inherited=(1-beta)*gramnorm/max(beta*covnorm,1e-24),
                relative_covariance_change=float(difference.norm())/max(covnorm,1e-24),
                prospective_covariance_norm=float(prospective.norm()),before_covariance_offdiag=before,
                before_normalized_covariance_offdiag=before/max(float(prospective.norm()),1e-24),
                prospective_sample=scalar_array(sample),clip_gram_relative=relative,
                policies=[measure_policy(runtime,policy,prospective,q,before,stats,(n,name,'basis_movement',axis,mode),owner,pool=pool) for policy in runtime.protocol['basis_response']['policies']])
            rows.append(row);del prospective,difference,gram,tile,sample,ri
        del mean,raw
    return rows
