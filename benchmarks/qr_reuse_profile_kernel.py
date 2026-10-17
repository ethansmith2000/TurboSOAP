"""Opt-in QR product reuse and isolated state helpers; original optimizer unchanged."""
import copy
import math
import weakref
import torch
from cifar_optimizers import build_optimizer
from soap_reference import ResearchSOAP
from hybrid_training_kernel import precision
from matched_training_runtime import tensor_values, optimizer_extras
from tensor_ownership import TensorOwnership


def qr_basis(cov,q,policy):
    if policy=='native':
        order=torch.argsort((q.T@cov@q).diagonal(),descending=True)
        return torch.linalg.qr(cov@q[:,order])[0],order
    if policy!='reuse':raise ValueError('Unknown QR implementation')
    product=cov@q
    order=torch.argsort((q*product).sum(dim=0),descending=True)
    return torch.linalg.qr(product[:,order])[0],order


def refresh_reuse(matrix,state,group):
    if group['basis_method']!='qr' or group['variance_policy']!='permutation' or group['transport_precision']!='highest':
        raise ValueError('Reuse candidate supports declared native QR/highest-M/permutation only')
    old=state['Q'];new=[];variance=state['exp_avg_sq']
    for axis,(cov,q) in enumerate(zip(state['GG'],old)):
        if q is None or q.dtype!=torch.float32 or cov.dtype!=torch.float32:
            raise ValueError('Two-sided FP32 factors required')
        candidate,order=qr_basis(cov,q,'reuse');new.append(candidate.float())
        variance=variance.index_select(axis,order)
    with precision('highest'):matrix._transport_first_moment(state,old,new,group)
    state['exp_avg_sq']=variance;state['Q']=new
    state['research_strict_transports']=state.get('research_strict_transports',0)+1
    state['research_qr_refreshes']=state.get('research_qr_refreshes',0)+1
    state['research_warm_refreshes']=state.get('research_warm_refreshes',0)


def install(matrix,policy,physical_rows=None):
    if policy not in ('native','reuse'):raise ValueError('Unknown QR implementation')
    if policy=='native' and physical_rows is None:return
    ref=weakref.ref(matrix)
    def refresh(state,group):
        obj=ref()
        if obj is None:raise RuntimeError('Released QR owner')
        owner=TensorOwnership();before=after=None
        try:
            if physical_rows is not None:
                with precision('highest'):before=obj._project_back_with_basis(state['exp_avg'],state['Q'],group)
                owner.watch(before,'physical.before')
            if policy=='native':ResearchSOAP._research_refresh(obj,state,group)
            else:refresh_reuse(obj,state,group)
            if physical_rows is not None:
                with precision('highest'):
                    after=obj._project_back_with_basis(state['exp_avg'],state['Q'],group)
                    owner.watch(after,'physical.after');error=float((after-before).norm()/before.norm().clamp_min(1e-12))
                physical_rows.append(dict(policy=policy,error=error,passed=math.isfinite(error) and error<=.01))
        finally:
            before=after=None
            release=owner.report() if physical_rows is not None else None
            owner.close()
            if release is not None and not release['all_registered_released']:raise RuntimeError('Physical QR buffers leaked')
    matrix._research_refresh=refresh


def shadow(model,config,base,policy,physical_rows=None):
    opt=build_optimizer(model,config,'qr20')
    private=copy.deepcopy(base);opt.load_state_dict(private);del private
    for group in opt.matrix.param_groups:
        group.update(basis_method='qr',precondition_frequency=1000000 if policy=='ordinary' else 200,
            hard_reset_interval=0,variance_policy='permutation',transport_precision='highest')
    if policy!='ordinary':install(opt.matrix,policy,physical_rows)
    return opt


def close_shadow(opt):
    """Release only private state; the live source owns parameters and gradients."""
    owner=TensorOwnership()
    for part in (opt.matrix,opt.auxiliary):
        owner.watch_tree(list(part.state.values()),'shadow.state')
        owner.watch_tree(optimizer_extras(part),'shadow.extras')
    del part
    opt.matrix.state.clear();opt.auxiliary.state.clear()
    opt.param_groups.clear();opt.matrix.param_groups.clear();opt.auxiliary.param_groups.clear()
    result=owner.report();owner.close()
    if not result['all_registered_released']:raise RuntimeError('Shadow state/storage leaked')
    return result


def following_update(matrix,state,group,grad):
    """Same hypothetical following gradient in both bases, before parameter rounding."""
    projected=matrix._project_with_basis(grad,state['Q'],group)
    b1,b2=group['betas']
    m=state['exp_avg']*b1+projected*(1-b1)
    v=state['exp_avg_sq']*b2+projected.square()*(1-b2)
    return matrix._project_back_with_basis(m/(v.sqrt()+group['eps']),state['Q'],group)


def compare(native,reuse,base,gates):
    factors=[];actions=[];index=0
    for ng,rg in zip(native.matrix.param_groups,reuse.matrix.param_groups):
        for name,p in zip(ng['param_names'],ng['params']):
            ns=native.matrix.state[p];rs=reuse.matrix.state[p];old=base['matrix']['state'][index];index+=1
            for axis,(cov,q,nq,rq) in enumerate(zip(ns['GG'],old['Q'],ns['Q'],rs['Q'])):
                diag=(q.T@cov@q).diagonal();product=cov@q
                no=torch.argsort(diag,descending=True);ro=torch.argsort((q*product).sum(0),descending=True)
                band=float((diag[no]-diag[ro]).abs().max()/diag.abs().max().clamp_min(1e-12))
                with precision('highest'):
                    gram=rq.T@rq;gram.diagonal().sub_(1.);absolute=float(gram.norm())
                    signs=torch.sign((nq*rq).sum(0));signs=torch.where(signs==0,torch.ones_like(signs),signs)
                    aligned=float((nq-rq*signs).norm()/nq.norm().clamp_min(1e-12))
                    raw=float((nq-rq).norm()/nq.norm().clamp_min(1e-12))
                factors.append(dict(parameter=name,axis=axis,dimension=len(q),order_disagreements=int((no!=ro).sum()),
                    order_score_band=band,raw_basis_relative=raw,sign_aligned_basis_relative=aligned,
                    absolute_gram_fro=absolute,normalized_gram=absolute/math.sqrt(len(q)),
                    passed=math.isfinite(absolute) and absolute<=.05 and absolute/math.sqrt(len(q))<=.005 and band<=gates['order_score_band_relative_to_max']))
            with precision('highest'):
                a=following_update(native.matrix,ns,ng,p.grad);b=following_update(reuse.matrix,rs,rg,p.grad)
                error=float((a-b).norm()/a.norm().clamp_min(1e-12))
                cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-24))
            actions.append(dict(parameter=name,relative=error,cosine=cosine,
                passed=math.isfinite(error) and math.isfinite(cosine) and error<=gates['following_gradient_update_relative'] and cosine>=gates['following_gradient_update_cosine']))
    return dict(factors=factors,actions=actions,passed=all(r['passed'] for r in factors+actions))
