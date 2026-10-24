"""Per-instance reuse installation and own-history physical-M diagnostics."""
import math
import weakref
import torch
from llm_sparse_screen_kernel import build_arm as native_arm, common_geometry_guard
from qr_reuse_profile_kernel import install
from hybrid_training_kernel import precision
from tensor_ownership import TensorOwnership
from stage_stability_contract import ARMS, PRIMARY_M_AGES

def build_arm(model,config,arm,router):
    if arm!='qr20':raise ValueError('Native source only')
    return native_arm(model,config,'qr20',router)

def qualify_actual_m(matrix,router,arm,phase):
    original=matrix._research_refresh
    ref=weakref.ref(matrix)
    # Retain a native bound method only through a weak reference to avoid an owner cycle.
    if getattr(original,'__self__',None) is matrix:
        method=original.__func__
        def refresh_original(obj,state,group):return method(obj,state,group)
    else:
        def refresh_original(obj,state,group):return original(state,group)
    def refresh(state,group):
        obj=ref()
        if obj is None:raise RuntimeError('Released optimizer owner')
        due=(not state.get('research_strict_transports',0)) if phase=='disposable' else state['step'] in PRIMARY_M_AGES
        if not due:return refresh_original(obj,state,group)
        owner=TensorOwnership();before=after=None;rng=router.rng_digest()
        try:
            with precision('highest'):
                before=obj._project_back_with_basis(state['exp_avg'],state['Q'],group);owner.watch(before,'actualM.before')
            refresh_original(obj,state,group)
            with precision('highest'):
                after=obj._project_back_with_basis(state['exp_avg'],state['Q'],group);owner.watch(after,'actualM.after')
                error=float((after-before).norm()/before.norm().clamp_min(1e-12))
            row=dict(arm=arm,phase=phase,age=int(state['step']),physical_m_relative=error,passed=math.isfinite(error) and error<=.01,rng_unchanged=rng==router.rng_digest())
            router.moments.append(row);router.emit('actual_m_before_gate',row)
            if not row['passed'] or not row['rng_unchanged']:raise FloatingPointError('Actual own-history momentum preservation failed')
        finally:
            before=after=None;release=owner.report();owner.close();router.emit('actual_m_release',release)
            if not release['all_registered_released']:raise RuntimeError('Physical-M buffers leaked')
    matrix._research_refresh=refresh


# Read-only stage observers. No instance projection hook is installed on source.
from stage_stability_contract import STAGES,LAGS
from projection_profile_kernel import project as bf16_project
from qr_reuse_profile_kernel import qr_basis,following_update
from qr_reuse_profile_core import flags
from matched_training_runtime import streaming_digest,failure_record


def covariance_metrics(current,old,owner):
    delta=current-old;owner.watch(delta,'metric.covariance_delta')
    tc=current.trace();to=old.trace()
    shape=current/tc.clamp_min(1e-30)-old/to.clamp_min(1e-30);owner.watch(shape,'metric.trace_normalized_delta')
    return dict(relative=float(delta.norm()/old.norm().clamp_min(1e-30)),trace_ratio=float(tc/to.clamp_min(1e-30)),trace_normalized_relative=float(shape.norm()/(old.norm()/to.clamp_min(1e-30)).clamp_min(1e-30)))


def gradient_metrics(current,old,owner):
    delta=current-old;owner.watch(delta,'metric.gradient_delta')
    return dict(cosine=float((current*old).sum()/(current.norm()*old.norm()).clamp_min(1e-30)),relative=float(delta.norm()/old.norm().clamp_min(1e-30)),norm_ratio=float(current.norm()/old.norm().clamp_min(1e-30)))


def residual(cov,q,owner):
    rotated=q.T@cov@q;owner.watch(rotated,'metric.rotated_covariance')
    rotated.diagonal().zero_()
    return float(rotated.norm()/cov.norm().clamp_min(1e-30))


def factor_metrics(cov,q,banks,index,axis,owner):
    fresh,order=qr_basis(cov,q,'native');owner.watch(fresh,'metric.native_qr');owner.watch(order,'metric.native_order')
    with precision('highest'):
        gram=fresh.T@fresh;owner.watch(gram,'metric.native_gram');gram.diagonal().sub_(1.)
        absolute=float(gram.norm());normalized=absolute/math.sqrt(len(fresh))
        if absolute>.05 or normalized>.005:raise FloatingPointError('Native reference geometry failed')
        row=dict(dimension=len(q),native_fresh_gram_absolute=absolute,native_fresh_gram_normalized=normalized,current_basis_residual=residual(cov,q,owner),fresh_native_qr_residual=residual(cov,fresh,owner),lags={})
        for lag,bank in banks.items():
            old=bank['data'][index]
            row['lags'][str(lag)]=dict(covariance=covariance_metrics(cov,old['GG'][axis],owner),held_basis_residual=residual(cov,old['Q'][axis],owner),capture_call=bank['call'],capture_matrix_age=old['age'],capture_last_refresh_call=old['last_refresh_call'])
    return row


def adaptive_metrics(matrix,state,group,grad,owner):
    # Native helper is retained exact. Candidate repeats its pointwise math with
    # the same187 BF16 forward/back projection arithmetic, without touching state.
    a=following_update(matrix,state,group,grad);owner.watch(a,'adaptive.native')
    z=bf16_project(grad,state['Q']);owner.watch(z,'adaptive.bf16projected')
    b1,b2=group['betas'];m=b1*state['exp_avg']+(1-b1)*z;v=b2*state['exp_avg_sq']+(1-b2)*z.square()
    owner.watch(m,'adaptive.privateM');owner.watch(v,'adaptive.privateV')
    projected=m/(v.sqrt()+group['eps']);owner.watch(projected,'adaptive.private_scaledM')
    b=bf16_project(projected,state['Q'],True);owner.watch(b,'adaptive.bf16world')
    with precision('highest'):
        relative=float((a-b).norm()/a.norm().clamp_min(1e-30));cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))
    return dict(relative=relative,cosine=cosine,passes_187_screen=relative<=.01 and cosine>=.9999)


def finite_scalars(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite_scalars(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(finite_scalars(v) for v in value)
    return True


class Recorder:
    def __init__(self,runtime,stages=STAGES,lags=LAGS):
        self.runtime=runtime;self.stages=tuple(stages);self.lags=tuple(lags);self.banks={};self.releases=[];self.observations=[]
        if min(self.stages)<=max(self.lags):raise ValueError('History capture must follow optimizer initialization')

    def _capture(self,opt,call,target,lag):
        owner=TensorOwnership();bank=dict(call=call,data=[],owner=owner)
        self.banks[(target,lag)]=bank
        for group in opt.matrix.param_groups:
            for name,p in zip(group['param_names'],group['params']):
                state=opt.matrix.state[p]
                age=int(state['step']);freq=group['precondition_frequency']
                row=dict(name=name,age=age,last_refresh_call=(age//freq)*freq+1 if age>=freq else 1,GG=[],Q=[])
                bank['data'].append(row)
                for key in ('GG','Q'):
                    for value in state[key]:
                        cloned=value.detach().clone();row[key].append(cloned)
                        owner.watch(cloned,'history.'+str(target)+'.'+str(lag)+'.'+key);del cloned
                row['gradient']=p.grad.detach().clone();owner.watch(row['gradient'],'history.gradient')
        self.runtime.emit('stage_history_capture',dict(source_call=call,target=target,lag=lag,matrices=len(bank['data'])))

    def _drop(self,key):
        bank=self.banks.pop(key);bank['data'].clear()
        owner=bank['owner'];release=owner.report() if owner._tensors else dict(all_registered_released=True,tracked_tensors=0)
        owner.close();self.releases.append(release)
        if not release['all_registered_released']:raise RuntimeError('Stage history bank leaked')
        return release

    def close(self):
        errors=[]
        for key in list(self.banks):
            try:self._drop(key)
            except BaseException as exc:errors.append(failure_record(exc))
        return dict(all_registered_released=not errors and all(r['all_registered_released'] for r in self.releases),released_history_banks=len(self.releases),errors=errors)

    def fingerprint(self,opt):
        r=self.runtime
        return dict(model=streaming_digest(r.model.state_dict()),optimizer=streaming_digest(opt.state_dict()),gradients=streaming_digest([p.grad for p in r.model.parameters()]),rng=streaming_digest(r.streams.capture()),flags=flags(),gradient_ids=[id(p.grad) for p in r.model.parameters()])

    def _measure(self,call,opt,owner):
        banks={lag:self.banks[(call,lag)] for lag in self.lags};factors=[];gradients=[];actions=[];index=0
        for group in opt.matrix.param_groups:
            for name,p in zip(group['param_names'],group['params']):
                state=opt.matrix.state[p]
                for axis,(cov,q) in enumerate(zip(state['GG'],state['Q'])):
                    row=factor_metrics(cov,q,banks,index,axis,owner);row.update(parameter=name,axis=axis,current_age=int(state['step']),current_last_refresh_call=(int(state['step'])//group['precondition_frequency'])*group['precondition_frequency']+1)
                    factors.append(row)
                with precision('highest'):
                    for lag,bank in banks.items():gradients.append(dict(parameter=name,lag=lag,capture_call=bank['call'],**gradient_metrics(p.grad,bank['data'][index]['gradient'],owner)))
                actions.append(dict(parameter=name,**adaptive_metrics(opt.matrix,state,group,p.grad,owner)))
                index+=1
        return dict(call=call,lr=float(opt.matrix.param_groups[0]['lr']),factors=factors,gradients=gradients,adaptive=actions,meaning='Read-only common native state/retained last gradient; no unseen gradient, candidate history or stale-update learning')

    @torch.no_grad()
    def observe(self,call,opt):
        for target in self.stages:
            for lag in self.lags:
                if call==target-lag:self._capture(opt,call,target,lag)
        if call not in self.stages:return None
        before=self.fingerprint(opt);owner=TensorOwnership();row=None;error=None
        try:
            row=self._measure(call,opt,owner)
            if not finite_scalars(row):raise FloatingPointError('Nonfinite stage diagnostic')
        except BaseException as exc:error=failure_record(exc)
        finally:
            after=self.fingerprint(opt)
            releases=[self._drop((call,lag)) for lag in self.lags]
            release=owner.report() if owner._tensors else dict(all_registered_released=True,tracked_tensors=0)
            owner.close()
            cleanup=dict(source_unchanged=before==after,workspace_release=release,history_releases=releases,error=error)
            self.runtime.emit('stage_observation_cleanup',dict(call=call,**cleanup))
        if error is not None:raise RuntimeError('Stage observation failed: '+str(error))
        if before!=after or not release['all_registered_released']:raise RuntimeError('Stage source/flags/gradient/storage guard failed')
        row['cleanup']=cleanup;self.observations.append(row);self.runtime.emit('stage_observation',row)
        return row
