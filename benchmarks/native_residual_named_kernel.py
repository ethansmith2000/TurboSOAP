"""Explicit instance hooks; live original M/V/QR paths untouched."""
from collections import Counter
from functools import partial
import gc
import math
import weakref
import torch
from common_y_kernel import warm_prefix,retract_two
from residual_retraction import residual_retract_two
from prefix_sensitivity_kernel import prefix_stages
from hybrid_training_kernel import precision,validate_group,configure_hybrid
from native_residual_named_contract import POLICIES,ARMS
from matched_training_runtime import streaming_digest


def current_prefix(cov,q):return warm_prefix(cov,q,cov_mixed=False,candidate_mixed=True)
def highest_prefix(cov,q):return warm_prefix(cov,q,cov_mixed=False,candidate_mixed=True)
def candidate_retraction(y):return residual_retract_two(y,correction_mixed=True)


class PolicyKernel:
    def __init__(self,name,router,compiler=torch.compile):
        if name not in POLICIES:raise ValueError('Unknown named native policy')
        self.name=name;self.router=router;self.mode='high' if name=='current_hybrid20' else 'highest'
        self.eager_prefix=current_prefix if name=='current_hybrid20' else highest_prefix
        self.eager_ret=retract_two if name=='current_hybrid20' else candidate_retraction
        self.prefix=compiler(self.eager_prefix,mode='default',fullgraph=True,dynamic=False)
        self.retraction=compiler(self.eager_ret,mode='default',fullgraph=True,dynamic=False)
        self.calls=Counter()

    def evaluate(self,cov,q,eager=False):
        with precision(self.mode):y=(self.eager_prefix if eager else self.prefix)(cov,q)
        with precision('highest'):return (self.eager_ret if eager else self.retraction)(y)

    def __call__(self,cov,q,group):
        validate_group(group)
        if cov.dtype!=torch.float32 or q.dtype!=torch.float32 or cov.shape!=q.shape or q.ndim!=2:
            raise ValueError('Square matching FP32 state required')
        new=self.evaluate(cov,q);self.calls[str(len(q))]+=1;self.router.calls[str(len(q))]+=1
        if self.router.probe:
            ref=self.evaluate(cov,q,eager=True)
            with precision('highest'):
                delta=new-q;target=ref-q
                gram=new.T@new;gram.diagonal().sub_(1.)
                row=dict(policy=self.name,dimension=len(q),basis_relative=float((new-ref).norm()/ref.norm().clamp_min(1e-12)),
                    movement_relative=float((delta-target).norm()/target.norm().clamp_min(1e-12)),
                    movement_cosine=float((delta*target).sum()/(delta.norm()*target.norm()).clamp_min(1e-24)),
                    gram=float(gram.norm()/math.sqrt(len(q))))
            row['strict_basis_pass']=row['basis_relative']<=1e-4
            row['passed']=all(math.isfinite(row[k]) for k in ('basis_relative','movement_relative','movement_cosine','gram')) and row['strict_basis_pass'] and row['movement_relative']<=.1 and row['movement_cosine']>=.995 and row['gram']<.005
            self.router.probes.append(row);self.router.emit('factor_probe_before_gate',row)
            if not row['passed']:raise FloatingPointError('Own-policy actual-state parity failed: '+str(row))
        return new


class PolicyRouter:
    def __init__(self,emit,rng_digest,compiler=torch.compile):
        self.emit=emit;self.rng_digest=rng_digest;self.calls=Counter();self.probes=[];self.moment_probes=[];self.probe=False
        self.kernels={a:PolicyKernel(a,self,compiler) for a in POLICIES}


def relative(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-12))


def diagnose_state(matrix,state,group,router,parameter,owners):
    def own(x):owners.append(weakref.ref(x));return x
    old=state['Q'];covs=state['GG'];m=state['exp_avg'];v=state['exp_avg_sq']
    qr=[];orders=[];bases={a:[] for a in POLICIES};factor_rows=[]
    for axis,(cov,q) in enumerate(zip(covs,old)):
        with precision('high'):
            order=torch.argsort((q.T@cov@q).diagonal(),descending=True)
            candidate,_=torch.linalg.qr(cov@q[:,order])
        qr.append(own(candidate));orders.append(order)
        for name in POLICIES:bases[name].append(own(router.kernels[name].evaluate(cov,q)))
        metrics={}
        with precision('highest'):
            b_before=q.T@cov@q;b_before.diagonal().zero_();before=float(b_before.norm())
            for name,basis in dict(qr_counterfactual=qr,**bases).items():
                actual=basis[-1];gram=actual.T@actual;gram.diagonal().sub_(1.)
                b=actual.T@cov@actual;b.diagonal().zero_()
                metrics[name]=dict(gram=float(gram.norm()/len(q)**.5),movement_norm=float((actual-q).norm()),
                    covariance_offdiag=float(b.norm()),tracking_vs_before=float(b.norm()/max(before,1e-12)))
            delta=bases['highest_residual20'][-1]-q;target=bases['current_hybrid20'][-1]-q
            policy_difference=dict(basis_relative=relative(bases['highest_residual20'][-1],bases['current_hybrid20'][-1]),
                movement_relative=float((delta-target).norm()/target.norm().clamp_min(1e-12)),
                movement_cosine=float((delta*target).sum()/(delta.norm()*target.norm()).clamp_min(1e-24)))
        caps={}
        for name in POLICIES:
            kernel=router.kernels[name]
            with precision(kernel.mode):
                stage=prefix_stages(cov,q,False,True);pure=kernel.eager_prefix(cov,q)
            with precision('highest'):observer_relative=relative(stage['candidate'],pure)
            bound=float(stage['bound']);caps[name]=dict(attribution='eager-only',observer_relative=observer_relative,
                valid=observer_relative<=1e-6,raw_bound=bound,raw_norm=float(stage['raw'].norm()),
                cap_active=bound>.5,applied_cap=min(1.,.5/max(bound,1e-12)))
            router.emit('eager_cap_before_gate',dict(policy=name,dimension=len(q),**caps[name]))
            if not caps[name]['valid']:raise FloatingPointError('Eager cap observer invalid: '+str(caps[name]))
            del stage,pure
        factor_rows.append(dict(axis=axis,dimension=len(q),before_covariance_offdiag=before,
                                policies=metrics,policy_difference=policy_difference,eager_caps=caps))
    with precision('highest'):
        world=own(matrix._project_back_with_basis(m,old,group))
        qv=v
        for axis,order in enumerate(orders):qv=qv.index_select(axis,order)
        # The inherited-state preview contains no future gradient and reconstructs no missing V cross-history.
        qprojected=matrix._project_with_basis(world,qr,group)
        qr_adaptive=own(matrix._project_back_with_basis(qprojected/(qv.sqrt()+group['eps']),qr,group))
        previews={}
        for name,basis in dict(qr_counterfactual=qr,**bases).items():
            projected=matrix._project_with_basis(world,basis,group)
            reconstructed=matrix._project_back_with_basis(projected,basis,group)
            variance=qv if name=='qr_counterfactual' else v
            adaptive=own(matrix._project_back_with_basis(projected/(variance.sqrt()+group['eps']),basis,group))
            previews[name]=dict(world_m_reconstruction_relative=relative(reconstructed,world),
                projected_m_norm=float(projected.norm()),inherited_adaptive_norm=float(adaptive.norm()),
                inherited_adaptive_relative_to_qr=relative(adaptive,qr_adaptive),
                inherited_adaptive_cosine_to_qr=float((adaptive*qr_adaptive).sum()/(adaptive.norm()*qr_adaptive.norm()).clamp_min(1e-24)),
                variance_policy='QR permutation' if name=='qr_counterfactual' else 'unchanged carry')
            if name=='current_hybrid20':current_adaptive=own(adaptive.clone())
            if name=='highest_residual20':candidate_vs_current=relative(adaptive,current_adaptive)
            del projected,reconstructed,adaptive
    return dict(parameter=parameter,source_step=int(state['step']),shape=list(m.shape),factors=factor_rows,
        full_actual_momentum=True,full_actual_diagonal_v=True,adaptive_preview_not_future_step=True,
        previews=previews,candidate_adaptive_relative_to_current=candidate_vs_current,
        all_finite=all(math.isfinite(value) for row in previews.values() for value in row.values() if isinstance(value,float)),
        physical_m_pass=all(row['world_m_reconstruction_relative']<=.01 for row in previews.values()))


def configure_diagnostics(matrix,router,arm):
    original=type(matrix)._research_refresh
    matrix_ref=weakref.ref(matrix)
    def refresh(state,group):
        self=matrix_ref()
        if self is None:raise RuntimeError('Diagnostic optimizer owner no longer alive')
        if router.probe:
            name,p=next((name,p) for name,p in zip(group['param_names'],group['params']) if self.state[p] is state)
            before=streaming_digest(dict(state=state,parameter=p,gradient=p.grad));rng_before=router.rng_digest();owners=[]
            allocated_before=torch.cuda.memory_allocated()
            with precision('highest'):row=diagnose_state(self,state,group,router,name,owners)
            gc.collect();alive=sum(ref() is not None for ref in owners)
            after=streaming_digest(dict(state=state,parameter=p,gradient=p.grad));rng_after=router.rng_digest()
            row.update(source_arm=arm,before_sha256=before,after_sha256=after,rng_before=rng_before,rng_after=rng_after,
                       nonmutation_pass=before==after and rng_before==rng_after,
                       registered_owners=len(owners),alive_after_release=alive,registered_release_pass=alive==0,
                       allocated_before=allocated_before,global_peak_after=torch.cuda.max_memory_allocated(),
                       conservative_peak_extra_bytes=max(0,torch.cuda.max_memory_allocated()-allocated_before))
            router.moment_probes.append(row);router.emit('full_moment_probe_before_gate',row)
            if not(row['nonmutation_pass'] and row['registered_release_pass'] and row['all_finite'] and row['physical_m_pass'] and row['conservative_peak_extra_bytes']<=16*2**30):
                raise FloatingPointError('Read-only full native diagnostic failed')
        return original(self,state,group)
    matrix._research_refresh=refresh


def build_arm(model,config,arm,router):
    from cadence_training_contract import build_arm as baseline
    if arm not in ARMS:raise ValueError('Unknown named native policy arm')
    opt=baseline(model,config,'qr20' if arm=='qr20' else 'warm20')
    if arm in POLICIES:
        configure_hybrid(opt.matrix,router.kernels[arm]);configure_diagnostics(opt.matrix,router,arm)
    return opt
