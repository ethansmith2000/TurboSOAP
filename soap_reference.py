"""Research QR-SOAP reference and matched warm variant.

Uses original SOAP update ordering, epsilon/bias convention, QR column sorting
and permutation-only v handling. Avoids redundant m round trips on non-refresh
steps. Verified against pinned upstream source in tests/reference/. Dense warm
updates share the same step path; this does not change the existing SOAP class.
"""
import torch
from soap import SOAP


class ResearchSOAP(SOAP):
    def __init__(self, params, *, lr=.001, betas=(.9,.999), shampoo_beta=.999,
                 eps=1e-8, weight_decay=.05, precondition_frequency=10,
                 precondition_mode='all', covariance_compute_dtype='float32',
                 max_precond_dim=10000, basis_method='qr', variance_policy='permutation',
                 hard_reset_interval=200):
        if basis_method not in ('qr','warm') or variance_policy not in ('permutation','overlap'):
            raise ValueError('Unknown research basis or variance policy')
        if type(precondition_frequency) is not int or precondition_frequency < 1:
            raise ValueError('Refresh frequency must be a positive integer')
        if precondition_mode not in ('all','smaller_side'):
            raise ValueError('Research comparison supports all or smaller_side factors')
        if type(hard_reset_interval) is not int or hard_reset_interval < 0 or (hard_reset_interval and hard_reset_interval % precondition_frequency):
            raise ValueError('Hard-reset interval must be zero or divisible by refresh frequency')
        super().__init__(params,lr=lr,betas=betas,shampoo_beta=shampoo_beta,eps=eps,
                         weight_decay=weight_decay,precondition_frequency=precondition_frequency,
                         precondition_mode=precondition_mode,covariance_compute_dtype=covariance_compute_dtype,
                         max_precond_dim=max_precond_dim,basis_track_stats=False,
                         basis_reset_frequency=0,basis_reset_stagger=False)
        for group in self.param_groups:
            unsupported=('merge_dims','precondition_1d','normalize_grads',
                         'precondition_frequency_after_warmup','basis_residual_threshold',
                         'basis_lr_age_compensation','basis_stall_pair_threshold')
            if any(group.get(key,False) for key in unsupported):
                raise ValueError('Unsupported parameter-group controller in ResearchSOAP')
            if group['precondition_frequency'] != precondition_frequency:
                raise ValueError('Per-group cadence overrides are outside the research protocol')
            group.update(basis_method=basis_method,variance_policy=variance_policy,
                         hard_reset_interval=hard_reset_interval)

    def _research_refresh(self,state,group):
        old=state['Q']
        reset=group['basis_method']=='qr' or (group['hard_reset_interval']>0 and
                 state['step'] % group['hard_reset_interval']==0)
        new=[]
        variance=state['exp_avg_sq']
        for axis,(cov,q) in enumerate(zip(state['GG'],old)):
            if q is None:
                new.append(None);continue
            if reset:
                order=torch.argsort((q.T@cov@q).diagonal(),descending=True)
                candidate,_=torch.linalg.qr(cov@q[:,order])
                if group['variance_policy']=='permutation':
                    variance=variance.index_select(axis,order)
            else:
                candidate=self._gauge_step_one(cov,q,group)
            new.append(candidate.float())
        self._transport_first_moment(state,old,new,group)
        if group['variance_policy']=='overlap':
            self._transport_second_moment_for_reset(state,old,new,group)
        else:
            state['exp_avg_sq']=variance
        state['Q']=new
        state['research_qr_refreshes']=state.get('research_qr_refreshes',0)+int(reset)
        state['research_warm_refreshes']=state.get('research_warm_refreshes',0)+int(not reset)

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        self.last_basis_refreshes=0
        self.last_hard_reset_events=0
        for group in self.param_groups:
            beta1,beta2=group['betas']
            for p in group['params']:
                if p.grad is None:continue
                if p.ndim!=2 or p.grad.is_sparse:
                    raise ValueError('ResearchSOAP accepts dense 2D matrix parameters only')
                grad=p.grad.detach()
                state=self.state[p]
                if 'step' not in state:
                    self._init_state(grad,state,group,p)
                    self._accumulate_covariance(grad,state,group)
                    self._cold_start_basis(state)
                    continue
                state['step']+=1
                projected=self._project_with_basis(grad,state['Q'],group)
                m,v=state['exp_avg'],state['exp_avg_sq']
                m.mul_(beta1).add_(projected,alpha=1-beta1)
                v.mul_(beta2).add_(projected.square(),alpha=1-beta2)
                denominator=v.sqrt().add_(group['eps'])
                update=self._project_back_with_basis(m/denominator,state['Q'],group)
                rate=group['lr']*(1-beta2**state['step'])**.5/(1-beta1**state['step'])
                p.add_(update,alpha=-rate)
                if group['weight_decay']>0:p.add_(p,alpha=-group['lr']*group['weight_decay'])
                self._accumulate_covariance(grad,state,group)
                if state['step'] % group['precondition_frequency']==0:
                    self._research_refresh(state,group)
                    self.last_basis_refreshes+=1
                    self.last_hard_reset_events+=int(group['basis_method']=='qr' or
                        (group['hard_reset_interval']>0 and state['step']%group['hard_reset_interval']==0))
        return loss
