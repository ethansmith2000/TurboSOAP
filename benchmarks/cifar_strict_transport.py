"""Shared implementation gate, cost kernels and actual-training transport probe."""
import math
import torch
from soap_reference import ResearchSOAP
try:
    from .cifar_tracking_probe import TrackingProbe,project,world,relative,factor_score
except ImportError:
    from cifar_tracking_probe import TrackingProbe,project,world,relative,factor_score


def shadow_refresh(optimizer,old,group,age):
    state=dict(Q=old['q'],GG=old['cov'],exp_avg=old['m'],exp_avg_sq=old['v'],step=age)
    # Bypass an observer attached to the instance. Execute the real implementation.
    ResearchSOAP._research_refresh(optimizer,state,group)
    return state


def update(m,v,qs,eps):
    return world(m/(v.sqrt()+eps),qs)


@torch.no_grad()
def diagnose(optimizer,old,group,age,actual):
    previous=torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        expected_m=project(world(old['m'],old['q']),actual['Q'])
        actual_update=update(actual['exp_avg'],actual['exp_avg_sq'],actual['Q'],group['eps'])
        expected_update=update(expected_m,actual['exp_avg_sq'],actual['Q'],group['eps'])
        reference=shadow_refresh(optimizer,old,group,age)
        scores=[factor_score(c,q) for c,q in zip(old['cov'],actual['Q'])]
        d=dict(selected_basis_update_error=relative(actual_update,expected_update),
               selected_basis_m_error=relative(actual['exp_avg'],expected_m),
               world_m_preservation_error=relative(world(actual['exp_avg'],actual['Q']),world(old['m'],old['q'])),
               full_strict_update_difference=relative(actual_update,update(reference['exp_avg'],reference['exp_avg_sq'],reference['Q'],group['eps'])),
               maximum_gram=max(f['gram'] for f in scores),mean_residual=sum(f['residual'] for f in scores)/len(scores))
        if not all(math.isfinite(x) for x in d.values()):raise FloatingPointError('Nonfinite transport diagnostic')
        d['targeted_pass']=(d['selected_basis_update_error']<1e-6 and d['world_m_preservation_error']<.01 and d['maximum_gram']<.05)
        d['full_strict_pass']=d['full_strict_update_difference']<.01
        return d
    finally:torch.set_float32_matmul_precision(previous)


class StrictTransportProbe(TrackingProbe):
    @torch.no_grad()
    def flush(self):
        rows=[]
        try:
            names={id(self.opt.state[p]):name for g in self.opt.param_groups for p,name in zip(g['params'],g['param_names'])}
            for state,group,old in self.pending:
                d=diagnose(self.opt,old,group,state['step'],state)
                strict=group.get('transport_precision','inherit')=='highest'
                if strict and not d['targeted_pass']:
                    raise FloatingPointError('Strict transport selected-basis/geometry gate failed')
                reset=group['basis_method']=='qr' or (group['hard_reset_interval'] and state['step']%group['hard_reset_interval']==0)
                rows.append(dict(matrix=names[id(state)],age=state['step'],shape=list(old['m'].shape),
                                 executed_method='qr' if reset else 'warm',strict_transport=strict,**d))
        finally:self.pending.clear()
        return rows
