"""Existing BF16 adapter with counted dispatch and a genuinely native preview reference."""
from collections import Counter
import math

import torch
from soap import SOAP
from hybrid_training_kernel import precision
from projection_profile_kernel import install,project
from deep_llm_learning_observer import LayerObserver,selected
from stage_stability_kernel import factor_metrics,gradient_metrics


def install_counted(matrix,counts,scope):
    install(matrix,'bf16')
    original_forward=matrix._project_with_basis;original_back=matrix._project_back_with_basis
    def forward(tensor,basis,group):
        kind='highest_forward' if torch.get_float32_matmul_precision()=='highest' else 'bf16_forward'
        counts[scope['value']+'.'+kind]+=1
        return original_forward(tensor,basis,group)
    def back(tensor,basis,group):
        kind='highest_back' if torch.get_float32_matmul_precision()=='highest' else 'bf16_back'
        counts[scope['value']+'.'+kind]+=1
        return original_back(tensor,basis,group)
    matrix._project_with_basis=forward;matrix._project_back_with_basis=back


def native_adaptive(matrix,state,group,grad):
    """Bypass instance BF16 hooks; preserve following_update's exact operation order."""
    projected=SOAP._project_with_basis(matrix,grad,state['Q'],group)
    b1,b2=group['betas']
    m=state['exp_avg']*b1+projected*(1-b1)
    v=state['exp_avg_sq']*b2+projected.square()*(1-b2)
    return SOAP._project_back_with_basis(matrix,m/(v.sqrt()+group['eps']),state['Q'],group)


def projection_metrics(matrix,state,group,grad,owner,check_live=False):
    a=native_adaptive(matrix,state,group,grad);owner.watch(a,'adaptive.unhooked_native')
    z=matrix._project_with_basis(grad,state['Q'],group) if check_live else project(grad,state['Q'])
    owner.watch(z,'adaptive.bf16_forward')
    if check_live:
        expected=project(grad,state['Q']);owner.watch(expected,'adapter.forward_reference')
        if not torch.equal(z,expected):raise RuntimeError('Live BF16 forward dispatch differs')
    b1,b2=group['betas']
    m=state['exp_avg']*b1+z*(1-b1);v=state['exp_avg_sq']*b2+z.square()*(1-b2)
    owner.watch(m,'adaptive.private_M');owner.watch(v,'adaptive.private_V')
    scaled=m/(v.sqrt()+group['eps']);owner.watch(scaled,'adaptive.private_scaled_M')
    b=matrix._project_back_with_basis(scaled,state['Q'],group) if check_live else project(scaled,state['Q'],True)
    owner.watch(b,'adaptive.bf16_reverse')
    if check_live:
        expected=project(scaled,state['Q'],True);owner.watch(expected,'adapter.back_reference')
        if not torch.equal(b,expected):raise RuntimeError('Live BF16 reverse dispatch differs')
    with precision('highest'):
        relative=float((a-b).norm()/a.norm().clamp_min(1e-30))
        cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))
    if not math.isfinite(relative) or not math.isfinite(cosine):raise FloatingPointError('Nonfinite projection preview')
    return dict(relative=relative,cosine=cosine,passes_187_screen=relative<=.01 and cosine>=.9999,
        reference='Unhooked native SOAP projections on common candidate state',live_adapter_parity=check_live)


class CandidateObserver(LayerObserver):
    def measure(self,opt,call,owner):
        banks={lag:self.banks[(call,lag)] for lag in self.lags};factors=[];gradients=[];actions=[]
        for index,(name,p,group) in enumerate(selected(opt,self.layers)):
            state=opt.matrix.state[p];layer=int(name.split('.')[1])
            for axis,(cov,q) in enumerate(zip(state['GG'],state['Q'])):
                row=factor_metrics(cov,q,banks,index,axis,owner)
                row.update(parameter=name,layer=layer,axis=axis,matrix_age=int(state['step']));factors.append(row)
            with precision('highest'):
                for lag,bank in banks.items():gradients.append(dict(parameter=name,layer=layer,lag=lag,**gradient_metrics(p.grad,bank['data'][index]['gradient'],owner)))
            actions.append(dict(parameter=name,layer=layer,**projection_metrics(opt.matrix,state,group,p.grad,owner)))
        return dict(call=call,lr=float(opt.matrix.param_groups[0]['lr']),factors=factors,gradients=gradients,adaptive=actions,
            meaning='Candidate-owned state; unhooked native versus BF16 hypothetical following retained gradient, not native-history replay or another training update')
