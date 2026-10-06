"""Explicit ViT matrix/auxiliary grouping and group-preserving schedules."""
import math
import torch
from soap_reference import ResearchSOAP


class SplitOptimizer:
    def __init__(self,matrix,auxiliary):
        self.matrix=matrix
        self.auxiliary=auxiliary
        self.param_groups=matrix.param_groups+auxiliary.param_groups

    def zero_grad(self,set_to_none=True):
        self.matrix.zero_grad(set_to_none=set_to_none)
        self.auxiliary.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.matrix.step();self.auxiliary.step()

    def state_dict(self):
        return {'matrix':self.matrix.state_dict(),'auxiliary':self.auxiliary.state_dict()}

    def load_state_dict(self,state):
        self.matrix.load_state_dict(state['matrix']);self.auxiliary.load_state_dict(state['auxiliary'])
        self.param_groups=self.matrix.param_groups+self.auxiliary.param_groups


def grouped_parameters(model,c,all_adam=False):
    groups={}
    seen=set()
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        parent,_,leaf=name.rpartition('.')
        module=model.get_submodule(parent)
        matrix=(not all_adam and name.startswith('blocks.') and
                isinstance(module,torch.nn.Linear) and leaf=='weight')
        decay=leaf=='weight' and isinstance(module,(torch.nn.Linear,torch.nn.Conv2d))
        role='matrix' if matrix else 'auxiliary'
        key=(role,decay)
        rate=c['matrix_lr'] if matrix else c['lr']
        group=groups.setdefault(key,dict(params=[],param_names=[],role=role,
                         lr=rate,peak_lr=rate,weight_decay=c['weight_decay'] if decay else 0.))
        assert id(p) not in seen
        seen.add(id(p));group['params'].append(p);group['param_names'].append(name)
    assert seen=={id(p) for p in model.parameters() if p.requires_grad}
    return list(groups.values())


def build_optimizer(model,c,policy='qr10'):
    groups=grouped_parameters(model,c,all_adam=policy=='adamw')
    fused=next(model.parameters()).is_cuda
    if policy=='adamw':
        return torch.optim.AdamW(groups,betas=tuple(c['betas']),eps=1e-8,fused=fused)
    matrix=ResearchSOAP([g for g in groups if g['role']=='matrix'],
        lr=c['matrix_lr'],betas=tuple(c.get('matrix_betas',c['betas'])),shampoo_beta=c['shampoo_beta'],
        precondition_mode=c['precondition_mode'],covariance_compute_dtype=c['covariance_compute_dtype'],
        transport_precision=c.get('transport_precision','inherit'))
    iterations=c.get('basis_ns_iterations',2)
    if type(iterations) is not int or iterations<1:
        raise ValueError('basis_ns_iterations must be a positive integer')
    for group in matrix.param_groups:group['basis_ns_iterations']=iterations
    cap=c.get('basis_rotation_cap',.1)
    if not isinstance(cap,(int,float)) or isinstance(cap,bool) or not math.isfinite(cap) or cap<=0:
        raise ValueError('basis_rotation_cap must be finite and positive')
    for group in matrix.param_groups:group['basis_rotation_cap']=cap
    auxiliary=torch.optim.AdamW([g for g in groups if g['role']=='auxiliary'],
        betas=tuple(c['betas']),eps=1e-8,fused=fused)
    result=SplitOptimizer(matrix,auxiliary)
    set_policy(result,policy)
    return result


def set_policy(optimizer,policy):
    policies={'qr10':('qr',10,0),'qr20':('qr',20,0),'qr40':('qr',40,0),
              'warm5':('warm',5,200),'warm10':('warm',10,200),
              'warm20':('warm',20,200),'warm40':('warm',40,200)}
    if policy not in policies:raise ValueError('Unknown calibration policy')
    method,frequency,reset=policies[policy]
    for group in optimizer.matrix.param_groups:
        group.update(basis_method=method,precondition_frequency=frequency,
                     hard_reset_interval=reset,variance_policy='permutation')


def apply_lr_multiplier(optimizer,multiplier):
    for group in optimizer.param_groups:
        group['lr']=group['peak_lr']*multiplier


def counters(optimizer):
    if not isinstance(optimizer,SplitOptimizer):return {}
    names=['research_qr_refreshes','research_warm_refreshes']
    if any(g.get('transport_precision','inherit')=='highest' for g in optimizer.matrix.param_groups):
        names.append('research_strict_transports')
    return {name:sum(int(s.get(name,0)) for s in optimizer.matrix.state.values()) for name in names}


def screen_arm_config(config,arm):
    """Apply matrix-only study changes without changing auxiliary Adam memory."""
    beta2=arm.get('matrix_beta2',config.get('matrix_betas',config['betas'])[1])
    covariance=arm.get('covariance_beta',config['shampoo_beta'])
    if beta2 not in (.99,.999) or covariance not in (.99,.999):
        raise ValueError('Memory screen permits .99 and .999 only')
    result={**config,'matrix_lr':arm['matrix_lr'],
            'matrix_betas':[config['betas'][0],beta2],'shampoo_beta':covariance}
    if 'basis_ns_iterations' in arm:result['basis_ns_iterations']=arm['basis_ns_iterations']
    for key in ['basis_rotation_cap','transport_precision']:
        if key in arm:result[key]=arm[key]
    return result


@torch.no_grad()
def check_basis_state(optimizer,maximum_error):
    """Offline diagnostic; caller excludes its synchronized time from training."""
    prior=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
    errors=[]
    try:
        for parameter,state in optimizer.matrix.state.items():
            if not all(bool(torch.isfinite(t).all()) for t in [parameter,state['exp_avg'],state['exp_avg_sq']]):
                raise FloatingPointError('Nonfinite optimizer state')
            for q in state['Q']:
                if q is not None:
                    errors.append((q.T@q-torch.eye(q.shape[0],device=q.device)).norm()/q.shape[0]**.5)
        values=torch.stack(errors)
        if not bool(torch.isfinite(values).all()) or float(values.max())>=maximum_error:
            raise FloatingPointError('Predeclared basis stability guard exceeded')
        return dict(maximum_gram_error=float(values.max()),mean_gram_error=float(values.mean()),factor_count=len(errors))
    finally:torch.set_float32_matmul_precision(prior)
