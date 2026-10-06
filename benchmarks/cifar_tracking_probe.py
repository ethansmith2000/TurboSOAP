"""Offline same-state refresh alternatives; never returned to the optimizer."""
import math
import torch
from soap import SOAP, _damped_jacobi_generator


def project(x,qs):
    return qs[0].T@x@qs[1]


def world(x,qs):
    return qs[0]@x@qs[1].T


def relative(x,y):
    return float((x-y).norm()/y.norm().clamp_min(1e-30))


def factor_score(cov,q):
    b=q.T@cov@q
    return dict(residual=float((b-torch.diag_embed(b.diagonal())).norm()/b.norm().clamp_min(1e-30)),
        gram=float((q.T@q-torch.eye(q.shape[0],device=q.device)).norm()/q.shape[0]**.5))


class TrackingProbe:
    def __init__(self,optimizer,ages):
        self.opt=optimizer.matrix;self.ages=set(ages);self.pending=[]
        self.original=self.opt._research_refresh
        def observed(state,group):
            selected=state['step'] in self.ages
            if selected:
                before=dict(q=state['Q'],m=state['exp_avg'],v=state['exp_avg_sq'],cov=state['GG'])
            self.original(state,group)
            if selected:self.pending.append((state,group,before))
        self.opt._research_refresh=observed

    @torch.no_grad()
    def flush(self):
        previous=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
        rows=[]
        try:
            names={id(self.opt.state[p]):name for group in self.opt.param_groups
                   for p,name in zip(group['params'],group.get('param_names',[f'matrix{i}' for i in range(len(group['params']))]))}
            for state,group,old in self.pending:
                if any(q is None for q in old['q']):raise ValueError('Tracking probe requires both factors')
                qr=[];warm=[];twice=[];v_qr=old['v'];factors=[]
                shadow_group={**group,'basis_ns_iterations':6}
                for axis,(cov,q) in enumerate(zip(old['cov'],old['q'])):
                    order=torch.argsort((q.T@cov@q).diagonal(),descending=True)
                    qr_q=torch.linalg.qr(cov@q[:,order]).Q
                    warm_q=SOAP._gauge_step_one(cov,q,shadow_group)
                    twice_q=SOAP._gauge_step_one(cov,warm_q,shadow_group)
                    qr.append(qr_q);warm.append(warm_q);twice.append(twice_q)
                    v_qr=v_qr.index_select(axis,order)
                    raw=group['basis_lr']*_damped_jacobi_generator(SOAP._basis_covariance(cov,q),group['basis_jacobi_damping'])
                    magnitude=float(raw.norm()/q.shape[0]**.5)
                    scores={key:factor_score(cov,value) for key,value in
                            [('before',q),('executed',state['Q'][axis]),('qr',qr_q),('warm1',warm_q),('warm2',twice_q)]}
                    factors.append(dict(axis=axis,dimension=q.shape[0],raw_rotation_rms=magnitude,
                        cap_binding=magnitude>group['basis_rotation_cap'],scores=scores,
                        warm1_movement_rms=float((warm_q-q).norm()/q.shape[0]**.5),
                        warm2_movement_rms=float((twice_q-q).norm()/q.shape[0]**.5)))
                old_world=world(old['m'],old['q'])
                executed=world(state['exp_avg']/(state['exp_avg_sq'].sqrt()+group['eps']),state['Q'])
                updates={}
                for label,qs,v in [('qr',qr,v_qr),('warm1',warm,old['v']),('warm2',twice,old['v'])]:
                    m=project(old_world,qs);update=world(m/(v.sqrt()+group['eps']),qs)
                    updates[label]=dict(relative_difference_from_executed=relative(update,executed),
                        cosine_with_executed=float((update.flatten()@executed.flatten())/
                            (update.norm()*executed.norm()).clamp_min(1e-30)),
                        world_m_preservation_error=relative(world(m,qs),old_world))
                row=dict(matrix=names[id(state)],age=state['step'],shape=list(state['exp_avg'].shape),
                    executed_method='qr' if group['basis_method']=='qr' or state['step']%200==0 else 'warm',
                    factors=factors,update_alternatives=updates)
                scalars=[score[k] for f in factors for score in f['scores'].values() for k in ['residual','gram']]
                scalars.extend(v for item in updates.values() for v in item.values())
                if not all(math.isfinite(x) for x in scalars):raise FloatingPointError('Nonfinite tracking diagnostic')
                rows.append(row)
        finally:
            torch.set_float32_matmul_precision(previous);self.pending.clear()
        return rows

    def close(self):
        self.opt._research_refresh=self.original;self.pending.clear()
