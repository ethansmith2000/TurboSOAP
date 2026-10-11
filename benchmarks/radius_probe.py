"""Offline radius/correction interventions; trained controls remain unchanged."""
import copy
import math
import statistics
import torch
from soap_reference import ResearchSOAP
try:
    from .movement_retraction import warm
    from .warm_retraction import Retraction
    from .cifar_strict_transport import StrictTransportProbe
    from .cifar_tracking_probe import factor_score, relative, world
except ImportError:
    from movement_retraction import warm
    from warm_retraction import Retraction
    from cifar_strict_transport import StrictTransportProbe
    from cifar_tracking_probe import factor_score, relative, world

POLICIES=('base05_cubic2','radius06_cubic2','radius06_cubic3','radius06_alternate','qr')
COST_AGES={20,40,220,240}
CHAIN_AGES={20,220}


def policy_spec(name,age):
    if name not in POLICIES or name=='qr':raise ValueError('Expected a warm policy')
    if age<=0 or age%20:raise ValueError('Expected a positive scheduled refresh age')
    iterations=3 if name=='radius06_cubic3' or (name=='radius06_alternate' and (age//20)%2==1) else 2
    return Retraction(name,order=3,iterations=iterations,cap_mode='skew_frobenius',cap=.5 if name=='base05_cubic2' else .6)


def make_proxy(optimizer,name):
    proxy=copy.copy(optimizer)
    if name!='qr':
        proxy._gauge_step_one=lambda cov,q,group: warm(cov,q,group,policy_spec(name,group['radius_probe_age']))
    return proxy


def refresh(proxy,old,group,age,name):
    settings={**group,'radius_probe_age':age,'basis_method':'qr' if name=='qr' else 'warm'}
    state=dict(Q=old['q'],GG=old['cov'],exp_avg=old['m'],exp_avg_sq=old['v'],step=age)
    ResearchSOAP._research_refresh(proxy,state,settings)
    return dict(q=state['Q'],m=state['exp_avg'],v=state['exp_avg_sq'],cov=old['cov'])


@torch.no_grad()
def frozen_covariance_chain(proxy,old,group,age,name):
    """Nine successive refreshes, no new gradients/EMA updates, real QR schedule."""
    original=world(old['m'],old['q']);current=old;errors=[];grams=[]
    for i in range(9):
        torch.set_float32_matmul_precision('high')
        current=refresh(proxy,current,group,age+20*i,name)
        torch.set_float32_matmul_precision('highest')
        errors.append(relative(world(current['m'],current['q']),original))
        grams.append(max(factor_score(c,q)['gram'] for c,q in zip(current['cov'],current['q'])))
    return dict(world_m_errors=errors,maximum_world_m_error=max(errors),maximum_gram=max(grams),
                passes=max(errors)<.01 and max(grams)<.05)


@torch.no_grad()
def time_refreshes(proxies,old,group,age,offset):
    samples={name:[] for name in POLICIES}
    torch.set_float32_matmul_precision('high')
    for name in POLICIES:refresh(proxies[name],old,group,age,name)
    torch.cuda.synchronize()
    for round_index in range(3):
        shift=(offset+round_index)%len(POLICIES);order=POLICIES[shift:]+POLICIES[:shift]
        for name in order:
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(3):refresh(proxies[name],old,group,age,name)
            end.record();end.synchronize()
            samples[name].append(start.elapsed_time(end)/3)
    return {name:dict(round_ms=values,median_ms=statistics.median(values)) for name,values in samples.items()}


class RadiusProbe(StrictTransportProbe):
    @torch.no_grad()
    def flush(self):
        pending=list(self.pending);rows=super().flush()
        previous=torch.get_float32_matmul_precision()
        proxies={name:make_proxy(self.opt,name) for name in POLICIES}
        try:
            for index,(row,(actual,group,old)) in enumerate(zip(rows,pending)):
                age=actual['step'];torch.set_float32_matmul_precision('highest')
                incoming=[factor_score(c,q) for c,q in zip(old['cov'],old['q'])]
                original=world(old['m'],old['q']);effects={}
                for name in POLICIES:
                    torch.set_float32_matmul_precision('high')
                    result=refresh(proxies[name],old,group,age,name)
                    if name=='base05_cubic2':
                        exact=all(torch.equal(a,b) for a,b in zip(result['q'],actual['Q'])) and torch.equal(result['m'],actual['exp_avg']) and torch.equal(result['v'],actual['exp_avg_sq'])
                        if not exact:raise AssertionError('Radius baseline differs from executed Frobenius control')
                        row['radius_control_exact']=exact
                    torch.set_float32_matmul_precision('highest')
                    factors=[]
                    for before,c,q,new_q in zip(incoming,old['cov'],old['q'],result['q']):
                        score=factor_score(c,new_q)
                        factors.append(dict(dimension=q.shape[0],movement_rms=float((new_q-q).norm())/math.sqrt(q.shape[0]),
                                            residual_change=score['residual']-before['residual'],**score))
                    error=relative(world(result['m'],result['q']),original)
                    effects[name]=dict(factors=factors,world_m_error=error,maximum_gram=max(x['gram'] for x in factors),
                        corrections=0 if name=='qr' else policy_spec(name,age).iterations,
                        passes=error<.01 and max(x['gram'] for x in factors)<.05)
                    assert math.isfinite(error) and all(math.isfinite(v) for f in factors for v in f.values())
                    if age in CHAIN_AGES:effects[name]['frozen_covariance_chain']=frozen_covariance_chain(proxies[name],old,group,age,name)
                if age in COST_AGES:
                    timings=time_refreshes(proxies,old,group,age,index)
                    for name in POLICIES:effects[name]['refresh_timing']=timings[name]
                row['radius_effects']=effects
        finally:torch.set_float32_matmul_precision(previous)
        return rows
