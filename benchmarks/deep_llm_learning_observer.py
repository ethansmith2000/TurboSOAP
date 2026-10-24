"""Bounded read-only layer/stage observations; native source updates stay untouched."""
import gc
import math

import torch

from hybrid_training_kernel import precision
from matched_training_runtime import streaming_digest
from qr_reuse_profile_core import flags
from stage_stability_kernel import factor_metrics,gradient_metrics,adaptive_metrics,finite_scalars
from tensor_ownership import TensorOwnership

STAGES=(260,1000,3500,10000,20000,30000,35000)
LAGS=(20,200)
LAYERS=(0,5,11)
MAX_BANK_BYTES=2*552075264


def selected(opt,layers=LAYERS):
    for group in opt.matrix.param_groups:
        for name,p in zip(group['param_names'],group['params']):
            if name.startswith('blocks.') and int(name.split('.')[1]) in layers:
                yield name,p,group


class LayerObserver:
    def __init__(self,model,emit,rng_digest,stages=STAGES,lags=LAGS,layers=LAYERS,bank_bound=MAX_BANK_BYTES):
        if not stages or min(stages)<=max(lags) or len(lags)!=2:
            raise ValueError('Two history lags must precede every observation')
        self.model=model;self.emit=emit;self.rng_digest=rng_digest
        self.stages=tuple(stages);self.lags=tuple(lags);self.layers=tuple(layers)
        self.bank_bound=bank_bound;self.banks={};self.releases=[];self.observations=[];self.bank_bytes=0

    def fingerprint(self,opt):
        return dict(model=streaming_digest(self.model.state_dict()),optimizer=streaming_digest(opt.state_dict()),
            gradients=streaming_digest([p.grad for p in self.model.parameters()]),
            gradient_ids=[id(p.grad) for p in self.model.parameters()],rng=self.rng_digest(),flags=flags())

    def capture(self,opt,call,target,lag):
        if len(self.banks)>=2:raise MemoryError('At most two history banks')
        owner=TensorOwnership();bank=dict(call=call,data=[],owner=owner,bytes=0)
        self.banks[(target,lag)]=bank
        for name,p,group in selected(opt,self.layers):
            state=opt.matrix.state[p];age=int(state['step'])
            row=dict(name=name,age=age,last_refresh_call=(age//20)*20+1 if age>=20 else 1,GG=[],Q=[])
            bank['data'].append(row)
            for key in ('GG','Q'):
                for value in state[key]:
                    cloned=value.detach().clone();row[key].append(cloned)
                    owner.watch(cloned,'history.'+key);bank['bytes']+=cloned.numel()*cloned.element_size();del cloned
            row['gradient']=p.grad.detach().clone();owner.watch(row['gradient'],'history.gradient')
            bank['bytes']+=row['gradient'].numel()*row['gradient'].element_size()
        self.bank_bytes+=bank['bytes']
        if len(bank['data'])!=4*len(self.layers) or self.bank_bytes>self.bank_bound:
            raise MemoryError('History coverage or byte budget differs')
        self.emit('layer_history_capture',dict(call=call,target=target,lag=lag,matrices=len(bank['data']),bytes=bank['bytes'],live_bank_bytes=self.bank_bytes))

    def drop(self,key):
        bank=self.banks.pop(key);bank['data'].clear();self.bank_bytes-=bank['bytes']
        owner=bank['owner'];report=owner.report() if owner._tensors else dict(all_registered_released=True,tracked_tensors=0)
        owner.close();self.releases.append(report)
        if not report['all_registered_released']:raise RuntimeError('History storage retained')
        return report

    def measure(self,opt,call,owner):
        banks={lag:self.banks[(call,lag)] for lag in self.lags}
        factors=[];gradients=[];actions=[]
        for index,(name,p,group) in enumerate(selected(opt,self.layers)):
            state=opt.matrix.state[p];layer=int(name.split('.')[1])
            for axis,(cov,q) in enumerate(zip(state['GG'],state['Q'])):
                row=factor_metrics(cov,q,banks,index,axis,owner)
                row.update(parameter=name,layer=layer,axis=axis,matrix_age=int(state['step']))
                factors.append(row)
            with precision('highest'):
                for lag,bank in banks.items():
                    gradients.append(dict(parameter=name,layer=layer,lag=lag,**gradient_metrics(p.grad,bank['data'][index]['gradient'],owner)))
            actions.append(dict(parameter=name,layer=layer,**adaptive_metrics(opt.matrix,state,group,p.grad,owner)))
        return dict(call=call,lr=float(opt.matrix.param_groups[0]['lr']),factors=factors,gradients=gradients,adaptive=actions,
            meaning='Common native state, repeating retained clipped gradient hypothetically; no candidate updates/backwards/history or stale-update learning')

    @torch.no_grad()
    def tick(self,call,opt):
        for target in self.stages:
            for lag in self.lags:
                if call==target-lag:self.capture(opt,call,target,lag)
        if call not in self.stages:return None
        owner=TensorOwnership();before=self.fingerprint(opt);row=None
        try:
            row=self.measure(opt,call,owner)
            if not finite_scalars(row):raise FloatingPointError('Nonfinite layer observation')
            if len(row['factors'])!=8*len(self.layers) or len(row['adaptive'])!=4*len(self.layers):
                raise RuntimeError('Layer observation coverage differs')
        finally:
            gc.collect();metric_release=owner.report() if owner._tensors else dict(all_registered_released=True,tracked_tensors=0)
            owner.close()
            bank_releases=[self.drop((call,lag)) for lag in self.lags if (call,lag) in self.banks]
            after=self.fingerprint(opt)
            self.emit('layer_observation_restoration',dict(call=call,source_unchanged=before==after,metric_release=metric_release,bank_releases=bank_releases))
            if before!=after or not metric_release['all_registered_released']:
                raise RuntimeError('Read-only layer observation mutated source or retained workspace')
        row.update(source_unchanged=True,all_registered_released=True)
        self.observations.append(row);self.emit('layer_observation',row)
        return row

    def close(self):
        for key in list(self.banks):self.drop(key)
        return dict(released_history_banks=len(self.releases),live_bank_bytes=self.bank_bytes,
            all_registered_released=not self.banks and all(r['all_registered_released'] for r in self.releases))
