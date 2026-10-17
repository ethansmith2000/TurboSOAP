"""Read-only, explicitly counted actual-state profiling; copies excluded from timers."""
from collections import Counter
import copy
import math
import time
import torch
from hybrid_training_kernel import precision
from matched_training_runtime import streaming_digest, failure_record
from tensor_ownership import TensorOwnership
from qr_reuse_profile_kernel import shadow, close_shadow, compare, qr_basis
from qr_reuse_profile_contract import PROFILE_COUNTS


def flags():
    return dict(matmul=torch.get_float32_matmul_precision(),tf32=torch.backends.cuda.matmul.allow_tf32,
        bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        grad_enabled=torch.is_grad_enabled(),recompile_limit=torch._dynamo.config.recompile_limit,
        accumulated_recompile_limit=torch._dynamo.config.accumulated_recompile_limit)


def gpu_measure(fn):
    torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    before=time.perf_counter();start.record();value=fn();end.record();torch.cuda.synchronize()
    return value,float(start.elapsed_time(end)),1000*(time.perf_counter()-before)


class Profile:
    def __init__(self,runtime,source):
        self.runtime,self.source=runtime,source
        self.base=copy.deepcopy(source.state_dict())
        self.parameters=list(runtime.model.parameters())
        self.backup=[p.detach().clone() for p in self.parameters]
        self.owner=TensorOwnership();self.owner.watch_tree(self.base,'profile.base');self.owner.watch_tree(self.backup,'profile.parameters')
        self.counts={k:0 for k in PROFILE_COUNTS};self.rows=[];self.releases=[]

    def restore(self):
        for p,value in zip(self.parameters,self.backup):p.copy_(value)

    def measure(self,fn,label):
        self.runtime.budget(resources=True)
        value,ms,host=self.runtime.measure_call(fn)
        if not math.isfinite(ms) or not math.isfinite(host) or ms<0 or host<0:
            raise FloatingPointError('Nonfinite profile timing')
        row=dict(label,device_ms=ms,host_ms=host)
        self.rows.append(row);self.runtime.emit('profile_timing',row)
        return value

    def step(self,policy,block,repeat,qualify=False):
        self.restore();physical=[];opt=None
        try:
            opt=shadow(self.runtime.model,self.runtime.config,self.base,policy,physical if qualify else None)
            self.measure(opt.step,dict(scope='full_step',policy=policy,block=block,repeat=repeat))
            self.counts['shadow_optimizer_steps']+=1
        finally:
            if opt is not None:self.releases.append(close_shadow(opt))
            opt=None;self.restore()

    def qualify(self):
        native=reuse=None;physical=[]
        try:
            self.restore();native=shadow(self.runtime.model,self.runtime.config,self.base,'native',physical)
            native.step();self.counts['shadow_optimizer_steps']+=1
            native_parameters=streaming_digest(self.runtime.model.state_dict())
            self.restore();reuse=shadow(self.runtime.model,self.runtime.config,self.base,'reuse',physical)
            reuse.step();self.counts['shadow_optimizer_steps']+=1
            reused_parameters=streaming_digest(self.runtime.model.state_dict())
            report=compare(native,reuse,self.base,self.runtime.protocol['profile']['qualification'])
            self.counts['following_gradient_previews']+=len(report['actions'])*2
            report.update(physical=physical,parameter_update_hash_equal=native_parameters==reused_parameters)
            report['passed'] &= report['parameter_update_hash_equal'] and all(r['passed'] for r in physical) and len(physical)==8
            self.runtime.emit('qr_reuse_qualification_before_gate',report)
            if not report['passed']:raise FloatingPointError('QR reuse failed prospective qualification')
            return report
        finally:
            if native is not None:self.releases.append(close_shadow(native))
            if reuse is not None:self.releases.append(close_shadow(reuse))
            native=reuse=None;self.restore()

    def factor(self,parameter,axis,cov,q,policy,block,repeat):
        prepared=None
        if policy=='decomposition':
            order=torch.argsort((q.T@cov@q).diagonal(),descending=True)
            prepared=cov@q[:,order]
            fn=lambda:torch.linalg.qr(prepared)[0]
        else:fn=lambda:qr_basis(cov,q,policy)
        owner=TensorOwnership();value=None
        try:
            value=self.measure(fn,dict(scope='factor',parameter=parameter,axis=axis,dimension=len(q),policy=policy,block=block,repeat=repeat))
            owner.watch_tree(value,'factor.output');value=None
            self.counts['decomposition_calls' if policy=='decomposition' else 'factor_basis_calls']+=1
        finally:
            value=prepared=None;release=owner.report();owner.close();self.releases.append(release)
            if not release['all_registered_released']:raise RuntimeError('Factor output leaked')

    def aggregate(self,kind,block,repeat):
        # Covariance mutates private GG copies only. Projection uses borrowed
        # gradients/M/Q and returns no persistent outputs; it performs no update.
        states=[]
        if kind=='covariance':
            for group in self.source.matrix.param_groups:
                for p in group['params']:
                    state=self.source.matrix.state[p]
                    states.append((p,dict(GG=[x.clone() for x in state['GG']]),group))
        def atom():
            if kind=='covariance':
                for p,state,group in states:self.source.matrix._accumulate_covariance(p.grad,state,group)
            else:
                for group in self.source.matrix.param_groups:
                    for p in group['params']:
                        state=self.source.matrix.state[p]
                        a=self.source.matrix._project_with_basis(p.grad,state['Q'],group)
                        b=self.source.matrix._project_back_with_basis(state['exp_avg'],state['Q'],group)
                        owner.watch(a,'aggregate.projected_gradient');owner.watch(b,'aggregate.world_momentum')
                        del a,b
        owner=TensorOwnership();owner.watch_tree([s['GG'] for _,s,_ in states],'aggregate.privateGG')
        try:
            self.measure(atom,dict(scope='aggregate',policy=kind,block=block,repeat=repeat))
            self.counts[kind+'_endpoints']+=1
        finally:
            states.clear();release=owner.report();owner.close();self.releases.append(release)
            if not release['all_registered_released']:raise RuntimeError('Aggregate buffers leaked')

    def factors(self):
        for group in self.source.matrix.param_groups:
            for name,p in zip(group['param_names'],group['params']):
                state=self.source.matrix.state[p]
                for axis,(cov,q) in enumerate(zip(state['GG'],state['Q'])):yield name,axis,cov,q

    def run(self):
        qualification=self.qualify()
        for policy in ('ordinary','native','reuse'):self.step(policy,-1,0)
        for name,axis,cov,q in self.factors():
            for policy in ('native','reuse','decomposition'):self.factor(name,axis,cov,q,policy,-1,0)
        for atom in ('covariance','projection'):self.aggregate(atom,-1,0)
        for block in range(3):
            policies=('ordinary','native','reuse')
            if block==1:policies=policies[1:]+policies[:1]
            if block==2:policies=tuple(reversed(policies))
            for repeat in range(3):
                for policy in policies:self.step(policy,block,repeat)
            for name,axis,cov,q in self.factors():
                policies=('native','reuse','decomposition')
                if block==1:policies=policies[1:]+policies[:1]
                if block==2:policies=tuple(reversed(policies))
                for repeat in range(3):
                    for policy in policies:self.factor(name,axis,cov,q,policy,block,repeat)
            for repeat in range(3):
                for atom in (('covariance','projection') if block%2==0 else ('projection','covariance')):self.aggregate(atom,block,repeat)
        if self.counts!=PROFILE_COUNTS:raise RuntimeError('Frozen profile endpoint counts differ: '+str(self.counts))
        return dict(qualification=qualification,counts=dict(self.counts),timings=self.rows,
            releases=self.releases,source_age=999,scope='Fixed natural-state diagnostic; no new learning result or deployed throughput qualification')


@torch.no_grad()
def run_profile(runtime,source):
    def endpoint():
        return dict(model=streaming_digest(runtime.model.state_dict()),optimizer=streaming_digest(source.state_dict()),
            gradients=streaming_digest([p.grad for p in runtime.model.parameters()]),rng=streaming_digest(runtime.streams.capture()))
    before=endpoint();before_flags=flags();gradient_ids=[id(p.grad) for p in runtime.model.parameters()]
    profile=None;result=None;error=None;release=None
    try:
        profile=Profile(runtime,source)
        result=profile.run()
    except BaseException as exc:
        error=failure_record(exc)
    finally:
        if profile is not None:
            profile.restore();profile.base=None;profile.backup=None
            release=profile.owner.report();profile.owner.close()
        after=endpoint()
        cleanup=dict(before=before,after=after,nonmutation_pass=before==after,
            gradient_identity_pass=gradient_ids==[id(p.grad) for p in runtime.model.parameters()],
            flags_restored=before_flags==flags(),registered_release=release,error=error)
        runtime.emit('profile_cleanup',cleanup)
    if error is not None:raise RuntimeError('Bounded QR/profile failed: '+str(error))
    if not cleanup['nonmutation_pass'] or not cleanup['gradient_identity_pass'] or not cleanup['flags_restored'] or not release['all_registered_released']:
        raise RuntimeError('Profile source/flags/ownership restoration failed')
    result['cleanup']=cleanup
    return result
