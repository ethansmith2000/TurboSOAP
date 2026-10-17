"""Native-only complete-step/atom profile; candidate rejection cannot block its scope."""
import torch
from qr_reuse_profile_core import Profile as PreviousProfile,flags
from qr_reuse_profile_kernel import shadow,close_shadow
from llm_sparse_screen_kernel import common_geometry_guard
from matched_training_runtime import streaming_digest,failure_record
from native_profile_contract import PROFILE_COUNTS

class Profile(PreviousProfile):
    def __init__(self,runtime,source):
        super().__init__(runtime,source);self.counts={k:0 for k in PROFILE_COUNTS}

    def qualify(self):
        opt=None;physical=[]
        try:
            self.restore();opt=shadow(self.runtime.model,self.runtime.config,self.base,'native',physical)
            opt.step();self.counts['shadow_optimizer_steps']+=1
            rows=common_geometry_guard(opt,self.runtime.model)
            report=dict(physical=physical,factors=rows,passed=len(physical)==4 and len(rows)==8 and all(r['passed'] for r in physical))
            self.runtime.emit('native_profile_qualification_before_gate',report)
            if not report['passed']:raise FloatingPointError('Native profile failed original qualification')
            return report
        finally:
            if opt is not None:self.releases.append(close_shadow(opt))
            opt=None;self.restore()

    def run(self):
        qualification=self.qualify()
        for policy in ('ordinary','native'):self.step(policy,-1,0)
        for name,axis,cov,q in self.factors():
            for policy in ('native','decomposition'):self.factor(name,axis,cov,q,policy,-1,0)
        for atom in ('covariance','projection'):self.aggregate(atom,-1,0)
        for block in range(3):
            policies=('ordinary','native') if block%2==0 else ('native','ordinary')
            for repeat in range(3):
                for policy in policies:self.step(policy,block,repeat)
            for name,axis,cov,q in self.factors():
                policies=('native','decomposition') if block%2==0 else ('decomposition','native')
                for repeat in range(3):
                    for policy in policies:self.factor(name,axis,cov,q,policy,block,repeat)
            for repeat in range(3):
                for atom in (('covariance','projection') if block%2==0 else ('projection','covariance')):self.aggregate(atom,block,repeat)
        if self.counts!=PROFILE_COUNTS:raise RuntimeError('Frozen native profile counts differ: '+str(self.counts))
        return dict(qualification=qualification,counts=dict(self.counts),timings=self.rows,releases=self.releases,source_age=999,
            scope='Unchanged native optimizer cost at fixed natural state; no candidate/speedup/learning promotion')

@torch.no_grad()
def run_profile(runtime,source):
    def endpoint():
        return dict(model=streaming_digest(runtime.model.state_dict()),optimizer=streaming_digest(source.state_dict()),
            gradients=streaming_digest([p.grad for p in runtime.model.parameters()]),rng=streaming_digest(runtime.streams.capture()))
    before=endpoint();before_flags=flags();gradient_ids=[id(p.grad) for p in runtime.model.parameters()]
    profile=None;result=None;error=None;release=None;shadow_releases=[]
    try:
        profile=Profile(runtime,source);result=profile.run()
    except BaseException as exc:error=failure_record(exc)
    finally:
        if profile is not None:
            profile.restore();profile.base=None;profile.backup=None
            release=profile.owner.report();profile.owner.close();shadow_releases=profile.releases
        after=endpoint();cleanup=dict(before=before,after=after,nonmutation_pass=before==after,
            gradient_identity_pass=gradient_ids==[id(p.grad) for p in runtime.model.parameters()],flags_restored=before_flags==flags(),
            registered_release=release,endpoint_releases=shadow_releases,error=error)
        runtime.emit('profile_cleanup',cleanup)
    if error is not None:raise RuntimeError('Bounded native profile failed: '+str(error))
    if not cleanup['nonmutation_pass'] or not cleanup['gradient_identity_pass'] or not cleanup['flags_restored'] or not release['all_registered_released'] or not all(r['all_registered_released'] for r in shadow_releases):
        raise RuntimeError('Native profile source/flags/ownership restoration failed')
    result['cleanup']=cleanup
    return result
