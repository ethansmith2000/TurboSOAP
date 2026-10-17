"""Common-state BF16projection observations and paired complete ordinary-step timers."""
import math
import torch
from qr_reuse_profile_core import Profile as PreviousProfile,flags
from qr_reuse_profile_kernel import shadow,close_shadow
from projection_profile_kernel import install
from llm_sparse_screen_kernel import common_geometry_guard
from hybrid_training_kernel import precision
from matched_training_runtime import streaming_digest,failure_record
from tensor_ownership import TensorOwnership
from projection_profile_contract import PROFILE_COUNTS


def compare_updates(native,candidate,gates):
    if len(native)!=4 or len(candidate)!=4:raise RuntimeError('Four actual adaptive updates required')
    rows=[]
    with precision('highest'):
        for i,(a,b) in enumerate(zip(native,candidate)):
            relative=float((a-b).norm()/a.norm().clamp_min(1e-12))
            cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-24))
            rows.append(dict(matrix_index=i,shape=list(a.shape),relative=relative,cosine=cosine,
                passed=math.isfinite(relative) and math.isfinite(cosine) and relative<=gates['adaptive_update_relative'] and cosine>=gates['adaptive_update_cosine']))
    return rows


class Profile(PreviousProfile):
    def __init__(self,runtime,source):
        super().__init__(runtime,source);self.counts={k:0 for k in PROFILE_COUNTS}

    def make_shadow(self,policy,qualification=False,physical=None,outputs=None):
        # Qualification crosses the original native QR refresh, so the strict
        # highest transport path is checked with the candidate adapter installed.
        opt=shadow(self.runtime.model,self.runtime.config,self.base,'native' if qualification else 'ordinary',physical)
        try:install(opt.matrix,policy,outputs)
        except BaseException:
            self.releases.append(close_shadow(opt));raise
        return opt

    def qualify(self):
        opt=None;physical=[];outputs={'native':[],'bf16':[]};digests={};factor_rows=[]
        owner=TensorOwnership();release=None
        try:
            for policy in ('native','bf16'):
                self.restore();error=None
                try:
                    physical_start=len(physical)
                    opt=self.make_shadow(policy,True,physical,outputs[policy])
                    opt.step();self.counts['shadow_optimizer_steps']+=1
                    for row in physical[physical_start:]:row['projection_policy']=policy
                    factor_rows.extend(dict(r,policy=policy) for r in common_geometry_guard(opt,self.runtime.model))
                    digests[policy]=dict(covariance=streaming_digest([s['GG'] for s in opt.matrix.state.values()]),
                        basis=streaming_digest([s['Q'] for s in opt.matrix.state.values()]))
                except BaseException as exc:
                    error=failure_record(exc)
                finally:
                    owner.watch_tree(outputs[policy],policy+'.adaptive_updates')
                    if opt is not None:self.releases.append(close_shadow(opt))
                    opt=None;self.restore()
                if error is not None:raise RuntimeError('Projection qualification failed: '+str(error))
            actions=compare_updates(outputs['native'],outputs['bf16'],self.runtime.protocol['profile']['qualification'])
            self.counts['adaptive_update_comparisons']+=len(actions)
            passed=(len(factor_rows)==16 and len(physical)==8 and all(v['passed'] for v in physical+actions) and digests['native']==digests['bf16'])
            report=dict(actions=actions,physical=physical,factors=factor_rows,unchanged_covariance_basis=digests['native']==digests['bf16'],digests=digests,passed=passed,
                meaning='Current adaptive world-update before parameter rounding with common retained gradient; no own-history or learning/default acceptance')
            self.runtime.emit('projection_qualification_before_gate',report)
            if not passed:raise FloatingPointError('BF16ordinary-projection qualification failed frozen gates')
            return report
        finally:
            if opt is not None:self.releases.append(close_shadow(opt))
            opt=None
            for values in outputs.values():values.clear()
            outputs.clear();self.restore()
            release=owner.report() if owner._tensors else dict(all_registered_released=True,tracked_tensors=0,scope='No adaptive update outputs produced')
            owner.close();self.releases.append(release)
            if not release['all_registered_released']:raise RuntimeError('Adaptive update outputs leaked')

    def step(self,policy,block,repeat):
        opt=None;self.restore()
        try:
            opt=self.make_shadow(policy)
            self.measure(opt.step,dict(scope='full_step',policy=policy,block=block,repeat=repeat))
            self.counts['shadow_optimizer_steps']+=1
        finally:
            if opt is not None:self.releases.append(close_shadow(opt))
            opt=None;self.restore()

    def aggregate(self,policy,block,repeat):
        # Private optimizer supplies identical untouched state/layout; copies and
        # binding lie outside the timer. All BF16casts lie inside the timer.
        opt=None;owner=TensorOwnership()
        def atom():
            for group in opt.matrix.param_groups:
                for p in group['params']:
                    s=opt.matrix.state[p]
                    a=opt.matrix._project_with_basis(p.grad,s['Q'],group)
                    b=opt.matrix._project_back_with_basis(s['exp_avg'],s['Q'],group)
                    owner.watch(a,'projection.gradient');owner.watch(b,'projection.momentum')
                    del a,b
        try:
            opt=self.make_shadow(policy)
            self.measure(atom,dict(scope='aggregate_projection',policy=policy,block=block,repeat=repeat))
            self.counts['projection_endpoints']+=1
        finally:
            if opt is not None:self.releases.append(close_shadow(opt))
            opt=None;release=owner.report();owner.close();self.releases.append(release)
            if not release['all_registered_released']:raise RuntimeError('Projection endpoint outputs leaked')

    def run(self):
        qualification=self.qualify()
        for policy in ('native','bf16'):self.step(policy,-1,0)
        for policy in ('native','bf16'):self.aggregate(policy,-1,0)
        for block in range(3):
            policies=('native','bf16') if block%2==0 else ('bf16','native')
            for repeat in range(3):
                for policy in policies:self.step(policy,block,repeat)
            for repeat in range(3):
                for policy in policies:self.aggregate(policy,block,repeat)
        if self.counts!=PROFILE_COUNTS:raise RuntimeError('Frozen projection profile counts differ: '+str(self.counts))
        return dict(qualification=qualification,counts=dict(self.counts),timings=self.rows,releases=self.releases,source_age=999,
            scope='Read-only common-state ordinary-projection fidelity/cost; no candidate source update, own history or learning/default promotion')


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
    if error is not None:raise RuntimeError('Bounded projection profile failed: '+str(error))
    if not cleanup['nonmutation_pass'] or not cleanup['gradient_identity_pass'] or not cleanup['flags_restored'] or not release['all_registered_released'] or not all(r['all_registered_released'] for r in shadow_releases):
        raise RuntimeError('Projection source/flags/ownership restoration failed')
    result['cleanup']=cleanup
    return result
