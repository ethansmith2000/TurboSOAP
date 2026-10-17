"""Three-age read-only approximation study, distinct from the frozen182screen."""
import math
from qr_reuse_profile_core import Profile as PreviousProfile
from qr_reuse_profile_kernel import shadow,close_shadow,compare
from matched_training_runtime import streaming_digest
from reuse_observation_contract import PROFILE_COUNTS

class Profile(PreviousProfile):
    def qualify(self):
        native=reuse=None;physical=[]
        try:
            self.restore();native=shadow(self.runtime.model,self.runtime.config,self.base,'native',physical)
            native.step();self.counts['shadow_optimizer_steps']+=1
            n_hash=streaming_digest(self.runtime.model.state_dict())
            self.restore();reuse=shadow(self.runtime.model,self.runtime.config,self.base,'reuse',physical)
            reuse.step();self.counts['shadow_optimizer_steps']+=1
            r_hash=streaming_digest(self.runtime.model.state_dict())
            report=compare(native,reuse,self.base,self.runtime.protocol['profile']['qualification'])
            self.counts['following_gradient_previews']+=len(report['actions'])*2
            legacy=report['passed']
            factors=report['factors'];actions=report['actions']
            geometry=len(factors)==8 and all(math.isfinite(r['absolute_gram_fro']) and r['absolute_gram_fro']<=.05 and r['normalized_gram']<=.005 for r in factors)
            finite=len(actions)==4 and all(math.isfinite(r['relative']) and math.isfinite(r['cosine']) for r in actions) and all(math.isfinite(r['order_score_band']) for r in factors)
            safe=geometry and finite and len(physical)==8 and all(r['passed'] for r in physical) and n_hash==r_hash
            legacy=legacy and len(physical)==8 and all(r['passed'] for r in physical) and n_hash==r_hash
            report.update(legacy_fidelity_pass=legacy,physical=physical,parameter_update_hash_equal=n_hash==r_hash,
                passed=safe,meaning='passed means read-only geometry/M/finite/source checks only; not adaptive fidelity acceptance or passing182')
            self.runtime.emit('reuse_observation_before_geometry_gate',report)
            if not safe:raise FloatingPointError('Original read-only geometry/M/finite guard failed')
            return report
        finally:
            if native is not None:self.releases.append(close_shadow(native))
            if reuse is not None:self.releases.append(close_shadow(reuse))
            native=reuse=None;self.restore()

import torch
from qr_reuse_profile_core import flags
from matched_training_runtime import failure_record

@torch.no_grad()
def run_profile(runtime,source,timed=True):
    def endpoint():
        return dict(model=streaming_digest(runtime.model.state_dict()),optimizer=streaming_digest(source.state_dict()),
            gradients=streaming_digest([p.grad for p in runtime.model.parameters()]),rng=streaming_digest(runtime.streams.capture()))
    before=endpoint();before_flags=flags();gradient_ids=[id(p.grad) for p in runtime.model.parameters()]
    profile=None;result=None;error=None;release=None;shadow_releases=[]
    try:
        profile=Profile(runtime,source)
        if timed:result=profile.run()
        else:
            qualification=profile.qualify()
            result=dict(qualification=qualification,counts=dict(profile.counts),timings=[],releases=profile.releases)
        result['source_call']=next(iter(source.matrix.state.values()))['step']+1
        result['scope']='Read-only common native-history observation, not candidate own-history learning or acceptance'
    except BaseException as exc:error=failure_record(exc)
    finally:
        if profile is not None:
            profile.restore();profile.base=None;profile.backup=None
            release=profile.owner.report();profile.owner.close();shadow_releases=profile.releases
        after=endpoint();cleanup=dict(before=before,after=after,nonmutation_pass=before==after,
            gradient_identity_pass=gradient_ids==[id(p.grad) for p in runtime.model.parameters()],flags_restored=before_flags==flags(),
            registered_release=release,endpoint_releases=shadow_releases,error=error)
        runtime.emit('profile_cleanup',cleanup)
    if error is not None:raise RuntimeError('Bounded reuse observation failed: '+str(error))
    if not cleanup['nonmutation_pass'] or not cleanup['gradient_identity_pass'] or not cleanup['flags_restored'] or not release['all_registered_released'] or not all(r['all_registered_released'] for r in shadow_releases):
        raise RuntimeError('Reuse observation source/flags/ownership restoration failed')
    result['cleanup']=cleanup
    return result
