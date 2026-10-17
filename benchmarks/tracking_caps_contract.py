"""Prospective common-state step-bound diagnostic; live cap and V unchanged."""
from pathlib import Path
from native_residual_named_contract import SNAPSHOT_CAP,PARAMETERS,HORIZON,WARMUP
from native_residual_named_contract import recipe as previous_recipe,protocol as previous_protocol
SEED=153
PARENTS=(Path('/workspace/optimizer_replay_results/cifar_soap_native_residual_named_seed152_frozen_20261012'),)
ARMS=('highest_residual20',)
POLICIES=ARMS
CAPS=(.5,2.,8.)

def recipe(base):
    c=previous_recipe(base);c.update(seed=SEED,max_wall_seconds=1800)
    c['matched_comparison']=dict(arms=list(ARMS),steps_per_arm=420,execution_order=list(ARMS),
        scope='Fresh fixed live policy state collection; fixed counterfactual step caps and QR V controls, not matched learning')
    return c

def protocol():
    p=previous_protocol();p.update(seed=SEED,arms=list(ARMS),execution_order=list(ARMS),empty_preflight_backward_calls=1,
        maximum_optimizer_updates=690)
    for phase in ('primary','disposable'):p['expected'][phase]={ARMS[0]:p['expected'][phase][ARMS[0]]}
    p['expected'].update(hybrid_kernel_calls=232,hybrid_calls_by_dimension={'2048':145,'6144':29,'8192':58},
        policy_calls_by_dimension={ARMS[0]:{'2048':145,'6144':29,'8192':58}},diagnostic_reference_calls=16,full_moment_transactions=8,
        counterfactual_complete_calls=432,operator_estimate_calls=64,exact_spectral_checks=1)
    p['implementation']['entry']='source/benchmarks/cifar_tracking_caps.py'
    p['implementation']['diagnostic_adapter']='Weak instance callable; declared disposable CF prewarm at21/221, read-only primary CF at21/281'
    p['tracking_diagnostic']=dict(caps=list(CAPS),live_cap=.5,candidate_compute='highest FP32 prefix/Gram,BF16 Q@A and residual corrections',
        scope='Counterfactuals only; no live policy/controller/EMA/V change',
        fullgraph_complete_entries=True,compiled_inputs='three distinct plain cap-specific complete-call functions',
        reference='One highest eager raw generator per common C,Q; cap-only scaling then BF16 candidate/residual retraction',
        disposable_prewarm_calls=[21,221],timing_warmup_calls=2,timing_samples=5,
        admitted_geometry=dict(normalized_gram=.005,absolute_gram_fro=.05),physical_m_limit=.01,
        invalid_counterfactual='Retain rejection/metrics; exclude from momentum/adaptive admission, never apply to live state',
        norm_estimator=dict(iterations=8,deterministic_vectors=2,kind='power lower estimate, never safety certificate/controller',
            timings=3,independent_check=dict(parameter='blocks.0.attn.to_out.weight',axis=0,source_step=20,dimension=2048,driver='gesvd',count=1)),
        variance_controls=['QRbasis+QRpermutedV','sameQRbasis+carriedV'],
        covariance_metrics=['absolute offdiag','offdiag/covariance Fro','after/before ratio'],
        partial_safe_journal=True,counterfactual_call_limit=500,norm_call_limit=80)
    p['full_moment_diagnostic'].update(source_arms=list(ARMS),policies=['qr_permutationV','qr_carryV','cap05','cap2','cap8'],
        preview='Actual M transported at highest on common C,Q; fixed carriedV for caps; QR carry/permutation on identical QRbasis; no future gradient/cross-history')
    p['artifacts']['probe'].update(purpose='Common actual C,Q,M,V cap-only tracking/geometry/cost/operator and same-QRbasis V diagnostic',
        expected_max_extra_bytes=24*8192**2*4+16*8192*2048*4)
    p['bound_rationale']='Fresh single live source690updates;8transactions/16factors,432completecounterfactual calls/64power probes/one2048SVD. Prior152362s/8.47GiB;1800/1860s,24GiB/512MiB initialCPU/100MiB compact/512MiB cache,expectedprobeextra7GiB and conservative16GiBguard. No changed gates for live policy; stricter absoluteGram bound for CF admission.'
    return p

def validate(config,declared,base):
    if config!=recipe(base) or declared!=protocol():raise ValueError('Undeclared tracking/cap diagnostic delta')
