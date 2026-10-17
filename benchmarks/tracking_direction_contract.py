"""Fresh fixed-cap direction comparison; the live Jacobi policy is unchanged."""
from pathlib import Path
from tracking_caps_contract import SNAPSHOT_CAP,PARAMETERS,HORIZON,WARMUP,ARMS,POLICIES
from tracking_caps_contract import recipe as old_recipe,protocol as old_protocol
SEED=154
PARENTS=(Path('/workspace/optimizer_replay_results/cifar_soap_tracking_caps_seed153_frozen_20261012'),)
DIRECTIONS=('jacobi','energy')

def recipe(base):
    c=old_recipe(base);c['seed']=SEED
    c['matched_comparison']['scope']='Fresh fixed live Jacobi state collection; read-only fixed-cap direction comparison, not learning'
    return c

def protocol():
    p=old_protocol();p['seed']=SEED
    p['expected'].update(counterfactual_complete_calls=288,operator_estimate_calls=128,exact_spectral_checks=2)
    p['implementation']['entry']='source/benchmarks/cifar_tracking_direction.py'
    t=p['tracking_diagnostic'];t.pop('caps')
    t.update(directions=list(DIRECTIONS),fixed_cap=.5,
        compiled_inputs='Two distinct plain direction-specific complete-call functions',
        reference='Validated eager-only common B; Jacobi gap-preconditioned or negative offdiag-energy gradient; same raw half-step scale and cap.5',
        raw_scaling='Both raw generators use .5 times antisymmetrized omega; cap is an upper bound, not forced movement when raw demand is below .5',
        directional_metrics=['2 sum B_ij gap_ij A_ij','cosine to negative offdiag-energy gradient','predicted fractional energy decrease'],
        counterfactual_call_limit=350,norm_call_limit=160)
    t['norm_estimator']['independent_check'].update(count=2,directions=list(DIRECTIONS))
    p['full_moment_diagnostic'].update(policies=['qr_permutationV','qr_carryV',*DIRECTIONS],
        preview='Actual M transported at highest on common C,Q; carriedV for both directions; same QRbasis carry/permutation controls; no future gradient/cross-history')
    p['artifacts']['probe']['purpose']='Common actual C,Q,M,V fixed-cap direction/slope/tracking/geometry/cost/operator and same-QRbasis V diagnostic'
    p['bound_rationale']='Fresh seed154,690liveupdates/16factors/8fullM transactions;288compiledCF/128powerlower/2exact2048SVD. Prior153197.825s/7.608GiB;1800/1860s,24GiB/512MiB initialCPU/100MiBcompact/512MiBcache,expected7GiB probeextra/conservative16GiBguard. Original live policy/gates fixed; no retry or controller.'
    return p

def validate(config,declared,base):
    if config!=recipe(base) or declared!=protocol():raise ValueError('Undeclared direction diagnostic delta')
