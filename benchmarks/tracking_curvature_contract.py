"""Fresh native-state damping diagnostic; live Jacobi remains fixed."""
from pathlib import Path
from tracking_direction_contract import SNAPSHOT_CAP,PARAMETERS,HORIZON,WARMUP,ARMS,POLICIES
from tracking_direction_contract import recipe as old_recipe,protocol as old_protocol
SEED=157
PARENTS=(Path('/workspace/optimizer_replay_results/cifar_soap_tracking_direction_seed154_frozen_20261012'),)
DIRECTIONS=('jacobi','energy','damped')

def recipe(base):
    c=old_recipe(base);c['seed']=SEED
    c['matched_comparison']['scope']='Fresh fixed live Jacobi state collection; read-only diagonal-curvature energy damping comparison, not learning'
    return c

def protocol():
    p=old_protocol();p['seed']=SEED
    p['expected'].update(counterfactual_complete_calls=432,operator_estimate_calls=192,exact_spectral_checks=3)
    p['implementation']['entry']='source/benchmarks/cifar_tracking_curvature.py'
    t=p['tracking_diagnostic'];t.update(directions=list(DIRECTIONS),
        compiled_inputs='Three distinct plain direction-specific complete-call functions',
        reference='Validated eager-only common B; unchanged Jacobi and energy anchors plus capped energy multiplied by diagonal-model alpha',
        counterfactual_call_limit=500,norm_call_limit=240,
        damping=dict(slope='2 sum(B*gap*A)',curvature='2 sum((gap*A)^2)',
            alpha='clip(-slope/h_diag,0,1) for capped A',
            equivalent_raw_scale='min(cap_scale,clip(-slope_raw/h_raw,0,1))',
            numerical_floor=1e-24,equivalence_scope='Away from numerical floors; zero/degenerate cases separately tested',scope='Diagonal-only surrogate, not full Hessian or guaranteed finite descent',
            added_operations='Elementwise products and scalar reductions only; no extra matrix products or persistent state'))
    t['norm_estimator']['independent_check'].update(count=3,directions=list(DIRECTIONS))
    p['full_moment_diagnostic'].update(policies=['qr_permutationV','qr_carryV',*DIRECTIONS],
        preview='Actual M transported at highest on common C,Q; carriedV for all directions; same QRbasis carry/permutation controls; no future gradient/cross-history')
    p['artifacts']['probe']['purpose']='Actual C,Q,M,V diagonal-curvature damping, finite tracking, geometry, inherited adaptive previews and complete costs'
    p['bound_rationale']='Fresh seed157:690 live updates/16 factors/8 fullM transactions;432 CF/192 powerlower/3 exact2048 checks. Prior154193.399s/8.343GiB.1800/1860s,24GiB peak,512MiB initialCPU,100MiBcompact,512MiBcache,expected7GiB probeextra/conservative16GiB guard. No changed live policy/gates, extra matmul, persistent EMA or retry.'
    return p

def validate(config,declared,base):
    if config!=recipe(base) or declared!=protocol():raise ValueError('Undeclared curvature diagnostic delta')
