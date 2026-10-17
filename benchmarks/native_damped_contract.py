"""Explicit live damped-policy pilot; historical source and defaults unchanged."""
from pathlib import Path
from native_residual_named_contract import SNAPSHOT_CAP,PARAMETERS,HORIZON,WARMUP
from native_residual_named_contract import recipe as old_recipe,protocol as old_protocol
from hybrid_training_contract import expected_counts
SEED=158
PARENTS=(Path('/workspace/optimizer_replay_results/cifar_soap_tracking_curvature_seed157_frozen_20261012'),)
ARMS=('highest_residual20','damped20','qr20')
POLICIES=ARMS[:2]
ORDER=ARMS

def recipe(base):
    c=old_recipe(base);c['seed']=SEED
    c['matched_comparison'].update(arms=list(ARMS),execution_order=list(ORDER),
        scope='Live damped20/Jacobi20/QR20 native420 feasibility and actual-M/V diagnostics, not learning qualification')
    return c

def protocol():
    p=old_protocol();p.update(seed=SEED,arms=list(ARMS),execution_order=list(ORDER),empty_preflight_backward_calls=3,maximum_optimizer_updates=2020)
    for key in ('ownership_fix','compiler_routing_fix'):p.pop(key,None)
    for phase,calls in (('primary',HORIZON),('disposable',WARMUP)):
        p['expected'][phase]={a:expected_counts(calls,'qr' if a=='qr20' else 'warm',20,0 if a=='qr20' else 200) for a in ARMS}
    p['expected'].update(hybrid_kernel_calls=464,hybrid_calls_by_dimension={'2048':290,'6144':58,'8192':116},
        policy_calls_by_dimension={a:{'2048':145,'6144':29,'8192':58} for a in POLICIES},
        diagnostic_reference_calls=32,full_moment_transactions=24,counterfactual_basis_calls=96)
    p['precision']['policies']={a:dict(prefix='highest',candidate='BF16 Q@A',retraction='BF16 residual correction, highest FP32 Gram, cubic2') for a in POLICIES}
    p['precision']['reference']='Each actual live policy versus its matching eager prefix/retraction; Jacobi compiler path preserved, damped qualified pure whole entry'
    p['implementation'].update(entry='source/benchmarks/cifar_native_damped.py',
        diagnostic_adapter='Weak instance hook on all three source arms; original ResearchSOAP step/M/V/QR retained',
        compiler_adapter='Original Jacobi separate named prefix/ret; qualified pure whole damped entry; default fullgraph/default recompile limit')
    p['full_moment_diagnostic'].update(source_arms=list(ARMS),policies=['qr_permutationV','qr_carryV',*POLICIES],
        stage_attribution='Validated matching eager-only Jacobi/damped stages; no compiled intermediate observer',
        preview='Full actual M/V on each source history; Jacobi/damped carryV and sameQRbasis permutation/carryV; not future gradients or identical histories across arms')
    p['live_damped_policy']=dict(direction='Negative offdiag-energy gradient',fixed_cap=.5,
        scalar='Diagonal-only curvature raw scale min(cap_scale,clip(-slope_raw/h_raw,0,1)), numerical floor1e-24',
        persistence='No added state, EMA or variance history',extra_matrix_products=0,
        parent='Qualified common-state seed157; not a full-Hessian/descent guarantee',
        actual_policy_parity=True,original_jacobi_compiler_path_unchanged=True)
    p['diagnostic_admission']=dict(normalized_gram=.005,absolute_gram_fro=.05,own_eager_basis=1e-4,physical_m=.01,
        policy='Both declared warm counterfactuals and QR geometry must pass; all failures retained, no relaxation/retry')
    p['artifacts']['probe'].update(purpose='Actual live-policy parity and read-only C/Q/M/V tracking/geometry/physical-M/inherited-adaptive transactions on all source arms',
        expected_max_extra_bytes=24*8192**2*4+16*8192*2048*4)
    p['full_moment_diagnostic']['expected_extra_bytes']=p['artifacts']['probe']['expected_max_extra_bytes']
    p['bound_rationale']='Fresh3arm420/245 pilot2020updates/3emptybackwards/32actualfactorchecks/24fullM transactions/96CF basis calls. Prior152362.451s/8.466GiB,157212.190s/8.356GiB.1800/1860s,24GiB,512MiBinitialCPU,100MiBcompact,512MiBcache;expected7GiBprobeextra/conservative16GiBguard. No checkpoints/tensors/retries or learning/default promotion.'
    return p

def validate(config,declared,base):
    if config!=recipe(base) or declared!=protocol():raise ValueError('Undeclared live damped-policy pilot delta')
