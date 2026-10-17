"""Fresh post-warmup matched learning; qualified pilot and defaults preserved."""
from pathlib import Path
from native_damped_contract import ARMS,POLICIES,SNAPSHOT_CAP,PARAMETERS,WARMUP
from native_damped_contract import recipe as pilot_recipe,protocol as pilot_protocol
from hybrid_training_contract import expected_counts
SEED=159
HORIZON=3500
ORDER=('damped20','qr20','highest_residual20')
PROBES=(1021,3481)
EVALUATIONS=(1000,2000,3000,3500)
PARENTS=(Path('/workspace/optimizer_replay_results/cifar_soap_native_damped_seed158_frozen_20261012'),)

def recipe(base):
    c=pilot_recipe(base);c.update(seed=SEED,max_wall_seconds=3600)
    c['matched_comparison'].update(steps_per_arm=HORIZON,execution_order=list(ORDER),
        scope='Fresh single-seed within-width2048 postwarmup matched learning, no convergence/statistical/wall equivalence')
    return c

def protocol():
    p=pilot_protocol();p.update(seed=SEED,steps_per_arm=HORIZON,execution_order=list(ORDER),maximum_optimizer_updates=11260,
        probe_calls=list(PROBES),evaluation_calls=list(EVALUATIONS),sampled_batch_calls=[21,401,1001,1021,2001,3001,3481],
        prefix_hashes=[1,20,21,200,201,1000,2000,3000,3500],
        evaluation_policy='Existing eager BF16, eval mode, fixed validation/train panel',
        evaluation_transaction='Hash weights/gradients/optimizer; preserve every module flag and all RNG streams; restore finally, no mutation')
    p['expected']['primary']={a:expected_counts(HORIZON,'qr' if a=='qr20' else 'warm',20,0 if a=='qr20' else 200) for a in ARMS}
    warm=sum(expected_counts(n,'warm',20,200)['research_warm_refreshes'] for n in (WARMUP,HORIZON))
    p['expected'].update(hybrid_kernel_calls=warm*8*len(POLICIES),
        hybrid_calls_by_dimension={'2048':warm*5*2,'6144':warm*2,'8192':warm*2*2},
        policy_calls_by_dimension={a:{'2048':warm*5,'6144':warm,'8192':warm*2} for a in POLICIES},
        diagnostic_reference_calls=len(PROBES)*8*len(POLICIES),full_moment_transactions=len(PROBES)*4*len(ARMS),
        counterfactual_basis_calls=len(PROBES)*4*len(ARMS)*2*len(POLICIES),evaluation_observations_per_arm=len(EVALUATIONS))
    p['full_moment_diagnostic']['calls']=list(PROBES)
    p['bounds'].update(internal_wall_seconds=3600,external_watchdog_seconds=3660)
    p['gates']['no_speed_or_learning_qualification']=False
    p['gates'].update(no_statistical_equivalence_or_speed_qualification=True,exact_disposable_primary_scalar_prefix=True,
        evaluation_model_optimizer_gradient_nonmutation=True,evaluation_rng_flags_restored=True)
    p['timing'].update(diagnostic_calls_excluded=list(PROBES),primary_within_lr_warmup=False,evaluations_and_guards_charged=True,
        interpretation='Descriptive serialized optimizer cost. Cadence estimates separate from measured wall; no throughput/time-to-quality qualification')
    p['artifacts']['compiler_cache'].update(expected_bytes=250*2**20,retention='Through matched-learning and replication review; arithmetic provenance, no model state')
    p['artifacts']['expected_compact_bytes']=40*2**20
    p['implementation']['entry']='source/benchmarks/cifar_damped_learning.py'
    p['decision']='Retain all four postwarmup validation observations and paired source-state regressions. Single-seed numerical completion is not statistical equivalence or convergence. Analyze before any fresh independently seeded confirmation; no queue/retry/default promotion.'
    p['bound_rationale']='Completed seed1582020updates389.844s/8.341GiB; proportional11260 projection2173s overestimates repeated cold/probe overhead. Prior width2048 matched3500 supports3600internal/3660external. Fixed24GiB/512MiBinitialCPU/100MiBcompact/512MiBcache;96CFbasis and24fullM transactions at1021/3481, no new persistent state or checkpoints. No after-launch loosening.'
    return p

def validate(config,declared,base):
    if config!=recipe(base) or declared!=protocol():raise ValueError('Undeclared damped matched-learning delta')
