"""Prospective matched learning, fixed observations and independent seed replication."""
from pathlib import Path
import copy
from hybrid_training_contract import expected_counts, protocol as pilot_protocol, recipe as pilot_recipe
PARENT=Path('/workspace/optimizer_replay_results/cifar_soap_hybrid_trainer_frozen_20261011')
HORIZON=3500
WARMUP=245
SEEDS=(142,143)
ORDERS={142:('hybrid20','warm20','qr20'),143:('qr20','warm20','hybrid20')}
PROBES=(21,1021,2021,3481)
EVALUATIONS=(1000,1750,2625,3500)

def recipe(base,seed=142):
    if seed not in SEEDS:raise ValueError('Undeclared seed')
    c=pilot_recipe(base);c.update(seed=seed,max_wall_seconds=2400)
    c['matched_comparison']=dict(arms=['qr20','warm20','hybrid20'],steps_per_arm=HORIZON,
        execution_order=list(ORDERS[seed]),scope='Fixed-step paired learning; no statistical or wall qualification')
    return c

def protocol(seed=142):
    if seed not in SEEDS:raise ValueError('Undeclared seed')
    p=copy.deepcopy(pilot_protocol());p.update(seed=seed,execution_order=list(ORDERS[seed]),
        steps_per_arm=HORIZON,maximum_optimizer_updates=11260,probe_calls=list(PROBES),
        sampled_batch_calls=[21,401,1001,1751,2021,2626,3481],
        prefix_hashes=[1,20,21,200,201,1000,1750,2625,3500],
        evaluation_calls=list(EVALUATIONS),evaluation_policy='Existing eager BF16, eval mode, fixed validation/train panel',
        evaluation_transaction='Hash weights/gradients/optimizer, preserve every module flag and all RNG streams, restore finally; no mutation')
    p['expected']['primary']={a:expected_counts(HORIZON,'qr' if a=='qr20' else 'warm',20,0 if a=='qr20' else 200) for a in p['arms']}
    warm=expected_counts(WARMUP,'warm',20,200)['research_warm_refreshes']+expected_counts(HORIZON,'warm',20,200)['research_warm_refreshes']
    p['expected'].update(hybrid_kernel_calls=warm*8,hybrid_calls_by_dimension={'1536':warm*5,'4608':warm,'6144':warm*2},diagnostic_reference_calls=len(PROBES)*8,
        evaluation_observations_per_arm=len(EVALUATIONS))
    p['gates']['no_speed_or_learning_qualification']=False
    p['gates'].update(no_statistical_equivalence_or_speed_qualification=True,exact_disposable_primary_scalar_prefix=True,
        evaluation_model_optimizer_gradient_nonmutation=True,evaluation_rng_flags_restored=True)
    p['bounds'].update(internal_wall_seconds=2400,external_watchdog_seconds=2460,maximum_evidence_bytes=100*2**20)
    p['artifacts']['initial_snapshot']['purpose']='Fresh smoke/disposable/preflight/primary branches, exact paired resets'
    p['artifacts']['compiler_cache'].update(expected_bytes=150*2**20,retention='Through learning comparison and independent replication review; no model state')
    p['artifacts']['expected_compact_bytes']=30*2**20
    p['timing'].update(diagnostic_calls_excluded=list(PROBES),primary_within_lr_warmup=False,
        evaluations_and_guards_charged=True,interpretation='Exploratory serialized optimizer cost; no throughput/time-to-quality claim')
    p['implementation']['entry']='source/benchmarks/cifar_hybrid_learning.py'
    p['decision']='Report all four paired validation observations. A completed numerical run motivates fresh seed143 replication without selecting by gap sign. Longer10000 and width2048 remain distinct tasks.'
    p['bound_rationale']='Prior2020 calls190.517s/2.919GiB/43.839MiB cache. Linear11260-call projection1062s;2400s includes evaluation/hash/compile overhead.24GiB/256MiB unchanged;100MiB compact evidence accommodates longer scalar trajectories. No after-launch loosening.'
    return p

def validate(config,declared,base):
    seed=declared.get('seed')
    if seed not in SEEDS or config!=recipe(base,seed) or declared!=protocol(seed):
        raise ValueError('Undeclared hybrid learning recipe/protocol delta')
