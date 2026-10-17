"""Independent predeclared seed160 confirmation; previous evidence immutable."""
from pathlib import Path
from damped_learning_contract import ARMS,POLICIES,SNAPSHOT_CAP,PARAMETERS,WARMUP,HORIZON,PROBES,EVALUATIONS
from damped_learning_contract import recipe as previous_recipe,protocol as previous_protocol
SEED=160
ORDER=('highest_residual20','qr20','damped20')
PARENTS=(Path('/workspace/optimizer_replay_results/cifar_soap_damped_learning_seed159_frozen_20261012'),)

def recipe(base):
    c=previous_recipe(base);c['seed']=SEED
    c['matched_comparison']['execution_order']=list(ORDER)
    return c

def protocol():
    p=previous_protocol();p.update(seed=SEED,execution_order=list(ORDER))
    p['implementation']['entry']='source/benchmarks/cifar_damped_confirmation.py'
    p['decision']='Independent seed160 with opposite armorder, chosen irrespective of seed159 gap sign. Report every within-seed observation and both-seed gaps; two seeds do not establish convergence/statistical equivalence. No queue/retry/sweep/default promotion; analyze before next intervention.'
    p['bound_rationale']='Completed same11260-update seed1591232.590s/8.346GiB, unchanged3500 recipe,12evaluations/32ownchecks/24fullM transactions/96CFbasis. Keep original3600internal/3660external/24GiB/512MiBinitialCPU/100MiBcompact/512MiBcache; no after-launch loosening or checkpoints.'
    return p

def validate(config,declared,base):
    if config!=recipe(base) or declared!=protocol():raise ValueError('Undeclared independent damped-confirmation delta')
