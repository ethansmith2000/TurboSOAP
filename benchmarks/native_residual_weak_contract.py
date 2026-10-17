"""New ownership-safe attempt; failed seed150 protocol/source stays frozen."""
from pathlib import Path
from native_residual_contract import ARMS,POLICIES,ORDER,HORIZON,WARMUP,SNAPSHOT_CAP,PARAMETERS
from native_residual_contract import recipe as prior_recipe,protocol as prior_protocol
SEED=151
PARENTS=(Path('/workspace/optimizer_replay_results/cifar_soap_native_residual_seed150_frozen_20261012'),)

def recipe(base):
    c=prior_recipe(base);c['seed']=SEED
    return c

def protocol():
    p=prior_protocol();p['seed']=SEED
    p['implementation']['entry']='source/benchmarks/cifar_native_residual_weak.py'
    p['implementation']['diagnostic_adapter']='Instance callable with weakref to matrix, no bound-self cycle; immediate OptimizerLease release unchanged'
    p['ownership_fix']=dict(source_parent_status='failed preflight immediate object-release',
        source_parent_seed=150,no_retry_of_parent=True,numerical_gates_unchanged=True,lease_gate_unchanged=True,
        cpu_immediate_release_empty_and_populated_required=True)
    return p

def validate(config,declared,base):
    if config!=recipe(base) or declared!=protocol():raise ValueError('Undeclared ownership-safe native policy delta')
