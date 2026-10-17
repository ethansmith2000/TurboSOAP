"""New compiler-routing-safe attempt; prior failed gates remain immutable."""
from pathlib import Path
from native_residual_weak_contract import ARMS,POLICIES,ORDER,HORIZON,WARMUP,SNAPSHOT_CAP,PARAMETERS
from native_residual_weak_contract import recipe as prior_recipe,protocol as prior_protocol
SEED=152
PARENTS=(Path('/workspace/optimizer_replay_results/cifar_soap_native_residual_weak_seed151_frozen_20261012'),)
def recipe(base):
    c=prior_recipe(base);c['seed']=SEED;return c
def protocol():
    p=prior_protocol();p['seed']=SEED
    p['implementation']['entry']='source/benchmarks/cifar_native_residual_named.py'
    p['implementation']['compiler_adapter']='Distinct plain current/highest prefix and candidate-ret function entries; no partial compiler input/shared external_utils.inner; default limit unchanged'
    p['compiler_routing_fix']=dict(source_parent_seed=151,no_retry_of_parent=True,global_limit_unchanged=True,
        numerical_and_release_gates_unchanged=True,cpu_shape_layout_routing_check_required=True)
    return p
def validate(config,declared,base):
    if config!=recipe(base) or declared!=protocol():raise ValueError('Undeclared named-entry native delta')
