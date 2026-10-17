"""Fresh single-seed within-width2048 post-warmup learning; no equivalence claim."""
from pathlib import Path
from hybrid_width_contract import recipe as width_recipe,SNAPSHOT_CAP,PARAMETERS
from hybrid_learning_contract import protocol as learning_protocol
PARENT=Path('/workspace/optimizer_replay_results/cifar_soap_hybrid_width2048_seed144_frozen_20261011')
SEED=145
ORDER=['warm20','hybrid20','qr20']

def recipe(base):
 c=width_recipe(base);c.update(seed=SEED,max_wall_seconds=3600)
 c['matched_comparison']=dict(arms=['qr20','warm20','hybrid20'],steps_per_arm=3500,execution_order=ORDER.copy(),
  scope='Single-seed within-width2048 fixed learning; no convergence or statistical/wall qualification')
 return c

def protocol():
 p=learning_protocol(142);p.update(seed=SEED,execution_order=ORDER.copy())
 p['coverage']['dimensions_per_refresh']={'2048':5,'6144':1,'8192':2}
 p['expected']['hybrid_calls_by_dimension']={'2048':840,'6144':168,'8192':336}
 p['bounds'].update(internal_wall_seconds=3600,external_watchdog_seconds=3660,maximum_initial_snapshot_purpose_cpu_bytes=SNAPSHOT_CAP)
 p['artifacts']['initial_snapshot'].update(expected_bytes=203186576,purpose_expected_bytes=414257552)
 p['artifacts']['probe']['expected_max_extra_bytes']=7*8192**2*4
 p['artifacts']['compiler_cache'].update(expected_bytes=200*2**20,retention='Through next larger-width learning review, arithmetic provenance only')
 p['implementation']['entry']='source/benchmarks/cifar_hybrid_width_learning.py'
 p['implementation']['snapshot_adapter']='Reuse tested width2048 explicit512MiB snapshot factory, restore binding finally; default256MiB unchanged'
 p['decision']='Retain every paired within-width validation observation and compare gaps descriptively with1536. No direct raw-loss cross-width ranking, single-seed statistical equivalence, retries, sweeps or default promotion. Replication/10000/step-controller/V interventions remain separate.'
 p['bound_rationale']='Preliminary projection from completed1536 learning599/618s and2.370x cubicwork is1420/1464s. Width2048 pilot actualcost/resource record must be included in frozen launch references before launch. Prospective3600/3660s,24GiB GPU/512MiB initialCPU/100MiB compact+512MiB cache; no after-launch bound loosening.'
 p['width_change']=dict(width=2048,heads=32,depth=1,batch_size=256,head_width=64,parameters=PARAMETERS,largest_factor=8192,
  matrix_cutoff=10000,all_four_weights_two_sided=True,comparison='Warm-minusQR within width; rawloss acrosswidths confounded')
 return p

def validate(config,declared,base):
 if config!=recipe(base) or declared!=protocol():raise ValueError('Undeclared width2048 learning recipe/protocol delta')
