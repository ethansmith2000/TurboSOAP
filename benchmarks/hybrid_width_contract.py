"""Fresh2048 feasibility; no cross-width quality ranking or convergence claim."""
from pathlib import Path
from hybrid_training_contract import protocol as pilot_protocol,recipe as pilot_recipe
PARENT=Path('/workspace/optimizer_replay_results/cifar_soap_hybrid_learning_seed143_frozen_20261011')
SEED=144
SNAPSHOT_CAP=512*2**20
PARAMETERS=50796644

def recipe(base):
 c=pilot_recipe(base);c.update(seed=SEED,width=2048,heads=32,max_wall_seconds=1800)
 c['matched_comparison']=dict(arms=['qr20','warm20','hybrid20'],steps_per_arm=420,
  execution_order=['hybrid20','warm20','qr20'],scope='Width2048 engineering feasibility; no converged quality ranking')
 return c

def protocol():
 p=pilot_protocol();p.update(seed=SEED,execution_order=['hybrid20','warm20','qr20'],evaluation_calls=[],
  prefix_hashes=[1,20,21,200,201,281,400,401,420])
 p['coverage']['dimensions_per_refresh']={'2048':5,'6144':1,'8192':2}
 p['expected']['hybrid_calls_by_dimension']={'2048':145,'6144':29,'8192':58}
 p['bounds'].update(internal_wall_seconds=1800,external_watchdog_seconds=1860,
  maximum_initial_snapshot_purpose_cpu_bytes=SNAPSHOT_CAP,maximum_evidence_bytes=100*2**20)
 p['artifacts']['initial_snapshot'].update(expected_bytes=203186576,purpose_expected_bytes=414257552)
 p['artifacts']['probe']['expected_max_extra_bytes']=7*8192**2*4
 p['artifacts']['compiler_cache'].update(expected_bytes=200*2**20,retention='Through next larger-width integration review, arithmetic provenance only')
 p['implementation']['entry']='source/benchmarks/cifar_hybrid_width.py'
 p['implementation']['snapshot_adapter']='Temporarily supply explicit512MiB to original SharedInitialState before allocation; restore module binding finally. Default256MiB unchanged.'
 p['gates']['exact_disposable_primary_scalar_prefix']=True
 p['width_change']=dict(width=2048,heads=32,depth=1,batch_size=256,head_width=64,
  parameters=PARAMETERS,largest_factor=8192,all_four_weights_two_sided=True,matrix_cutoff=10000,
  purpose='Feasibility, numerical parity and complete optimizer cost;420steps inside875LRwarmup',
  caveat='Raw validation loss across widths confounds capacity/gradients; longer learning compares each warm arm against QR within width')
 p['bound_rationale']='1536 pilot190.517s/2.919GiB. Cubic factor-work projection2.370x, square-storage1.778x, unqualified.1800/1860s and24GiB GPU are prospective guards,512MiB initial CPU purpose required by414257552B exact audit.'
 return p

def validate(config,declared,base):
 if config!=recipe(base) or declared!=protocol():raise ValueError('Undeclared width2048 recipe/protocol delta')
