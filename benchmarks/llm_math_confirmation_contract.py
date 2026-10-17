"""Fresh independently seeded confirmation; qualified recipe and evaluator retained."""
from pathlib import Path
from llm_math_learning_contract import ARMS,POLICIES,HORIZON,WARMUP,SNAPSHOT_CAP,PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS,CACHE,QUALIFICATION,PROBES,EVALUATIONS,TRAIN_BLOCKS,VALIDATION_BLOCKS,TRAIN_PROBE_BLOCKS,ids_sha
from llm_math_learning_contract import recipe as previous_recipe,protocol as previous_protocol,evaluation_ids as previous_ids
SEED=172
ORDER=('qr20','highest_residual20','damped20')
PARENTS=(Path('/workspace/optimizer_replay_results/llm_soap_math_learning_seed171_frozen_20261013'),)
def evaluation_ids(seed=SEED,train_blocks=TRAIN_BLOCKS,validation_blocks=VALIDATION_BLOCKS):
 return previous_ids(seed,train_blocks,validation_blocks)
def recipe(base=None):
 c=previous_recipe();c['seed']=SEED
 c['matched_comparison'].update(execution_order=list(ORDER),scope='Fresh independent-seed postwarmup matched LLM confirmation; no convergence/statistical/repeated-wall qualification')
 return c
def protocol():
 p=previous_protocol();p.update(seed=SEED,execution_order=list(ORDER))
 p['implementation']['entry']='source/benchmarks/llm_math_confirmation.py'
 p['implementation']['trainer_delta']='Fresh independent seed and declared order only; reuse original qualified token evaluator/model/compiled SOAP, original exact gates and bounds'
 p['llm_evaluation']['panels']={n:dict(blocks=len(ids),valid_tokens=len(ids)*512,ids_sha256=ids_sha(ids)) for n,ids in evaluation_ids().items()}
 p['bound_rationale']='Completed matched3500seed17111260updates1872.190s/10.140GiB and12evaluations10-12s each. SAME3600internal/3660external/24GiBGPU/3GiBinitialpurposeCPU/100MiBcompact/512MiBcache/min5GiBdisk; original RAM-only24MV/96CF/393216samplebytes with immediate release. No extra model state/checkpoints or postlaunch loosening.'
 p['decision']='Compare all four validation observations and endpoints across fresh172 and preserved171, retain all sign changes/regressions and tracking/adaptive limitations. Independent seed is not convergence/statistical equivalence or counterbalanced repeated wall qualification. No beta/history/precision sweep or default promotion; never retry spent bundles.'
 return p
def validate(config,declared,base=None):
 if config!=recipe() or declared!=protocol():raise ValueError('Undeclared matched LLM confirmation delta')
