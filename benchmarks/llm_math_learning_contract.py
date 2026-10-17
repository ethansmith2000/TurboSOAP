"""Fresh postwarmup matched LLM learning; original qualified arithmetic retained."""
from pathlib import Path
import hashlib,random,struct
from llm_math_attention_contract import ARMS,POLICIES,WARMUP,SNAPSHOT_CAP,PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS,CACHE,QUALIFICATION
from llm_math_attention_contract import recipe as previous_recipe,protocol as previous_protocol
from hybrid_training_contract import expected_counts
SEED=171
HORIZON=3500
ORDER=('damped20','highest_residual20','qr20')
PROBES=(1021,3481)
EVALUATIONS=(1000,2000,3000,3500)
PARENTS=(Path('/workspace/optimizer_replay_results/llm_soap_math_attention_seed170_frozen_20261013'),)
TRAIN_BLOCKS=78124
VALIDATION_BLOCKS=511
TRAIN_PROBE_BLOCKS=256

def evaluation_ids(seed=SEED,train_blocks=TRAIN_BLOCKS,validation_blocks=VALIDATION_BLOCKS):
 return dict(train_probe=random.Random(seed+5001).sample(range(train_blocks),TRAIN_PROBE_BLOCKS),validation=list(range(validation_blocks)))
def ids_sha(ids):return hashlib.sha256(struct.pack('<'+'q'*len(ids),*ids)).hexdigest()
def recipe(base=None):
 c=previous_recipe();c['seed']=SEED
 c['matched_comparison'].update(steps_per_arm=HORIZON,execution_order=list(ORDER),scope='Fresh single-seed postwarmup common-math-attention LLM validation; no convergence/statistical/repeated-wall qualification')
 return c

def protocol():
 p=previous_protocol();p.update(seed=SEED,steps_per_arm=HORIZON,execution_order=list(ORDER),maximum_optimizer_updates=11260,
  probe_calls=list(PROBES),evaluation_calls=list(EVALUATIONS),sampled_batch_calls=[21,401,1001,1021,2001,3001,3481],prefix_hashes=[1,20,21,200,201,1000,2000,3000,3500],
  evaluation_policy='Fixed token-weighted CE/accuracy, eager original model, same math-only attention/BF16 autocast under eval/no_grad',
  evaluation_transaction='Hash model/optimizer/gradients before/after; restore all module flags and global/private RNG in finally and verify even on evaluator failure; no tensor copies/checkpoints')
 p['expected']['primary']={a:expected_counts(HORIZON,'qr' if a=='qr20' else 'warm',20,0 if a=='qr20' else 200) for a in ARMS}
 warm=sum(expected_counts(n,'warm',20,200)['research_warm_refreshes'] for n in (WARMUP,HORIZON))
 p['expected'].update(hybrid_kernel_calls=warm*8*len(POLICIES),hybrid_calls_by_dimension={n:k*warm*len(POLICIES) for n,k in DIMENSIONS.items()},policy_calls_by_dimension={a:{n:k*warm for n,k in DIMENSIONS.items()} for a in POLICIES},evaluation_observations_per_arm=len(EVALUATIONS))
 p['full_moment_diagnostic']['calls']=list(PROBES)
 p['covariance_samples'].update(calls=list(PROBES),scope='Fixed physical coordinate sample drift across2460updates includes model change/QRrepair, not isolated noise or full covariance')
 p['artifacts']['probe']['purpose']='Read-only actual C/Q/M/V and fixed64x64 covariance samples at1021/3481; original numerical/ownership gates'
 p['artifacts']['compiler_cache']['retention']='Through matched LLM learning/replication review; arithmetic provenance only, no model state'
 p['llm_data'].update(batch_plan='Freshseed+2001 permutation, fixed3500batches/56000unique blocks, same across arms and disposable prefix; no wrap',training_tokens_per_primary_arm=HORIZON*8192,validation='Original262144-token cache:511complete512-target blocks,261632 valid targets; discarded partial suffix unchanged')
 p['llm_evaluation']=dict(batch_size=16,sequence_length=512,train_blocks=TRAIN_BLOCKS,validation_blocks=VALIDATION_BLOCKS,
  panels={n:dict(blocks=len(ids),valid_tokens=len(ids)*512,ids_sha256=ids_sha(ids)) for n,ids in evaluation_ids().items()},
  selection='All complete validation blocks; train_probe256 sample without replacement from independent local Python Random(seed+5001), may overlap training and is not heldout',
  metrics='ce=sum FP32 token cross entropy / valid_tokens; accuracy=correct tokens / valid_tokens; examples=sequence blocks; explicit valid_tokens retained',
  last_batch='Validation511blocks produces31full16sequence batches and one15sequence batch; token weighting, no padding/ignored targets')
 p['gates']['no_speed_or_learning_qualification']=False
 p['gates'].update(no_statistical_equivalence_or_speed_qualification=True,evaluation_model_optimizer_gradient_nonmutation=True,evaluation_rng_flags_restored=True,evaluation_token_weighting=True)
 p['timing'].update(diagnostic_calls_excluded=list(PROBES),primary_within_lr_warmup=False,evaluations_and_guards_charged=True,interpretation='Descriptive serialized optimizer cost; cadence estimate separate from measured wall, no throughput/time-to-quality qualification')
 p['implementation'].update(entry='source/benchmarks/llm_math_learning.py',trainer_delta='Add token-weighted LLM fixed evaluation and finally nonmutation/RNG/flags checks; original qualified eager math model and compiled SOAP unchanged')
 p['bound_rationale']='Qualifiedseed1702020updates626.058s/10.140GiB. Max primary phase119.460s/420 conservatively projects10500primary2986.512s; disposable169.640s and other overhead109.272s total3265.424s before evaluation. Twelve panels each767blocks (~48forward batches) plus streamed before/after state hashes have334.5s remaining under3600internal/3660external. No compilation for variable eager eval batch. Fixed24GiBGPU/3GiBinitialpurposeCPU/100MiBcompact/512MiBcache; exact shared model/no extra persistent GPU tensors, RAM-only original24M/96CF and393216byte samples released; no afterlaunch loosening.'
 p['decision']='Retain all four postwarmup validations and paired source-state regressions; compare endpoint and trajectory descriptively. One seed is not convergence/statistical equivalence; no token averaging causal attribution, counterbalanced wall speed or default promotion. Never retry spent bundles.'
 return p

def validate(config,declared,base=None):
 if config!=recipe() or declared!=protocol():raise ValueError('Undeclared matched LLM learning delta')
