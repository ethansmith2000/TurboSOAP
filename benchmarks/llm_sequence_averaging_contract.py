"""Fresh fixed-state sequence-averaging diagnostic; no optimizer intervention."""
from pathlib import Path
import random
from llm_math_confirmation_contract import ARMS,POLICIES,WARMUP,SNAPSHOT_CAP,PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS,CACHE,QUALIFICATION,TRAIN_BLOCKS,VALIDATION_BLOCKS,TRAIN_PROBE_BLOCKS,ids_sha
from llm_math_confirmation_contract import recipe as previous_recipe,protocol as previous_protocol,evaluation_ids as old_evaluation_ids
from hybrid_training_contract import expected_counts
SEED=173
HORIZON=1000
ORDER=('qr20',)
PARENTS=(Path('/workspace/optimizer_replay_results/llm_soap_math_confirmation_seed172_frozen_20261013'),)
SIZES=(1,2,4,8,16)
PARAMETER_BYTES=PARAMETERS*4
MATRIX_BYTES=41943040*4
LARGEST_MATRIX_BYTES=8192*2048*4
EXTRA_BYTES=3*PARAMETER_BYTES+MATRIX_BYTES+2*LARGEST_MATRIX_BYTES+12*2**20

def pools(seed=SEED,blocks=TRAIN_BLOCKS):
 ids=random.Random(seed+6001).sample(range(blocks),64)
 return [ids[k:k+16] for k in range(0,64,16)]
def evaluation_ids(seed=SEED,train_blocks=TRAIN_BLOCKS,validation_blocks=VALIDATION_BLOCKS):
 return old_evaluation_ids(seed,train_blocks,validation_blocks)
def recipe(base=None):
 c=previous_recipe();c['seed']=SEED
 c['matched_comparison'].update(steps_per_arm=HORIZON,execution_order=list(ORDER),scope='Fresh QR1000 state for paired sequence-averaging gradient/covariance diagnostics; no optimizer intervention or learning comparison')
 return c

def protocol():
 p=previous_protocol();p.update(seed=SEED,steps_per_arm=HORIZON,execution_order=list(ORDER),maximum_optimizer_updates=1270,
  probe_calls=[],evaluation_calls=[1000],sampled_batch_calls=[21,401,1000],prefix_hashes=[1,20,21,200,201,1000])
 p['expected'].update(primary={'qr20':expected_counts(HORIZON,'qr',20,0)},disposable={'qr20':expected_counts(WARMUP,'qr',20,0)},hybrid_kernel_calls=0,hybrid_calls_by_dimension={},policy_calls_by_dimension={a:{} for a in POLICIES},diagnostic_reference_calls=0,full_moment_transactions=0,counterfactual_basis_calls=0,evaluation_observations_per_arm=1)
 p['full_moment_diagnostic']['calls']=[]
 p['covariance_samples'].update(calls=[],retained_cpu_bytes=0,scope='Old full-MV/basis diagnostic not invoked; new fixed-coordinate gradient/covariance input sketches declared separately')
 p['llm_evaluation']['panels']={n:dict(blocks=len(ids),valid_tokens=len(ids)*512,ids_sha256=ids_sha(ids)) for n,ids in evaluation_ids().items()}
 p['llm_data'].update(batch_plan='Freshseed+2001 permutation, fixed1000batches/16000unique blocks, same disposable prefix; no wrap',training_tokens_per_primary_arm=HORIZON*8192)
 p['implementation'].update(entry='source/benchmarks/llm_sequence_averaging.py',trainer_delta='Reuse exact QR training/empty3arm preflight/reset/evaluator; append fixed-state autograd.grad diagnostic before lease release; original sources unchanged')
 p['timing'].update(diagnostic_calls_excluded=[],interpretation='QR training timing descriptive only; all gradient diagnostics charged to all-in wall, no optimizer speed/learning comparison')
 p['sequence_diagnostic']=dict(after_training_call=1000,pool_seed=SEED+6001,pools=[dict(ids=v,ids_sha256=ids_sha(v)) for v in pools()],sizes=list(SIZES),
  group_definition='Nested FIRST-N prefixes within each16sequence pool; four pools disjoint in block IDs; groups within pools not independent',
  sequences=64,reference_full_batches=4,maximum_backward_calls=68,tokens_per_sequence=512,optimizer_updates=0,
  arithmetic='Eager FP32/highest TF32-off/autocast-disabled/MATH-only attention, eval mode; restore original global precision/SDPA flags and all RNG/module flags afterward; not BF16 training arithmetic',
  estimator='Mean raw per-sequence mean-token-loss gradients; global clip once after mean across ALLuniqueparameters; unnormalized G@G.T/G.T@G covariance inputs; never average outer products instead',
  observations='Full matrix mean-gradient norms/traces and within-pool Welford variation;64x64 physical gradient/covariance input samples, all columns/rows retained in covariance contractions; cross-pool variation on sketches only',
  assumptions='Variance/N is an algebraic iid-sequence plug-in, not independent token count or population CI; empirical four-pool sketch variation retained separately',
  gates=dict(gradient_relative=1e-4,gradient_cosine=.99999,norm_relative=1e-5,loss_absolute=1e-5,loss_relative=1e-5,all_parameters_finite=True,each_matrix_finite=True,exact_state_rng_flags_gradients_and_identity=True,compiler_counters_unchanged=True,all_registered_temporaries_released=True),
  memory=dict(borrowed_training_gradient_bytes=PARAMETER_BYTES,one_all_parameter_sum_bytes=PARAMETER_BYTES,current_or_reference_gradient_bytes=PARAMETER_BYTES,matrix_variation_bytes=MATRIX_BYTES,maximum_two_matrix_temporaries_bytes=2*LARGEST_MATRIX_BYTES,sketch_norm_fingerprint_workspace_bytes=12*2**20,conservative_tensor_purpose_bytes=EXTRA_BYTES,maximum_tensor_purpose_bytes=2*2**30,cpu_sketch_statistics_expected_bytes=6553600,cpu_sketch_statistics_cap_bytes=8*2**20),
  retention='RAM-only sum/variation/current gradients/sketch statistics; immediate release per pool/diagnostic, preserve borrowed current gradients unchanged; scalar/hash/provenance only, no tensor serialization')
 p['artifacts']['probe'].update(purpose=p['sequence_diagnostic']['retention'],expected_max_extra_bytes=EXTRA_BYTES)
 p['decision']='Fixed postwarmup state, four disjoint sequence pools and paired prefixes; quantify gradient/covariance inputs only. No token-independent samples, cross-domain cause, future-learning prediction, basis/MV-history intervention, default promotion, retries or checkpoint recovery.'
 p['bound_rationale']='Fresh1270liveQRupdates supports fixedpostwarmup state; completed17211260updates1880.839s/10.140GiB.68declaredFP32backwards and streamed aggregation/norm/state checks fit unchanged3600/3660s,24GiBGPU,3GiBinitialCPU,100MiBcompact,512MiBcache. Tensor purpose2053080576B under2GiB includes borrowedtrain gradients/currentref/allsum/SOAPvariation/two matrixtemporaries/workspaces;CPUstatistics<=8MiB. No weights/checkpoints or afterlaunch loosening.'
 return p

def validate(config,declared,base=None):
 if config!=recipe() or declared!=protocol():raise ValueError('Undeclared sequence-averaging diagnostic delta')
