"""Fresh exact-chosen-A2 Jacobi prototype; same-state qualification only."""
from pathlib import Path
from llm_step_scale_contract import ARMS,POLICIES,WARMUP,SNAPSHOT_CAP,PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS,CACHE,QUALIFICATION,FACTORS
from llm_step_scale_contract import recipe as previous_recipe,protocol as previous_protocol
from llm_basis_response_contract import evaluation_ids,ids_sha
SEED=176
HORIZON=1000
PARENTS=(Path('/workspace/optimizer_replay_results/llm_soap_step_scale_seed175_frozen_20261013'),)
NAMES=('qr20','highest_residual20','damped20','jacobi2')
REPEATS=5
OBSERVATIONS=dict(warmup_geometry=48,own_eager_comparisons=72,factor_metrics=40,before_offdiag=8,full_m_previews=20,full_m_projection_calls=72)
GATES=dict(reference_basis_relative=1e-4,movement_relative=.1,movement_cosine=.995,normalized_gram=.005,absolute_gram_fro=.05,physical_m_relative=.01,scale2_overlay_relative=1e-6)
MEMORY=dict(borrowed_gradient_bytes=PARAMETERS*4,largest_factor_workspace_bytes=32*8192**2*4,
 full_basis_banks_bytes=4*(8192**2+2048**2)*4,moment_preview_workspace_bytes=16*8192*2048*4,
 fingerprint_workspace_bytes=12*2**20,maximum_tensor_purpose_bytes=12*2**30)
PURPOSE=sum(v for k,v in MEMORY.items() if k!='maximum_tensor_purpose_bytes')
COUNTS=dict(warmup_compiled=48,warmup_eager_references=48,qualification_compiled=24,qualification_eager_references=24,scale2_overlay_references=8,qualification_qr=8,timing_complete_basis=160)
def calibration_contract():
 return dict(status='PASS_CPU_JACOBI2_CALIBRATION',seed=SEED,cuda_initialized=False,gates=GATES,counts=COUNTS,
  independent_scale_before_bf16_and_retraction=True,cap_active_and_uncapped_controls=True,full_physical_m_and_v_permutation_controls=True,
  tiny_whole_transaction_and_counts=True,nonmutation_and_registered_release=True,partial_allocation_failure_and_admission_rejection=True)
def recipe(base=None):
 c=previous_recipe();c['seed']=SEED;c['matched_comparison']['scope']='Fresh QR1000 state: compiled Jacobi2 exact chosen-A scaling, complete basis cost/full physicalM/inheritedV preview; no candidate trajectory';return c
def protocol():
 p=previous_protocol();p['seed']=SEED;p.pop('step_scale')
 p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in evaluation_ids(SEED).items()}
 p['jacobi2']=dict(after_training_call=1000,factors=[dict(parameter=n,axis=a,dimension=d) for n,a,d in FACTORS],policies=list(NAMES),gates=GATES,
  counts=COUNTS,observation_counts=OBSERVATIONS,maximum_complete_basis_calls=sum(COUNTS.values()),timing_repeats=REPEATS,
  optimizer_updates=0,backward_calls=0,compile=dict(mode='default',fullgraph=True,dynamic=False,default_specialization_limit_unchanged=True,recompile_limit=8,accumulated_recompile_limit=256),
  warmup='After245disposable BEFOREprimary: each8actualC/Q factors and each3compiledpolicies, Actual QR output layout then a canonical row-major copy of its own warm output; output strides recorded, both QR/warm layouts covered, own eager comparison each of48calls. Never compile in primary.',
  numerical='Jacobi2 chosen A exactly2*original capped JacobiA BEFORE BF16 Q@A; effective aggregate cap1.0; normalized C/B highestFP32, cubic2 original BF16 residual correction/highestGram. Damped and Jacobi1 unchanged.',
  qualification='8QR+24compiled+24own eager+8independent frozen175scale2 overlays; fixed inherited state. Keep every scalar before gate. Gates aggregated after all4matrices for complete failure evidence; no injection or promotion.',
  timing='5fixed interleaved repeats/factor/policy, cyclic policy order; fixed actual C/Q, CUDA events whole prefix+retraction or entire QRsort/product/decomposition. No metric/transport work in timer, no profiler/traces; one device/state no repeat-qualified wall claim.',
  physical_m='Original highestFP32 project_back inherited fullM then project/reconstruct in each two-sided candidate bank; <=1%. All four matrices. Inherited actual diagonalV carry versus QR permutation; no missing cross-history reconstruction or future gradient.',
  memory=dict(MEMORY,conservative_tensor_purpose_bytes=PURPOSE,workspace_breakdown='Phase-exclusive32largestfactor equivalents for normalization/rotatedC/gap/skew/BF16/retraction/reference/Gram/tracking; four full two-axis basisbanks sized by largest8192x2048matrix; sixteen largestmatrix buffers for fullM/QRV/projections/adaptive comparisons; borrowed gradients and12MiBfingerprints; vendor scratch covered by24GiBallocated global bound.'),
  retention='All full C/Q/M/V references/counterfactual outputs RAM-only streamed by matrix and immediately released. No optimizer insertion, CPU tensor banks, checkpoints/weights/tensors/traces/EMA. Compact scalar/source/hash/call records only. Compiler cache retained through this qualification review under512MiB inventory.')
 p['implementation'].update(entry='source/benchmarks/llm_jacobi2_qualification.py',trainer_delta='Additive unchanged QR/evaluator trainer; seed176;96pre-primary basiswarmup/ref calls,224primary diagnostic basiscalls; zero extra backwards/updates')
 p['artifacts']['probe'].update(purpose=p['jacobi2']['retention'],expected_max_extra_bytes=PURPOSE)
 p['decision']='One fixed actual state; compiled cost/fidelity/MVpreview qualification only; original gates unchanged. Failure spends bundle; no silent retraction correction/threshold relaxation/retry. No learning/default/adaptive-history/population or causal-width claim.'
 p['bound_rationale']='1270originalQRupdates and320bounded complete basis calls. Original3600/3660seconds,24GiBallocatedGPU,3GiBinitialCPU,100MiBcompact/512MiBcache/min5GiBfree. Explicit streamed tensor purpose<12GiB; no source/bounds/gate changes afterlaunch.'
 return p
def validate(config,declared,base=None):
 if config!=recipe() or declared!=protocol():raise ValueError('Undeclared Jacobi2 qualification delta')
