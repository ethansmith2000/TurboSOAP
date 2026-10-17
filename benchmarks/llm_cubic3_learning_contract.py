"""Frozen actual-history Jacobi2cubic3 comparison, original model/recipe/gates."""
from pathlib import Path
from llm_math_confirmation_contract import WARMUP,HORIZON,SNAPSHOT_CAP,PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS,CACHE,QUALIFICATION,ids_sha
from llm_math_confirmation_contract import recipe as previous_recipe,protocol as previous_protocol,evaluation_ids as previous_ids
from llm_jacobi2_contract import FACTORS,GATES,MEMORY
from hybrid_training_contract import expected_counts
SEED=179
ARMS=('qr20','highest_residual20','jacobi2_cubic3')
POLICIES=ARMS[1:]
ORDER=ARMS
PROBES=(21,1021,3481)
PARENTS=(Path('/workspace/optimizer_replay_results/llm_soap_cubic3_qualification_seed178_frozen_20261014'),)
WARMUP_COUNTS=dict(compiled_basis=32,eager_references=32)
DIAGNOSTIC_COUNTS=dict(qr_controls=72,compiled_counterfactuals=144,eager_counterfactuals=144,scale2_overlay_references=72,own_actual_references=48,full_moment_transactions=36,full_m_previews=144,full_m_projection_calls=540,third_correction_overlay_references=72)
from llm_cubic3_contract import MEM as QUALIFIED_MEMORY
MEM=dict(QUALIFIED_MEMORY,full_basis_banks_bytes=3*(8192**2+2048**2)*4)
PURPOSE=sum(v for k,v in MEM.items() if k not in ('maximum_tensor_purpose_bytes','persistent_cpu_statistics_bytes'))
def evaluation_ids(seed=SEED):return previous_ids(seed)
def calibration_contract():
 return dict(status='PASS_CPU_CUBIC3_LIVE_CALIBRATION',seed=SEED,cuda_initialized=False,prototype_gates=GATES,
  exact_reference_before_bf16_scaling=True,real_two_sided_M_transport_and_QR_V_permutation=True,
  actual_cadence_and_counters=True,whole_diagnostic_and_failure_release=True,own_actual_and_counterfactual_original_gates=True,
  warmup_counts=WARMUP_COUNTS,diagnostic_counts=DIAGNOSTIC_COUNTS,compile_limits=dict(recompile_limit=8,accumulated_recompile_limit=256))
def recipe(base=None):
 c=previous_recipe();c['seed']=SEED;c['matched_comparison'].update(arms=list(ARMS),execution_order=list(ORDER),scope='Fresh matched actual-history Jacobi1/Jacobi2cubic3/QR3500 learning; no default/statistical-equivalence/repeat-qualified wall promotion');return c
def protocol():
 p=previous_protocol();p.update(seed=SEED,arms=list(ARMS),execution_order=list(ORDER),probe_calls=list(PROBES))
 for key,h in [('primary',HORIZON),('disposable',WARMUP)]:p['expected'][key]={a:expected_counts(h,'qr' if a=='qr20' else 'warm',20,0 if a=='qr20' else 200) for a in ARMS}
 warm=sum(expected_counts(n,'warm',20,200)['research_warm_refreshes'] for n in (WARMUP,HORIZON))
 p['expected'].update(hybrid_kernel_calls=warm*8*2,hybrid_calls_by_dimension={n:k*warm*2 for n,k in DIMENSIONS.items()},
  policy_calls_by_dimension={a:{n:k*warm for n,k in DIMENSIONS.items()} for a in POLICIES},diagnostic_reference_calls=48,full_moment_transactions=36,counterfactual_basis_calls=144)
 p['full_moment_diagnostic'].update(calls=list(PROBES),source_arms=list(ARMS),policies=list(POLICIES),transactions=36,compiled_counterfactual_basis_calls=144,preview='Full actualM/V on QR/Jacobi1/Jacobi2cubic3 source histories; bothwarm carryV and sameQRbasis permutation/carryV. Not futuregradient or cross-history oracle.',stage_attribution='Own pure eager endpoints and exact frozen175chosen-A2 overlay; no compiled stage observer',expected_extra_bytes=PURPOSE)
 p['covariance_samples'].update(calls=[],retained_cpu_bytes=0,scope='No CPU covariance/statistics banks in this live cubic3 control')
 p['timing']['diagnostic_calls_excluded']=list(PROBES)
 p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in evaluation_ids().items()}
 p['cubic3_live']=dict(policies=list(POLICIES),factors=[dict(parameter=n,axis=a,dimension=d) for n,a,d in FACTORS],gates=dict(GATES),
  warmup_counts=WARMUP_COUNTS,diagnostic_counts=DIAGNOSTIC_COUNTS,compiler_limits=dict(recompile_limit=8,accumulated_recompile_limit=256),
  warmup='At end QR245disposable BEFOREprimary: all8actualC/Qfactors,2policies,actualQRlayout then canonicalrowmajor warmoutput;32compiled+32own-eager refs; compiled calls held outside native live/CF counters. Other disposables exercise native live warm and QR repair.',
  prototype='Exact frozen176 prefix_jacobi2, chosenA multiplied2 BEFOREBF16product, effective aggregatecap1.0, qualified178 three highestFP32Gram/BF16 residual corrections. Default/fullgraphTrue/dynamicFalse, original highestMtransport/carryV/QRpermutation/all other state paths unchanged.',
  probes='Actualown reference8factors at21/1021/3481 on two warm arms=48. Read-only full actual C/Q/M/V before actual refresh for all3arms/3calls/4matrices=36transactions; each all2factors/QR/J1/J2+own eager+scale2 overlay plus one explicit third correction. Emit complete matrix scalars before admission, no silent correction/gate change.',
  native_basis_call_budget=dict(compiled_live_warm=2688,actual_qr_refresh_factor_calls=1792,cold_eigen_factor_calls=56,compiler_warmup_and_references=64,
   diagnostic_basis_calls=552,total_training_warm_qr_eigen_and_diagnostic_basis_calls=5152),
  native_basis_budget_note='1792QRcalls=8*(QRprimary174+two repairs17each+QRdisposable12+two repairs1each+QR10smoke2);56cold factors=8*(smoke1+3disposable+3primary). Diagnostics72QR+144compiled+144eager+72scale2overlays+72thirdcorrectionoverlays+48actualrefs=552; these are charged along with all native step counters, no extra gradient backwards/optimizer updates.',
  memory=dict(MEM,conservative_tensor_purpose_bytes=PURPOSE,retained_cpu_covariance_tile_bytes=0),
  retention='Stream one fullmatrix moment/basis bank; no files/fullCPUtensorbanks/checkpoints/finalweights/traces/EMA. Retain only compact scalars/config/source/seed/input/software/code hashes/logs/inventories.Compiler cache retained through matchedJ2learning review under512MiB inventory.')
 p['precision'].update(candidate='OriginalBF16 Q@A;Jacobi2cubic3 chosenA*2 beforeproduct',reference='Own pure originalJacobi1 / frozen176Jacobi2cubic3 prefix plus three qualified mixed corrections; original limits unchanged')
 p['artifacts']['probe'].update(location='GPU RAM only, streamed full two-axis matrix bank',purpose=p['cubic3_live']['retention'],expected_max_extra_bytes=PURPOSE)
 p['artifacts']['compiler_cache']['retention']='Through actual-history Jacobi2cubic3 learning review; arithmetic provenance only, no model state'
 p['implementation'].update(entry='source/benchmarks/llm_cubic3_learning.py',trainer_delta='Additive original three-arm/evaluator/recipe trainer with Jacobi2cubic3 replacingdamped, fresh179,64pre-primarywarmup/ref calls,additional21probe/36fullMtransactions; all source/counts/gates/bounds frozenbeforelaunch')
 p['bound_rationale']='Original11260updates/12token-evaluations; seed171/172completed1872/1881s. Add64preprimarybasiswarmup/ref (~seed17695swarmup96calls),36instead24fullMtransactions andfrozen175scale2overlays72: budget remains3600/3660s,24GiBallocated,3GiBinitialCPU,100MiBcompact/512MiBcache,min5GiBfree. Streamedpurpose<12GiBpluszero retainedCPU covariance statistics; no checkpoint/tensorfiles/retry/gate relaxation.'
 p['decision']='Evaluate all paired token panels and early/late actualhistory diagnostics inclregressions; one freshseed, no populationCI/equivalence/default/counterbalancedrepeatwall claim. PriorD171/172 contextnotfreshmatchedarm. Any admission failure spends bundle; fourthcorrection/differentV/lr/beta only separately frozen later.'
 # Resolve inherited historical labels prospectively; frozen178 remains untouched.
 p['width_change'].update(batch_size=16,parameters=PARAMETERS,purpose='Matched width2048 actual-history QR/J1/cubic3 learning and safety over3500 calls')
 p['implementation'].update(cpu116_after_smoke=True,diagnostic_adapter='Weak instance hook on all three actual-history arms; original native QR/covariance/M/V/step paths preserved',compiler_adapter='Named pure J1 prefix/cubic2 and exact A2 prefix/cubic3, default fullgraph/dynamicFalse/8-256 limits')
 p['gates']['no_speed_or_learning_qualification']=True
 p['precision']['policies']={n:dict(prefix='highest',candidate='BF16 Q@A, chosenA*2 only cubic3',retraction='highest FP32 Gram / BF16 correction, 2 for J1, 3 for cubic3') for n in POLICIES}
 p['full_moment_diagnostic']['stage_attribution']='Own pure endpoints and independent frozen175 two-correction A2 overlay plus ONE third correction; no compiled observer'
 p['full_moment_diagnostic']['registered_gpu_ram_limit_extra_bytes']=MEM['maximum_tensor_purpose_bytes']
 p['diagnostic_admission']['policy']='QR/J1/cubic3 geometry and physicalM must pass, with both absolute and normalized Gram. Failures retained, no relaxation/retry.'
 p['cubic3_live']['native_basis_call_budget']['total_read_only_diagnostic_basis_endpoints']=616
 p['cubic3_live']['memory']['persistent_cpu_statistics_bytes']=0
 p['cubic3_live']['retention']='RAM-only streamed factor/fullmatrix M/V previews, immediate release; no CPU covariance samples/fullbanks, weights/checkpoints/tensorfiles/traces/EMA. Compact scalars/hashes/config/source/logs/inventories only. Compiler arithmetic cache retained through actual-history cubic3 review under512MiB separate inventory.'
 p['artifacts']['probe']['purpose']=p['cubic3_live']['retention']
 p['bound_rationale']='Original11260 optimizer updates/12 token evaluations plus64 preprimary endpoints and552 diagnostics; seed171/172 completed~1880s, additional third correction (~26% basis cost in178) and observations remain within original3600/3660s,24GiB allocation/3GiBinitialCPU/100MiBcompact/512MiBcache/min5GiBfree. Streamed purpose<12GiB; no checkpoints or CPU covariance statistics. All source/counts/gates/bounds frozen before launch.'
 return p
def validate(config,declared,base=None):
 if config!=recipe() or declared!=protocol():raise ValueError('Undeclared actual-history Jacobi2cubic3 delta')
