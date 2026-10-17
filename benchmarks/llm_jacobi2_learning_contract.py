"""Frozen actual-history Jacobi2 comparison, original model/recipe/gates."""
from pathlib import Path
from llm_math_confirmation_contract import WARMUP,HORIZON,SNAPSHOT_CAP,PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS,CACHE,QUALIFICATION,ids_sha
from llm_math_confirmation_contract import recipe as previous_recipe,protocol as previous_protocol,evaluation_ids as previous_ids
from llm_jacobi2_contract import FACTORS,GATES,MEMORY
from hybrid_training_contract import expected_counts
SEED=177
ARMS=('qr20','highest_residual20','jacobi2')
POLICIES=ARMS[1:]
ORDER=ARMS
PROBES=(21,1021,3481)
PARENTS=(Path('/workspace/optimizer_replay_results/llm_soap_jacobi2_qualification_seed176_frozen_20261014'),)
WARMUP_COUNTS=dict(compiled_basis=32,eager_references=32)
DIAGNOSTIC_COUNTS=dict(qr_controls=72,compiled_counterfactuals=144,eager_counterfactuals=144,scale2_overlay_references=72,own_actual_references=48,full_moment_transactions=36,full_m_previews=144,full_m_projection_calls=540,covariance_sample_tiles=72)
# Includes the separately observed frozen175 exact scale2 overlay at every CF factor.
MEM=dict(MEMORY,full_basis_banks_bytes=3*(8192**2+2048**2)*4)
PURPOSE=sum(v for k,v in MEM.items() if k!='maximum_tensor_purpose_bytes')
def evaluation_ids(seed=SEED):return previous_ids(seed)
def calibration_contract():
 return dict(status='PASS_CPU_JACOBI2_LIVE_CALIBRATION',seed=SEED,cuda_initialized=False,prototype_gates=GATES,
  exact_reference_before_bf16_scaling=True,real_two_sided_M_transport_and_QR_V_permutation=True,
  actual_cadence_and_counters=True,whole_diagnostic_and_failure_release=True,own_actual_and_counterfactual_original_gates=True,
  warmup_counts=WARMUP_COUNTS,diagnostic_counts=DIAGNOSTIC_COUNTS,compile_limits=dict(recompile_limit=8,accumulated_recompile_limit=256))
def recipe(base=None):
 c=previous_recipe();c['seed']=SEED;c['matched_comparison'].update(arms=list(ARMS),execution_order=list(ORDER),scope='Fresh matched actual-history Jacobi1/Jacobi2/QR3500 learning; no default/statistical-equivalence/repeat-qualified wall promotion');return c
def protocol():
 p=previous_protocol();p.update(seed=SEED,arms=list(ARMS),execution_order=list(ORDER),probe_calls=list(PROBES))
 for key,h in [('primary',HORIZON),('disposable',WARMUP)]:p['expected'][key]={a:expected_counts(h,'qr' if a=='qr20' else 'warm',20,0 if a=='qr20' else 200) for a in ARMS}
 warm=sum(expected_counts(n,'warm',20,200)['research_warm_refreshes'] for n in (WARMUP,HORIZON))
 p['expected'].update(hybrid_kernel_calls=warm*8*2,hybrid_calls_by_dimension={n:k*warm*2 for n,k in DIMENSIONS.items()},
  policy_calls_by_dimension={a:{n:k*warm for n,k in DIMENSIONS.items()} for a in POLICIES},diagnostic_reference_calls=48,full_moment_transactions=36,counterfactual_basis_calls=144)
 p['full_moment_diagnostic'].update(calls=list(PROBES),source_arms=list(ARMS),policies=list(POLICIES),transactions=36,compiled_counterfactual_basis_calls=144,preview='Full actualM/V on QR/Jacobi1/Jacobi2 source histories; bothwarm carryV and sameQRbasis permutation/carryV. Not futuregradient or cross-history oracle.',stage_attribution='Own pure eager endpoints and exact frozen175chosen-A2 overlay; no compiled stage observer',expected_extra_bytes=PURPOSE)
 p['covariance_samples'].update(calls=list(PROBES),scope='Fixed physical64x64 tiles across initial/1021/3481; drift includes model changes/repairs, not noise or full covariance')
 p['timing']['diagnostic_calls_excluded']=list(PROBES)
 p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in evaluation_ids().items()}
 p['jacobi2_live']=dict(policies=list(POLICIES),factors=[dict(parameter=n,axis=a,dimension=d) for n,a,d in FACTORS],gates=dict(GATES),
  warmup_counts=WARMUP_COUNTS,diagnostic_counts=DIAGNOSTIC_COUNTS,compiler_limits=dict(recompile_limit=8,accumulated_recompile_limit=256),
  warmup='At end QR245disposable BEFOREprimary: all8actualC/Qfactors,2policies,actualQRlayout then canonicalrowmajor warmoutput;32compiled+32own-eager refs; compiled calls held outside native live/CF counters. Other disposables exercise native live warm and QR repair.',
  prototype='Exact frozen176 prefix_jacobi2, chosenA multiplied2 BEFOREBF16product, effective aggregatecap1.0, same original cubic2 residual retraction. Default/fullgraphTrue/dynamicFalse, original highestMtransport/carryV/QRpermutation/all other state paths unchanged.',
  probes='Actualown reference8factors at21/1021/3481 on two warm arms=48. Read-only full actual C/Q/M/V before actual refresh for all3arms/3calls/4matrices=36transactions; each all2factors/QR/J1/J2+own eager+scale2 overlay. Emit complete matrix scalars before admission, no silent correction/gate change.',
  native_basis_call_budget=dict(compiled_live_warm=2688,actual_qr_refresh_factor_calls=1792,cold_eigen_factor_calls=56,compiler_warmup_and_references=64,
   diagnostic_basis_calls=480,total_training_warm_qr_eigen_and_diagnostic_basis_calls=5080),
  native_basis_budget_note='1792QRcalls=8*(QRprimary174+two repairs17each+QRdisposable12+two repairs1each+QR10smoke2);56cold factors=8*(smoke1+3disposable+3primary). Diagnostics72QR+144compiled+144eager+72scale2overlays+48actualrefs=480; these are charged along with all native step counters, no extra gradient backwards/optimizer updates.',
  memory=dict(MEM,conservative_tensor_purpose_bytes=PURPOSE,retained_cpu_covariance_tile_bytes=393216,maximum_cpu_diagnostic_purpose_bytes=8*2**20),
  retention='Stream one fullmatrix moment/basis bank; no files/fullCPUtensorbanks/checkpoints/finalweights/traces/EMA. Retain only compact scalars/config/source/seed/input/software/code hashes/logs/inventories.64x64physical covariance samplesRAM-only393216B,lastperarm/factor for descriptive drift, clearedcompletion/abort. Compiler cache retained through matchedJ2learning review under512MiB inventory.')
 p['precision'].update(candidate='OriginalBF16 Q@A;Jacobi2 chosenA*2 beforeproduct',reference='Own pure originalJacobi1 / frozen176Jacobi2 prefix plus original residualcubic2; original limits unchanged')
 p['artifacts']['probe'].update(location='GPU RAM only, streamed full two-axis matrix bank',purpose=p['jacobi2_live']['retention'],expected_max_extra_bytes=PURPOSE)
 p['artifacts']['compiler_cache']['retention']='Through actual-history Jacobi2 learning review; arithmetic provenance only, no model state'
 p['implementation'].update(entry='source/benchmarks/llm_jacobi2_learning.py',trainer_delta='Additive original three-arm/evaluator/recipe trainer with Jacobi2 replacingdamped, fresh177,64pre-primarywarmup/ref calls,additional21probe/36fullMtransactions; all source/counts/gates/bounds frozenbeforelaunch')
 p['bound_rationale']='Original11260updates/12token-evaluations; seed171/172completed1872/1881s. Add64preprimarybasiswarmup/ref (~seed17695swarmup96calls),36instead24fullMtransactions andfrozen175scale2overlays72: budget remains3600/3660s,24GiBallocated,3GiBinitialCPU,100MiBcompact/512MiBcache,min5GiBfree. Streamedpurpose<12GiBplus393216BRAMtiles; no checkpoint/tensorfiles/retry/gate relaxation.'
 p['decision']='Evaluate all paired token panels and early/late actualhistory diagnostics inclregressions; one freshseed, no populationCI/equivalence/default/counterbalancedrepeatwall claim. PriorD171/172 contextnotfreshmatchedarm. Any admission failure spends bundle; thirdcorrection/differentV/lr/beta onlyseparatelyfrozenlater.'
 return p
def validate(config,declared,base=None):
 if config!=recipe() or declared!=protocol():raise ValueError('Undeclared actual-history Jacobi2 delta')
