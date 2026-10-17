"""Fresh early/late cubic3 read-only admission and whole-basis cost contract."""
from pathlib import Path
from llm_jacobi2_learning_contract import ARMS,POLICIES,WARMUP,HORIZON,SNAPSHOT_CAP,PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS,CACHE,QUALIFICATION,PROBES,evaluation_ids,ids_sha,GATES
from llm_jacobi2_learning_contract import recipe as previous_recipe,protocol as previous_protocol
from llm_jacobi2_contract import FACTORS
from hybrid_training_contract import expected_counts
SEED=178
PARENTS=(Path('/workspace/optimizer_replay_results/llm_soap_jacobi2_learning_seed177_frozen_20261014'),)
NAMES=('qr20','highest_residual20','jacobi2','jacobi2_cubic3')
MANDATORY=('qr20','highest_residual20','jacobi2_cubic3')
REPEATS=3
COUNTS=dict(warmup_compiled=48,warmup_eager=48,diagnostic_qr=24,diagnostic_compiled=72,diagnostic_eager=72,scale2_cubic2_overlays=24,third_correction_overlays=24,timing_whole_basis=288)
OBS=dict(warmup_comparisons=48,warmup_geometry=48,own_comparisons=72,factor_metrics=120,before_offdiag=24,full_m_previews=60,full_m_projection_calls=216,matrix_transactions=12,rotated_covariance_and_generator_calls=48)
MEM=dict(borrowed_gradient_bytes=PARAMETERS*4,largest_factor_workspace_bytes=36*8192**2*4,
 full_basis_banks_bytes=4*(8192**2+2048**2)*4,moment_preview_workspace_bytes=16*8192*2048*4,fingerprint_workspace_bytes=12*2**20,maximum_tensor_purpose_bytes=12*2**30,persistent_cpu_statistics_bytes=0)
PURPOSE=sum(v for k,v in MEM.items() if k not in ('maximum_tensor_purpose_bytes','persistent_cpu_statistics_bytes'))
def calibration_contract():return dict(status='PASS_CPU_CUBIC3_CALIBRATION',seed=SEED,cuda_initialized=False,gates=GATES,counts=COUNTS,observations=OBS,
 independent_mixed_retraction_and_exact_residual_recurrence=True,concentrated_skew_and_capped_uncapped_controls=True,whole_matrix_admission_rejected_shadows_preserved=True,
 full_M_V_permutation_controls=True,nonmutation_gradient_identity_flags_RNG_compiler_release=True,partial_allocation_and_proposed_admission_failure=True,compile_limits=dict(recompile_limit=8,accumulated_recompile_limit=256))
def recipe(base=None):
 c=previous_recipe();c['seed']=SEED;c['matched_comparison'].update(arms=['qr20'],execution_order=['qr20'],scope='FreshQR3500 source, early21/postwarm1021/late3481read-only all-factor Jacobi2cubic3 admission/fullM/wholebasis cost; no candidate history');return c
def protocol():
 p=previous_protocol();p.pop('jacobi2_live');p.update(seed=SEED,arms=['qr20'],execution_order=['qr20'],maximum_optimizer_updates=3770)
 p['expected'].update(primary={'qr20':expected_counts(3500,'qr',20,0)},disposable={'qr20':expected_counts(245,'qr',20,0)},hybrid_kernel_calls=0,hybrid_calls_by_dimension={},policy_calls_by_dimension={},diagnostic_reference_calls=0,full_moment_transactions=12,counterfactual_basis_calls=72)
 p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in evaluation_ids(SEED).items()}
 p['full_moment_diagnostic'].update(calls=list(PROBES),source_arms=['qr20'],policies=list(NAMES),transactions=12,compiled_counterfactual_basis_calls=72,
  preview='FullactualM/V in QR source atsameC/Q; mandatory QR/J1/J2cubic3, explicitlynonpromoted J2cubic2shadows includingrejections',expected_extra_bytes=PURPOSE,
  stage_attribution='Pure own-eager endpoints plusfrozen175scale2two-correction overlay and oneexplicit third-correction overlay; no compiledobserver')
 p['covariance_samples'].update(calls=[],retained_cpu_bytes=0,scope='No covariance samples/statistics retained in this retraction control')
 p['cubic3']=dict(factors=[dict(parameter=n,axis=a,dimension=d) for n,a,d in FACTORS],policies=list(NAMES),mandatory=list(MANDATORY),shadow='jacobi2',gates=dict(GATES),
  counts=COUNTS,observations=OBS,maximum_diagnostic_basis_endpoints=sum(COUNTS.values()),timing_repeats=REPEATS,
  compile=dict(mode='default',fullgraph=True,dynamic=False,recompile_limit=8,accumulated_recompile_limit=256),
  numerical='EXACTfrozen176prefix_jacobi2chosenA2 BEFOREBF16product; cube2unchanged; namedcubic3addsONEhighestFP32Gram/BF16residualcorrection; effectivecap1.0 unchanged. No newB/direction/beta/LR/Vscheme.',
  preflight='OriginalemptyQR/J1/J2three-armbackward preflightonly; actualtraining QRsourceonly, no candidate optimizertrajectory',
  warmup='AfterQR245disposable BEFOREprimary:8factors/3compiledpolicies/QR+canonicalrowmajor layouts=48compiled+48eager; no primary recompilation orraised8/256limits.',
  probes=list(PROBES),all_matrix_admission='Read-only eachmatrix BEFOREnativeQRrefresh; capture all4matrix scalarrows before aggregate proposedadmission. Safety/reference/compiler/release failures abort immediately. Cube2shadowgeometry/M rejection expected andretained; neverpromoted. QR/J1/Cube3mandatory.',
  overlays='For eachfactor independentlyreconstructchosenA2 and originaltwo mixedcorrections; thenONEexplicitmixedthirdcorrection. Overlaycube3reusescube2output/onecommoncandidate;24+24endpoints, notseparatecomputedprefix orspeed claim. Allworkcharged to all-in.',
  timing='3fixedcyclicinterleavedrepeats/factor/policy/state, originalcommonC/Q, CUDAevents WHOLEprefix+all2or3corrections or QRsort/product/decomposition. Everytimedcallrecomputesits ownprefix/retraction. Excludesmetrics/fullM/gradient/parameter work; minmaxnotCI, no trainerwall claim.',
  native_basis_counts=dict(qr_factor_calls=1504,cold_eigen_factors=24,total_training_and_diagnostic_basis_endpoints=2128),
  memory=dict(MEM,conservative_tensor_purpose_bytes=PURPOSE,workspace_breakdown='Borrowedallgradients;36phase-exclusive8192factorFP32equivalents inclrotatedC/skew/generator/quantization/3retractions/pairedpure/overlay/Gram/tracking;4two-axisbankslargest8192x2048;16fullmatrixmomentbuffers;12MiBfingerprints. Vendorscratch covered by24GiBallocated global bound.'),
  retention='RAM-only streamedfullmatrix/factorbanks andmomentpreviews, immediate release. No modelstate/tensorfiles/CPUfullbanks/covtiles/EMA/checkpoints/weights/traces; compactscalar/hash/source/config/log/inventoryonly. Compiler cache retainedthroughcubic3early-late/cost review.')
 p['artifacts']['probe'].update(location='GPU RAM only, one fulltwo-axis matrixbank at a time',purpose=p['cubic3']['retention'],expected_max_extra_bytes=PURPOSE)
 p['artifacts']['compiler_cache']['retention']='Through cubic3early-late/cost review; separatelyinventoried arithmeticcache, no modelstate'
 p['precision'].update(reference='OwnpureJ1/cubic2/cubic3endpoints; samechosenA, cube3explicitthirdcorrection; originalhighestFP32M andhighQR',retraction='Cube2original orseparatelynamedcube3highestGram/BF16correction')
 p['implementation'].update(entry='source/benchmarks/llm_cubic3_qualification.py',trainer_delta='Additive QR-only3770updates/threefullmatrixprobe points/600read-onlybasisendpoints including96prewarm, unchanged originalQR/evaluator; candidate shadows only; explicit terminal peakallocation persisted even onfailure; declaredcompact/cache budgets checkedseparately')
 p['bound_rationale']='FreshsingleQR3500 trajectory3770maxupdates and600bounded diagnostics endpoints; prior3arm11260updates~1880s,178oneQR plus new3retraction/ownrefs/288timing calls under ORIGINAL3600/3660s,24GiBallocated,3GiBinitialCPU,100MiBcompact/512MiBcache,min5GiBfree. Named streamedpurpose12470354432B<12GiB, no CPUtensorstatistics/checkpoints; no afterlaunchchanges.'
 p['decision']='Originalgates unchanged: mandatoryproposedcube3fails=>spentabort, no silentfourthcorrection. Cube2knownunsafecontrolremainsnonpromoted withallrejections retained. OneQRhistory, no candidatelearning/repeatedrefresh/default/CI/crosshistory or repeat-qualifiedwall claim.'
 return p
def validate(config,declared,base=None):
 if config!=recipe() or declared!=protocol():raise ValueError('Undeclared cubic3 qualification delta')
