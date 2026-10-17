"""Fixed-state chosen-generator scalar sweep; no training/optimizer intervention."""
from pathlib import Path
from llm_basis_response_contract import ARMS,POLICIES,WARMUP,SNAPSHOT_CAP,PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS,CACHE,QUALIFICATION,ids_sha
from llm_basis_response_contract import recipe as previous_recipe,protocol as previous_protocol,evaluation_ids as prior_ids
SEED=175
HORIZON=1000
ORDER=('qr20',)
PARENTS=(Path('/workspace/optimizer_replay_results/llm_soap_basis_response_seed174_frozen_20261013'),)
FACTORS=(('blocks.0.attn.to_qkv.weight',0,6144),('blocks.0.attn.to_qkv.weight',1,2048),
         ('blocks.0.attn.to_out.weight',0,2048),('blocks.0.attn.to_out.weight',1,2048),
         ('blocks.0.ff.proj_in.weight',0,8192),('blocks.0.ff.proj_in.weight',1,2048),
         ('blocks.0.ff.proj_out.weight',0,2048),('blocks.0.ff.proj_out.weight',1,4096))
MULTIPLIERS=(.5,1.,2.,4.)
FACTOR_BYTES=8192**2*4
EXTRA_BYTES=PARAMETERS*4+30*FACTOR_BYTES+12*2**20
STEP_GATES=dict(prefix_relative=1e-6,reference_basis_relative=1e-6,movement_relative=.1,movement_cosine=.995,
                normalized_gram=.005,absolute_gram_fro=.05,exact_state_rng_flags_gradients_and_identity=True,
                compiler_counters_unchanged=True,all_registered_temporaries_released=True)
def recipe(base=None):
    c=previous_recipe();c['seed']=SEED;c['matched_comparison']['scope']='Fresh QR1000 state; fixed inherited C/Q chosen-generator scale/retraction curves, no optimizer intervention'
    return c
def protocol():
    p=previous_protocol();p['seed']=SEED;p.pop('sequence_diagnostic');p.pop('basis_response')
    p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in prior_ids(SEED).items()}
    p['step_scale']=dict(after_training_call=1000,factors=[dict(parameter=n,axis=a,dimension=d) for n,a,d in FACTORS],
        factor_selection='All eight actual factors, no outcome selection',policies=list(POLICIES),multipliers=list(MULTIPLIERS),
        execution_order=[1.,.5,2.,4.],factor_observations=8,direction_observations=16,sweep_candidates=64,pure_eager_references=16,qr_controls=8,
        maximum_complete_basis_calls=88,diagnostic_backward_calls=0,optimizer_updates=0,
        generator='One highest normalized rotated C per factor; unchanged eager Jacobi or damped raw/chosen generator; multiply CHOSEN A, do not recompute direction at each scale',
        retraction='Unchanged BF16 Q@A candidate and BF16 residual correction/highest Gram cubic2; retain candidate/first/final geometry/tracking; alias-only eager stages, endpoint+prefix neutral-reference gates first',
        references='Original pure eager prefix plus original pure retraction for each policy; strongest1e-6 prefix/final relative overlay plus original movement/geometry gates; QR original high FP32 sorting/product',
        admission='Unmodified1x and pure references must pass unchanged original geometry/reference gates. Other finite candidates may be rejected; retain all64proposals and reasons; never relax geometry or interpret rejected residuals as fidelity wins.',
        movement='Known QR sort undone and column signs aligned for reporting only; near-degenerate subspace ambiguity remains; no M/V/adaptive preview',
        arithmetic='Highest FP32 B/metrics and original eager warm arithmetic; original high QR. All original flags/RNG/compiler counters restore. No new basis compilation or timing qualification.',
        gates=dict(STEP_GATES),
        memory=dict(borrowed_training_gradient_bytes=PARAMETERS*4,largest_factor_fp32_bytes=FACTOR_BYTES,
            full_factor_workspace_equivalents=30,workspace_breakdown='Borrowed C/Q2; normalized-C/B/sym/products4; gap/omega/raw/chosen/scaledA5; candidate/first/final/residual/correction6; BF16 equivalents3; QR4; metrics/geometry/alignment5; reference comparison1. Conservative phase-exclusive30.',
            scalar_fingerprint_workspace_bytes=12*2**20,conservative_tensor_purpose_bytes=EXTRA_BYTES,maximum_tensor_purpose_bytes=9*2**30,
            persistent_cpu_statistics_bytes=0),
        retention='Stream one factor, one direction, one candidate at a time. Reference/candidate/first/final/generator matrices RAM-only, immediate release; scalar/hash/provenance only, no snapshots or serialized tensors')
    p['implementation'].update(entry='source/benchmarks/llm_step_scale.py',trainer_delta='Additive exact QR1000/evaluator trainer copy; new fresh175 and read-only88basis-call fixed-C/Q diagnostic; no sequence pools/extra backwards')
    p['artifacts']['probe'].update(purpose=p['step_scale']['retention'],expected_max_extra_bytes=EXTRA_BYTES)
    p['decision']='One fixed state/eight factors/scalar sweep; rejected proposals retained. No new optimizer/beta/step/cap/retraction defaults, full-curvature guarantee, M/V-history, future-learning/compiled-basis-speed qualification or retry.'
    p['bound_rationale']='Original1270QRupdates, zero extra gradient backwards,88bounded eager basis calls including8192; parent174204basis+68backwards312.687s/11.883GiB. Original3600/3660s,24GiBGPU,3GiBinitialCPU,100MiBcompact/512MiBcache unchanged. Named-purpose8645149184B<9GiB includes borrowed gradients+30phase-exclusive largest-factor equivalents+12MiB; wholeGPUbound includes model activations/vendor scratch. No checkpoints/weights/afterlaunch changes.'
    return p
def validate(config,declared,base=None):
    if config!=recipe() or declared!=protocol():raise ValueError('Undeclared step-scale diagnostic delta')
