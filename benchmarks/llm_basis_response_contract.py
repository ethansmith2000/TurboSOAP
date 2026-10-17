"""Frozen prospective covariance/basis response at one fresh QR state."""
from pathlib import Path
from llm_sequence_averaging_contract import (
    ARMS,POLICIES,WARMUP,SNAPSHOT_CAP,PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,
    DIMENSIONS,CACHE,QUALIFICATION,TRAIN_BLOCKS,VALIDATION_BLOCKS,TRAIN_PROBE_BLOCKS,
    ids_sha,EXTRA_BYTES,
)
from llm_sequence_averaging_contract import evaluation_ids as previous_evaluation_ids,pools as previous_pools
from llm_sequence_averaging_contract import recipe as previous_recipe,protocol as previous_protocol

SEED=174
HORIZON=1000
ORDER=('qr20',)
PARENTS=(Path('/workspace/optimizer_replay_results/llm_soap_sequence_averaging_seed173_frozen_20261013'),)
SIZES=(1,16)
BASIS_POLICIES=('highest_residual20','damped20','qr20')
FACTORS=(
    ('blocks.0.attn.to_qkv.weight',0,6144),
    ('blocks.0.attn.to_out.weight',1,2048),
    ('blocks.0.ff.proj_in.weight',0,8192),
    ('blocks.0.ff.proj_out.weight',1,4096),
)
FACTOR_BYTES=8192**2*4
BASIS_EXTRA_BYTES=EXTRA_BYTES+28*FACTOR_BYTES
CPU_STATS_BYTES=104*64**2*8*2
BASIS_GATES=dict(clip_gram_relative=1e-5,finite=True,normalized_gram=.005,
                 absolute_gram_fro=.05,exact_state_rng_flags_gradients_and_identity=True,
                 compiler_counters_unchanged=True,registered_release=True)

def pools(seed=SEED,blocks=TRAIN_BLOCKS):return previous_pools(seed,blocks)
def evaluation_ids(seed=SEED,train_blocks=TRAIN_BLOCKS,validation_blocks=VALIDATION_BLOCKS):
    return previous_evaluation_ids(seed,train_blocks,validation_blocks)

def recipe(base=None):
    c=previous_recipe();c['seed']=SEED
    c['matched_comparison']['scope']='Fresh QR1000 state; paired prospective covariance and eager basis response only, no training intervention'
    return c

def protocol():
    p=previous_protocol();p['seed']=SEED
    p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in evaluation_ids(SEED).items()}
    s=p['sequence_diagnostic'];s.update(pool_seed=SEED+6001,pools=[dict(ids=v,ids_sha256=ids_sha(v)) for v in pools(SEED)],sizes=list(SIZES),
        group_definition='Four disjoint16sequence pools; nested FIRST1/FIRST16 prefixes; paired not independent; fixed QR1000 state',
        observations='Original full gradient energies/within-pool variation and fixed64x64 input samples; new full-factor prospective tracking and aligned physical movement samples',
        retention='RAM-only sums/variation/current references, streamed single-factor Gram/prospective C/policy/metric workspaces; scalar/hash provenance only; preserve borrowed state/gradients')
    s['memory'].update(conservative_tensor_purpose_bytes=BASIS_EXTRA_BYTES,maximum_tensor_purpose_bytes=9*2**30,
        additional_largest_factor_workspace_equivalents=28,largest_factor_fp32_bytes=FACTOR_BYTES,
        largest_factor_workspace_breakdown='Borrowed C/Q2; raw+clipped Gram2; prospective C/change2; QR sorted-input/Q/R/product4; warm FP32 normalized-C/B/sym/gap/omega/raw/A/candidate/correction intermediates12; BF16 Q/A/correction equivalents3; metrics/delta/Gram/rotated-C3. Conservative sum across phase-exclusive purposes28.',
        cpu_sketch_statistics_expected_bytes=CPU_STATS_BYTES)
    p['basis_response']=dict(after_training_call=1000,factors=[dict(parameter=n,axis=a,dimension=d) for n,a,d in FACTORS],
        factor_selection='Exactly one factor per SOAP weight; left/right choice covers unique6144/8192/4096 and attention-output2048; frozen before new seed, no outcome selection',
        policies=list(BASIS_POLICIES),modes=['raw','clipped'],sizes=list(SIZES),groups=8,prospective_factor_inputs=64,
        inherited_only_factor_inputs=4,prospective_policy_calls=192,inherited_only_policy_calls=12,maximum_policy_calls=204,
        maximum_cross_pool_statistics=104,
        covariance='Clone inherited C, then lerp_(explicit FP32 G@G.T or G.T@G,1-beta); mean raw gradients first, then one all-unique-parameter global clip; no state injection',
        arithmetic='Diagnostic gradient/Gram/EMA/metrics highest FP32. Eager unchanged Jacobi/damped highest prefix, BF16 Q@A and residual correction/highest Gram cubic2; original QR ordering/product at high FP32. All contexts restore. No new basis compilation or compiled-basis/speed qualification.',
        clipping_control='Compute full raw and explicitly clipped FP32 Gram; chunked FP64 relative check clippedGram against c^2*rawGram, limit1e-5. Seed173 algebraic-sample semantics remain separately documented.',
        movement='For reporting only, undo QR sort permutation and align each column sign to inherited Q; same sign rule for warm. Track full offdiag against unmodified actual policy basis. Alignment does not resolve near-degenerate subspace ambiguity.',
        statistics='104 online cross-pool fixed64x64 arrays:40 original gradient/Gram,16 prospective-C,48 aligned basis-delta; four pools each; no full covariance/movement tensor serialization',
        gates=dict(BASIS_GATES),no_optimizer_updates=True,no_moment_transfer_or_adaptive_preview=True)
    p['implementation'].update(entry='source/benchmarks/llm_basis_response.py',trainer_delta='Additive copy of exact seed173 QR/evaluator trainer; fresh174 state and68FP32 backwards; stream read-only prospective C/Q response before lease release')
    p['artifacts']['probe'].update(purpose=s['retention'],expected_max_extra_bytes=BASIS_EXTRA_BYTES)
    p['decision']='Paired input-to-basis mechanistic response, one state/four pools. No independent tokens/populationCI/cross-domain cause/future learning/MV history/compiled basis timing/default promotion/retry.'
    p['bound_rationale']='Original1270QRupdates,68backwards,204 bounded eager basis calls; four streamed factors including8192. Original3600/3660s,24GiB process,3GiB initial snapshot CPU,100MiBcompact/512MiBcache unchanged. Named gradient/statistic/factor-purpose estimate9569273344B<9GiB includes28 phase-exclusive largest-factor equivalents; model activations/vendor scratch remain inside24GiB whole-process bound. CPUstatistics6815744B<8MiB. No weights/checkpoints/afterlaunch changes.'
    return p

def validate(config,declared,base=None):
    if config!=recipe() or declared!=protocol():raise ValueError('Undeclared basis-response diagnostic delta')
