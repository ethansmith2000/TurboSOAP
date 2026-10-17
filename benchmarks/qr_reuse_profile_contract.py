"""One fresh natural state for QR reuse qualification and bounded cost profiling."""
import copy
import json
from pathlib import Path
from llm_math_learning_contract import evaluation_ids, ids_sha
from llm_feasibility_contract import PARAMETERS, SNAPSHOT_BYTES, SNAPSHOT_PURPOSE, SNAPSHOT_CAP, DIMENSIONS, CACHE

SEED=182
SEEDS=(SEED,)
PARENT=Path('/workspace/optimizer_replay_results/llm_soap_sparse_screen_seed181_frozen_20261015')
HORIZON=1000
WARMUP=245
BLOCKS=3
REPEATS=3
STEP_POLICIES=('ordinary','native','reuse')
FACTOR_POLICIES=('native','reuse','decomposition')
PROFILE_COUNTS=dict(shadow_optimizer_steps=32, factor_basis_calls=160, decomposition_calls=80,
    covariance_endpoints=10, projection_endpoints=10, following_gradient_previews=8)


def recipe(seed=SEED):
    if seed!=SEED:raise ValueError('Only fresh182 profile declared')
    c=json.loads((PARENT/'config.json').read_text());c['seed']=seed;c['max_wall_seconds']=900
    c['matched_comparison']=dict(arms=['hold200'],steps_per_arm=HORIZON,execution_order=['hold200'],
        scope='Fresh natural hold200 state, read-only QR implementation/profile diagnostic, no learning comparison')
    return c


def protocol(seed=SEED):
    if seed!=SEED:raise ValueError('Only fresh182 profile declared')
    p=json.loads((PARENT/'protocol.json').read_text())
    p.update(seed=seed,arms=['hold200'],execution_order=['hold200'],steps_per_arm=HORIZON,
        disposable_smoke_calls=25,disposable_calls_per_arm=WARMUP,empty_preflight_backward_calls=1,
        maximum_optimizer_updates=1270,maximum_shadow_optimizer_steps=32,maximum_all_optimizer_calls=1302,
        evaluation_calls=[],guard_calls=[1,21,201,501,1000],sampled_batch_calls=[21,201,501,1000],prefix_hashes=[245,1000])
    p['expected']={phase:{'hold200':dict(step=n-1,research_qr_refreshes=(n-1)//200,research_warm_refreshes=0,research_strict_transports=(n-1)//200)} for phase,n in [('disposable',245),('primary',1000)]}
    p['expected'].update(hybrid_calls_by_dimension={},own_eager_factor_references=0,disposable_actual_m_transactions=4,
        primary_counterfactual_basis_calls=0,evaluations_per_arm=0)
    p['bounds'].update(internal_wall_seconds=900,external_watchdog_seconds=960)
    p['llm_data'].update(training_tokens_per_primary_arm=8192000,batch_plan='Fresh182 fixed1000batches/16000unique blocks; no wrap. State needed for profiling, not a rerun or learner comparison.')
    p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in evaluation_ids(seed).items()}
    p['profile']=dict(blocks=BLOCKS,repeats=REPEATS,counts=PROFILE_COUNTS,
        input='Actual common C/Q/M/V and retained gradients after fresh hold1000 calls (matrix age999). Original source/model/optimizer/gradient/RNG/flags hash restored exactly.',
        complete_step='Private copies of original optimizer state share model parameters; restore parameter GPU backup after every shadow step. Native/reuse trigger QR at age1000. Ordinary selects the native nonrefresh branch with shadow frequency1000000; no deployed cadence change.',
        atoms='Actual factor shapes: native orderedQR, CQ-reuse QR, decomposition alone on prepared native input. Aggregate matrix covariance/projection endpoints. No assumed additive latency attribution.',
        setup='Two qualification shadow steps, one warmup per3 full-step kinds/3factor kinds/2aggregate atoms, then3counterbalanced blocks x3repeats.',
        qualification=dict(absolute_gram_fro=.05,normalized_gram=.005,physical_m_relative=.01,
            following_gradient_update_relative=.001,following_gradient_update_cosine=.99999,
            order_score_band_relative_to_max=.0001,raw_basis_difference='Report raw and sign-aligned; not an exact-coordinate gate. Any tolerated order changes are a candidate approximation requiring a later learning check.'),
        decision='Proceed only if original numerical checks plus prospective action/order-band criteria pass. A promising speed candidate needs reuse complete-refresh mean at least5% below native in all3blocks; otherwise report unresolved. No deployed/learning promotion, retry or extra variant.',
        observer_policy='Qualification and source hashes outside all timers. Time CUDA and host complete endpoints, including casts/M/V for full steps. State copies/restores outside timer, charged to all-in. No profiler trace.',
        memory=dict(maximum_tensor_purpose_bytes=20*2**30,largest_factor_workspace_bytes=32*8192**2*4,
            four_optimizer_state_banks_bytes=4*(2*142606336*4+2*144875648*4),
            model_gradient_parameter_backup_bytes=3*144875648*4,following_update_buffers_bytes=2*8192*2048*4,
            fingerprint_workspace_bytes=12*2**20),
        retention='All trained parameters, optimizer banks, gradients and temporaries RAM-only. Release before source lease release; no tensor/model/checkpoint files. Compact scalar/timing/source/hash manifests; arithmetic cache separately retained through this profile review.')
    p['artifacts']['probe']=dict(location='GPU RAM only',purpose=p['profile']['retention'],expected_max_extra_bytes=20*2**30,retention='Release each shadow and all banks before source lease closes')
    p['artifacts']['compiler_cache']['retention']='Through QR/profile review; arithmetic provenance only'
    p['memory']=p['profile']['memory']
    p['qualification']=dict(phase='disposable source firstQR200 physicalM + read-only profile',no_primary_diagnostics=False)
    p['implementation']=dict(entry='source/benchmarks/qr_reuse_profile.py',cpu116_after_smoke=True,
        delta='Original hold source training unchanged. QR reuse opt-in shadow instance only; covariance, parameter update, auxiliary AdamW, M transport and V permutation retained.')
    p['decision']='Paired180/181 selected sparse candidate. This run qualifies a new QR implementation and profiles actual shapes; no fresh learner ranking or default change.'
    p['bound_rationale']='1270live updates plus32shadow steps/240factor endpoints/20aggregate atoms, fixed3blocks.900/960s,24GiBallocated,3GiBinitialCPU,100MiBcompact/512MiBcache/min5GiBfree. Explicit RAM purpose<20GiB, no stored weights or checkpoints.'
    return p


def validate(c,p):
    if c!=recipe() or p!=protocol():raise ValueError('Undeclared QR/profile delta')
