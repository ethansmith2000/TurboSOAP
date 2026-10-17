"""Single-state read-only BF16 ordinary-projection fidelity and complete-cost screen."""
import json
from pathlib import Path
from llm_math_learning_contract import evaluation_ids,ids_sha
from llm_feasibility_contract import PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,SNAPSHOT_CAP,DIMENSIONS,CACHE
SEED=187
SEEDS=(187,)
PARENT=Path('/workspace/optimizer_replay_results/llm_soap_reuse_confirmation_seed186_frozen_20261017')
SOURCE=Path('/workspace/optimizer_replay_results/llm_soap_native_profile_seed183_frozen_20261017')
HORIZON=1000
WARMUP=245
PROFILE_COUNTS=dict(shadow_optimizer_steps=22,projection_endpoints=20,adaptive_update_comparisons=4)
def checked(seed):
    if type(seed) is not int or seed!=SEED:raise ValueError('Only fresh187 projection profile declared')
def recipe(seed=SEED):
    checked(seed);c=json.loads((SOURCE/'config.json').read_text());c['seed']=seed;c['max_wall_seconds']=900
    c['matched_comparison']=dict(arms=['hold200'],steps_per_arm=HORIZON,execution_order=['hold200'],scope='Native hold1000 source, read-only FP32/BF16ordinary-projection qualification; no candidate learning')
    return c
def protocol(seed=SEED):
    checked(seed);p=json.loads((SOURCE/'protocol.json').read_text());p.update(seed=seed,maximum_shadow_optimizer_steps=22,maximum_all_optimizer_calls=1292)
    p['llm_data']['batch_plan']='Fresh187native1000batches/16000unique blocks; no wrap. Native source state required for new ordinary-projection arithmetic/cost question, not spent-state recovery.'
    p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in evaluation_ids(seed).items()}
    memory=dict(maximum_tensor_purpose_bytes=20*2**30,initial_snapshot_purpose_bytes=SNAPSHOT_PURPOSE,three_optimizer_state_banks_bytes=3*(2*142606336*4+2*PARAMETERS*4),parameter_backup_bytes=PARAMETERS*4,borrowed_gradients_bytes=PARAMETERS*4,adaptive_outputs_bytes=2*41943040*4,largest_qr_workspace_bytes=32*8192**2*4,projection_cast_and_outputs_bytes=6*8192**2*4,fingerprint_workspace_bytes=12*2**20)
    p['profile']=dict(counts=PROFILE_COUNTS,policies=['native','bf16'],blocks=3,repeats=3,
        input='Native source aftercall1000(matrixage999),actualQ/C/M/V and retained last gradient. Shadows repeat retained gradient, not unseen next-gradient or own candidate history.',
        complete_step='One qualification refresh-shadow per policy, then one ordinary-step warmup and3counterbalanced blocks x3ordinary repeats per policy. Native covariance/pointwise/auxiliary/Q/M/V unchanged. Full-step timings include per-call BF16casts, no cached basis.',
        atoms='Aggregate forward and reverse projections across all4matrices, native/BF16, one warmup+3blocks x3repeats. Separate atom times are not an additive decomposition.',
        approximation='Only ordinary high-context forward/reverse projection GEMMs use BF16inputs/output and return FP32. Stored state and covariance stay FP32. Every highest-context call, including physical-M and strict transport, uses original FP32method. No ordinary pointwise fusion, caching, cadence or QRreuse change.',
        qualification=dict(absolute_gram_fro=.05,normalized_gram=.005,physical_m_relative=.01,adaptive_update_relative=.01,adaptive_update_cosine=.9999,covariance_and_basis_hash_equal=True),
        decision='Original geometry/M/finite/source-restoration/release are hard guards. New1%relative/.9999cosine applies to current complete adaptive world-update BEFORE parameter rounding, under common retained gradient; independent BF16screen, not relaxation of182criterion. If it passes and complete ordinary-step saving >=5%in EVERY block, consider separate own-history learning; no learning/default acceptance from this diagnostic.',
        memory=memory,retention='Source/shadow states, backups, BF16casts and8adaptive outputs RAM-only. No tensorfiles/weights/checkpoints/traces; release all before source lease closes; compact evidence and arithmetic cache retained through this profile review.')
    p['memory']=memory;p['artifacts']['probe']=dict(purpose=p['profile']['retention'],location='GPU RAM only',expected_max_extra_bytes=20*2**30,retention='Release each shadow/endpoint and all profile banks before source lease closes')
    p['artifacts']['compiler_cache']['retention']='Arithmetic provenance through ordinary-projection review; no model state'
    p['implementation']=dict(entry='source/benchmarks/projection_profile.py',cpu116_after_smoke=True,delta='248parent sources unchanged plus7additive files. Native source and native QR; per-instance BF16 ordinary-projection adapter in private shadows only.')
    p['decision']='185/186support efficientQR20short-recipe baseline. Ordinary projections measured13.5ms of24.3ms ordinary-step endpoints; test one BF16arithmetic candidate read-only before learning. No new learner ranking.'
    p['bound_rationale']='1270live+22shadow=1292optimizer calls;20aggregate projection endpoints,4adaptive comparisons/8outputs,8shadow physicalM plus4source disposableM checks.900/960seconds,24GiBallocated/3GiBinitialCPU/100MiBcompact/512MiBcache/min5GiBdisk;explicitRAMpurpose<20GiB;no saved state/retry/queue/grid.'
    return p
def validate(c,p):
    if c!=recipe() or p!=protocol():raise ValueError('Undeclared projection-profile delta')
