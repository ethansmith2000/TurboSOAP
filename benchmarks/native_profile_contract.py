"""Bounded native-only cost diagnostic following frozen product-reuse rejection."""
import json
from pathlib import Path
from llm_math_learning_contract import evaluation_ids,ids_sha
from llm_feasibility_contract import PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,SNAPSHOT_CAP,DIMENSIONS,CACHE
SEED=183
SEEDS=(183,)
PARENT=Path('/workspace/optimizer_replay_results/llm_soap_qr_reuse_profile_seed182_frozen_20261016')
REJECTION=Path('/workspace/optimizer_replay_results/soap_qr_reuse_profile_readout_20261017/completion_audit.json')
HORIZON=1000
WARMUP=245
PROFILE_COUNTS=dict(shadow_optimizer_steps=21,factor_basis_calls=80,decomposition_calls=80,covariance_endpoints=10,projection_endpoints=10,following_gradient_previews=0)

def recipe(seed=SEED):
    if seed!=SEED:raise ValueError('Only fresh183 native profile declared')
    c=json.loads((PARENT/'config.json').read_text());c['seed']=seed
    c['matched_comparison']=dict(arms=['hold200'],steps_per_arm=HORIZON,execution_order=['hold200'],scope='Fresh183 natural state; native-only cost profile, no candidate retry or learning comparison')
    return c

def protocol(seed=SEED):
    if seed!=SEED:raise ValueError('Only fresh183 native profile declared')
    p=json.loads((PARENT/'protocol.json').read_text());p.update(seed=seed,maximum_shadow_optimizer_steps=21,maximum_all_optimizer_calls=1291)
    p['llm_data']['batch_plan']='Fresh183 fixed1000 batches/16000unique blocks; no wrap. New native-only cost question; do not rerun spent182.'
    p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in evaluation_ids(seed).items()}
    f=p['profile'];f.update(counts=PROFILE_COUNTS,input='Actual C/Q/M/V and retained gradients after fresh183 hold1000 calls; exact source restoration.',
        complete_step='Original native ordinary and QR200 branches only, no product reuse. Private optimizer copies share source parameters; restore GPU parameter backup after every step.',
        atoms='Actual eight factors: unchanged native orderedQR and decomposition alone on native prepared input. Aggregate covariance/projection endpoints; times not assumed additive.',
        setup='One native qualification step, one warmup each for2full-step kinds/2factor kinds/2aggregate kinds, then3counterbalanced blocks x3repeats.',
        qualification=dict(absolute_gram_fro=.05,normalized_gram=.005,physical_m_relative=.01),
        decision='Native-only measurements identify cost; no new implementation, numerical approximation, speedup claim or candidate promotion. Reuse182 stays rejected. No precision/cadence/learning grid or retry.',
        policies=['ordinary','native'],factor_policies=['native','decomposition'])
    f['memory']['four_optimizer_state_banks_bytes']=0
    f['memory']['three_optimizer_state_banks_bytes']=3*(2*142606336*4+2*144875648*4)
    f['memory']['following_update_buffers_bytes']=0
    f['memory'].pop('four_optimizer_state_banks_bytes')
    f['retention']='All trained parameters/state, three optimizer banks, gradients and temporary outputs RAM-only. Release before source lease closes. No tensors/weights/checkpoints/traces; compact evidence retained; arithmetic cache through native profile review.'
    p['memory']=f['memory'];p['artifacts']['probe']['purpose']=f['retention']
    p['artifacts']['compiler_cache']['retention']='Through native profile review; arithmetic provenance only'
    p['implementation']=dict(entry='source/benchmarks/native_profile.py',cpu116_after_smoke=True,delta='Native-only opt-in diagnostic; all225parent files retained unchanged, no reuse invocation.')
    p['decision']='Reuse182 failed action fidelity; preserve native sparse policy and complete independent cost profile. No relaxed threshold or reuse retry.'
    p['bound_rationale']='1270live+21shadow=1291optimizer calls,160factor endpoints/20aggregate atoms, fixed3blocks.900/960s,24GiBallocated,3GiBinitialCPU,100MiBcompact/512MiBcache/min5GiBdisk. RAM tensor purpose<20GiB; no saved weights.'
    return p

def validate(c,p):
    if c!=recipe() or p!=protocol():raise ValueError('Undeclared native-profile delta')
