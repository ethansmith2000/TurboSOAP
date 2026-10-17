"""Read-only reuse observations, original guards and separately reported old fidelity screen."""
import json
from pathlib import Path
from llm_math_learning_contract import evaluation_ids,ids_sha
from llm_feasibility_contract import PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,SNAPSHOT_CAP,DIMENSIONS,CACHE
SEED=184
SEEDS=(184,)
PARENT=Path('/workspace/optimizer_replay_results/llm_soap_native_profile_seed183_frozen_20261017')
HORIZON=1000
WARMUP=245
OBSERVATION_CALLS=(200,400,1000)
PROFILE_COUNTS=dict(shadow_optimizer_steps=32,factor_basis_calls=160,decomposition_calls=80,covariance_endpoints=10,projection_endpoints=10,following_gradient_previews=8)
TOTAL_COUNTS=dict(PROFILE_COUNTS,shadow_optimizer_steps=36,following_gradient_previews=24)

def recipe(seed=SEED):
    if seed!=SEED:raise ValueError('Only fresh184 observation declared')
    c=json.loads((PARENT/'config.json').read_text());c['seed']=seed
    c['matched_comparison']=dict(arms=['hold200'],steps_per_arm=HORIZON,execution_order=['hold200'],scope='Fresh native source; read-only reuse at3ages plus fixed-state paired timing, no candidate learning')
    return c

def protocol(seed=SEED):
    if seed!=SEED:raise ValueError('Only fresh184 observation declared')
    p=json.loads((PARENT/'protocol.json').read_text());p.update(seed=seed,maximum_shadow_optimizer_steps=36,maximum_all_optimizer_calls=1306)
    p['llm_data']['batch_plan']='Fresh184 native1000batches/16000unique blocks; no wrap. State needed for three-age approximation observation and new paired timing, not spent-state recovery.'
    p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in evaluation_ids(seed).items()}
    f=p['profile'];f.update(counts=PROFILE_COUNTS,total_counts=TOTAL_COUNTS,observation_calls=list(OBSERVATION_CALLS),
        input='Actual native C/Q/M/V and retained gradients after source calls200/400/1000, matrix ages199/399/999. Each shadow repeats retained gradient to reach a refresh age200/400/1000; not the unseen actual next gradient.',
        complete_step='Native source only. Isolated native/reuse QR shadow steps and ordinary branch; restore parameters and all source/RNG/gradient/flag hashes. Candidate never inserted into source.',
        atoms='Eight actual factors: native orderedQR, CQ-reuse QR, decomposition on native prepared input. Aggregate covariance and two projection endpoints.',
        setup='At200/400: two qualification shadow steps only. At1000:2shadow observations+3fullstep warmups+27timed steps,240factor and20aggregate endpoints. Three cyclic blocks x3repeats.',
        qualification=dict(absolute_gram_fro=.05,normalized_gram=.005,physical_m_relative=.01,
            following_gradient_update_relative=.001,following_gradient_update_cosine=.99999,order_score_band_relative_to_max=.0001,
            adaptive_and_order_policy='Report original182criteria unchanged as legacy_fidelity_pass; finite action/order differences are observations, not hard guards or deployment/learning acceptance.'),
        decision='Original geometry/M, finite outputs, source restoration and parameter-hash agreement are hard read-only safety guards. Legacy182fidelity criteria are logged without stopping this separate observational task. No candidate acceptance threshold, learning/default promotion or retry. At least5% whole-refresh saving in every block is timing-only evidence for considering a later bounded learning check.',
        policies=['ordinary','native','reuse'],factor_policies=['native','reuse','decomposition'])
    f['memory'].pop('three_optimizer_state_banks_bytes')
    f['memory']['four_optimizer_state_banks_bytes']=4*(2*142606336*4+2*144875648*4)
    f['memory']['following_update_buffers_bytes']=2*8192*2048*4
    f['retention']='All source/candidate tensors and four optimizer banks RAM-only. Release after each three-age transaction and before source lease closes. No saved weights/checkpoints/tensorfiles/traces; compact evidence retained and arithmetic cache through observation review.'
    p['memory']=f['memory'];p['artifacts']['probe']['purpose']=f['retention'];p['artifacts']['compiler_cache']['retention']='Through three-age reuse observation review; arithmetic provenance only'
    p['implementation']=dict(entry='source/benchmarks/reuse_observation.py',cpu116_after_smoke=True,delta='Separate read-only task after user clarification; original237source lineage, legacy182gate explicitly reported rather than relabeled. No candidate live updates.')
    p['decision']='Native183cost measured. User regards0.134% as potentially acceptable; observe approximation at3ages and actual paired timing before deciding whether to spend on learning.182still failed originalscreen.'
    p['bound_rationale']='1270live+36shadow=1306optimizer calls,240factor endpoints/20aggregate atoms/24adaptive previews, fixed3ages/3timing blocks.900/960s,24GiBallocated,3GiBinitialCPU,100MiBcompact/512MiBcache/min5GiBdisk; explicit RAM<20GiB, no stored state.'
    return p

def validate(c,p):
    if c!=recipe() or p!=protocol():raise ValueError('Undeclared reuse-observation delta')
