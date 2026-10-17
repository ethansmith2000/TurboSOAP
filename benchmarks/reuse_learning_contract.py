"""One prospective fresh QR20 native/reuse own-history learning screen."""
import json
from pathlib import Path
from llm_math_learning_contract import evaluation_ids, ids_sha
from llm_feasibility_contract import CACHE, PARAMETERS, SNAPSHOT_CAP, SNAPSHOT_BYTES, SNAPSHOT_PURPOSE, DIMENSIONS
from llm_sparse_screen_contract import counts as old_counts, event_kind as old_event_kind
PARENT=Path('/workspace/optimizer_replay_results/llm_soap_reuse_observation_seed184_frozen_20261017')
SCREEN=Path('/workspace/optimizer_replay_results/llm_soap_sparse_screen_seed180_frozen_20261015')
SEED=185
SEEDS=(185,)
ARMS=('qr20','reuse20')
HORIZON=3500
WARMUP=245
OWN_REFERENCE_CALLS=()
PRIMARY_M_AGES=(20,200,1000,3000)
def checked(seed):
    if type(seed) is not int or seed!=SEED:raise ValueError('Only fresh185 declared')
def counts(calls,arm):
    if arm not in ARMS:raise ValueError('Undeclared learning arm')
    return old_counts(calls,'qr20')
def event_kind(call,arm):
    if arm not in ARMS:raise ValueError('Undeclared learning arm')
    return old_event_kind(call,'qr20')
def recipe(seed=SEED):
    checked(seed);c=json.loads((SCREEN/'config.json').read_text());c['seed']=seed;c['max_wall_seconds']=1800
    c['matched_comparison']=dict(arms=list(ARMS),execution_order=list(ARMS),steps_per_arm=HORIZON,scope='Single fresh paired native/reuseQR20 own-history learning, not equivalence/default promotion')
    return c
def protocol(seed=SEED):
    checked(seed);p=json.loads((SCREEN/'protocol.json').read_text())
    p.update(seed=seed,arms=list(ARMS),execution_order=list(ARMS),maximum_optimizer_updates=7515,empty_preflight_backward_calls=2,probe_calls=[])
    p['expected']={phase:{a:counts(n,a) for a in ARMS} for phase,n in [('disposable',WARMUP),('primary',HORIZON)]}
    p['expected'].update(hybrid_calls_by_dimension={},own_eager_factor_references=0,disposable_actual_m_transactions=8,primary_actual_m_transactions=32,actual_m_transactions=40,primary_counterfactual_basis_calls=0,evaluations_per_arm=7)
    p['cadence']={a:dict(method='qr',frequency=20,repair=0) for a in ARMS}
    p['gates'].pop('reference_basis_relative');p['gates'].pop('movement_relative');p['gates'].pop('movement_cosine')
    p['qualification']=dict(actual_m='Both arms: first disposable refresh per matrix, primary ages20/200/1000/3000 per matrix. Highest physical-M roundtrip original1% gate; own histories, no counterfactuals.',primary_m_ages=list(PRIMARY_M_AGES),exact_prefix='All model/optimizer/RNG hashes and245loss/gradient norm pairs match own disposable prefix',no_primary_counterfactuals=True)
    p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in evaluation_ids(seed).items()}
    p['timing']['primary_diagnostic_calls']=[v+1 for v in PRIMARY_M_AGES]
    p['timing']['interpretation']='Serialized optimizer/events and synchronized iteration costs. Separately report primary M-check calls; compare steady ordinary/refresh and full training costs. Setup/observations/all-in retained. One fixed arm order, no repeated-wall or deployed speed qualification.'
    p['precision']=dict(state='Original FP32 covariance/basis/M/V',ordinary='Original high/TF32',momentum='Original highest FP32 world transport',variance='Original QR permutation',reuse='Same qr_reuse_profile_kernel.refresh_reuse as184; reassociated score/product, not exact native arithmetic',compile='Original default/fullgraph/dynamic=False limits8/256; no new compiled path')
    p['implementation']=dict(entry='source/benchmarks/reuse_learning.py',cpu116_after_smoke=True,source_delta='Additive runner/adapter only; frozen parent237sources unchanged. NativeQR20 untouched, reuseQR20 uses already-observed reuse kernel. Same model/data/evaluator/native step/cadence/M/V.')
    p['decision']=dict(primary_question='Does CQ-reuseQR20 retain useful learning quality on its own history and reduce complete cost?',quality_margin_ce=.01,quality_calls=[1000,1500,2000,2500,3000,3500],practical_screen='Provisional survivor only if reuse validationCE <= native+0.01 at EVERY declared postwarmup panel and lower steady optimizer/iteration mean excluding declared M-check calls; report every panel including500 and complete phase/all-in. Otherwise tradeoff/unresolved; no automatic expansion.',limits='One seed and fixed order do not prove equivalence, long-run harmlessness, population effect or deployed speed.182originalcriterion failure remains. No reuse action-error threshold relabeled or promoted.',next='Review once; any confirmation/sparse-policy insertion/ordinary optimization requires separate bounded task.')
    p['artifacts']['probe']=dict(purpose='40 actual own-history physical-M transactions; two streamed rectangular buffers, no shadow states',location='GPU RAM only',expected_max_extra_bytes=2*8192*2048*4,retention='Release each refresh, completion or failure')
    p['artifacts']['compiler_cache']['retention']='Arithmetic provenance through this learning review; no model state'
    p['bounds']['internal_wall_seconds']=1800;p['bounds']['external_watchdog_seconds']=1860
    p['memory']['retention']='Initial snapshot CPU RAM only, active optimizer GPU RAM only, physical-M buffers released per diagnostic. No weights/checkpoints/tensorfiles/traces.'
    p['bound_rationale']='7515calls=25smoke+2*(245disposable+3500primary);14fixed token panels;8disposable+32primary actual-M checks.1800internal/1860external seconds,24GiBallocated/3GiBinitialCPU/100MiBcompact/512MiBcache/min5GiBdisk. No retry/queue/statefiles/counterfactuals/extra seed.'
    return p
def validate(c,p):
    if c!=recipe() or p!=protocol():raise ValueError('Undeclared reuse-learning delta')
