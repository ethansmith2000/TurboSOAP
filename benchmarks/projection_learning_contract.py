"""Fresh188 nativeQR20/BF16projectionQR20 own-history learning screen."""
import json
from pathlib import Path
from reuse_learning_contract import recipe as old_recipe,protocol as old_protocol,counts as old_counts,event_kind as old_kind
from llm_math_learning_contract import evaluation_ids,ids_sha
from llm_feasibility_contract import CACHE,PARAMETERS,SNAPSHOT_CAP,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS
PARENT=Path('/workspace/optimizer_replay_results/llm_soap_projection_profile_seed187_frozen_20261017')
SCREEN=Path('/workspace/optimizer_replay_results/soap_projection_profile_readout_20261017')
SEED=188
SEEDS=(188,)
ARMS=('qr20','bf1620')
HORIZON=3500
WARMUP=245
OWN_REFERENCE_CALLS=()
PRIMARY_M_AGES=(20,200,1000,3000)
def checked(seed):
    if type(seed) is not int or seed!=SEED:raise ValueError('Only fresh188 declared')
def counts(calls,arm):
    if arm not in ARMS:raise ValueError('Undeclared projection-learning arm')
    return old_counts(calls,'qr20')
def event_kind(call,arm):
    if arm not in ARMS:raise ValueError('Undeclared projection-learning arm')
    return old_kind(call,'qr20')
def recipe(seed=SEED):
    checked(seed);c=old_recipe();c['seed']=seed
    c['matched_comparison']=dict(arms=list(ARMS),execution_order=list(ARMS),steps_per_arm=HORIZON,scope='Fresh paired nativeQR20/BF16projectionQR20 own-history learning; isolate projection precision, no QR product reuse/default promotion')
    return c
def protocol(seed=SEED):
    checked(seed);p=old_protocol();p.update(seed=seed,arms=list(ARMS),execution_order=list(ARMS))
    p['expected']={phase:{a:counts(n,a) for a in ARMS} for phase,n in [('disposable',WARMUP),('primary',HORIZON)]}
    p['expected'].update(hybrid_calls_by_dimension={},own_eager_factor_references=0,disposable_actual_m_transactions=8,primary_actual_m_transactions=32,actual_m_transactions=40,primary_counterfactual_basis_calls=0,evaluations_per_arm=7)
    p['cadence']={a:dict(method='qr',frequency=20,repair=0) for a in ARMS}
    p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in evaluation_ids(seed).items()}
    p['precision']=dict(state='Original FP32 covariance/basis/M/V',ordinary='Native high/TF32 versus same187 adapter: BF16 operands/intermediate/output for forward/reverse projections, return FP32; casts inside every call, no BF16basis cache',momentum='Original highest FP32 world transport and actual-M observations; adapter bypasses BF16 in highest context',variance='Original QR permutation',refresh='Original native QR20 in both arms; no covariance-product reuse',compile='Original default/fullgraph/dynamic=False limits8/256, no new compiled path')
    p['implementation']=dict(entry='source/benchmarks/projection_learning.py',cpu116_after_smoke=True,source_delta='255parent sources unchanged plus6additive files; runner follows185 with per-instance187projection adapter only; original physical-M sampler unchanged')
    p['decision']=dict(primary_question='Does BF16 ordinary-projection QR20 retain useful learning quality on its own history and reduce complete cost?',quality_margin_ce=.01,quality_calls=[1000,1500,2000,2500,3000,3500],practical_screen='Provisional survivor only if BF16 validationCE <= native+0.01 at EVERY postwarmup panel and lower steady optimizer/iteration means excluding first50 and actual-M calls21/201/1001/3001; report500 and every panel/full phase/all-in. Otherwise tradeoff/unresolved; no automatic expansion.',limits='187qualified one common native source with0.45–0.51% adaptive discrepancy and20.4% ordinary-step saving; it did not establish own-history learning or training speed. One seed/fixed order cannot prove equivalence/population benefit/deployed throughput.182originalthresholdfailure unchanged.',next='Review once before any confirmation, combined QRreuse/BF16, sparse-policy insertion or default promotion; no automatic extra seed.')
    p['memory']=dict(maximum_tensor_purpose_bytes=16*2**30,initial_snapshot_purpose_bytes=SNAPSHOT_PURPOSE,active_optimizer_state_bytes=2*142606336*4+2*PARAMETERS*4,largest_factor_workspace_bytes=32*8192**2*4,rectangular_m_workspace_bytes=2*8192*2048*4,borrowed_gradients_bytes=PARAMETERS*4,projection_cast_and_outputs_bytes=6*8192**2*4,fingerprint_workspace_bytes=12*2**20,retention='Initial snapshot CPU RAM, one active optimizer GPU RAM, per-call BF16casts and diagnostic physical-M buffers RAM-only. Release per call/refresh or completion/failure. No saved state.')
    p['artifacts']['probe']['purpose']='40actual own-history physical-M transactions plus transient BF16projection casts; no shadow optimizer states'
    p['artifacts']['probe']['expected_max_extra_bytes']=2*8192*2048*4+6*8192**2*4
    p['artifacts']['probe']['retention']='Physical-M buffers per diagnostic and BF16casts/outputs per projection call, all RAM-only; release on completion/failure'
    p['artifacts']['compiler_cache']['retention']='Arithmetic provenance through BF16 own-history learning review; no model state'
    p['bound_rationale']='7515calls=25smoke+2*(245disposable+3500primary),14fixed token panels,40actual-M checks.1800/1860s,24GiBallocated/3GiBinitialCPU/100MiBcompact/512MiBcache/min5GiBdisk,explicit tensor purpose<16GiB. No retry/queue/statefiles/counterfactuals/extra seed.'
    return p
def validate(c,p):
    if c!=recipe() or p!=protocol():raise ValueError('Undeclared projection-learning delta')
