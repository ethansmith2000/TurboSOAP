"""One native full-schedule trajectory; all approximation observations read-only."""
import json
from pathlib import Path
from projection_learning_contract import recipe as old_recipe,protocol as old_protocol
from projection_learning_contract import CACHE,PARAMETERS,SNAPSHOT_CAP,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS
from reuse_learning_contract import counts as parent_counts,event_kind as parent_kind
from llm_math_learning_contract import evaluation_ids,ids_sha
PARENT=Path('/workspace/optimizer_replay_results/llm_soap_projection_confirmation_seed189_frozen_20261017')
SCREEN=Path('/workspace/optimizer_replay_results/soap_projection_learning_paired_20261018')
SEED=190
SEEDS=(190,)
ARMS=('qr20',)
HORIZON=35000
WARMUP=245
STAGES=(260,1000,3500,10000,20000,30000,35000)
LAGS=(20,200)
PRIMARY_M_AGES=(20,200,1000,3000,10000,20000,30000,34980)
OWN_REFERENCE_CALLS=()
def checked(seed):
 if type(seed) is not int or seed!=SEED:raise ValueError('Only fresh190 declared')
def counts(calls,arm):
 if arm!='qr20':raise ValueError('Native QR20 source only')
 return parent_counts(calls,arm)
def event_kind(call,arm):
 if arm!='qr20':raise ValueError('Native QR20 source only')
 return parent_kind(call,arm)
def recipe(seed=SEED):
 checked(seed);c=old_recipe();c.update(seed=seed,max_wall_seconds=5400)
 c['matched_comparison']=dict(arms=list(ARMS),execution_order=list(ARMS),steps_per_arm=HORIZON,scope='Native QR20 full35000schedule with deterministic repeated data; read-only stage diagnostics, no candidate learning or promotion')
 return c
def protocol(seed=SEED):
 checked(seed);p=old_protocol();p.update(seed=seed,arms=list(ARMS),execution_order=list(ARMS),steps_per_arm=HORIZON,maximum_optimizer_updates=35270,empty_preflight_backward_calls=1)
 p['expected']={phase:{'qr20':counts(n,'qr20')} for phase,n in [('disposable',WARMUP),('primary',HORIZON)]}
 p['expected'].update(hybrid_calls_by_dimension={},own_eager_factor_references=0,disposable_actual_m_transactions=4,primary_actual_m_transactions=32,actual_m_transactions=36,primary_counterfactual_basis_calls=0,evaluations_per_arm=6)
 p['cadence']={'qr20':dict(method='qr',frequency=20,repair=0)}
 p['guard_calls']=sorted(set(p['guard_calls']+list(STAGES)+[a+1 for a in PRIMARY_M_AGES]))
 p['evaluation_calls']=[1000,3500,10000,20000,30000,35000]
 p['sampled_batch_calls']=sorted(set(p['sampled_batch_calls']+list(STAGES)))
 p['qualification']['primary_m_ages']=list(PRIMARY_M_AGES)
 p['qualification']['actual_m']='Native source first disposable refresh and primary ages20/200/1000/3000/10000/20000/30000/34980, four matrices each; original highest physical-M1%gate'
 p['timing']['primary_diagnostic_calls']=[a+1 for a in PRIMARY_M_AGES]
 p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in evaluation_ids(seed).items()}
 p['llm_data']['batch_plan']='Full batches from deterministic seed+2001permutations of fixed train blocks, reshuffle each exhausted full-batch epoch, drop incomplete tail; final partial epoch stops at35000batches. Same initial prefix for preflight/disposable/primary. No fresh data; repeated-data scope, epoch boundaries/planhash journaled.'
 p['llm_data']['training_tokens_per_primary_arm']=HORIZON*8192
 p['stage_stability']=dict(stages=list(STAGES),lags=list(LAGS),source='Native QR20, original35000schedule/875warmup; no candidate writes or skipped covariance updates',covariance='Relative Frobenius drift, trace ratio and trace-normalized shape drift over20/200source updates',gradient='Cosine, relative drift and norm ratio of retained clipped gradients; gradients can stay noisy despite stable covariance',basis='Highest-FP32 offdiagonal-Frobenius/||C|| for currentQ,lag20Q,lag200Q and one read-only native QR iteration on currentC/currentQ. Actual capture/source/refresh calls recorded; no raw-coordinate basis-distance claim.',adaptive='Same hypothetical following retained gradient: original native forward/back versus exact187BF16projection adapter, current native M/V/Q. Scalar rate cancels. Four comparisons per stage, no parameter writes/candidate history.',bf16_screen=dict(relative=.01,cosine=.9999,interpretation='Report passes/exceedances as observations; finite/source/geometry/M/release remain hard, candidate screen does not abort source'),counts=dict(observations=7,history_banks=14,factor_rows=56,gradient_rows=56,adaptive_comparisons=28,native_qr_factor_endpoints=56,extra_optimizer_steps=0,extra_backwards=0),retention='At most two C/Q/gradient history banks in GPU RAM for200/20call windows; release at each stage or failure. No tensorfiles/checkpoints/weights/traces/CPU covariance banks.')
 p['precision']=dict(source='Unchanged native high/TF32 QR20 and FP32storedstate/covariance/pointwise, highest physical-M transport',diagnostic='Highest-FP32 metrics; native QR reference uses original high-context arithmetic; BF16ordinary projection preview only, no source insertion',compile='Original defaults8/256/fullgraph/dynamicFalse; no new compiled path')
 p['memory']=dict(maximum_tensor_purpose_bytes=20*2**30,initial_snapshot_purpose_bytes=SNAPSHOT_PURPOSE,active_optimizer_state_bytes=2*142606336*4+2*PARAMETERS*4,active_model_bytes=PARAMETERS*4,largest_factor_workspace_bytes=32*8192**2*4,history_banks_bytes=2*(2*142606336+41943040)*4,borrowed_gradients_bytes=PARAMETERS*4,projection_cast_outputs_bytes=6*8192**2*4,metric_workspace_bytes=4*8192**2*4,adaptive_workspace_bytes=6*8192*2048*4,physical_m_workspace_bytes=2*8192*2048*4,fingerprint_workspace_bytes=12*2**20,retention=p['stage_stability']['retention'])
 p['bounds'].update(internal_wall_seconds=5400,external_watchdog_seconds=5460)
 p['artifacts']['probe']=dict(purpose=p['stage_stability']['retention'],location='GPU RAM only',expected_max_extra_bytes=p['memory']['history_banks_bytes']+p['memory']['projection_cast_outputs_bytes']+p['memory']['metric_workspace_bytes']+p['memory']['adaptive_workspace_bytes'],retention='Release at each observed stage or completion/failure')
 p['artifacts']['compiler_cache']['retention']='Arithmetic provenance through full-schedule stage-stability review; no model state'
 p['implementation']=dict(entry='source/benchmarks/stage_stability.py',cpu116_after_smoke=True,source_delta='266parent sources unchanged plus8additive files; original native runner with declared long source/data plan and read-only observers. Same model/eager MATH attention/native optimizer/evaluator/guards.')
 p['decision']=dict(question='Do covariance/basis statistics stabilize across full schedule while gradients stay noisy, and does BF16update fidelity change with stage?',analysis='Descriptive per-factor/per-matrix early260/near1000/pilot3500/mid10000/late20000,30000,35000, fixed20/200lags and common QR refresh phase; report scale/shape/residual/gradient/update/LR/dataepochs and source validation. No monotonicity gate or population causal claim; correlated factors are not independent seeds.',limits='Repeated40Mtoken data over286.72Mtraining tokens may memorize; LR decay/EMA initialization/data reuse confound age. Not new-data pretraining, BF16own-history learning, stale-update learning, adaptive policy or combinedQRreuse qualification.',next='Review once; any staged refresh/covariance skipping/long BF16learning/combined policy requires a separate bounded task. No automatic grid or extra seed.')
 p['bound_rationale']='35270optimizer calls=25smoke+245disposable+35000nativeprimary,6token panels,36actualM,7read-only observations/14boundedRAMbanks/56nativeQRfactorendpoints/28adaptivecomparisons,zeroextraupdates/backwards.5400/5460s,24GiBallocated/3GiBinitialCPU/100MiBcompact/512MiBcache/min5GiBdisk,<20GiBexplicitpurpose;no saved state/retry/queue.'
 return p
def validate(c,p):
 if c!=recipe() or p!=protocol():raise ValueError('Undeclared stage-stability delta')
