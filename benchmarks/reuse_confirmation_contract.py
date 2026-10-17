"""Fresh186same-recipe confirmation; seed and arm order are the interventions."""
import json
from pathlib import Path
from reuse_learning_contract import recipe as previous_recipe, protocol as previous_protocol
from reuse_learning_contract import SCREEN,HORIZON,WARMUP,PRIMARY_M_AGES,OWN_REFERENCE_CALLS,counts,event_kind,CACHE,PARAMETERS,SNAPSHOT_CAP,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS
from llm_math_learning_contract import evaluation_ids,ids_sha
SEED=186
SEEDS=(186,)
ARMS=('reuse20','qr20')
PARENT=Path('/workspace/optimizer_replay_results/llm_soap_reuse_learning_seed185_frozen_20261017')
def checked(seed):
    if type(seed) is not int or seed!=SEED:raise ValueError('Only fresh186 confirmation declared')
def recipe(seed=SEED):
    checked(seed);c=previous_recipe();c['seed']=seed
    c['matched_comparison']['execution_order']=list(ARMS)
    return c
def protocol(seed=SEED):
    checked(seed);p=previous_protocol();p.update(seed=seed,execution_order=list(ARMS))
    p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in evaluation_ids(seed).items()}
    p['implementation']=dict(entry='source/benchmarks/reuse_confirmation.py',cpu116_after_smoke=True,source_delta='243parent files unchanged; five additive confirmation files. Exact185runner/kernel with explicit entrypoint binding for186launch verification and reversed execution order. No math/cadence/M/V/evaluator change.')
    p['confirmation']=dict(parent_seed=185,parent_bundle=str(PARENT),execution_order=list(ARMS),scope='User-authorized separate confirmation after185audit. Same7515calls/14panels/40actualM/gates/bounds/practicalmargin. Different seed and reversed arm order; not an isolated arm-order causal test.',paired_decision='Report185and186individually. Useful short-recipe support only if BOTH seeds independently pass every postwarmup0.01CEpanel and lower measured optimizer/iteration cost. Do not average away a failed panel/seed. No automatic third seed/grid/default promotion.',next='If supported, retain efficientQR20as a short-recipe baseline and prioritize measured ordinary costs or a separately scoped longer/deeper/backend transfer. No more confirmation seeds by default.')
    return p
def validate(c,p):
    if c!=recipe() or p!=protocol():raise ValueError('Undeclared reuse-confirmation delta')
