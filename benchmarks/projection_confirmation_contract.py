"""Fresh189same-recipe confirmation; seed and arm order are the interventions."""
import json
from pathlib import Path
from projection_learning_contract import recipe as previous_recipe, protocol as previous_protocol
from projection_learning_contract import SCREEN,HORIZON,WARMUP,PRIMARY_M_AGES,OWN_REFERENCE_CALLS,counts,event_kind,CACHE,PARAMETERS,SNAPSHOT_CAP,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS
from llm_math_learning_contract import evaluation_ids,ids_sha
SEED=189
SEEDS=(189,)
ARMS=('bf1620','qr20')
PARENT=Path('/workspace/optimizer_replay_results/llm_soap_projection_learning_seed188_frozen_20261017')
REVIEW=Path('/workspace/optimizer_replay_results/soap_projection_learning_readout_20261017')
def checked(seed):
    if type(seed) is not int or seed!=SEED:raise ValueError('Only fresh189 confirmation declared')
def recipe(seed=SEED):
    checked(seed);c=previous_recipe();c['seed']=seed
    c['matched_comparison']['execution_order']=list(ARMS)
    return c
def protocol(seed=SEED):
    checked(seed);p=previous_protocol();p.update(seed=seed,execution_order=list(ARMS))
    p['llm_evaluation']['panels']={n:dict(blocks=len(v),valid_tokens=len(v)*512,ids_sha256=ids_sha(v)) for n,v in evaluation_ids(seed).items()}
    p['implementation']=dict(entry='source/benchmarks/projection_confirmation.py',cpu116_after_smoke=True,source_delta='261parent files unchanged; five additive confirmation files. Exact188runner/kernel with explicit entrypoint binding for189launch verification and reversed execution order. No math/cadence/M/V/evaluator change.')
    p['confirmation']=dict(parent_seed=188,parent_bundle=str(PARENT),execution_order=list(ARMS),scope='User-authorized separate confirmation after188audit. Same7515calls/14panels/40actualM/gates/bounds/practicalmargin. Different seed and reversed arm order; not an isolated arm-order causal test.',paired_decision='Report188and189individually. Useful short-recipe support only if BOTH seeds independently pass every postwarmup0.01CEpanel and lower measured optimizer/iteration cost. Do not average away a failed panel/seed. No automatic third seed/grid/default promotion.',next='If supported, retain BF16projectionQR20as a short-recipe survivor; scope any combination with QRreuse/sparse policy or longer/deeper/backend transfer separately. No more confirmation seeds by default.')
    return p
def validate(c,p):
    if c!=recipe() or p!=protocol():raise ValueError('Undeclared projection-confirmation delta')
