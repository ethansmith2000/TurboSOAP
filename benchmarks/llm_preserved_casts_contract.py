"""Fresh model-only precision-cast LLM pilot; optimizer math/gates retained."""
from pathlib import Path
from llm_feasibility_contract import ARMS,POLICIES,HORIZON,WARMUP,SNAPSHOT_CAP,PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS,CACHE
from llm_feasibility_contract import recipe as previous_recipe,protocol as previous_protocol
SEED=167
ORDER=('qr20','damped20','highest_residual20')
PARENTS=(Path('/workspace/optimizer_replay_results/llm_soap_feasibility_seed162_frozen_20261013'),)
QUALIFICATION=Path('/workspace/optimizer_replay_results/llm_compilation_diagnostic_seed166_frozen_20261013')
MODEL_OPTIONS={'emulate_precision_casts':True}
def recipe(base=None):
 c=previous_recipe();c['seed']=SEED;c['model_compile_options']=MODEL_OPTIONS.copy();c['matched_comparison']['execution_order']=list(ORDER)
 c['matched_comparison']['scope']='Fresh model-only preserved-cast causal-LM420 engineering/actualMV feasibility, not a spent-bundle retry or learning qualification'
 return c
def protocol():
 p=previous_protocol();p.update(seed=SEED,execution_order=list(ORDER))
 p['implementation']['entry']='source/benchmarks/llm_preserved_casts.py'
 p['implementation']['model_compile']=dict(options=MODEL_OPTIONS.copy(),mode='Implicit default, options cannot be combined with explicit mode',fullgraph=True,dynamic=False,scope='Model FlatOutput only, no global config patch or optimizer compiler options')
 p['implementation']['trainer_delta']='Subclass original LLM runtime: replace unused compiled model wrapper before any forward; preserve all optimizer/data/probe paths. New entry retains PID on failure.'
 p['compiler_isolation']=dict(global_emulate_precision_casts=False,model_emulate_precision_casts=True,optimizer_options='Original named policy kernels unchanged',guards='Before model construction and every budget call; model options before/after measured')
 p['artifacts']['compiler_cache']['retention']='Through next matched LLM learning/token-batch review; arithmetic provenance only'
 p['decision']='Fresh seed167 after zero-update seed166 GPU numerical qualification. Original failed seed162 gate/bundle retained. All original geometry/MV/parity/reset/ownership/default compiler-limit gates unchanged. Different arm order is a single pass, not wall-speed or statistical qualification.'
 return p
def validate(config,declared,base=None):
 if config!=recipe() or declared!=protocol():raise ValueError('Undeclared preserved-cast LLM delta')
