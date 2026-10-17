"""Common eager model, original compiled SOAP; fresh bounded LLM pilot."""
from pathlib import Path
from llm_feasibility_contract import ARMS,POLICIES,HORIZON,WARMUP,SNAPSHOT_CAP,PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS,CACHE
from llm_feasibility_contract import recipe as previous_recipe,protocol as previous_protocol
SEED=168
ORDER=('qr20','damped20','highest_residual20')
PARENTS=(Path('/workspace/optimizer_replay_results/llm_soap_preserved_casts_seed167_frozen_20261013'),)
QUALIFICATION=Path('/workspace/optimizer_replay_results/llm_compilation_diagnostic_seed166_frozen_20261013')
def recipe(base=None):
 c=previous_recipe();c['seed']=SEED;c['model_execution']='Common eager FlatOutput Transformer reference; SOAP kernels remain compiled';c['matched_comparison']['execution_order']=list(ORDER)
 c['matched_comparison']['scope']='Fresh common-eager-model causal-LM420 optimizer feasibility, no compiled-model/learning/end-to-end timing qualification'
 return c
def protocol():
 p=previous_protocol();p.update(seed=SEED,execution_order=list(ORDER))
 p['implementation']['entry']='source/benchmarks/llm_eager_model.py'
 p['implementation']['model_execution']='Common original eager FlatOutput Transformer across all arms; unused original compile wrapper replaced before any forward, model math unchanged'
 p['implementation']['trainer_delta']='Eager model execution only; original compiled optimizer/probe/data paths. Retain PID on all progress/failure status writes.'
 p['compiler_isolation']=dict(global_emulate_precision_casts=False,model='eager reference, not compiled',optimizer_options='Original named policy kernels unchanged',guards='Global setting every budget; actual model wrapper type before/after measured')
 p['model_parity']['reference']='Same eager model implementation consistency only, not compiled-model qualification; original tolerance unchanged'
 p['artifacts']['compiler_cache'].update(expected_bytes=250*2**20,retention='Through next matched LLM learning/token-batch review; optimizer arithmetic provenance only')
 p['decision']='After trained-state compiled-model fidelity rejection seed167, use common eager model to study optimizer fidelity. No failed-bundle retry or gate relaxation. Compiler fidelity remains separately unresolved; no end-to-end compiled speed/model/learning/default qualification.'
 return p
def validate(config,declared,base=None):
 if config!=recipe() or declared!=protocol():raise ValueError('Undeclared common-eager LLM delta')
