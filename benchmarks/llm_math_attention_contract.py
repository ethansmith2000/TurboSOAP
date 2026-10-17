"""Fresh common math-only attention LLM optimizer pilot; exact gates preserved."""
from pathlib import Path
from llm_eager_model_contract import ARMS,POLICIES,HORIZON,WARMUP,SNAPSHOT_CAP,PARAMETERS,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE,DIMENSIONS,CACHE,ORDER
from llm_eager_model_contract import recipe as previous_recipe,protocol as previous_protocol
SEED=170
PARENTS=(Path('/workspace/optimizer_replay_results/llm_soap_eager_model_seed168_frozen_20261013'),)
QUALIFICATION=Path('/workspace/optimizer_replay_results/llm_backward_repeatability_seed169_frozen_20261013')
def recipe(base=None):
 c=previous_recipe();c['seed']=SEED;c['attention_backend']='Common SDPA math-only context; eager model, original compiled SOAP'
 return c
def protocol():
 p=previous_protocol();p['seed']=SEED;p['implementation']['entry']='source/benchmarks/llm_math_attention.py'
 p['implementation']['attention_backend']=dict(policy='SDPBackend.MATH only, scoped to model autocast/forward for ALL arms',outside_flags='Restore original flags on exit; original SOAP/global numerical precision unchanged',qualification='Seed169initialstate3mathpasses exact; smoke/preflight/disposable/primary must requalify')
 p['decision']='Freshseed170 after measured Flashbackward tiny nonrepeatability/Math exact control. No spentbundle retry or exactgate relaxation. Common declared mathattention arithmetic is not bit-identical to Flash; allarms use the same model backend. No compiledmodel/end-to-end timing/quality/noise-cause/default qualification.'
 return p
def validate(config,declared,base=None):
 if config!=recipe() or declared!=protocol():raise ValueError('Undeclared math-attention LLM delta')
