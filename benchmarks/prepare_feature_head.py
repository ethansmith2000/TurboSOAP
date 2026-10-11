"""Freeze four fresh contexts with unchanged primary and later feature observer."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from prepare_cifar_benchmark import sha,REPO
from feature_head_contract import validate_config,validate_child_change


def main():
    b=Path(sys.argv[1]).resolve();prior=Path('/workspace/optimizer_replay_results/cifar_soap_cudnn_deterministic_20261008')
    index=int(sys.argv[2]);uuid=sys.argv[3]
    assert index==7 and uuid=='GPU-866cad81-dc9b-cc40-afb8-c387d3ffba5f'
    if b.exists():raise ValueError('Use a new bundle')
    assert shutil.disk_usage(b.parent).free>5*2**30
    pm=json.loads((prior/'manifest.json').read_text())
    for name,h in pm['source_sha256'].items():assert sha(prior/'source'/name)==h,name
    for name,key in [('config.json','config_sha256'),('split.json','split_sha256'),('data_prepared.json','base_prepared_sha256'),('PROTOCOL.md','protocol_sha256')]:assert sha(prior/name)==pm[key]
    validate_child_change((prior/'source/benchmarks/cifar_cudnn_deterministic_child.py').read_text(),(REPO/'benchmarks/cifar_feature_head_child.py').read_text())
    assert (REPO/'benchmarks/cifar_feature_head.py').read_text()==(prior/'source/benchmarks/cifar_cudnn_deterministic.py').read_text().replace('cifar_cudnn_deterministic_child.py','cifar_feature_head_child.py')
    b.mkdir();shutil.copytree(prior/'source',b/'source');shutil.copytree(prior/'upstream_reference',b/'upstream_reference')
    for name in ['split.json','data_prepared.json']:shutil.copy2(prior/name,b/name)
    names=['feature_head_replay.py','feature_head_contract.py','cifar_feature_head_child.py','cifar_feature_head.py','prepare_feature_head.py','summarize_feature_head.py']
    for name in names:shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    c=json.loads((prior/'config.json').read_text());c['max_wall_seconds']=900
    c['execution_diagnostic']['per_process_seconds']=210
    c['execution_diagnostic']['contexts']=[dict(label=f'plan{h}_flush25_r{r}_feature_head',allocation_horizon=h,timing_chunk=25,repeat=r) for h,r in [(420,0),(3500,0),(3500,1),(420,1)]]
    validate_config(c);(b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    next_task=Path('/workspace/optimizer_replay_results/soap_cudnn_deterministic_round_20261008/NEXT_TASK.md')
    (b/'PROTOCOL.md').write_text(next_task.read_text()+'''\n## Prospective implementation and execution declaration

Now prepared, not yet launched. AST audit requires exact preceding primary,
final25 and initial-state replays after stripping only one added import and the
new feature/head observer region after the preceding replay. Exact outer source
with child entry rename. Four fresh contexts200updates total; prior studies are
not rerun for context. Instrumented timings are not throughput qualification.

Separately compiled FeatureReturn shares the original model parameters, reproduces
ordered ViT operations and explicitly casts CLS features to bf16 before passing
that SAME operand to the original head module. Returning features and this
explicit equivalent autocast cast change the diagnostic graph; CPU independent
original-forward checks cover zero/nonzero classifier gradients, but actualGPU
primary-gradient equality remains mandatory for interpretation. Diagnostic feature
calls: first/second plain and third RAM profile. Head calls: first/second plain,
third RAM profile on one detached cloned operand from the first feature call.
The detached operand removes upstream input-gradient work from the isolated head
graph, another declared graph confound; require exact head-weight/bias gradient,
loss/logit parity to the feature diagnostic before attributing its result.
All feature calls must match both actual measured1 and existing initial warm
plain execution; all repeated feature/held-head calls must be exact for supported
interpretation. Unsupported parity is recorded as INCONCLUSIVE, not numerical
failure/retry; complete the remaining declared contexts. Primary exact repetition
remains a separate unchanged gate, not required to collect this diagnostic.

Hash FP32 CLS features and the actual BF16 head operand. Keep only one cloned
BF16 operand (~.75MiB), finalmodelcopy114,641,296bytes, firstinput/target3,248,128
bytes: expected118,675,856 retained bytes (<128MiB) during this observer. These
RAM-only copies are purpose-specific; no optimizer tensors copied. Transient
model activations/profiler/compiler work are additional, governed by24GiBallocated
cap. Complete parameter/optimizer/currentRNG invariants, exact rebuiltinput/target,
finite observer records and own release/masks mandatory. No disk tensors/weights/
checkpoints/traces. Each completed observer stage is flushed as compact partial
evidence; timeout/exception retains primary records and completed observer stages.

GPU7 RTX5090 UUID GPU-866cad81-dc9b-cc40-afb8-c387d3ffba5f, same as the previous
control and free at preparation. Baseline backend metadata must have ZERO deltas;
within-control primary/code/backend metadata exact, diagnostic loaded-code
exactness separately required for interpretation. Ambient load/cache history is
recorded, not forced identical; other projects remain independent. Exclusive
nonqueued claim/private manifold supervisor/autostartfalse only. No sharedcache
edits/clears, flag/environment/library/driver changes, clocks/GC/otherjob changes.

Internal900/external960s, child210 withinremaining900; no retries or extensions.
Prior four contexts401.354s; additive compilation/feature/head replay may consume
more, not a runtime promise. About321GiB free. Expected compact<25MiB/hard100MiB.
Preserve211 existingPythonfiles/29 inventories+members/dirtytrees; no agentcommit/
reset/push/delete. Compact config/source/seed/provenance/metrics/logs/summaries/
manifests retained until explicitcleanup. No automatic next job or long learning.
''')
    m={**pm,'source_sha256':{str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},'config_sha256':sha(b/'config.json'),'protocol_sha256':sha(b/'PROTOCOL.md'),
       'baseline_bundle':str(prior),'baseline_manifest_sha256':sha(prior/'manifest.json'),'baseline_summary_sha256':sha(prior/'summary.json'),
       'single_backend_flag_change':'None; primary backend flags unchanged','declared_baseline_backend_deltas':[],
       'device':dict(index=index,uuid=uuid,baseline_index=7,reason='Same device as preceding control, free at preparation'),
       'base_repository_head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
       'next_task_sha256':sha(next_task)}
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (b/'source_worktree.diff').write_text(subprocess.check_output(['git','-C',str(REPO),'diff'],text=True))
    (b/'source_worktree_status.txt').write_text(subprocess.check_output(['git','-C',str(REPO),'status','--short'],text=True))
    print(b)


if __name__=='__main__':main()
