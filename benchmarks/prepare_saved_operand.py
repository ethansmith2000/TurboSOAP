"""Freeze original-graph saved-operand capture, preserving completed evidence."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from prepare_cifar_benchmark import sha,REPO
from saved_operand_contract import validate_config,validate_child_change


def main():
    b=Path(sys.argv[1]).resolve();prior=Path('/workspace/optimizer_replay_results/cifar_soap_cudnn_deterministic_20261008')
    index=int(sys.argv[2]);uuid=sys.argv[3]
    assert index==7 and uuid=='GPU-866cad81-dc9b-cc40-afb8-c387d3ffba5f'
    if b.exists():raise ValueError('Use a new bundle')
    assert shutil.disk_usage(b.parent).free>5*2**30
    pm=json.loads((prior/'manifest.json').read_text())
    for name,h in pm['source_sha256'].items():assert sha(prior/'source'/name)==h,name
    for name,key in [('config.json','config_sha256'),('split.json','split_sha256'),('data_prepared.json','base_prepared_sha256'),('PROTOCOL.md','protocol_sha256')]:assert sha(prior/name)==pm[key]
    validate_child_change((prior/'source/benchmarks/cifar_cudnn_deterministic_child.py').read_text(),(REPO/'benchmarks/cifar_saved_operand_child.py').read_text())
    assert (REPO/'benchmarks/cifar_saved_operand.py').read_text()==(prior/'source/benchmarks/cifar_cudnn_deterministic.py').read_text().replace('cifar_cudnn_deterministic_child.py','cifar_saved_operand_child.py')
    b.mkdir();shutil.copytree(prior/'source',b/'source');shutil.copytree(prior/'upstream_reference',b/'upstream_reference')
    for name in ['split.json','data_prepared.json']:shutil.copy2(prior/name,b/name)
    names=['saved_operand_abi.py','saved_operand_replay.py','saved_operand_contract.py','cifar_saved_operand_child.py','cifar_saved_operand.py','prepare_saved_operand.py','summarize_saved_operand.py']
    for name in names:shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    c=json.loads((prior/'config.json').read_text());c['max_wall_seconds']=900;c['execution_diagnostic']['per_process_seconds']=210
    c['execution_diagnostic']['contexts']=[dict(label=f'plan{h}_flush25_r{r}_saved_operand',allocation_horizon=h,timing_chunk=25,repeat=r) for h,r in [(420,0),(3500,0),(3500,1),(420,1)]]
    validate_config(c);(b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    next_task=Path('/workspace/optimizer_replay_results/soap_feature_head_round_20261008/NEXT_TASK.md')
    (b/'PROTOCOL.md').write_text(next_task.read_text()+'''\n## Prospective implementation/execution declaration

Now prepared, not launched. The original deterministic child is unchanged after
stripping one new import/post-trajectory observer region. Old final25 and initial
replay precede the new capture. No feature-return graph is compiled in this task.
Exact outer with child entry rename. New four fresh processes/200updates provide
new saved-operand evidence, not completed historical reruns for context.

Read-only AST audit of actually loaded source identifies the classifier addmm
operand allocated bf16(256,1536), returned at the saved-index position. Verify the
corresponding backward input's shape/stride and use in the head-weight mm. One
forward/backward pair required. For the retained original code this is saved
index22 of30, not an unverified hardcoded selection. Local torch2.11 AOT code
combines node.saved_tensors and node._tensors_no_vc_check in that order; inspect
both existing collections without hooks/mutation. Require direct
CompiledFunctionBackward, total count/index/unique shape(256,1536)/bf16/stride
(1536,1)/offset0 to agree with verified ABI. Small CPUaot_eager check verifies
actual interface access and unchanged backward gradients; explicit tests reject
missing/ambiguous/wrong-index and corrupted source/result evidence.

Three original-graph calls: first/second plain, third RAM profile. Before backward
use ONLY tensor metadata and a reference to the existing selected operand.
No new graph outputs, hooks, retain_graph, CUDA clones/hashes/tensor math then.
After complete backward+sync record original fullgrad/output/loss/operand hashes.
Clone only the first operand afterward for three isolated-head calls
plain/plain/profile. Detached head input removes upstream gradient work; require
exact head/full-original output/loss/head-gradient parity before interpretation.
Require all captures match actual measured1 and old initial warm replay, and
repeat exactly. A held reference can change tensor lifetime/allocation; parity
remains mandatory. ABI/access failure still collects three original executions,
skips isolated head if no operand, records specific inconclusive failure, restores
state and continues remaining planned processes. No guessed input or flag retry.
All completed stages flushed as compact partial evidence. Diagnostic full code
metadata exactness is independently required for supported interpretation; no
trimming configurations after launch. Runtime names/counts recorded separately.

Purpose-specific retained finalweight/input/onebf16operand118,675,856bytes
(<128MiB), GPU RAM-only until restore then discard. Existing initial-state replay
retained copy alsoRAMonly and sequentially discarded. No optimizer snapshots,
weights/checkpoints/features/tensor/trace files. Transientactivations/compiler/
profiles governed separately by24GiBallocated. Exact full model/optimizer/current
RNG restoration and held invariants, rebuilt input/target, finite/Gram checks,
owned release and own masks mandatory. Primary exact repetition is a separate
unchanged gate, not required to collect the observer. No timing/learning claim.

GPU7 RTX5090 UUID GPU-866cad81-dc9b-cc40-afb8-c387d3ffba5f, free at preparation,
same device as baseline and latest study. Baseline backend ZERO deltas required.
Other projects/cache history/ambient load remain independent. Exclusive nonqueued
claim/private manifold supervisor/autostartfalse. Internal900/external960s,
child210 withinglobal900; no extension/retry. Expectedcompact<25MiB/hard100MiB,
about321GiB free. Preserve218existingPythonfiles/31 inventories+members/dirtytrees.
No cache edits/clears, environment/backend/library/driver/clocks/GC/otherjob changes,
commit/reset/push/delete or automatic next job/long learning. Compact seed/config/
source/provenance/log/metric/summary/manifests retained until explicitcleanup.
''')
    m={**pm,'source_sha256':{str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},'config_sha256':sha(b/'config.json'),'protocol_sha256':sha(b/'PROTOCOL.md'),
       'baseline_bundle':str(prior),'baseline_manifest_sha256':sha(prior/'manifest.json'),'baseline_summary_sha256':sha(prior/'summary.json'),
       'single_backend_flag_change':'None; primary flags unchanged','declared_baseline_backend_deltas':[],
       'device':dict(index=index,uuid=uuid,baseline_index=7,reason='Same free device as baseline/latest study'),
       'base_repository_head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'next_task_sha256':sha(next_task)}
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (b/'source_worktree.diff').write_text(subprocess.check_output(['git','-C',str(REPO),'diff'],text=True))
    (b/'source_worktree_status.txt').write_text(subprocess.check_output(['git','-C',str(REPO),'status','--short'],text=True))
    print(b)


if __name__=='__main__':main()
