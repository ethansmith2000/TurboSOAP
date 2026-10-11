"""Freeze the declared cuDNN flag control and post-trajectory RAM replay."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from prepare_cifar_benchmark import sha,REPO
from cudnn_deterministic_contract import validate_config,validate_child_change


def main():
    b=Path(sys.argv[1]).resolve();prior=Path('/workspace/optimizer_replay_results/cifar_soap_benchmark_off_20261008')
    index=int(sys.argv[2]);uuid=sys.argv[3]
    assert index==7 and uuid=='GPU-866cad81-dc9b-cc40-afb8-c387d3ffba5f'
    if b.exists():raise ValueError('Use a new bundle')
    assert shutil.disk_usage(b.parent).free>5*2**30
    pm=json.loads((prior/'manifest.json').read_text())
    for name,h in pm['source_sha256'].items():assert sha(prior/'source'/name)==h,name
    for name,key in [('config.json','config_sha256'),('split.json','split_sha256'),('data_prepared.json','base_prepared_sha256'),('PROTOCOL.md','protocol_sha256')]:assert sha(prior/name)==pm[key]
    validate_child_change((prior/'source/benchmarks/cifar_benchmark_off_child.py').read_text(),(REPO/'benchmarks/cifar_cudnn_deterministic_child.py').read_text())
    assert (REPO/'benchmarks/cifar_cudnn_deterministic.py').read_text()==(prior/'source/benchmarks/cifar_benchmark_off.py').read_text().replace('cifar_benchmark_off_child.py','cifar_cudnn_deterministic_child.py')
    b.mkdir();shutil.copytree(prior/'source',b/'source');shutil.copytree(prior/'upstream_reference',b/'upstream_reference')
    for name in ['split.json','data_prepared.json']:shutil.copy2(prior/name,b/name)
    names=['initial_state_replay.py','cudnn_deterministic_contract.py','cifar_cudnn_deterministic_child.py','cifar_cudnn_deterministic.py','prepare_cudnn_deterministic.py','summarize_cudnn_deterministic.py']
    for name in names:shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    c=json.loads((prior/'config.json').read_text())
    c['execution_diagnostic']['contexts']=[dict(label=f'plan{h}_flush25_r{r}_cudnn_deterministic',allocation_horizon=h,timing_chunk=25,repeat=r) for h,r in [(420,0),(3500,0),(3500,1),(420,1)]]
    validate_config(c);(b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    next_task=Path('/workspace/optimizer_replay_results/soap_benchmark_off_round_20261008/NEXT_TASK.md')
    (b/'PROTOCOL.md').write_text(next_task.read_text()+'''\n## Prospective execution declaration (before launch)

The above proposed task is now prepared, not yet launched. Source audit permits
only the cuDNN deterministic=True assignment in the primary path plus the
specified import/post-trajectory replay region. Outer source differs solely by
child entry name. Existing final25 replay remains required and precedes the new
initial-state replay. Initial plain/profile mismatch is recorded separately,
not silently substituted for trajectory repetition qualification.

Device amendment: GPU3 remains held by adalora, GPU6 by altnorm_research. Use one
free RTX5090 GPU7 UUID GPU-866cad81-dc9b-cc40-afb8-c387d3ffba5f instead. No queue or
other-job changes. All four fresh contexts use this single device. Compared with
old GPU3 evidence, allow exactly cudnn_flags.deterministic and
 environment.CUDA_VISIBLE_DEVICES differences; require every other recorded
backend field identical. Cross-study differences cannot uniquely establish flag
causality because device and ambient load/cache history differ. Within-control
backend/code exactness remains mandatory. Do not relax any post-launch gate.

Private manifold supervisor, autostart=false, one nonqueued exclusive claim;
internal600/external660s/child150 remain fixed. About322GiB free. No retry or
extension. Expected compact artifacts<25MiB, hard100MiB; allocated cap24GiB.
RAM-only final-model copy plus regenerated initial input<120MiB, retained only
for the specified same-state diagnostic. No weights/checkpoints/tensor/trace
files. Compact source/config/seed/provenance/log/metrics/summary/manifests retained
until explicit cleanup. Preserve185 existing Python files and27 prior inventories
and all members. No commit/reset/cleanup or historical GPU rerun.
''')
    m={**pm,'source_sha256':{str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},'config_sha256':sha(b/'config.json'),'protocol_sha256':sha(b/'PROTOCOL.md'),
       'baseline_bundle':str(prior),'baseline_manifest_sha256':sha(prior/'manifest.json'),'baseline_summary_sha256':sha(prior/'summary.json'),
       'single_backend_flag_change':'cudnn.deterministic False -> True',
       'declared_baseline_backend_deltas':['cudnn_flags.deterministic','environment.CUDA_VISIBLE_DEVICES'],
       'device':dict(index=index,uuid=uuid,baseline_index=3,reason='Baseline device held by another project; one free same-model GPU, prospectively declared'),
       'base_repository_head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
       'next_task_sha256':sha(next_task)}
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (b/'source_worktree.diff').write_text(subprocess.check_output(['git','-C',str(REPO),'diff'],text=True))
    (b/'source_worktree_status.txt').write_text(subprocess.check_output(['git','-C',str(REPO),'status','--short'],text=True))
    print(b)


if __name__=='__main__':main()
