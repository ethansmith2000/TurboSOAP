"""Freeze a four-process benchmark=False control without changing old evidence."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from prepare_cifar_benchmark import sha,REPO
from benchmark_off_contract import validate_config,validate_child_change


def main():
    b=Path(sys.argv[1]).resolve();prior=Path('/workspace/optimizer_replay_results/cifar_soap_execution_diagnostic_20261007')
    if b.exists():raise ValueError('Use a new bundle')
    if shutil.disk_usage(b.parent).free<5*2**30:raise RuntimeError('Require5GiB free')
    pm=json.loads((prior/'manifest.json').read_text())
    for name,h in pm['source_sha256'].items():assert sha(prior/'source'/name)==h,name
    for name,key in [('config.json','config_sha256'),('split.json','split_sha256'),('data_prepared.json','base_prepared_sha256'),('PROTOCOL.md','protocol_sha256')]:assert sha(prior/name)==pm[key]
    before=(prior/'source/benchmarks/cifar_execution_child.py').read_text()
    after=(REPO/'benchmarks/cifar_benchmark_off_child.py').read_text();validate_child_change(before,after)
    old_outer=(prior/'source/benchmarks/cifar_execution_diagnostic.py').read_text()
    new_outer=(REPO/'benchmarks/cifar_benchmark_off.py').read_text()
    assert new_outer==old_outer.replace('cifar_execution_child.py','cifar_benchmark_off_child.py')
    b.mkdir();shutil.copytree(prior/'source',b/'source');shutil.copytree(prior/'upstream_reference',b/'upstream_reference')
    for name in ['split.json','data_prepared.json']:shutil.copy2(prior/name,b/name)
    names=['cifar_benchmark_off_child.py','cifar_benchmark_off.py','benchmark_off_contract.py',
           'prepare_benchmark_off.py','summarize_benchmark_off.py']
    for name in names:shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    c=json.loads((prior/'config.json').read_text());d=c['execution_diagnostic']
    d['contexts']=[dict(label=f'plan{h}_flush25_r{r}_benchmark_off',allocation_horizon=h,
        timing_chunk=25,repeat=r) for h,r in [(420,0),(3500,0),(3500,1),(420,1)]]
    validate_config(c);(b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (b/'PROTOCOL.md').write_text('''# Four-process cuDNN benchmark=False execution control

Prospective2026-10-08. Four fresh sequential processes420/3500/3500/420 allocation
plans, two each, flush25 fixed. Change ONLY torch.backends.cudnn.benchmark=True
toFalse in the exact frozen preceding child source. AST audit rejects other
functional changes. Exact preceding outer source with only child entry rename.
All baseline source remains untouched. Default cuDNN deterministic=False, TF32,
matmul high/strict M highest, reduced-precision/SDP/otherflags/environment fixed;
record and require baseline metadata differ only at cudnn_flags.benchmark.
No deterministic-mode toggle, kernel replacement, controller/EMA/recipe change,
shared-cache edits/clears, clock/GC settings or other-job changes. Natural
framework compilation/cache use is recorded, not forced cold or treated as proof
of identical backend algorithms. This is a new control, not a missing-repeat
retry or a completed historical experiment rerun for context.

ViT1536/depth1/heads24, seed139, original frozen CIFAR100 split/strongaugmentation,
batch256, LR .0005 on200epoch/5epoch LRwarmup. Exactly25 discarded QR10
compilation/parity smoke plus25 measured QR20/.99covariance/Adam(.9,.999)/strictM/
currentV carry-reindex/bothfactors perprocess:100smoke+100measured=200total.
Restore originalmodel/completeRNG and freshoptimizer aftersmoke. No400 disposable
warmup,100historicalgate recovery,3500learning endpoint, radius/seed/backendflag
sweep, automatic extension/retry or long learning campaign. Allocation before
compile:525device batchindex tensors for420,3500for3500; identical first25ids.
Own main/autograd116/116 aftersmoke, original masks restored at exit.

Use exact preceding caller diagnostics: smoke1/25, measured1/2/20/21/25 with
input/target/ids/RNG/logits/loss/preclip/postclipgradients/weights/fulloptimizer/GG/Q
hashes and finite/norm/max scalars. RAM copies discarded, no graph hooks. These
synchronized diagnostic captures are inside event/host spans; those spans are
NOT throughput or optimizer timing qualification. Include every diagnostic,
initialization/compile/hash/profile/cleanup in total all-in time. Geometry at
measured1/21:finite matrixweights/moments andGram<.05. No additionalfullstrict
QRshadow or physicalM/selectedupdate gate implied; all earlier failures remain.

Record actual loaded Inductor source identifiers/hashes/Triton configurations/
PTX/cubin hashes, Torch/CUDA/cuDNN/librarypath/backend/environment metadata.
One final25 RAM-only model forward/backward profiler window and precedingplain
call, no optimizerstep. Require exact logits/loss/gradients and unchanged
model/fulloptimizer/RNG. Kernel names/counts at25 do not identify actual2 kernels,
first historical divergence, every library-internal algorithm ID or a unique
cause. Retain compact actually loaded wrapper source after run by verified hash;
no trace/model weights/checkpoint/tensor snapshots saved.

Require all four new sampled trajectories, loaded code metadata and backend
metadata to match exactly within this control; complete prescribed observations,
finite/Gram guards, profile invariants, owned release and ownmask restoration.
Compare to BOTH earlier trajectorygroups descriptively, never require eitherold
bittrajectory or relabel its failed gate. Exact within this25-step control only
supports a prospective baseline; it does not establish universaldeterminism,
historical100 equality or larger-factor learning/wall quality. If negative,
preserve it and choose a separately bounded deterministic-mode or same-state
convolution replay. No automatic next flag/control/21k training run.

One exclusive nonqueued freeGPU3 RTX5090 UUID GPU-55fc0669-93bd-9223-674d-212086d4258b,
private manifold supervisor. Capacity checked, about322GiB free. Internal600/
external660s, child150s inside remainingglobal600;24GiBallocated cap/process,
expectedcompact<25MiB/hard100MiB. Earlier child~75s plusimport/teardown suggests
fourcontexts~320s, not a promise. Timeout/failure preservespartials, no extension
or retry. Generic checkpoints/finalweights disabled; all tensor copiesRAMonly.
Retain compact source/config/seeds/provenance/logs/metrics/summaries/manifests
until explicitcleanup. Preserve179 pre-existingPythonfiles/25 prior inventories
and dirtytrees, no commit/reset/delete.
''')
    m={**pm,'source_sha256':{str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        'config_sha256':sha(b/'config.json'),'protocol_sha256':sha(b/'PROTOCOL.md'),
        'baseline_bundle':str(prior),'baseline_manifest_sha256':sha(prior/'manifest.json'),
        'baseline_summary_sha256':sha(prior/'summary.json'),'single_backend_flag_change':'cudnn.benchmark True -> False',
        'base_repository_head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()}
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (b/'source_worktree.diff').write_text(subprocess.check_output(['git','-C',str(REPO),'diff'],text=True))
    (b/'source_worktree_status.txt').write_text(subprocess.check_output(['git','-C',str(REPO),'status','--short'],text=True))
    print(b)


if __name__=='__main__':main()
