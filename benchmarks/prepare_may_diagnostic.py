"""CPU-only preparation of the short diagnostic. Never binds a GPU or launches."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
try:
    from .prepare_matched_training import verify_runnable
    from .launch_environment_contract import load_contract
except ImportError:
    from prepare_matched_training import verify_runnable
    from launch_environment_contract import load_contract
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/workspace/optimizer_replay_results/cifar_soap_matched_ownership_20261010')
ARCHIVE=Path('/workspace/optimizer_replay_results/soap_geometry_diagnostics_stats_20261010/cpu_source_archive')
NEW=('benchmarks/may_diagnostic_runtime.py','benchmarks/cifar_may_diagnostic.py',
     'benchmarks/prepare_may_diagnostic.py','tests/test_may_diagnostic.py')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
# Literal hashes filled at authoring time, not inferred from mutable files at launch.
BASE_MANIFEST_SHA='8b373be17b9cc575741fc95c76f8ed8584685021a750a3d268367da3b849cf94'
ARCHIVE_MANIFEST_SHA='767433c2b364594ba9957ffa7c0bdfac1d917b0c5a341378af60ee05b3515342'

def protocol():
    p=copy.deepcopy(read(BASE/'protocol.json'))
    p.update(scope='Short literal-May refresh diagnostic; no learning or timing comparison',
        arms=['may_literal'],preflight_arms=['qr20','may_literal','fro_cubic2_cap05','qr20_repeat'],
        steps_per_arm=3500,batch_plan_reference_calls=3500,may_calls=291,
        total_measured_updates=291,maximum_optimizer_updates=316,observation_steps=[0,100],
        refresh_bracket_calls=[251,261,271,281],quality_qualified=False,timing_qualified=False,
        full_comparison_completed=False,qr_repeat_completed=False)
    p['gate'].update(qr_repeat='NOT_IN_SHORT_DIAGNOSTIC',quality_qualification='Never qualified by this diagnostic',
        comparison='No optimizer ranking, statistical winner, default promotion or speedup')
    p['bounds'].update(internal_wall_seconds=240,external_watchdog_seconds=300)
    p['implementation'].update(entry='source/benchmarks/cifar_may_diagnostic.py',
        exact_qr_repeat_full_loss_sequence_required=False,
        full_batch_plan_and_initialization_preserved=True,
        short_completion_backend=dict(phase='short_completion',historical_reference_phase='backend_after_measured',
            completed_full_comparison=False,completed_after_measured_gate=False),
        abort_backend=dict(phase='abort',completed_after_measured_gate=False))
    p['artifact_policy']['initial_model_snapshot'].update(purpose='Exact smoke/preflight/single-May initial reset',
        retention='Only until short diagnostic endpoint fingerprints or abort; release on all exits')
    p['artifact_policy'].update(expected_compact_evidence_bytes=10*2**20,
        compact_evidence_retention='Retain compact failed/successful diagnostic audit; no model/optimizer tensors')
    p['timing']['interpretation']='Instrumented short May diagnostic; no qualified timing speedup'
    return p

PROTOCOL_TEXT='''# Short literal-May refresh diagnostic, prepared only

291May calls maximum+25disposableQR10smoke=316optimizer calls,4independent empty
all-arm preflightbackwards/no preflightsteps. OnlyMay updates after exactactualfirst
gate. Full3500batchplan/model/data initialization and original35000schedule/875LR
warmup/common5e-4/.99/(.9,.999) recipe remain. Config.json is the unchanged reference
recipe; its historical fullcomparison/max_wall fields are not the short driver's
execution limits. Operational limits come only from verified protocol.json.

LiteralMay frequency10/warmup250/BF16/columnEMA.99/controller/fallback unchanged.
Pinned model1536/depth1/heads24/data/split/strongaugmentation/eagerBF16evaluation,
observations0/100; no extra model forward before271. Scalarpre/post251/261/271/281
refresh evidence, finiteallstate and highestFP32 normalizedGramstrict<.05.
First failure aborts; no tolerance/familyselection/retry/filling missing arms.
No QR3500 rerun, fullcomparison/QRrepeat/quality/timing/default/blog claim.

240continuous internal/300external kill-after10,24GiBallocated/256MiBpurposehost/
100MiBcompact evidence; initial109.33MiB/conservative226.62MiB RAM-only snapshot,
release on all exits, streamed1MiB hashes. No checkpoint/weights/optimizersnapshot/
tensorfile/profilertrace. Expected compact evidence<10MiB, retain for researchaudit.

Before-smoke exactJSON-wire historicalbackend gate, only laterdeclaredGPUUUID
delta. Successful short_completion uses strict historicalafterphase values for
consistency without qualifying the old fullcomparison/aftermeasured/repeat gate.
Failure records phaseabortbackend/code/resourcepeak/purpose/evidence/wall and
specific structuredtrigger before ownership/mask cleanup. Audit failure cannot
replace the original trigger. Branch reports show static eligibility, not an
observed fallback or causal explanation. GPUobserver invariance unqualified.

Private supervisor/nonqueuedgpu-claim, autostartfalse/autorestartfalse, fixed
compilerthreads4/OMP4/MKL4/no-bytecode1; main/existingautograd pinCPU116aftersmoke,
restore ownmasks. No device/launchmanifest/claim/queue/program installed during
preparation. Later prospectivelybindoneindex/UUID and verify actualsupervisor
inheritance before one managed launch. Check live state after disconnect; never
duplicate or retry. All prior source/evidence/failures remain unchanged.
'''

def template(b):
    env=read(BASE/'environment_contract.json')['phases']['backend_before_smoke']['environment']
    envtext=','.join(k+'='+json.dumps(v) for k,v in sorted(env.items()) if k!='CUDA_VISIBLE_DEVICES')
    return f'''[program:manifold_may_diagnostic]
command=/workspace/bin/gpu-claim run --owner optimizer-research --job may-refresh-291-20261010 --gpu __GPU_INDEX__ -- /usr/bin/timeout --kill-after=10s 300 /venv/main/bin/python -u {b}/source/benchmarks/cifar_may_diagnostic.py --bundle {b}
directory={b}
environment={envtext}
autostart=false
autorestart=false
startsecs=0
stopasgroup=true
killasgroup=true
stopwaitsecs=20
stdout_logfile={b}/run.log
stdout_logfile_maxbytes=0
redirect_stderr=true
'''


def verify_inputs():
    if sha(BASE/'manifest.json')!=BASE_MANIFEST_SHA or sha(ARCHIVE/'manifest.json')!=ARCHIVE_MANIFEST_SHA:
        raise ValueError('Original comparison or corrected CPU archive lineage drift')
    bm=verify_runnable(BASE);am=read(ARCHIVE/'manifest.json')
    if set(am['source_sha256'])!=set(bm['source_sha256'])|set(am['additive_sources']):
        raise ValueError('CPU archive source set changed')
    for n,h in am['source_sha256'].items():
        if sha(ARCHIVE/'source'/n)!=h:raise ValueError('CPU archive source drift: '+n)
    for n,h in bm['source_sha256'].items():
        if am['source_sha256'].get(n)!=h:raise ValueError('Original executed source replaced')
    return bm,am


def verify_bundle(bundle):
    b=Path(bundle).resolve();m=read(b/'manifest.json');bm,am=verify_inputs()
    if m['baseline_manifest_sha256']!=BASE_MANIFEST_SHA or m['cpu_archive_manifest_sha256']!=ARCHIVE_MANIFEST_SHA:
        raise ValueError('Short bundle lineage changed')
    if set(m['source_sha256'])!=set(am['source_sha256'])|set(NEW):raise ValueError('Short source dependency set changed')
    for n,h in m['source_sha256'].items():
        if sha(b/'source'/n)!=h:raise ValueError('Short source drift: '+n)
    for n,h in am['source_sha256'].items():
        if m['source_sha256'].get(n)!=h:raise ValueError('Archived diagnostic source replaced: '+n)
    for n,h in m['artifacts'].items():
        if sha(b/n)!=h:raise ValueError('Short artifact drift: '+n)
    for n in ['config.json','split.json','data_prepared.json','environment_contract.json']:
        if (b/n).read_bytes()!=(BASE/n).read_bytes():raise ValueError('Reference recipe/data/environment changed: '+n)
    if read(b/'protocol.json')!=protocol():raise ValueError('Short protocol changed')
    if (b/'PROTOCOL.md').read_text()!=PROTOCOL_TEXT or (b/'supervisor_template.conf').read_text()!=template(b):
        raise ValueError('Short protocol text/supervisor changed')
    load_contract(b)
    return m


def prepare(bundle):
    b=Path(bundle).resolve()
    if b.exists():raise ValueError('New unexecuted bundle required')
    if shutil.disk_usage(b.parent).free<5*2**30:raise OSError('Shared disk capacity bound')
    bm,am=verify_inputs();b.mkdir()
    for n,h in am['source_sha256'].items():
        target=b/'source'/n;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ARCHIVE/'source'/n,target)
    for n in NEW:
        target=b/'source'/n
        if target.exists():raise ValueError('Cannot replace archived source')
        shutil.copy2(ROOT/n,target)
    for n in ['config.json','split.json','data_prepared.json','environment_contract.json']:
        shutil.copy2(BASE/n,b/n)
    write(b/'protocol.json',protocol());(b/'PROTOCOL.md').write_text(PROTOCOL_TEXT)
    (b/'supervisor_template.conf').write_text(template(b))
    artifacts={str(p.relative_to(b)):sha(p) for p in b.rglob('*') if p.is_file() and b/'source' not in p.parents}
    m=dict(status='PREPARED_NOT_LAUNCHED',gpu_selected=False,device=None,
        baseline_bundle=str(BASE),baseline_manifest_sha256=BASE_MANIFEST_SHA,
        cpu_archive=str(ARCHIVE),cpu_archive_manifest_sha256=ARCHIVE_MANIFEST_SHA,
        source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},artifacts=artifacts,
        config_sha256=sha(b/'config.json'),protocol_sha256=sha(b/'PROTOCOL.md'),split_sha256=sha(b/'split.json'),
        base_prepared_sha256=sha(b/'data_prepared.json'),environment_contract_sha256=sha(b/'environment_contract.json'),
        supervisor_template_sha256=sha(b/'supervisor_template.conf'))
    write(b/'manifest.json',m);verify_bundle(b)
    write(b/'PREPARATION_STATUS.json',dict(status='PREPARED_NOT_LAUNCHED',gpu_selected=False,launch_manifest_exists=False,
        may_calls=291,smoke_calls=25,maximum_optimizer_calls=316,preflight_backward_calls=4,batch_plan_calls=3500))
    git_status=subprocess.run(['git','-C',str(ROOT),'status','--porcelain=v1'],capture_output=True,text=True)
    if git_status.returncode and (ROOT/'.git').exists():
        raise RuntimeError('Git provenance failed in source checkout: '+git_status.stderr)
    provenance=git_status.stdout if not git_status.returncode else 'Hash-pinned frozen source snapshot; Git checkout metadata unavailable here.\n'
    (b/'source_worktree_status.txt').write_text(provenance)
    return m

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    m=prepare(parser.parse_args().bundle);print(json.dumps(dict(status=m['status'],sources=len(m['source_sha256']))))
