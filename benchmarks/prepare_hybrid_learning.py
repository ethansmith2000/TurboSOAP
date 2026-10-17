"""Stdlib-only immutable preparation, semantic verification and launch gates."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
from hybrid_training_contract import BASELINE
from hybrid_learning_contract import PARENT, protocol, recipe, validate

ROOT=Path(__file__).resolve().parents[1]
NEW=('benchmarks/hybrid_learning_contract.py','benchmarks/cifar_hybrid_learning.py',
     'benchmarks/prepare_hybrid_learning.py','tests/test_hybrid_learning.py')
PREFIXES=('TORCH','TRITON','CUDA','CUBLAS','CUDNN','NVIDIA_TF32','OMP','MKL','PYTHONDONTWRITEBYTECODE')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def expected_environment(b):
    return dict(CUDA_HOME='/usr/local/cuda',CUDA_VERSION='12.8.1',MKL_NUM_THREADS='4',
        OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TORCHINDUCTOR_COMPILE_THREADS='4',
        TORCHINDUCTOR_CACHE_DIR=str(Path(b)/'compile_cache'),TRITON_CACHE_DIR=str(Path(b)/'compile_cache/triton'))

def backend_contract():
    old=read(BASELINE/'environment_contract.json')['phases']['backend_before_smoke']
    # Exact numerical/software fields; library mappings retained as runtime
    # provenance, not assumed identical before/after compiling extra kernels.
    numerical={k:v for k,v in old.items() if k not in ('environment','loaded_libraries')}
    numerical['matmul']['allow_bf16_reduced_precision_reduction']=False
    return dict(numerical=numerical,common_delta={'allow_bf16_reduced_precision_reduction':False},
        environment_delta=['single claimed device UUID','fresh isolated Inductor/Triton cache'],
        loaded_libraries='Capture complete before/after mappings without exact-set equality')

def template(b):
    b=Path(b);seed=read(b/'config.json')['seed']
    env=','.join(k+'="'+v+'"' for k,v in sorted(expected_environment(b).items()))
    return f'''[program:manifold_hybrid_learning_seed{seed}]
command=/workspace/bin/gpu-claim run --owner optimizer-research --job hybrid-learning-seed{seed}-20261011 --gpu __GPU_INDEX__ -- /usr/bin/timeout --kill-after=10s 2460 /venv/main/bin/python -u {b}/source/benchmarks/cifar_hybrid_learning.py --bundle {b}
directory={b}
environment={env}
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

def verify(b):
    b=Path(b);m=read(b/'manifest.json')
    parents=(PARENT,)
    if m['parents']!=[dict(path=str(p),manifest_sha256=sha(p/'manifest.json')) for p in parents]:
        raise ValueError('Parent provenance changed')
    for p in parents:
        for name,digest in read(p/'manifest.json')['source_sha256'].items():
            if sha(p/'source'/name)!=digest or m['source_sha256'].get(name)!=digest:
                raise ValueError('Preserved parent source changed: '+name)
    for name,digest in m['source_sha256'].items():
        if sha(b/'source'/name)!=digest:raise ValueError('Frozen source changed: '+name)
    for name,digest in m['artifacts'].items():
        if sha(b/name)!=digest:raise ValueError('Frozen input changed: '+name)
    for name,digest in m['references'].items():
        if sha(name)!=digest:raise ValueError('Frozen reference changed: '+name)
    validate(read(b/'config.json'),read(b/'protocol.json'),read(BASELINE/'config.json'))
    for name in ('split.json','data_prepared.json'):
        if (b/name).read_bytes()!=(BASELINE/name).read_bytes():raise ValueError('Shared data input changed')
    if read(b/'environment_contract.json')!=backend_contract() or (b/'supervisor_template.conf').read_text()!=template(b):
        raise ValueError('Undeclared environment/launch policy')
    return m

def verify_launch(b, env=None):
    b=Path(b);verify(b);env=os.environ if env is None else env
    if not (b/'initial_reference.json').exists():raise ValueError('CPU initial reference required before GPU launch')
    launch=read(b/'launch_manifest.json');gpu=launch['gpu']
    if set(gpu)!={'index','uuid'} or type(gpu.get('index')) is not int or not 0<=gpu['index']<8 or not re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',gpu.get('uuid','')):
        raise ValueError('Single physical GPU binding required')
    for name in ('manifest.json','config.json','protocol.json','environment_contract.json','supervisor_template.conf'):
        if launch.get('sha256',{}).get(name)!=sha(b/name):raise ValueError('Launch not bound to '+name)
    expected=expected_environment(b);expected['CUDA_VISIBLE_DEVICES']=gpu['uuid']
    actual={k:v for k,v in env.items() if k.startswith(PREFIXES)}
    if actual!=expected or env.get('GPU_CLAIM_INDICES')!=str(gpu['index']):raise ValueError('Actual launch environment differs')
    return launch

def verify_backend(b, record):
    numerical=json.loads(json.dumps({k:v for k,v in record.items() if k not in ('environment','loaded_libraries')},allow_nan=False))
    if numerical!=read(Path(b)/'environment_contract.json')['numerical']:
        raise ValueError('Numerical/software backend differs from prospective contract')
    verify_launch(b)

def prepare(b, initial_reference=None, seed=142):
    b=Path(b).resolve()
    if b.exists():raise ValueError('Fresh bundle required; no overwrite')
    if shutil.disk_usage(b.parent).free<5*2**30:raise OSError('Insufficient shared disk')
    for p in (PARENT,):
        if read(p/'execution_status.json')['status']!='completed':raise ValueError('Parent must be completed')
    b.mkdir();(b/'source').mkdir()
    for p in (PARENT,):
        for name,digest in read(p/'manifest.json')['source_sha256'].items():
            source=p/'source'/name;target=b/'source'/name
            if sha(source)!=digest:raise ValueError('Parent source hash differs')
            if target.exists():
                if sha(target)!=digest:raise ValueError('Conflicting parent source')
            else:
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for name in NEW:
        target=b/'source'/name
        if target.exists():raise ValueError('Cannot replace parent source')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    write(b/'config.json',recipe(read(BASELINE/'config.json'),seed));write(b/'protocol.json',protocol(seed))
    write(b/'environment_contract.json',backend_contract())
    if initial_reference is not None:
        record=read(initial_reference)
        if record.get('status')!='PASS_CPU_INITIAL_REFERENCE' or record.get('seed')!=seed or record.get('parameters')!=28660324 or record.get('cuda_initialized') is not False or record.get('snapshot_purpose_bytes',2**30)>256*2**20:
            raise ValueError('Invalid CPU initial reference')
        shutil.copy2(initial_reference,b/'initial_reference.json')
    for name in ('split.json','data_prepared.json'):shutil.copy2(BASELINE/name,b/name)
    (b/'supervisor_template.conf').write_text(template(b))
    (b/'PROTOCOL.md').write_text('Fresh matched hybrid learning. Full prospective policy in protocol.json. No weights/checkpoints. Each seed is a separate one-attempt bundle.\n')
    sources={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()}
    write(b/'manifest.json',dict(status='PREPARED_NOT_LAUNCHED',source_sha256=sources,
        parents=[dict(path=str(p),manifest_sha256=sha(p/'manifest.json')) for p in (PARENT,)],
        references={str(p/name):sha(p/name) for p in (PARENT,) for name in ('summary.json','results.json','execution_status.json') if (p/name).exists()},
        artifacts={p.name:sha(p) for p in b.iterdir() if p.is_file()},production_changed=False))
    return verify(b)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--initial-reference',type=Path)
    p.add_argument('--seed',type=int,default=142)
    args=p.parse_args();m=prepare(args.bundle,args.initial_reference,args.seed);print(json.dumps(dict(status=m['status'],sources=len(m['source_sha256']))))
