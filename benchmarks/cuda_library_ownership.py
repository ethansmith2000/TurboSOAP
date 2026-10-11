"""New library-only control; no optimizer, trainer, cache clearing or retry."""
import time
STARTED=time.monotonic()

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

import torch

try:
    from .tensor_ownership import TensorOwnership
except ImportError:
    from tensor_ownership import TensorOwnership


def config():
    return dict(shape=[32,32],dtypes=['float32','bfloat16'],seed=173,
                internal_seconds=60,external_seconds=90,allocated_memory_bytes=2*1024**3,
                evidence_bytes=5*1024**2,expected_default_workspace_bytes=8519680,
                optimizer_steps=0,training_steps=0,cache_clearing=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,data):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(data,indent=2,allow_nan=False)+'\n')


def verify(bundle,require_launch=True):
    m=json.loads((bundle/'manifest.json').read_text())
    if sha(bundle/'config.json')!=m['config_sha256']:
        raise ValueError('Frozen config mismatch')
    for name,digest in m['source_sha256'].items():
        if sha(bundle/'source'/name)!=digest:
            raise ValueError('Frozen source mismatch')
    if json.loads((bundle/'config.json').read_text())!=config():
        raise ValueError('Outside library-control scope')
    if require_launch:
        launch=json.loads((bundle/'launch_manifest.json').read_text())
        if launch['manifest_sha256']!=sha(bundle/'manifest.json'):
            raise ValueError('Device manifest mismatch')
        if (os.environ.get('CUDA_VISIBLE_DEVICES')!=launch['gpu']['uuid'] or
                os.environ.get('GPU_CLAIM_INDICES')!=str(launch['gpu']['index'])):
            raise ValueError('Claimed device mismatch')
        return launch


def library_scope(observer,device,dtype):
    gen=torch.Generator().manual_seed(config()['seed'])
    a=torch.randn(*config()['shape'],generator=gen).to(device=device,dtype=dtype)
    b=torch.randn(*config()['shape'],generator=gen).to(device=device,dtype=dtype)
    c=a@b
    observer.watch_tree(dict(a=a,b=b,c=c),'library_only')
    assert torch.isfinite(c).all()
    return dict(checksum=float(c.float().sum()),inside_scope=observer.report())


def alias_control(device):
    observer=TensorOwnership()
    original=torch.zeros(32,32,device=device)
    observer.watch(original,'alias_source')
    # A regular view keeps the base tensor object alive. Detach first to
    # isolate the native storage-lifetime case after the original object dies.
    alias=original.detach().view(-1)
    del original
    retained=observer.report()
    assert retained['live_tensors']==0 and retained['live_unique_storages']==1
    assert not retained['all_registered_released']
    del alias
    released=observer.report()
    assert released['all_registered_released']
    observer.close()
    return dict(alias_form='detach_then_view',intentional_alias_retention_detected=retained,
                after_release=released)


def guard(device):
    if time.monotonic()-STARTED>config()['internal_seconds']:
        raise TimeoutError('Internal control deadline exceeded')
    if device.type=='cuda' and torch.cuda.max_memory_allocated(device)>config()['allocated_memory_bytes']:
        raise MemoryError('Allocated-memory cap exceeded')


def main(bundle):
    launch=verify(bundle)
    if (bundle/'started.json').exists():
        raise ValueError('No repeat of an executed library control')
    write_new(bundle/'started.json',dict(status='running',gpu=launch['gpu']))
    result=dict(status='FAILED',optimizer_steps=0,training_steps=0,phases=[],
                scope='New library-only allocations and explicitly registered tensor/storage lifetimes; '
                      'not direct ownership attribution of the prior exited process')
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('Require exactly one claimed GPU')
        device=torch.device('cuda:0')
        result['backend']=dict(torch_version=torch.__version__,torch_git_version=torch.version.git_version,
            device_name=torch.cuda.get_device_name(device),gpu=launch['gpu'],
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            cublaslt_workspace_size=os.environ.get('CUBLASLT_WORKSPACE_SIZE'),
            unified_workspace=os.environ.get('TORCH_CUBLASLT_UNIFIED_WORKSPACE'),
            matmul_precision=torch.get_float32_matmul_precision(),
            allow_tf32=torch.backends.cuda.matmul.allow_tf32)
        if result['backend']['cublas_workspace_config'] is not None:
            raise ValueError('Declared default-workspace control requires absent override')
        torch.cuda.reset_peak_memory_stats(device)
        cold=torch.cuda.memory_allocated(device)
        result['cold_allocated_bytes']=cold
        for name in config()['dtypes']:
            guard(device)
            observer=TensorOwnership()
            baseline=torch.cuda.memory_allocated(device)
            record=library_scope(observer,device,getattr(torch,name))
            torch.cuda.synchronize(device)
            released=observer.report()
            assert released['all_registered_released']
            after=torch.cuda.memory_allocated(device)
            record.update(dtype=name,allocated_before_bytes=baseline,
                          allocated_after_bytes=after,allocator_delta_bytes=after-baseline,
                          registered_ownership_after_scope=released)
            result['phases'].append(record)
            observer.close()
            guard(device)
        result['alias_control']=alias_control(device)
        torch.cuda.synchronize(device)
        first=result['phases'][0]
        result['first_library_delta_matches_default_workspace']=(first['allocator_delta_bytes']==
                                                               config()['expected_default_workspace_bytes'])
        result['allocated_after_all_scopes_bytes']=torch.cuda.memory_allocated(device)
        result['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        result['status']='PASS' if result['first_library_delta_matches_default_workspace'] else 'INCONCLUSIVE'
        result['interpretation']='Library-only matmul can retain allocator bytes after all registered '
        result['interpretation']+='tensors/storages expire; whole allocator equality is a different measurement. '
        result['interpretation']+='The old release-gate failure remains FAILED; no old allocation block was inspected.'
    except Exception as error:
        result['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    result['elapsed_seconds']=time.monotonic()-STARTED
    if result['elapsed_seconds']>config()['internal_seconds']:
        result['status']='FAILED'
    write_new(bundle/'result.json',result)
    if sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())>config()['evidence_bytes']:
        raise RuntimeError('Compact evidence cap exceeded')
    print(json.dumps({k:v for k,v in result.items() if k not in ('backend','phases','alias_control')}),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args();raise SystemExit(main(a.bundle.resolve()))
