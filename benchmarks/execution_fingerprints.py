"""Caller-side compact numerical and loaded-code provenance; no graph hooks."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import torch


def tensor_record(tensor):
    """Hash exact bytes, including bf16, and compute norms on disposable CPU copies."""
    value=tensor.detach().cpu().contiguous()
    raw=value.reshape(-1).view(torch.uint8).numpy().tobytes()
    header=str((str(value.dtype),tuple(value.shape))).encode()
    floating=value.float()
    return dict(sha256=hashlib.sha256(header+raw).hexdigest(),dtype=str(value.dtype),
                shape=list(value.shape),finite=bool(torch.isfinite(floating).all()),
                norm=float(torch.linalg.vector_norm(floating)),
                maximum_absolute=float(floating.abs().max()) if value.numel() else 0.)


def gradient_records(model):
    return {name:tensor_record(p.grad) if p.grad is not None else None
            for name,p in model.named_parameters()}


def basis_records(opt):
    result={}
    for group in opt.matrix.param_groups:
        for p,name in zip(group['params'],group['param_names']):
            state=opt.matrix.state[p]
            result[name]={key:[tensor_record(v) if v is not None else None for v in state[key]]
                          for key in ['GG','Q']}
    return result


def loaded_code():
    """Snapshot only modules actually loaded by this process, never scan/edit caches."""
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    rows=[]
    for module in PyCodeCache.modules:
        path=Path(module.__file__)
        source=path.read_bytes()
        kernels=[]
        for name,value in vars(module).items():
            if isinstance(value,CachingAutotuner):
                configs=[]
                for compiled in value.compile_results:
                    cfg=compiled.config
                    kernel=compiled.kernel
                    assembly={}
                    for kind in ['ptx','cubin']:
                        content=getattr(kernel,'asm',{}).get(kind)
                        if content is not None:
                            if isinstance(content,str):content=content.encode()
                            assembly[kind]=hashlib.sha256(content).hexdigest()
                    configs.append(dict(kwargs=cfg.kwargs,num_warps=cfg.num_warps,
                        num_stages=cfg.num_stages,kernel_name=getattr(kernel,'name',None),
                        kernel_hash=getattr(kernel,'hash',None),assembly_sha256=assembly))
                kernels.append(dict(name=name,compiled_configurations=configs,
                                    launcher_count=len(value.launchers)))
        rows.append(dict(module=module.__name__,path=str(path),
                         source_sha256=hashlib.sha256(source).hexdigest(),bytes=len(source),
                         kernels=kernels))
    return rows


def backend_record():
    cuda=torch.backends.cuda.matmul
    flags={name:getattr(cuda,name,None) for name in
           ['allow_tf32','allow_fp16_reduced_precision_reduction',
            'allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']}
    libraries=[]
    for line in Path('/proc/self/maps').read_text().splitlines():
        path=line.split()[-1]
        if path.startswith('/') and re.search(r'lib(cudnn|cublas|cuda|nvrtc|torch|cusolver)',path):
            libraries.append(path)
    return dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        python=platform.python_version(),matmul_precision=torch.get_float32_matmul_precision(),
        matmul=flags,cudnn_flags=dict(allow_tf32=torch.backends.cudnn.allow_tf32,
            benchmark=torch.backends.cudnn.benchmark,deterministic=torch.backends.cudnn.deterministic,
            benchmark_limit=torch.backends.cudnn.benchmark_limit,enabled=torch.backends.cudnn.enabled),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        float32_precision=torch.backends.fp32_precision,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),
        environment={k:v for k,v in os.environ.items() if k.startswith(('TORCH','TRITON','CUDA','CUBLAS','CUDNN','NVIDIA_TF32','OMP','MKL','PYTHONDONTWRITEBYTECODE'))},
        loaded_libraries=sorted(set(libraries)))


def first_difference(left,right,path=''):
    """Return first exact structural mismatch; execution metadata is compared separately."""
    if type(left)!=type(right):return path or '<root>'
    if isinstance(left,dict):
        if left.keys()!=right.keys():return (path+'.keys').lstrip('.')
        for key in left:
            found=first_difference(left[key],right[key],f'{path}.{key}'.lstrip('.'))
            if found is not None:return found
        return None
    if isinstance(left,list):
        if len(left)!=len(right):return path+'.length'
        for i,(a,b) in enumerate(zip(left,right)):
            found=first_difference(a,b,f'{path}[{i}]')
            if found is not None:return found
        return None
    return None if left==right else path
