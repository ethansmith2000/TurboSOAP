"""Bounded compatibility check for literal May defaults, not a training benchmark."""
import time
STARTED = time.monotonic()

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

import torch

try:
    from .legacy_soap import build_legacy_optimizer, effective_policy
    from .audit_legacy_soap import assert_exact, independent_class, observe_calls, METHODS
except ImportError:
    from legacy_soap import build_legacy_optimizer, effective_policy
    from audit_legacy_soap import assert_exact, independent_class, observe_calls, METHODS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data):
    payload = json.dumps(data, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(payload)


def verify_bundle(bundle, require_launch=True):
    bundle = Path(bundle)
    manifest = json.loads((bundle/'manifest.json').read_text())
    if sha(bundle/'config.json') != manifest['config_sha256']:
        raise ValueError('Frozen config mismatch')
    for name, digest in manifest['source_sha256'].items():
        if sha(bundle/'source'/name) != digest:
            raise ValueError(f'Frozen source mismatch: {name}')
    config = json.loads((bundle/'config.json').read_text())
    if config != expected_config():
        raise ValueError('Configuration outside declared compatibility scope')
    launch = None
    if require_launch:
        launch = json.loads((bundle/'launch_manifest.json').read_text())
        if launch['config_sha256'] != manifest['config_sha256']:
            raise ValueError('Launch config mismatch')
        if launch['manifest_sha256'] != sha(bundle/'manifest.json'):
            raise ValueError('Launch manifest mismatch')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != launch['gpu']['uuid']:
            raise ValueError('Claim GPU UUID mismatch')
        if os.environ.get('GPU_CLAIM_INDICES') != str(launch['gpu']['index']):
            raise ValueError('Claim GPU index mismatch')
    return config, launch


def expected_config():
    return dict(shapes=[[4,6],[8,8]], seed=172, gradient_calls_per_shape=331,
                optimizer_overrides={}, internal_seconds=120,
                external_seconds=150, allocated_memory_bytes=2*1024**3,
                evidence_bytes=5*1024**2, observations=[0,10,250,260,270,310,330],
                fail_fast=True, compile=False, controller_observer='untimed_delegate_only',
                artifacts='Scalar/JSON/log/source evidence only; no tensors/checkpoints/weights/traces')


def check_bounds(device, deadline, memory_cap):
    if time.monotonic() > deadline:
        raise TimeoutError('Internal compatibility-probe deadline exceeded')
    if device.type=='cuda' and torch.cuda.max_memory_allocated(device)>memory_cap:
        raise MemoryError('Allocated CUDA memory cap exceeded')


def check_completion(record, calls):
    if record.get('failure') is not None:
        return False
    if (calls != 331 or record['completed_gradient_calls'] != calls or
            record['observed_gradient_calls'] != calls or
            not record['full_state_exact_at_every_observed_call'] or
            record['completed_parameter_updates'] != 330):
        return False
    counts=record['method_counts']
    return (record['observer_restored'] and record['first_call_skipped'] and
            counts.get('_accumulate_outer_products')==331 and
            counts.get('_fallback_basis_list')==25 and
            counts.get('_reorder_exp_avg_sq')==33 and
            counts.get('_finalize_basis_change')==33 and
            counts.get('_project_back_inside_preconditioner')==330 and
            counts.get('_project_inside_preconditioner')==332 and
            [e['step'] for e in record['warm_events']]==list(range(260,331,10)) and
            [e['step'] for e in record['warm_events'] if e['metrics']]==[260,310])


def run_pair(shape, calls, seed, device, deadline, memory_cap, overrides=None):
    """CPU test entry supports short/FP32 fixtures; GPU main forbids overrides."""
    generator=torch.Generator().manual_seed(seed)
    initial=torch.randn(*shape,generator=generator).to(device)
    p=torch.nn.Parameter(initial.clone())
    q=torch.nn.Parameter(initial.clone())
    kwargs={} if overrides is None else overrides
    live=build_legacy_optimizer([p],**kwargs)
    original=independent_class()([q],**kwargs)
    record=dict(shape=list(shape), planned_gradient_calls=calls,
                observed_gradient_calls=0, completed_gradient_calls=0, call=0,
                warm_events=[], observations=[], failure=None,
                first_call_skipped=False, effective_policy=effective_policy(live),
                full_state_exact_at_every_observed_call=True)
    counts={}
    try:
        with observe_calls(live,record) as counts:
            for index in range(calls):
                record['call']=index+1
                check_bounds(device,deadline,memory_cap)
                grad=torch.randn(*shape,generator=generator).to(device)
                p.grad=grad.clone();q.grad=grad.clone()
                live_error=original_error=None
                try:
                    live.step()
                except Exception as error:
                    live_error=error
                    live_trace=traceback.format_exc()
                record['observed_gradient_calls']=index+1
                if live_error is not None and isinstance(live_error,torch.OutOfMemoryError):
                    raise live_error
                try:
                    original.step()
                except Exception as error:
                    original_error=error
                assert_exact(p,q,'parameter')
                assert_exact(live.state[p],original.state[q])
                assert_exact(live.latest_basis_stats,original.latest_basis_stats,'stats')
                assert effective_policy(live)==effective_policy(original)
                if live_error is not None or original_error is not None:
                    record['failure']=dict(kind='literal_operation_error',call=index+1,
                        optimizer_step=live.state[p].get('step'),
                        error_type=type(live_error).__name__,message=str(live_error),
                        original_error_type=type(original_error).__name__,
                        original_message=str(original_error),
                        matched_original_exception=(type(live_error) is type(original_error)
                                                    and str(live_error)==str(original_error)),
                        partial_state_exact=True,
                        traceback=live_trace if live_error is not None else '')
                    break
                record['completed_gradient_calls']+=1
                if index==0:
                    record['first_call_skipped']=torch.equal(p,initial)
                    assert record['first_call_skipped']
                if index in expected_config()['observations']:
                    grams=[float((basis.float().T@basis.float()-
                            torch.eye(basis.shape[1],device=device)).norm())
                           for basis in live.state[p]['Q'] if not isinstance(basis,list)]
                    record['observations'].append(dict(call=index+1,optimizer_step=index,
                        stats=dict(live.latest_basis_stats),gram_frobenius_norms=grams,
                        note='Unnormalized diagnostic; no orthogonality/transport qualification'))
                check_bounds(device,deadline,memory_cap)
    except Exception as error:
        record['failure']=dict(kind='qualification_or_resource_error',
                               call=record['call'],error_type=type(error).__name__,
                               message=str(error),traceback=traceback.format_exc())
        record['full_state_exact_at_every_observed_call']=False
    finally:
        record.pop('call')
        record['observer_restored']=not any(name in live.__dict__ for name in METHODS)
        record['method_counts']=dict(counts)
        record['completed_parameter_updates']=max(0,record['completed_gradient_calls']-1)
        record['direction_ema']=[{k:dict(shape=list(v.shape),dtype=str(v.dtype),numel=v.numel())
                                if torch.is_tensor(v) else v for k,v in moment.items()}
                                for moment in live.state[p].get('basis_direction_moments',[])]
        record['status']='PASS' if check_completion(record,calls) else 'FAILED'
        # All parameter/state tensors are purpose-specific RAM and die with this
        # scope. No checkpoint, tensor snapshot or allocator-cache clearing.
        p.grad=None;q.grad=None
        live.state.clear();original.state.clear()
        live.param_groups.clear();original.param_groups.clear()
        # Drop retained exception/traceback references before returning so a
        # failure does not retain this frame's small CUDA tensors in a cycle.
        live_error=original_error=None
    return record


def main(bundle):
    config,launch=verify_bundle(bundle)
    if (bundle/'result.json').exists() or (bundle/'started.json').exists():
        raise ValueError('No retry/overwrite of an executed probe')
    write_new(bundle/'started.json',dict(status='running',gpu=launch['gpu'],
                                       timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    result=dict(status='FAILED',cases=[],declared_scope='Literal default CUDA compatibility, '
                'tiny prescribed-gradient fixtures, source/observer/full-state parity only; '
                'not training, H200, compiler, convergence, transport or timing qualification')
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('Require exactly one claimed visible CUDA GPU')
        device=torch.device('cuda:0')
        result['backend']=dict(torch_version=torch.__version__,wheel_cuda=torch.version.cuda,
            device_name=torch.cuda.get_device_name(device),gpu=launch['gpu'],
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        torch.cuda.reset_peak_memory_stats(device)
        baseline=torch.cuda.memory_allocated(device)
        for shape in config['shapes']:
            record=run_pair(shape,config['gradient_calls_per_shape'],config['seed'],device,
                            STARTED+config['internal_seconds'],config['allocated_memory_bytes'])
            result['cases'].append(record)
            if record['status']!='PASS':
                break
        torch.cuda.synchronize(device)
        result['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        result['allocated_baseline_bytes']=baseline
        result['allocated_after_scope_bytes']=torch.cuda.memory_allocated(device)
        result['owned_allocations_released']=result['allocated_after_scope_bytes']==baseline
        result['status']='PASS' if (len(result['cases'])==len(config['shapes']) and
                         all(r['status']=='PASS' for r in result['cases']) and
                         result['owned_allocations_released']) else 'FAILED'
    except Exception as error:
        result['failure']=dict(type=type(error).__name__,message=str(error),
                               traceback=traceback.format_exc())
    result['elapsed_seconds_from_script_entry']=time.monotonic()-STARTED
    result['completed_gradient_calls_per_copy']=sum(r['completed_gradient_calls'] for r in result['cases'])
    if result['elapsed_seconds_from_script_entry']>config['internal_seconds']:
        result['status']='FAILED'
        result['runtime_limit_exceeded']=True
    write_new(bundle/'result.json',result)
    total=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    if total>config['evidence_bytes']:
        raise RuntimeError('Compact evidence cap exceeded')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','backend')}),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(main(args.bundle.resolve()))
