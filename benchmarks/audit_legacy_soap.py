"""CPU-only behavior audit of literal May SOAP on prescribed gradients.

Call observation is explicitly diagnostic and untimed. An independently loaded
original trajectory verifies that observation preserves full numerical state.
No training data, GPU, checkpoint or tensor artifact is used.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch

try:
    from .legacy_soap import SOURCE, build_legacy_optimizer, effective_policy
except ImportError:
    from legacy_soap import SOURCE, build_legacy_optimizer, effective_policy


METHODS = ('_update_preconditioner', '_accumulate_outer_products',
           '_reorder_exp_avg_sq', '_finalize_basis_change', '_run_stiefel_tracker',
           '_fallback_basis_list', '_fallback_basis_one', '_basis_step_tangent',
           '_project', '_project_back')


@contextmanager
def observe_calls(optimizer, record):
    """Restore instance bindings even on failure; delegate every call unchanged."""
    previous = {name:optimizer.__dict__.get(name) for name in METHODS}
    present = {name:name in optimizer.__dict__ for name in METHODS}
    counts = Counter()
    preconditioner_depth = 0

    def wrap(name, original):
        @functools.wraps(original)
        def call(*args, **kwargs):
            nonlocal preconditioner_depth
            counts[name] += 1
            if name in ('_project', '_project_back') and preconditioner_depth:
                counts[name+'_inside_preconditioner'] += 1
            if name == '_run_stiefel_tracker':
                metrics = kwargs.get('use_metrics', args[6] if len(args)>6 else None)
                event = dict(call=record['call'], step=args[0]['step'],
                             metrics=bool(metrics), scales_before=list(args[3]))
                record['warm_events'].append(event)
            if name == '_update_preconditioner':
                preconditioner_depth += 1
            try:
                result = original(*args, **kwargs)
            finally:
                if name == '_update_preconditioner':
                    preconditioner_depth -= 1
            if name == '_run_stiefel_tracker':
                event['scales_after'] = list(args[3])
            return result
        return call

    try:
        for name in METHODS:
            setattr(optimizer, name, wrap(name, getattr(optimizer,name)))
        yield counts
    finally:
        for name in METHODS:
            if present[name]:
                setattr(optimizer, name, previous[name])
            else:
                optimizer.__dict__.pop(name, None)


def assert_exact(a, b, path='state'):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape, path
        assert torch.equal(a,b), path
        assert torch.isfinite(a).all(), path
    elif isinstance(a, dict):
        assert a.keys()==b.keys(), path
        for key in a:
            assert_exact(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (list,tuple)):
        assert type(a) is type(b) and len(a)==len(b), path
        for i,(x,y) in enumerate(zip(a,b)):
            assert_exact(x,y,f'{path}[{i}]')
    else:
        assert a==b, path


def independent_class():
    spec = importlib.util.spec_from_file_location('independent_may_soap', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SOAP


def run_case(label, kwargs, calls=331, expected_error=None):
    generator = torch.Generator().manual_seed(172)
    initial = torch.randn(4,6,generator=generator)
    p = torch.nn.Parameter(initial.clone())
    q = torch.nn.Parameter(initial.clone())
    live = build_legacy_optimizer([p], **kwargs)
    original = independent_class()([q], **kwargs)
    record = dict(label=label, overrides=kwargs, call=0, warm_events=[],
                  refresh_observations=[], effective_policy=effective_policy(live))
    failure = None
    completed_calls = 0
    with observe_calls(live, record) as counts:
        for index in range(calls):
            record['call'] = index+1
            grad = torch.randn(4,6,generator=generator)
            p.grad = grad.clone()
            q.grad = grad.clone()
            live_error = original_error = None
            try:
                live.step()
            except NotImplementedError as error:
                live_error = error
            try:
                original.step()
            except NotImplementedError as error:
                original_error = error
            assert_exact(p,q,'parameter')
            assert_exact(live.state[p],original.state[q])
            assert_exact(live.latest_basis_stats,original.latest_basis_stats,'stats')
            assert effective_policy(live)==effective_policy(original)
            if live_error is not None or original_error is not None:
                assert expected_error is not None, 'Unexpected CPU limitation'
                assert type(live_error) is type(original_error), 'Exception mismatch'
                assert str(live_error)==str(original_error)==expected_error
                failure = dict(type=type(live_error).__name__, message=str(live_error),
                               call=index+1, optimizer_step=live.state[p]['step'],
                               original_exception_and_partial_state_exact=True)
                break
            completed_calls += 1
            if index==0:
                assert torch.equal(p,initial), 'First call must skip parameter update'
            if index and index % kwargs.get('precondition_frequency',10)==0:
                record['refresh_observations'].append(dict(
                    call=index+1, step=live.state[p]['step'],
                    stats=dict(live.latest_basis_stats)))
    assert not any(name in live.__dict__ for name in METHODS), 'Observer leaked'
    state = live.state[p]
    record.pop('call')
    observed_calls = index+1
    assert (failure is not None)==(expected_error is not None)
    record.update(status='CPU_UNSUPPORTED_DEFAULTS' if failure else 'PASS',
                  planned_gradient_calls=calls, observed_gradient_calls=observed_calls,
                  completed_gradient_calls=completed_calls,
                  completed_parameter_updates=completed_calls-1,
                  optimizer_step_counter_at_stop=state['step'], failure=failure,
                  full_state_exact_every_observed_call=True,
                  observer_restored=True, method_counts=dict(counts),
                  final_parameter_sha256=hashlib.sha256(p.detach().numpy().tobytes()).hexdigest(),
                  direction_ema=[{k:dict(shape=list(v.shape),dtype=str(v.dtype),
                                      numel=v.numel()) if torch.is_tensor(v) else v
                                  for k,v in moment.items()}
                                 for moment in state.get('basis_direction_moments',[])],
                  final_basis_scales=state['basis_scales'])
    assert counts['_accumulate_outer_products']==observed_calls
    assert counts['_project_back_inside_preconditioner']==observed_calls-1
    assert counts['_project_inside_preconditioner']==observed_calls+1-int(bool(failure))
    assert counts['_reorder_exp_avg_sq']==(observed_calls-1)//kwargs.get('precondition_frequency',10)
    assert counts['_finalize_basis_change']==counts['_reorder_exp_avg_sq']-int(bool(failure))
    return record


def audit():
    started = time.monotonic()
    cpu_error = '"trace" not implemented for \'BFloat16\''
    rows = [run_case('literal_constructor_defaults',{},expected_error=cpu_error),
            run_case('recalled_beta2_tentative_lr_default_beta1',
                     dict(lr=.0006,betas=(.95,.999),shampoo_beta=.999),
                     expected_error=cpu_error),
            run_case('fp32_dtype_only_reconstruction',
                     dict(basis_compute_dtype='float32',basis_storage_dtype='float32',
                          covariance_compute_dtype='float32')),
            run_case('fp32_forced_residual_fallback_reconstruction',
                     dict(basis_compute_dtype='float32',basis_storage_dtype='float32',
                          covariance_compute_dtype='float32',basis_warmup_steps=0,
                          basis_metric_frequency=1,
                          basis_fallback_diag_err_threshold=1e-12),calls=41)]
    for row in rows[:2]:
        assert row['method_counts']['_fallback_basis_list']==25
        assert row['failure']['call']==271
        assert row['completed_gradient_calls']==270
        assert [e['step'] for e in row['warm_events']]==[260,270]
        assert [e['step'] for e in row['warm_events'] if e['metrics']]==[260]
    assert [e['step'] for e in rows[2]['warm_events']]==list(range(260,331,10))
    assert [e['step'] for e in rows[2]['warm_events'] if e['metrics']]==[260,310]
    assert rows[3]['method_counts']['_fallback_basis_one']==8
    assert all(e['metrics'] for e in rows[3]['warm_events'])
    elapsed = time.monotonic()-started
    assert elapsed<120, 'CPU audit exceeded protocol bound'
    return dict(status='PASS_ADAPTER_SCOPE_WITH_CPU_DEFAULT_LIMITATION',
                first_attempt_status='FAILED/incomplete, retained separately',
                literal_default_full_horizon=False,
                device='cpu',torch_version=torch.__version__,
                elapsed_seconds=elapsed,cases=rows,
                scope='Prescribed gradients / literal source / full state parity and event behavior only. Not training, H200 reproduction, convergence, throughput or a GPU qualification.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit('Preserve existing evidence: output already exists')
    result=audit()
    payload=json.dumps(result,indent=2,allow_nan=False)+'\n'
    assert len(payload.encode())<5*1024**2
    with args.output.open('x') as stream:
        stream.write(payload)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
