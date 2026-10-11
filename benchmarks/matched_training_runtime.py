"""Shared CPU/GPU orchestration for the frozen matched comparison design."""
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import time
import traceback
import weakref

import numpy as np
import torch
from torch.nn import functional as F

try:
    from .matched_training_contract import (ARMS, build_arm, all_arm_preupdate_gate,
        optimizer_metadata, digest, wire, tensor_fingerprint)
    from .cifar_optimizers import apply_lr_multiplier, build_optimizer
    from .cifar_support.cifar100_qualification import lr_at
    from .tensor_ownership import TensorOwnership
except ImportError:
    from matched_training_contract import (ARMS, build_arm, all_arm_preupdate_gate,
        optimizer_metadata, digest, wire, tensor_fingerprint)
    from cifar_optimizers import apply_lr_multiplier, build_optimizer
    from cifar100_qualification import lr_at
    from tensor_ownership import TensorOwnership


def streaming_tensor_record(tensor, chunk_bytes=2**20):
    """Full logical-byte hash matching the design, with <=3 chunks host workspace.

    Slice rows BEFORE transferring/contiguizing. Large covariance factors must
    never be copied wholesale to host RAM. No tensor-dependent numerical change.
    """
    if tensor.layout != torch.strided or type(chunk_bytes) is not int or chunk_bytes < 8:
        raise ValueError('Dense tensors and a positive chunk byte bound required')
    value = tensor.detach()
    row_elements = math.prod(value.shape[1:]) if value.ndim else 1
    row_bytes = row_elements * value.element_size()
    if row_bytes > chunk_bytes:
        raise MemoryError('Single logical row exceeds fingerprint workspace contract')
    rows_per_chunk = max(1, chunk_bytes // max(1, row_bytes))
    h = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)]).encode())
    finite = True
    if value.ndim == 0:
        value = value.reshape(1)
    for start in range(0, len(value), rows_per_chunk):
        part = value[start:start+rows_per_chunk].contiguous().cpu()
        raw = part.reshape(-1).view(torch.uint8).numpy().tobytes()
        h.update(raw)
        finite = finite and bool(torch.isfinite(part).all())
        del part, raw
    return dict(sha256=h.hexdigest(), dtype=str(tensor.dtype),
                shape=list(tensor.shape), finite=finite)


def streaming_compact(value):
    if torch.is_tensor(value):
        return {'tensor': streaming_tensor_record(value)}
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        return dict(numpy_dtype=str(a.dtype), shape=list(a.shape),
                    sha256=hashlib.sha256(a.tobytes()).hexdigest())
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            if not all(type(k) in (int, str) for k in value):
                raise TypeError('State keys must be text or integer indices')
            return {'typed_mapping': [[type(k).__name__, k, streaming_compact(v)]
                for k, v in sorted(value.items(), key=lambda item: (type(item[0]).__name__, str(item[0])))]}
        return {k: streaming_compact(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [streaming_compact(v) for v in value]
    if isinstance(value, np.generic):
        return streaming_compact(value.item())
    return wire(value)


def streaming_digest(value):
    return hashlib.sha256(json.dumps(streaming_compact(value), sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def tensor_values(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensor_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensor_values(child)


def assert_scalar_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError('Nonfinite scalar optimizer/controller state')
    if isinstance(value, dict):
        for child in value.values():
            assert_scalar_finite(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            assert_scalar_finite(child)


def optimizer_extras(part):
    result = {name: getattr(part, name) for name in ('latest_basis_stats', '_basis_stats_accum',
        'just_gathered_basis_stats', 'last_basis_refreshes', 'last_hard_reset_events') if hasattr(part, name)}
    cache = getattr(part, '_basis_weight_cache', {})
    result['basis_weight_cache'] = {repr(k): v for k, v in cache.items()}
    if getattr(part, '_basis_compiled_fns', {}):
        raise ValueError('Compiled optimizer kernels are outside this literal-default protocol')
    return result


@torch.no_grad()
def check_state(model, opt, maximum_gram_error=.05):
    """May [] / current None skipped factors, plus all moments/controller tensors."""
    prior = torch.get_float32_matmul_precision()
    errors = []; tensors = 0; skipped = 0
    try:
        torch.set_float32_matmul_precision('highest')
        for tensor in model.state_dict().values():
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError('Nonfinite model parameter/buffer')
        for part in (opt.matrix, opt.auxiliary):
            assert_scalar_finite(optimizer_extras(part))
            for tensor in tensor_values(optimizer_extras(part)):
                tensors += 1
                if not bool(torch.isfinite(tensor).all()):
                    raise FloatingPointError('Nonfinite optimizer cache/diagnostic tensor')
            for state in part.state.values():
                assert_scalar_finite(state)
                for tensor in tensor_values(state):
                    tensors += 1
                    if not bool(torch.isfinite(tensor).all()):
                        raise FloatingPointError('Nonfinite optimizer state, including covariance/controller')
        for state in opt.matrix.state.values():
            factors = state.get('Q')
            if factors is None:
                continue
            for q in factors:
                if q is None or (isinstance(q, list) and not q):
                    skipped += 1; continue
                if not torch.is_tensor(q) or q.ndim != 2 or q.shape[0] != q.shape[1] or not q.numel():
                    raise ValueError('Unsupported basis factor/sentinel')
                q32 = q.float()
                error = (q32.T @ q32 - torch.eye(len(q32), device=q.device)).norm() / len(q32)**.5
                value = float(error)
                if not math.isfinite(value) or value >= maximum_gram_error:
                    raise FloatingPointError('Predeclared basis Gram error exceeded')
                errors.append(value)
        return dict(factor_count=len(errors), skipped_factors=skipped,
                    checked_state_tensors=tensors, maximum_gram_error=max(errors, default=0.))
    finally:
        torch.set_float32_matmul_precision(prior)


def refresh_due(opt):
    for group in opt.matrix.param_groups:
        for p in group['params']:
            state = opt.matrix.state.get(p, {})
            if not state or state.get('Q') is None:
                return True
            next_step = int(state.get('step', 0)) + 1
            if next_step % int(group['precondition_frequency']) == 0:
                return True
    return False


class OptimizerLease:
    """Own one optimizer. Release checks cover registered state/storage aliases.

    Model parameters are persistent and excluded; gradients are explicitly cleared.
    External aliases cause release failure. No gc.collect or allocator-zero claim.
    """
    def __init__(self, model, optimizer):
        self.model, self.opt = model, optimizer
        self.refs = [weakref.ref(p) for p in (optimizer, optimizer.matrix, optimizer.auxiliary)]
        self.closed = False

    def release(self):
        if self.closed:
            raise RuntimeError('Optimizer lease already released')
        ownership = TensorOwnership()
        opt = self.opt
        for name, part in (('matrix', opt.matrix), ('auxiliary', opt.auxiliary)):
            ownership.watch_tree(list(part.state.values()), name + '.state_values')
            ownership.watch_tree(optimizer_extras(part), name + '.algorithm_extras')
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                ownership.watch(p.grad, 'gradient.' + name)
        del part
        self.model.zero_grad(set_to_none=True)
        opt.matrix.state.clear(); opt.auxiliary.state.clear()
        opt.param_groups.clear(); opt.matrix.param_groups.clear(); opt.auxiliary.param_groups.clear()
        self.opt = None; del opt
        objects_released = all(ref() is None for ref in self.refs)
        if ownership._tensors:
            result = ownership.report()
        else:
            result = dict(tracked_tensors=0, tracked_unique_storages=0,
                all_registered_released=True, scope='Empty optimizer state, no registered tensors')
        ownership.close(); self.closed = True
        result['optimizer_objects_released'] = objects_released
        result['passed'] = result['all_registered_released'] and objects_released
        return result


def failure_record(exc):
    result = dict(error=repr(exc), traceback=traceback.format_exc())
    # Preserve textual evidence first, then release tensor/optimizer references in
    # traceback frames before checking ownership. Never retain exception objects.
    seen = set(); current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current)); next_exc = current.__context__
        current.__traceback__ = None; current.__context__ = None; current.__cause__ = None
        current = next_exc
    return result


def captured_backward(model, active, make_batch, streams, *, clip_norm,
                      autocast=lambda: nullcontext()):
    """Same full observation and training math as frozen preupdate_observation.

    Also return detached loss/norm for history; do not recompute a second forward.
    CPU parity test compares every field and every clipped gradient to that helper.
    """
    before = digest(streams.capture())
    indices, x, target = make_batch()
    model.zero_grad(set_to_none=True)
    result = dict(indices=tensor_fingerprint(indices), input=tensor_fingerprint(x),
                  target=tensor_fingerprint(target), rng_before=before,
                  rng_after_augmentation=digest(streams.capture()))
    with autocast():
        logits = active(x)
        loss = F.cross_entropy(logits.float(), target)
    loss.backward()
    result.update(output=tensor_fingerprint(logits), loss=tensor_fingerprint(loss),
                  gradients_before_clip={n: tensor_fingerprint(p.grad) if p.grad is not None else None
                                         for n, p in model.named_parameters()})
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, foreach=True)
    result.update(clip_norm=tensor_fingerprint(norm),
                  gradients_after_clip={n: tensor_fingerprint(p.grad) if p.grad is not None else None
                                        for n, p in model.named_parameters()},
                  rng_after_backward=digest(streams.capture()))
    records = [result[k] for k in ('indices', 'input', 'target', 'output', 'loss', 'clip_norm')]
    records += [v for k in ('gradients_before_clip', 'gradients_after_clip')
                for v in result[k].values() if v is not None]
    if any(not r['finite'] for r in records):
        raise FloatingPointError('Nonfinite pre-update input/output/gradient')
    return wire(result), loss.detach(), norm.detach()


@contextmanager
def evaluation_scope(runtime):
    model = runtime.model
    flags = {n: m.training for n, m in model.named_modules()}
    rng = runtime.streams.capture(); before = streaming_digest(model.state_dict())
    try:
        model.eval()
        yield
        if streaming_digest(model.state_dict()) != before:
            raise RuntimeError('Evaluation mutated model state')
    finally:
        for n, m in model.named_modules():
            m.training = flags[n]
        runtime.streams.restore(rng)


def observe(runtime, step):
    started = time.monotonic()
    with evaluation_scope(runtime):
        values = runtime.evaluate()
    return dict(step=step, evaluation=wire(values), model_sha256=streaming_digest(runtime.model.state_dict()),
                rng_sha256=streaming_digest(runtime.streams.capture()), seconds=time.monotonic()-started)


def quality_view(result):
    return dict(observations=[{k: r[k] for k in ('step', 'evaluation', 'model_sha256', 'rng_sha256')}
                             for r in result['observations']],
                endpoint=result['endpoint'], loss_sequence_sha256=result['loss_sequence_sha256'],
                sampled_batches=result['sampled_batches'])


def preflight(runtime, factory):
    leases = []; arms = []; error = None; gate = None
    try:
        for arm in ARMS:
            leases.append(OptimizerLease(runtime.model, factory(runtime.model, runtime.config, arm)))
            arms.append((arm, leases[-1].opt))
        def probe(label, opt):
            runtime.before_probe(label)
            row, _, _ = captured_backward(runtime.model, runtime.active,
                lambda: runtime.make_batch(0), runtime.streams, clip_norm=runtime.config['clip_norm'],
                autocast=runtime.autocast)
            runtime.emit('preflight_probe', dict(arm=label, observation=row))
            return row
        gate = all_arm_preupdate_gate(runtime.snapshot, arms, probe)
    except BaseException as exc:
        error = failure_record(exc)
    finally:
        arms.clear()
        release = [lease.release() for lease in leases]
        runtime.emit('preflight_release', release)
    if error is not None:
        runtime.emit('preflight_failed', error)
        raise RuntimeError('All-arm preflight failed; no comparison updates')
    if not all(r['passed'] for r in release):
        raise RuntimeError('Preflight optimizer release failed')
    runtime.emit('preflight_passed', gate)
    return gate['arms'][0]['observation']


def train_arm(runtime, arm, expected_first, *, steps, observation_steps, factory):
    runtime.snapshot.reset()
    lease = OptimizerLease(runtime.model, factory(runtime.model, runtime.config, arm))
    result = dict(arm=arm, status='running', steps=0, observations=[], history=[], guards=[],
                  sampled_batches={}, timing_samples=[], effective_policy=optimizer_metadata(lease.opt, arm))
    error = None; loss_hash = hashlib.sha256(); pending = []
    try:
        result['observations'].append(observe(runtime, 0))
        runtime.emit('observation', dict(arm=arm, **result['observations'][-1]))
        for index in range(steps):
            runtime.budget()
            runtime.before_training_step(arm, index)
            refresh = refresh_due(lease.opt)
            apply_lr_multiplier(lease.opt, lr_at(index+1, runtime.schedule_steps,
                runtime.warmup_steps, 1., runtime.config['min_lr_ratio']))
            marks = runtime.marks()
            marks[0].record()
            if index == 0:
                row, loss, norm = captured_backward(runtime.model, runtime.active,
                    lambda: runtime.make_batch(index), runtime.streams,
                    clip_norm=runtime.config['clip_norm'], autocast=runtime.autocast)
                runtime.emit('actual_first_preupdate', dict(arm=arm, observation=row))
                if row != expected_first:
                    raise RuntimeError('Actual first pre-update differs from all-arm preflight: ' + arm)
                marks[1].record(); marks[2].record()
            else:
                indices, x, target = runtime.make_batch(index)
                marks[1].record()
                lease.opt.zero_grad(set_to_none=True)
                with runtime.autocast():
                    logits = runtime.active(x); loss = F.cross_entropy(logits.float(), target)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), runtime.config['clip_norm'], foreach=True)
                marks[2].record()
                loss = loss.detach(); norm = norm.detach()
                del logits
            runtime.before_optimizer_step(arm, index)
            lease.opt.step(); marks[3].record()
            if index > 0:
                if index+1 in observation_steps:
                    result['sampled_batches'][str(index+1)] = dict(indices=tensor_fingerprint(indices),
                        input=tensor_fingerprint(x), target=tensor_fingerprint(target))
                del x, target, indices
            result['steps'] = index+1
            pending.append((loss, norm, marks, index+1))
            if refresh:
                started = time.monotonic()
                guard = check_state(runtime.model, lease.opt)
                guard.update(step=index+1, seconds=time.monotonic()-started)
                result['guards'].append(guard); runtime.emit('refresh_guard', dict(arm=arm, **guard))
                runtime.budget(resources=True)
            if (index+1) % 50 == 0 or index+1 == steps:
                runtime.synchronize()
                losses = torch.stack([p[0] for p in pending]).cpu().numpy()
                norms = torch.stack([p[1] for p in pending]).cpu().numpy()
                if not np.isfinite(losses).all() or not np.isfinite(norms).all():
                    raise FloatingPointError('Nonfinite training loss/gradient norm')
                loss_hash.update(losses.astype(np.float32).tobytes())
                for _, _, ev, step in pending:
                    result['timing_samples'].append(dict(step=step, instrumented=step == 1,
                        phases_ms=runtime.elapsed(ev)))
                history = dict(step=index+1, mean_loss=float(losses.mean()), maximum_norm=float(norms.max()))
                result['history'].append(history); runtime.emit('history', dict(arm=arm, **history))
                pending.clear(); del losses, norms
                runtime.budget(resources=True)
            if index+1 in observation_steps:
                started = time.monotonic()
                full_check = check_state(runtime.model, lease.opt)
                row = observe(runtime, index+1)
                row['state_check'] = full_check; row['all_observation_seconds'] = time.monotonic()-started
                result['observations'].append(row); runtime.emit('observation', dict(arm=arm, **row))
        result['endpoint'] = dict(model_sha256=streaming_digest(runtime.model.state_dict()),
            optimizer_sha256=streaming_digest(dict(state=lease.opt.state_dict(),
                extras={'matrix':optimizer_extras(lease.opt.matrix),'auxiliary':optimizer_extras(lease.opt.auxiliary)})),
            rng_sha256=streaming_digest(runtime.streams.capture()))
        result['loss_sequence_sha256'] = loss_hash.hexdigest()
        steady = [r['phases_ms'] for r in result['timing_samples'] if r['step'] > 50 and not r['instrumented']]
        result['steady_timing'] = dict(excluded_first_calls=50, samples=len(steady),
            phase_mean_ms=np.asarray(steady).mean(0).tolist() if steady else [],
            qualified_speedup=False)
        result['status'] = 'completed'
    except BaseException as exc:
        error = failure_record(exc); result['status'] = 'failed'; result['failure'] = error
    finally:
        pending.clear()
        result['release'] = lease.release()
        runtime.emit('arm_result', result)
    if error is not None or not result['release']['passed']:
        raise RuntimeError('Arm failed or registered ownership release failed: ' + arm)
    return result


def disposable_smoke(runtime, calls):
    lease = OptimizerLease(runtime.model, build_optimizer(runtime.model, runtime.config, 'qr10'))
    error = None
    try:
        for index in range(calls):
            runtime.budget()
            _, x, target = runtime.make_batch(index)
            lease.opt.zero_grad(set_to_none=True)
            apply_lr_multiplier(lease.opt, lr_at(index+1, runtime.schedule_steps, runtime.warmup_steps,
                                                 1., runtime.config['min_lr_ratio']))
            with runtime.autocast():
                logits = runtime.active(x); loss = F.cross_entropy(logits.float(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), runtime.config['clip_norm'], foreach=True)
            lease.opt.step()
            check_state(runtime.model, lease.opt)
        runtime.synchronize()
        parity = runtime.smoke_parity(x)
        runtime.emit('smoke_parity', parity)
        del x, target, logits, loss
    except BaseException as exc:
        error = failure_record(exc)
    finally:
        release = lease.release(); runtime.emit('smoke_release', release)
        runtime.snapshot.reset()
    if error is not None:
        runtime.emit('smoke_failed', error)
        raise RuntimeError('Disposable smoke failed')
    if not release['passed']:
        raise RuntimeError('Disposable optimizer ownership release failed')


def run_shared(runtime, protocol, *, factory=build_arm):
    """One code path, tested with a tiny CPU runtime; production horizon is frozen."""
    results = []; status = 'failed'; error = None
    snapshots = TensorOwnership(); snapshots.watch_tree(runtime.snapshot.weights, 'initial_weights')
    snapshots.watch_tree(runtime.snapshot.rng, 'initial_rng')
    try:
        runtime.emit('started', dict(initial_model_sha256=runtime.snapshot.initial_digest,
            snapshot_bytes=runtime.snapshot.snapshot_bytes, purpose_budget_bytes=runtime.snapshot.budget_bytes))
        runtime.before_smoke()
        disposable_smoke(runtime, protocol['disposable_qr10_smoke_updates'])
        runtime.after_smoke()
        expected_first = preflight(runtime, factory)
        for arm in ARMS:
            runtime.current_arm = arm
            result = train_arm(runtime, arm, expected_first, steps=protocol['steps_per_arm'],
                observation_steps=protocol['observation_steps'], factory=factory)
            results.append(result)
            if arm != ARMS[0] and result['sampled_batches'] != results[0]['sampled_batches']:
                raise RuntimeError('Paired augmentation/index stream differs: ' + arm)
            if arm == 'qr20_repeat' and quality_view(result) != quality_view(results[0]):
                raise RuntimeError('QR-repeat exact trajectory/evaluation/endpoint gate failed')
        runtime.after_measured()
        runtime.budget(resources=True)
        status = 'completed'
    except BaseException as exc:
        error = failure_record(exc)
    finally:
        runtime.model.zero_grad(set_to_none=True)
        if status != 'completed':
            runtime.snapshot.reset()
        runtime.snapshot.close()
        snapshot_release = snapshots.report(); snapshots.close()
        placement = runtime.close()
    if not snapshot_release['all_registered_released'] or not placement['passed']:
        status = 'failed'
    report = dict(status=status, runs=results, failure=error, snapshot_release=snapshot_release,
                  placement_release=placement, quality_qualified=status == 'completed', timing_qualified=False)
    runtime.emit('finished', report)
    return report
