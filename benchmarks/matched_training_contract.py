"""CPU-tested prerequisites for a future shared-process training comparison.

This module does not launch training or change any existing trainer. Initial
weights live only in CPU RAM for declared paired resets, never in checkpoint files.
"""
import copy
from contextlib import contextmanager, nullcontext
import hashlib
import json
import random

import numpy as np
import torch
from torch.nn import functional as F

try:
    from .cifar_optimizers import SplitOptimizer, build_optimizer, grouped_parameters
    from .legacy_soap import build_legacy_optimizer, effective_policy
    from .movement_retraction import configure_matrix
except ImportError:
    from cifar_optimizers import SplitOptimizer, build_optimizer, grouped_parameters
    from legacy_soap import build_legacy_optimizer, effective_policy
    from movement_retraction import configure_matrix


ARMS = ('qr20', 'may_literal', 'fro_cubic2_cap05', 'qr20_repeat')
HORIZON = 3500
OBSERVATIONS = (0, 100, 400, 875, 1000, 1750, 2625, 3500)
RAM_CAP = 256 * 2**20
COMMON = dict(lr=.0005, matrix_lr=.0005, betas=[.9, .999],
              shampoo_beta=.99, weight_decay=.05)
MAY_OVERRIDES = dict(lr=.0005, betas=(.9, .999), shampoo_beta=.99,
                     weight_decay=.05, eps=1e-8)


def wire(value):
    return json.loads(json.dumps(value, allow_nan=False))


def tensor_fingerprint(tensor):
    """Logical full bytes, including BF16; no retained tensor or sampled digest."""
    value = tensor.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(sha256=hashlib.sha256(header + raw).hexdigest(),
                dtype=str(value.dtype), shape=list(value.shape),
                finite=bool(torch.isfinite(value).all()))


def compact(value):
    if torch.is_tensor(value):
        return {'tensor': tensor_fingerprint(value)}
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        return dict(numpy_dtype=str(a.dtype), shape=list(a.shape),
                    sha256=hashlib.sha256(a.tobytes()).hexdigest())
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            if not all(type(k) in (int, str) for k in value):
                raise TypeError('State dictionary keys must be strings or integer parameter indices')
            return {'typed_mapping': [[type(k).__name__, k, compact(v)]
                    for k, v in sorted(value.items(), key=lambda item: (type(item[0]).__name__, str(item[0])))]}
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [compact(v) for v in value]
    if isinstance(value, np.generic):
        return compact(value.item())
    return wire(value)


def digest(value):
    return hashlib.sha256(json.dumps(compact(value), sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


class RandomStreams:
    """Includes global NumPy, crop generator and augmentation counters as well.

    CUDA is opt-in and must already be initialized by the managed GPU caller.
    CPU tests/imports never query CUDA availability or initialize a device.
    """
    def __init__(self, crop, augmentation, *, include_cuda=False):
        if include_cuda and not torch.cuda.is_initialized():
            raise RuntimeError('CUDA RNG capture requires an initialized caller')
        self.crop, self.augmentation, self.include_cuda = crop, augmentation, include_cuda

    def capture(self):
        return dict(torch=torch.get_rng_state().clone(), python=random.getstate(),
                    numpy=copy.deepcopy(np.random.get_state()),
                    crop=self.crop.get_state().clone(),
                    augmentation=copy.deepcopy(self.augmentation.rng.bit_generator.state),
                    counts=copy.deepcopy(self.augmentation.counts),
                    cuda=torch.cuda.get_rng_state_all() if self.include_cuda else [])

    def restore(self, state):
        torch.set_rng_state(state['torch'])
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        self.crop.set_state(state['crop'])
        self.augmentation.rng.bit_generator.state = copy.deepcopy(state['augmentation'])
        self.augmentation.counts = copy.deepcopy(state['counts'])
        if self.include_cuda:
            torch.cuda.set_rng_state_all(state['cuda'])


class SharedInitialState:
    """One purpose-specific initial model snapshot; no optimizer state snapshots.

    Caller must start with all gradients None. Restores parameters, persistent
    buffers, each module's training flag and all declared RNG streams in place.
    Fingerprint workspace is conservatively budgeted at 3x the largest tensor.
    """
    def __init__(self, model, streams, *, extra_workspace_bytes=0, ram_cap=RAM_CAP):
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError('Clear gradients before capturing shared initial state')
        if type(extra_workspace_bytes) is not int or extra_workspace_bytes < 0:
            raise ValueError('Workspace bytes must be a nonnegative integer')
        live = model.state_dict()
        if not all(torch.is_tensor(v) for v in live.values()):
            raise TypeError('Model extra state needs an explicit snapshot contract')
        # Nonpersistent buffers need a separately declared contract, never omission.
        if any(m._non_persistent_buffers_set for m in model.modules()):
            raise ValueError('Nonpersistent buffers are outside this snapshot contract')
        sizes = [v.numel() * v.element_size() for v in live.values()]
        self.snapshot_bytes = sum(sizes)
        self.budget_bytes = self.snapshot_bytes + 3 * max(sizes, default=0) + extra_workspace_bytes
        if self.budget_bytes > ram_cap:
            raise MemoryError('Declared initial-state purpose RAM cap exceeded')
        self.model, self.streams = model, streams
        self.identities = self._identities()
        self.requirements = {n: p.requires_grad for n, p in model.named_parameters()}
        self.training = {n: m.training for n, m in model.named_modules()}
        self.weights = {n: v.detach().cpu().clone() for n, v in live.items()}
        self.rng = streams.capture()
        self.initial_digest = digest(self.weights)
        self.closed = False

    def _identities(self):
        return dict(parameters={n: id(v) for n, v in self.model.named_parameters()},
                    buffers={n: id(v) for n, v in self.model.named_buffers()},
                    modules={n: id(v) for n, v in self.model.named_modules()})

    def reset(self):
        if self.closed:
            raise RuntimeError('Initial snapshot already released')
        if self._identities() != self.identities:
            raise RuntimeError('Shared model identities changed')
        self.model.zero_grad(set_to_none=True)
        self.model.load_state_dict(self.weights, strict=True)
        for n, p in self.model.named_parameters():
            p.requires_grad_(self.requirements[n])
        for n, m in self.model.named_modules():
            m.training = self.training[n]
        self.streams.restore(self.rng)
        if digest(self.model.state_dict()) != self.initial_digest:
            raise RuntimeError('Initial model reset failed')
        if digest(self.streams.capture()) != digest(self.rng):
            raise RuntimeError('Initial RNG reset failed')

    @contextmanager
    def probe_scope(self):
        self.reset()
        try:
            yield
        finally:
            self.reset()

    def close(self):
        self.weights = self.rng = None
        self.closed = True


def build_arm(model, config, arm):
    if arm not in ARMS:
        raise ValueError('Unknown matched arm')
    if any(wire(config.get(k)) != v for k, v in COMMON.items()):
        raise ValueError('Common recipe drift; recipe experiments need separate design')
    if config.get('precondition_mode') != 'all' or config.get('covariance_compute_dtype') != 'float32':
        raise ValueError('Current reference requires both factors and FP32 covariance')
    if 'matrix_betas' in config and wire(config['matrix_betas']) != COMMON['betas']:
        raise ValueError('Matrix beta override changes the common recipe')
    if arm == 'may_literal':
        groups = grouped_parameters(model, config)
        matrix = build_legacy_optimizer([g for g in groups if g['role'] == 'matrix'],
                                        **MAY_OVERRIDES)
        auxiliary = torch.optim.AdamW([g for g in groups if g['role'] == 'auxiliary'],
            betas=tuple(config['betas']), eps=1e-8, fused=next(model.parameters()).is_cuda)
        return SplitOptimizer(matrix, auxiliary)
    c = {**config, 'transport_precision': 'highest'}
    opt = build_optimizer(model, c, 'warm20' if arm == 'fro_cubic2_cap05' else 'qr20')
    if arm == 'fro_cubic2_cap05':
        configure_matrix(opt.matrix, 'fro_cubic2_cap05')
    return opt


def optimizer_metadata(opt, arm):
    matrix = effective_policy(opt.matrix) if arm == 'may_literal' else dict(
        implementation='ResearchSOAP', groups=[dict(settings=wire(
            {k: v for k, v in g.items() if k != 'params'})) for g in opt.matrix.param_groups])
    return dict(arm=arm, matrix=matrix, auxiliary=[wire(
        {k: v for k, v in g.items() if k != 'params'}) for g in opt.auxiliary.param_groups])


def assert_fresh_independent(optimizers):
    components = [part for opt in optimizers for part in (opt.matrix, opt.auxiliary)]
    if len({id(part) for part in components}) != len(components):
        raise RuntimeError('Optimizer component alias across arms')
    groups = [g for part in components for g in part.param_groups]
    if len({id(g) for g in groups}) != len(groups):
        raise RuntimeError('Optimizer group alias across arms')
    if any(part.state for part in components):
        raise RuntimeError('Every comparison optimizer must start empty')
    if len({id(part.state) for part in components}) != len(components):
        raise RuntimeError('Optimizer state alias across arms')


@contextmanager
def forbid_steps(optimizers):
    """The all-arm numerical preflight is explicitly forbidden to update weights."""
    saved = []
    def forbidden(*args, **kwargs):
        raise RuntimeError('Optimizer steps forbidden until all-arm pre-update gate passes')
    try:
        for opt in optimizers:
            for part in (opt, opt.matrix, opt.auxiliary):
                previous = part.__dict__.get('step')
                had = 'step' in part.__dict__
                saved.append((part, had, previous))
                part.step = forbidden
        yield
    finally:
        for part, had, previous in reversed(saved):
            if had:
                part.step = previous
            else:
                del part.step


def preupdate_observation(model, active, make_batch, streams, *, clip_norm,
                          autocast=lambda: nullcontext()):
    """Caller-side hashes; the original model output/graph is unchanged.

    make_batch returns (batch indices, augmented input, smoothed mixed target).
    CUDA caller supplies autocast('cuda', dtype=bfloat16). No optimizer step.
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
    return wire(result)


def all_arm_preupdate_gate(snapshot, arms, observe):
    """Returns only after ALL declared arms pass, before ANY arm may train.

    No tolerance or retrospective selection of matching hash families is allowed.
    Failed results must be persisted by the future managed caller, with no retry.
    """
    if tuple(label for label, _ in arms) != ARMS:
        raise ValueError('All four prospective arms required in declared order')
    optimizers = [opt for _, opt in arms]
    assert_fresh_independent(optimizers)
    policies = [optimizer_metadata(opt, label) for label, opt in arms]
    records = []
    with forbid_steps(optimizers):
        for label, opt in arms:
            with snapshot.probe_scope():
                record = wire(observe(label, opt))
                required = {'indices', 'input', 'target', 'rng_before', 'rng_after_augmentation',
                            'output', 'loss', 'gradients_before_clip', 'clip_norm',
                            'gradients_after_clip', 'rng_after_backward'}
                if set(record) != required:
                    raise ValueError('Incomplete full pre-update observation')
                names = {n for n, _ in snapshot.model.named_parameters()}
                if any(set(record[k]) != names for k in ('gradients_before_clip', 'gradients_after_clip')):
                    raise ValueError('Incomplete gradient coverage')
                if digest(snapshot.model.state_dict()) != snapshot.initial_digest:
                    raise RuntimeError('Pre-update probe mutated model weights/buffers')
                assert_fresh_independent(optimizers)
                if [optimizer_metadata(o, n) for n, o in arms] != policies:
                    raise RuntimeError('Pre-update probe mutated optimizer policy')
                if records and record != records[0]['observation']:
                    raise RuntimeError('Exact all-arm pre-update comparison failed: ' + label)
                records.append(dict(arm=label, observation=record))
    return dict(status='PASS_PREUPDATE_ONLY', arms=records,
                initial_model_sha256=snapshot.initial_digest,
                snapshot_bytes=snapshot.snapshot_bytes, purpose_budget_bytes=snapshot.budget_bytes,
                optimizer_steps=0)
