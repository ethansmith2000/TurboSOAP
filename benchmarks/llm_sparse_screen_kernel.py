"""Cadence construction and disposable qualification, without changing native math."""
from collections import Counter
import math
import weakref
import torch
from cifar_optimizers import build_optimizer
from movement_retraction import configure_matrix
from hybrid_training_kernel import configure_hybrid, precision
from native_damped_kernel import LiveKernel
from matched_training_runtime import tensor_values, assert_scalar_finite, optimizer_extras
from tensor_ownership import TensorOwnership
from llm_sparse_screen_contract import SPECS


class Router:
    def __init__(self, emit, rng_digest, compiler=torch.compile):
        self.emit, self.rng_digest = emit, rng_digest
        self.calls, self.probes, self.moments = Counter(), [], []
        self.probe = False
        self.kernel = LiveKernel('highest_residual20', self, compiler)


def build_arm(model, config, arm, router):
    if arm not in SPECS:
        raise ValueError('Undeclared cadence arm')
    opt = build_optimizer(model, {**config, 'transport_precision': 'highest'},
                          'warm20' if arm == 'highest_residual20' else 'qr20')
    method, frequency, repair = SPECS[arm]
    for group in opt.matrix.param_groups:
        group.update(basis_method=method, precondition_frequency=frequency, hard_reset_interval=repair)
    if method == 'warm':
        configure_matrix(opt.matrix, 'fro_cubic2_cap05')
        configure_hybrid(opt.matrix, router.kernel)
    return opt


def qualify_actual_m(matrix, router, arm):
    """Disposable only: inspect actual transport, including both sides, not a shadow."""
    original = type(matrix)._research_refresh
    ref = weakref.ref(matrix)

    def refresh(state, group):
        obj = ref()
        if obj is None:
            raise RuntimeError('Released optimizer owner')
        first = not state.get('research_strict_transports', 0)
        repair = arm == 'highest_residual20' and state['step'] == 200
        if not (first or repair):
            return original(obj, state, group)
        owner = TensorOwnership()
        world = reconstructed = None
        row = dict(arm=arm, age=int(state['step']))
        before_rng = router.rng_digest()
        try:
            with precision('highest'):
                world = obj._project_back_with_basis(state['exp_avg'], state['Q'], group)
                owner.watch(world, 'actualM.before')
            original(obj, state, group)
            with precision('highest'):
                reconstructed = obj._project_back_with_basis(state['exp_avg'], state['Q'], group)
                owner.watch(reconstructed, 'actualM.after')
                error = float((reconstructed-world).norm()/world.norm().clamp_min(1e-12))
            row.update(physical_m_relative=error, passed=math.isfinite(error) and error <= .01,
                       rng_unchanged=before_rng == router.rng_digest())
            router.moments.append(row)
            router.emit('disposable_actual_m_before_gate', row)
            if not row['passed'] or not row['rng_unchanged']:
                raise FloatingPointError('Actual native momentum preservation failed')
        finally:
            world = reconstructed = None
            release = owner.report(); owner.close()
            router.emit('disposable_m_release', release)
            if not release['all_registered_released']:
                raise RuntimeError('Actual momentum qualification leaked tensors')

    matrix._research_refresh = refresh


def common_geometry_guard(opt, model=None):
    """Same factor coverage, absolute/normalized thresholds and ages for every arm."""
    rows = []
    with torch.no_grad(), precision('highest'):
        trees = [model.state_dict()] if model is not None else []
        for part in (opt.matrix, opt.auxiliary):
            trees.extend([part.state, optimizer_extras(part)])
        for tree in trees:
            assert_scalar_finite(tree)
            if not all(bool(torch.isfinite(t).all()) for t in tensor_values(tree)):
                raise FloatingPointError('Nonfinite model or optimizer state')
        for group in opt.matrix.param_groups:
            for name, p in zip(group['param_names'], group['params']):
                state = opt.matrix.state[p]
                if not all(bool(torch.isfinite(t).all()) for t in [p, state['exp_avg'], state['exp_avg_sq'], *state['GG']]):
                    raise FloatingPointError('Nonfinite matrix state')
                for axis, q in enumerate(state['Q']):
                    if q is None:
                        raise ValueError('Two-sided factor skipped')
                    gram = q.T @ q; gram.diagonal().sub_(1.)
                    absolute = float(gram.norm()); normalized = absolute/math.sqrt(len(q))
                    row = dict(parameter=name, axis=axis, dimension=len(q), absolute_gram_fro=absolute, normalized_gram=normalized)
                    rows.append(row)
                    if not math.isfinite(absolute) or absolute > .05 or normalized > .005:
                        raise FloatingPointError('Original geometry gates failed: '+str(row))
    return rows
