"""Native QR200 hold policy and read-only cadence-aware observations for trial195."""
import math
import weakref

import torch
from cifar_optimizers import build_optimizer as native_builder
from deep_llm_learning_observer import LayerObserver as NativeObserver
from hybrid_training_kernel import precision
from tensor_ownership import TensorOwnership

FREQUENCY = 200
PREFIX = 225
PRIMARY_M_AGES = (200, 400, 1000, 3000, 10000, 20000, 30000, 34800)


def event_kind(call):
    return 'cold' if call == 1 else ('qr' if (call - 1) % FREQUENCY == 0 else 'ordinary')


def build_optimizer(model, config, policy):
    if policy != 'qr20':
        raise ValueError('Parent construction label must remain qr20 before explicit QR200 override')
    opt = native_builder(model, config, policy)
    for group in opt.matrix.param_groups:
        group.update(basis_method='qr', precondition_frequency=FREQUENCY, hard_reset_interval=0)
    return opt


def qualify_actual_m(matrix, router, arm, phase):
    """Inspect actual first/selected QR200 transports, retaining only a weak owner."""
    method = matrix._research_refresh.__func__
    ref = weakref.ref(matrix)

    def refresh(state, group):
        obj = ref()
        if obj is None:
            raise RuntimeError('Released optimizer owner')
        due = not state.get('research_strict_transports', 0) if phase == 'disposable' else state['step'] in PRIMARY_M_AGES
        if not due:
            return method(obj, state, group)
        owner = TensorOwnership()
        before = after = None
        rng = router.rng_digest()
        try:
            with precision('highest'):
                before = obj._project_back_with_basis(state['exp_avg'], state['Q'], group)
                owner.watch(before, 'actualM.before')
            method(obj, state, group)
            with precision('highest'):
                after = obj._project_back_with_basis(state['exp_avg'], state['Q'], group)
                owner.watch(after, 'actualM.after')
                error = float((after-before).norm()/before.norm().clamp_min(1e-12))
            row = dict(arm='hold200', phase=phase, age=int(state['step']), physical_m_relative=error,
                       passed=math.isfinite(error) and error <= .01, rng_unchanged=rng == router.rng_digest())
            router.moments.append(row)
            router.emit('actual_m_before_gate', row)
            if not row['passed'] or not row['rng_unchanged']:
                raise FloatingPointError('Actual own-history momentum preservation failed')
        finally:
            before = after = None
            release = owner.report()
            owner.close()
            router.emit('actual_m_release', release)
            if not release['all_registered_released']:
                raise RuntimeError('Physical-M buffers leaked')
    matrix._research_refresh = refresh


class LayerObserver(NativeObserver):
    def capture(self, opt, call, target, lag):
        super().capture(opt, call, target, lag)
        bank = self.banks[(target, lag)]
        # The original observer computes QR20 metadata. Correct only that scalar;
        # the captured tensors, ownership, source fingerprint and release are unchanged.
        for row in bank['data']:
            row['last_refresh_call'] = (row['age']//FREQUENCY)*FREQUENCY + 1

    def measure(self, opt, call, owner):
        row = super().measure(opt, call, owner)
        row['meaning'] = 'Candidate-owned native-precision QR200 state; retained-gradient BF16 previews are read-only and do not change source arithmetic or qualify cadence learning'
        row['source_policy'] = 'hold200'
        row['current_last_refresh_call'] = ((call-1)//FREQUENCY)*FREQUENCY + 1
        return row
