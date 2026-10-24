"""FastSOAP builder and sparse diagnostics; production warm kernel stays clean."""
import math
import weakref

import torch
from cifar_optimizers import SplitOptimizer, grouped_parameters
from deep_llm_learning_observer import LayerObserver as NativeObserver
from fast_soap import FastSOAP, fast_warm_refresh
from hybrid_training_kernel import precision
from tensor_ownership import TensorOwnership

FREQUENCY = 10
REPAIR = 50
PREFIX = 75
PRIMARY_M_AGES = (10, 50, 1000, 3000, 10000, 20000, 30000, 34800)


def event_kind(call):
    age = call-1
    return 'cold' if call == 1 else ('qr' if age % REPAIR == 0 else
                                    'warm' if age % FREQUENCY == 0 else 'ordinary')


def build_optimizer(model, config, policy='fast10', *, compile_warm=True):
    if policy != 'fast10':
        raise ValueError('Only the declared FastSOAP warm10/QR50 candidate')
    groups = grouped_parameters(model, config)
    matrix = FastSOAP([g for g in groups if g['role'] == 'matrix'],
                      lr=config['matrix_lr'], betas=tuple(config.get('matrix_betas', config['betas'])),
                      shampoo_beta=config['shampoo_beta'], precondition_frequency=10,
                      hard_reset_interval=50, compile_warm=compile_warm)
    auxiliary = torch.optim.AdamW([g for g in groups if g['role'] == 'auxiliary'],
                                 betas=tuple(config['betas']), eps=1e-8,
                                 fused=next(model.parameters()).is_cuda)
    return SplitOptimizer(matrix, auxiliary)


def qualify_actual_m(matrix, router, arm, phase):
    method = type(matrix)._research_refresh
    ref = weakref.ref(matrix)

    def refresh(state, group):
        obj = ref()
        age = int(state['step'])
        check_m = age in ((10, 50) if phase == 'disposable' else PRIMARY_M_AGES)
        check_own = phase == 'disposable' and age in (10, 60)
        if not (check_m or check_own):
            return method(obj, state, group)
        owner = TensorOwnership()
        eager = []
        world = after = target = new = None
        rng = router.rng_digest()
        try:
            if check_own:
                for cov, q in zip(state['GG'], state['Q']):
                    target = fast_warm_refresh(cov, q)
                    owner.watch(target, 'own_eager.basis')
                    eager.append(target)
                target = None
            if check_m:
                with precision('highest'):
                    world = obj._project_back_with_basis(state['exp_avg'], state['Q'], group)
                    owner.watch(world, 'actualM.before')
            method(obj, state, group)
            if check_own:
                for axis, (new, target) in enumerate(zip(state['Q'], eager)):
                    with precision('highest'):
                        error = float((new-target).norm()/target.norm().clamp_min(1e-12))
                    row = dict(age=age, axis=axis, dimension=len(new), own_eager_basis_relative=error,
                               finite=math.isfinite(error), acceptance='Report-only; geometry/physical-M/finite are hard guards')
                    router.parities.append(row)
                    router.emit('own_warm_factor_report', row)
                    if not row['finite']:
                        raise FloatingPointError('Nonfinite own-policy comparison')
                new = target = None
            if check_m:
                with precision('highest'):
                    after = obj._project_back_with_basis(state['exp_avg'], state['Q'], group)
                    owner.watch(after, 'actualM.after')
                    error = float((after-world).norm()/world.norm().clamp_min(1e-12))
                row = dict(arm='fast10', phase=phase, age=age, physical_m_relative=error,
                           passed=math.isfinite(error) and error <= .01,
                           rng_unchanged=rng == router.rng_digest())
                router.moments.append(row)
                router.emit('actual_m_before_gate', row)
                if not row['passed'] or not row['rng_unchanged']:
                    raise FloatingPointError('Actual FastSOAP momentum preservation failed')
            if rng != router.rng_digest():
                raise RuntimeError('Disposable diagnostics changed RNG')
        finally:
            eager.clear()
            world = after = target = new = None
            release = owner.report()
            owner.close()
            router.emit('actual_m_release', release)
            if not release['all_registered_released']:
                raise RuntimeError('FastSOAP diagnostic buffers retained')
    matrix._research_refresh = refresh


class LayerObserver(NativeObserver):
    def capture(self, opt, call, target, lag):
        super().capture(opt, call, target, lag)
        for row in self.banks[(target, lag)]['data']:
            row['last_refresh_call'] = (row['age']//FREQUENCY)*FREQUENCY + 1

    def measure(self, opt, call, owner):
        row = super().measure(opt, call, owner)
        row.update(meaning='Candidate-owned FastSOAP warm10/QR50 history; BF16 ordinary-projection previews are read-only',
                   source_policy='fast10', current_last_refresh_call=((call-1)//FREQUENCY)*FREQUENCY+1)
        return row
