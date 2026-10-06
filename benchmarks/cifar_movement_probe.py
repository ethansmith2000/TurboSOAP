"""Read-only common-state clipping/retraction attribution on actual CIFAR states."""
import math
import torch
from soap import SOAP, _damped_jacobi_generator, _clip_rotation
from warm_retraction import Retraction, retract
from movement_retraction import clip_skew_frobenius
from cifar_strict_transport import StrictTransportProbe
from cifar_tracking_probe import factor_score, world, project, relative


class MovementProbe(StrictTransportProbe):
    @torch.no_grad()
    def flush(self):
        pending = list(self.pending)
        rows = super().flush()
        previous = torch.get_float32_matmul_precision()
        try:
            for row, (state, group, old) in zip(rows, pending):
                factors = []
                qs = {k: [] for k in ('ns6', 'row_cubic2', 'row_quintic1',
                      'fro_cubic2', 'fro_quintic1', 'oracle_cubic2')}
                for axis, (cov, q) in enumerate(zip(old['cov'], old['q'])):
                    # Candidate arithmetic matches the declared high-basis path;
                    # all diagnostic norms/eigensolves are strict FP32.
                    torch.set_float32_matmul_precision('high')
                    b = SOAP._basis_covariance(cov, q)
                    raw = group['basis_lr'] * _damped_jacobi_generator(b, group['basis_jacobi_damping'])
                    torch.set_float32_matmul_precision('highest')
                    spectral = torch.linalg.eigvalsh(raw.T @ raw)[-1].clamp_min(0).sqrt()
                    before = factor_score(cov, q)
                    raw_fro = float(raw.norm())
                    row_bound = float(raw.abs().sum(-1).amax())
                    policies = [
                        ('ns6', _clip_rotation(raw, .1, 'average'), Retraction('ns6', iterations=6, scaled=True)),
                        ('row_cubic2', _clip_rotation(raw, .5, 'spectral_bound'), Retraction('r', order=3, iterations=2)),
                        ('row_quintic1', _clip_rotation(raw, .5, 'spectral_bound'), Retraction('r')),
                        ('fro_cubic2', clip_skew_frobenius(raw, .5), Retraction('f', order=3, iterations=2)),
                        ('fro_quintic1', clip_skew_frobenius(raw, .4), Retraction('f')),
                        ('oracle_cubic2', raw*(.5/spectral.clamp_min(1e-12)).clamp(max=1), Retraction('oracle', order=3, iterations=2)),
                    ]
                    alternatives = {}
                    for name, a, policy in policies:
                        torch.set_float32_matmul_precision('high')
                        x = q.float() + q.float() @ a
                        y = retract(x, policy)
                        torch.set_float32_matmul_precision('highest')
                        scale = float(a.norm() / raw.norm().clamp_min(1e-30))
                        score = factor_score(cov, y)
                        alternatives[name] = dict(scale=scale, rotation_spectral=float(spectral)*scale,
                            movement_rms=float((y-q).norm())/math.sqrt(q.shape[0]),
                            residual_change=score['residual']-before['residual'], **score)
                        qs[name].append(y)
                    factors.append(dict(axis=axis, dimension=q.shape[0], before=before,
                        raw_spectral=float(spectral), raw_fro=raw_fro,
                        row_bound=row_bound, skew_fro_bound=raw_fro/math.sqrt(2),
                        alternatives=alternatives))
                old_world = world(old['m'], old['q'])
                effects = {}
                for name, bases in qs.items():
                    m = project(old_world, bases)
                    effects[name] = dict(world_m_error=relative(world(m, bases), old_world),
                        max_gram=max(f['alternatives'][name]['gram'] for f in factors))
                row.update(movement_factors=factors, movement_effects=effects)
        finally:
            torch.set_float32_matmul_precision(previous)
        return rows
