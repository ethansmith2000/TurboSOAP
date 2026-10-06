"""Read-only local refresh interventions, including moment handling and timing."""
import math
import statistics
import torch
from soap import SOAP
try:
    from .cifar_tracking_probe import TrackingProbe, factor_score, project, world, relative
except ImportError:
    from cifar_tracking_probe import TrackingProbe, factor_score, project, world, relative


VARIANTS = ('qr', 'warm1', 'warm2_final', 'warm2_sequential', 'warm1_cap02')


def refresh(old, group, variant):
    """Pure refresh: both bases, first-moment transport and the matching v policy.

    No incoming tensor is modified. Two inner steps reuse the same covariance.
    Final transport uses the original world m; sequential transport passes through
    the intermediate basis and therefore also its numerical Gram error.
    """
    if variant not in VARIANTS:
        raise ValueError('Unknown refresh variant')
    qs = old['q']
    m, v = old['m'], old['v']
    settings = {**group, 'basis_ns_iterations': 6,
                'basis_rotation_cap': .2 if variant == 'warm1_cap02' else .1}
    count = 2 if variant.startswith('warm2') else 1
    for inner in range(count):
        new = []
        for axis, (cov, q) in enumerate(zip(old['cov'], qs)):
            if q is None:
                raise ValueError('Efficiency probe requires both factors')
            if variant == 'qr':
                order = torch.argsort((q.T @ cov @ q).diagonal(), descending=True)
                candidate = torch.linalg.qr(cov @ q[:, order]).Q
                v = v.index_select(axis, order)
            else:
                candidate = SOAP._gauge_step_one(cov, q, settings)
            new.append(candidate.float())
        if variant == 'warm2_sequential':
            m = project(world(m, qs), new)
        elif inner == count - 1:
            m = project(world(old['m'], old['q']), new)
        qs = new
    return dict(q=qs, m=m, v=v)


def implied_update(state, eps):
    return world(state['m'] / (state['v'].sqrt() + eps), state['q'])


@torch.no_grad()
def diagnose(old, group, variant):
    """High-matmul-precision candidate versus its own strict-FP32 shadow."""
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('high')
        candidate = refresh(old, group, variant)
        torch.set_float32_matmul_precision('highest')
        reference = refresh(old, group, variant)
        incoming_world = world(old['m'], old['q'])
        candidate_world = world(candidate['m'], candidate['q'])
        reference_world = world(reference['m'], reference['q'])
        scores = [factor_score(c, q) for c, q in zip(old['cov'], candidate['q'])]
        result = dict(
            factors=scores,
            maximum_gram=max(s['gram'] for s in scores),
            world_m_preservation_error=relative(candidate_world, incoming_world),
            world_m_shadow_error=relative(candidate_world, reference_world),
            update_shadow_error=relative(implied_update(candidate, group['eps']),
                                         implied_update(reference, group['eps'])),
        )
        scalars = [v for k, v in result.items() if k != 'factors']
        scalars += [v for score in scores for v in score.values()]
        result['finite'] = all(math.isfinite(x) for x in scalars)
        result['passes'] = (result['finite'] and result['maximum_gram'] < .05
                            and result['world_m_preservation_error'] < .01
                            and result['world_m_shadow_error'] < .01
                            and result['update_shadow_error'] < .01)
        return result
    finally:
        torch.set_float32_matmul_precision(previous)


@torch.no_grad()
def time_variants(old, group, offset=0):
    """Three rotated rounds, three repetitions each; CUDA refresh latency only.

    Includes covariance projection used by the refresh, basis work, all m
    transport and QR v reindexing. Excludes covariance accumulation, Adam update,
    diagnostics and candidate comparison. No compilation or inference_mode.
    """
    if not old['m'].is_cuda:
        raise ValueError('CUDA event timing requires CUDA state')
    previous = torch.get_float32_matmul_precision()
    samples = {name: [] for name in VARIANTS}
    try:
        torch.set_float32_matmul_precision('high')
        for name in VARIANTS:
            refresh(old, group, name)
        torch.cuda.synchronize()
        for round_index in range(3):
            shift = (offset + round_index) % len(VARIANTS)
            order = VARIANTS[shift:] + VARIANTS[:shift]
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(3):
                    refresh(old, group, name)
                end.record(); end.synchronize()
                samples[name].append(start.elapsed_time(end) / 3)
        return {name: dict(round_ms=values, median_ms=statistics.median(values))
                for name, values in samples.items()}
    finally:
        torch.set_float32_matmul_precision(previous)


class EfficiencyProbe(TrackingProbe):
    @torch.no_grad()
    def flush(self):
        previous = torch.get_float32_matmul_precision()
        rows = []
        try:
            names = {id(self.opt.state[p]): name for group in self.opt.param_groups
                     for p, name in zip(group['params'], group['param_names'])}
            for index, (state, group, old) in enumerate(self.pending):
                reset = group['basis_method'] == 'qr' or (group['hard_reset_interval'] > 0
                        and state['step'] % group['hard_reset_interval'] == 0)
                torch.set_float32_matmul_precision('high')
                control = refresh(old, group, 'qr' if reset else 'warm1')
                exact = (all(torch.equal(a, b) for a, b in zip(control['q'], state['Q']))
                         and torch.equal(control['m'], state['exp_avg'])
                         and torch.equal(control['v'], state['exp_avg_sq']))
                if not exact:
                    raise ValueError('Local control differs from actual refresh')
                torch.set_float32_matmul_precision('highest')
                incoming = [factor_score(c, q) for c, q in zip(old['cov'], old['q'])]
                variants = {name: diagnose(old, group, name) for name in VARIANTS}
                timing = time_variants(old, group, offset=index)
                for name in VARIANTS:
                    variants[name]['timing'] = timing[name]
                torch.set_float32_matmul_precision('high')
                final = refresh(old, group, 'warm2_final')
                sequential = refresh(old, group, 'warm2_sequential')
                torch.set_float32_matmul_precision('highest')
                transport_difference = relative(implied_update(sequential, group['eps']),
                                                implied_update(final, group['eps']))
                rows.append(dict(matrix=names[id(state)], age=state['step'],
                                 shape=list(old['m'].shape), executed_method='qr' if reset else 'warm',
                                 executed_control_exact=exact,
                                 incoming=incoming, variants=variants,
                                 sequential_vs_final_update_difference=transport_difference))
        finally:
            torch.set_float32_matmul_precision(previous)
            self.pending.clear()
        return rows
