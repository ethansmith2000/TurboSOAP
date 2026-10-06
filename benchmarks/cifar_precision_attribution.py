"""Local precision attribution with fixed-basis transport and QR-order controls."""
import math
import torch
try:
    from .cifar_refresh_efficiency import refresh, implied_update, VARIANTS
    from .cifar_tracking_probe import TrackingProbe, project, world, relative
except ImportError:
    from cifar_refresh_efficiency import refresh, implied_update, VARIANTS
    from cifar_tracking_probe import TrackingProbe, project, world, relative


def transport_at_fixed_bases(old, final_q, intermediate_q=None):
    m, qs = old['m'], old['q']
    if intermediate_q is not None:
        m = project(world(m, qs), intermediate_q)
        qs = intermediate_q
    return project(world(m, qs), final_q)


def qr_orders(old):
    return [torch.argsort((q.T @ cov @ q).diagonal(), descending=True)
            for cov, q in zip(old['cov'], old['q'])]


def qr_at_fixed_order(old, orders):
    return [torch.linalg.qr(cov @ q[:, order]).Q
            for cov, q, order in zip(old['cov'], old['q'], orders)]


def chain_metrics(states, names, eps):
    """All output projections use the caller's strict precision setting.

    Vector differences telescope. Their norms generally do not add; these
    path-dependent counterfactuals are not percentages of causal importance.
    """
    updates = [implied_update(state, eps) for state in states]
    ref_norm = updates[-1].norm().clamp_min(1e-30)
    total = updates[0] - updates[-1]
    deltas = [a - b for a, b in zip(updates[:-1], updates[1:])]
    closure = sum(deltas, torch.zeros_like(total)) - total
    result = dict(full_update_difference=relative(updates[0], updates[-1]),
                  fixed_basis_transport_difference=relative(updates[0], updates[1]),
                  strict_transport_remaining_difference=relative(updates[1], updates[-1]),
                  chain_closure_relative=float(closure.norm() / ref_norm),
                  stages={name: float(delta.norm() / ref_norm) for name, delta in zip(names, deltas)})
    return result


@torch.no_grad()
def attribute(old, group, variant):
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('high')
        high = refresh(old, group, variant)
        middle = refresh(old, group, 'warm1')['q'] if variant == 'warm2_sequential' else None
        orders_high = qr_orders(old) if variant == 'qr' else None
        torch.set_float32_matmul_precision('highest')
        strict = refresh(old, group, variant)
        fixed_m = transport_at_fixed_bases(old, high['q'], middle)
        fixed = dict(q=high['q'], m=fixed_m, v=high['v'])
        states = [high, fixed]
        stage_names = ['transport_arithmetic']
        order_details = None
        if variant == 'qr':
            orders_strict = qr_orders(old)
            fixed_order_q = qr_at_fixed_order(old, orders_high)
            fixed_order = dict(q=fixed_order_q,
                               m=transport_at_fixed_bases(old, fixed_order_q), v=high['v'])
            strict_basis_high_v = dict(q=strict['q'], m=strict['m'], v=high['v'])
            states.extend([fixed_order, strict_basis_high_v, strict])
            stage_names.extend(['basis_arithmetic_fixed_order', 'basis_ordering_fixed_v', 'v_permutation_fixed_basis'])
            order_details = dict(axes_equal=[bool(torch.equal(a, b)) for a, b in zip(orders_high, orders_strict)],
                                 mismatch_fractions=[float((a != b).float().mean()) for a, b in zip(orders_high, orders_strict)])
        else:
            states.append(strict)
            stage_names.append('basis_path')
        result = chain_metrics(states, stage_names, group['eps'])
        result.update(fixed_basis_m_difference=relative(high['m'], fixed_m),
                      fixed_basis_world_m_difference=relative(world(high['m'], high['q']), world(fixed_m, high['q'])),
                      variance_equal=bool(torch.equal(high['v'], strict['v'])),
                      qr_order=order_details,
                      full_update_pass=result['full_update_difference'] < .01,
                      fixed_basis_transport_pass=result['fixed_basis_transport_difference'] < .01,
                      strict_transport_remaining_pass=result['strict_transport_remaining_difference'] < .01)
        scalars = [result[k] for k in ['full_update_difference', 'fixed_basis_transport_difference',
                   'strict_transport_remaining_difference', 'chain_closure_relative',
                   'fixed_basis_m_difference', 'fixed_basis_world_m_difference']]
        scalars += list(result['stages'].values())
        if not all(math.isfinite(x) for x in scalars):
            raise FloatingPointError('Nonfinite precision attribution')
        if result['chain_closure_relative'] >= 1e-5:
            raise ValueError('Precision attribution chain does not close')
        return result
    finally:
        torch.set_float32_matmul_precision(previous)


class PrecisionProbe(TrackingProbe):
    @torch.no_grad()
    def flush(self):
        previous = torch.get_float32_matmul_precision()
        rows = []
        try:
            names = {id(self.opt.state[p]): name for group in self.opt.param_groups
                     for p, name in zip(group['params'], group['param_names'])}
            for state, group, old in self.pending:
                reset = group['basis_method'] == 'qr' or (group['hard_reset_interval'] > 0
                        and state['step'] % group['hard_reset_interval'] == 0)
                torch.set_float32_matmul_precision('high')
                control = refresh(old, group, 'qr' if reset else 'warm1')
                exact = (all(torch.equal(a, b) for a, b in zip(control['q'], state['Q']))
                         and torch.equal(control['m'], state['exp_avg'])
                         and torch.equal(control['v'], state['exp_avg_sq']))
                if not exact:
                    raise ValueError('Attribution control differs from actual refresh')
                rows.append(dict(matrix=names[id(state)], age=state['step'], shape=list(old['m'].shape),
                                 executed_method='qr' if reset else 'warm', executed_control_exact=exact,
                                 variants={name: attribute(old, group, name) for name in VARIANTS}))
        finally:
            torch.set_float32_matmul_precision(previous)
            self.pending.clear()
        return rows
