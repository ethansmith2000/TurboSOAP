"""Offline diagonal-V implementation and same-basis policy diagnostics."""
import torch
try:
    from .cifar_strict_transport import StrictTransportProbe, update
    from .cifar_tracking_probe import relative
except ImportError:
    from cifar_strict_transport import StrictTransportProbe, update
    from cifar_tracking_probe import relative


def diagonal_variance(old, new, group, age, policy):
    """Two-sided explicit formula; caller sets the matched arithmetic precision."""
    if policy == 'overlap':
        left=(old['q'][0].float().T@new[0].float()).square()
        right=(old['q'][1].float().T@new[1].float()).square()
        return ((left.T@old['v'])@right).clamp_min(0).float()
    if policy != 'permutation':
        raise ValueError('Unknown diagonal variance policy')
    v=old['v']
    reset=group['basis_method']=='qr' or (group['hard_reset_interval'] and age%group['hard_reset_interval']==0)
    if reset:
        for axis,(q,cov) in enumerate(zip(old['q'],old['cov'])):
            order=torch.argsort((q.T@cov@q).diagonal(),descending=True)
            v=v.index_select(axis,order)
    return v


class VarianceProbe(StrictTransportProbe):
    @torch.no_grad()
    def flush(self):
        items=list(self.pending)
        rows=super().flush()
        assert len(rows)==len(items)
        previous=torch.get_float32_matmul_precision()
        try:
            for row,(state,group,old) in zip(rows,items):
                # The actual V operation runs high, after strict M restores it.
                torch.set_float32_matmul_precision('high')
                policies={p:diagonal_variance(old,state['Q'],group,state['step'],p)
                          for p in ['permutation','overlap']}
                expected=policies[group['variance_policy']]
                exact=torch.equal(state['exp_avg_sq'],expected)
                implementation_error=relative(state['exp_avg_sq'],expected)
                if not exact or not bool(torch.isfinite(expected).all()) or bool((expected<0).any()):
                    raise FloatingPointError('Diagonal variance implementation gate failed')
                torch.set_float32_matmul_precision('highest')
                a=update(state['exp_avg'],policies['permutation'],state['Q'],group['eps'])
                z=update(state['exp_avg'],policies['overlap'],state['Q'],group['eps'])
                row['variance']=dict(policy=group['variance_policy'],formula_exact=exact,
                    implementation_relative_error=implementation_error,
                    minimum=float(state['exp_avg_sq'].min()),
                    mass_ratio=float(state['exp_avg_sq'].sum()/old['v'].sum().clamp_min(1e-30)),
                    relative_v_difference=relative(policies['overlap'],policies['permutation']),
                    same_basis_update_relative_difference=relative(z,a),
                    same_basis_update_cosine=float((a.flatten()@z.flatten())/(a.norm()*z.norm()).clamp_min(1e-30)))
        finally:
            torch.set_float32_matmul_precision(previous)
        return rows
