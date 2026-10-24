"""Sparse, state-preserving covariance/basis measurements for seed replication."""
import math

import torch

from reference_llama import ReferenceLlama


def model_for_seed(seed, device='cuda', **kwargs):
    torch.manual_seed(seed)
    return ReferenceLlama(device=device, **kwargs)


@torch.no_grad()
def factor_metrics(covariance, basis, reference_covariance, reference_basis):
    # Copies promote only these selected factors. No optimizer tensors are mutated.
    c, q, c0, q0 = [t.detach().to(dtype=torch.float64) for t in
                    (covariance, basis, reference_covariance, reference_basis)]
    assert c.ndim == 2 and c.shape[0] == c.shape[1]
    assert all(t.shape == c.shape and bool(torch.isfinite(t).all()) for t in (c, q, c0, q0))
    norm, norm0 = float(torch.linalg.vector_norm(c)), float(torch.linalg.vector_norm(c0))
    change = float(torch.linalg.vector_norm(c-c0))

    def basis_values(matrix):
        transformed = matrix.T @ c @ matrix
        diagonal = torch.diag_embed(transformed.diagonal())
        den = float(torch.linalg.vector_norm(transformed))
        off = float(torch.linalg.vector_norm(transformed-diagonal))
        gram = matrix.T @ matrix - torch.eye(len(matrix), dtype=matrix.dtype, device=matrix.device)
        return (off/den if den else 0.), float(torch.linalg.vector_norm(gram))

    current_error, current_gram = basis_values(q)
    held_error, held_gram = basis_values(q0)
    trace = float(c.trace())
    result = dict(covariance_frobenius=norm, reference_covariance_frobenius=norm0,
                  covariance_change_frobenius=change,
                  relative_covariance_change=change/norm0 if norm0 else (0. if not norm else None),
                  zero_reference_covariance=not bool(norm0),
                  current_relative_offdiagonal=current_error, held_relative_offdiagonal=held_error,
                  current_absolute_gram=current_gram, held_absolute_gram=held_gram,
                  participation_rank=trace*trace/(norm*norm) if norm else 0.)
    assert all(value is None or isinstance(value, bool) or math.isfinite(value) for value in result.values())
    return result


def resolve_factors(opt, specifications):
    resolved = []
    for spec in specifications:
        matches = [(i, g) for i, g in enumerate(opt.groups) if spec['parameter'] in g['names']]
        assert len(matches) == 1, spec
        gi, group = matches[0]
        side = spec['side']
        assert side in ('l', 'r')
        tiles_per_parameter = group['meta'][2] * group['meta'][3]
        tile = spec.get('tile', 0)
        assert isinstance(tile, int) and 0 <= tile < tiles_per_parameter
        fi = group['names'].index(spec['parameter']) * tiles_per_parameter + tile
        assert len(group['state'][side+'c']) > fi, spec
        resolved.append(dict(**spec, group_index=gi, factor_index=fi,
                             dimension=group['state'][side+'c'].shape[-1]))
    assert len({(x['group_index'], x['side'], x['factor_index']) for x in resolved}) == len(resolved)
    return resolved


class BasisDiagnostics:
    def __init__(self, opt, specifications, windows, owner=None):
        self.factors = resolve_factors(opt, specifications)
        self.windows = [tuple(w) for w in windows]
        assert all(a < b-1 for a, b in self.windows)
        assert all(b < c for (_, b), (c, _) in zip(self.windows, self.windows[1:]))
        self.calls = {n for a, b in self.windows for n in (a, b-1, b)}
        self.references = {}
        self.owner = owner

    @torch.no_grad()
    def measure(self, opt, call):
        assert call in self.calls
        window = next((a, b) for a, b in self.windows if call in (a, b-1, b))
        start, end = window
        if call == start:
            assert not self.references
            for i, factor in enumerate(self.factors):
                state = opt.groups[factor['group_index']]['state']
                side, index = factor['side'], factor['factor_index']
                pair = (state[side+'c'][index].detach().clone(), state[side+'q'][index].detach().clone())
                self.references[i] = pair
                if self.owner is not None:
                    for label, tensor in zip(('covariance', 'basis'), pair):
                        self.owner.watch(tensor, f'basis_diagnostic.{start}.{i}.{label}')
        assert len(self.references) == len(self.factors)
        rows = []
        for i, factor in enumerate(self.factors):
            state = opt.groups[factor['group_index']]['state']
            side, index = factor['side'], factor['factor_index']
            rows.append(dict(**factor, **factor_metrics(state[side+'c'][index], state[side+'q'][index], *self.references[i])))
        reference_bytes = sum(t.numel()*t.element_size() for pair in self.references.values() for t in pair)
        if call == end:
            self.references.clear()
        return dict(call=call, window_start=start, window_end=end, elapsed_updates=call-start,
                    phase='window_start' if call == start else 'last_before_repair' if call == end-1 else 'after_repair',
                    factors=rows, reference_bytes=reference_bytes, references_released=not bool(self.references))

    def close(self):
        assert not self.references, 'Incomplete diagnostic window'
        self.owner = None
