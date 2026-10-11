"""Tiny CPU basis/history mechanism fixture, never a trainer or optimizer repair."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch

SEED = 20261010
BETA1, BETA2, EPS = .9, .999, 1e-8
EXPECTED = {
    'soap.py': '7e0d811daac8247f28284c61846b85dfa290a8faa4182979cc870051ee32070e',
    'soap_reference.py': '10f8b77a83742b849946177ee7c3f28cc169c638beedce4949132082ee0e119d',
    'benchmarks/movement_retraction.py': '0f62b9c8fdb0252b60b52a66dd3087ebfec43c8966420402f4622f7d4af23a60',
    'benchmarks/warm_retraction.py': 'ba6c7bc089c33534046a1c964168e9b838220bf6f0b4a6ec03a0a1002c7a607f',
}


def cpu_guard():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('Requires hidden, uninitialized CUDA')
    def prohibited(*args, **kwargs):
        raise RuntimeError('CUDA initialization prohibited')
    torch.cuda._lazy_init = prohibited


def load_frozen(root):
    root = Path(root).resolve()
    for relative, digest in EXPECTED.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen source mismatch: {relative}')
    sys.path[:0] = [str(root), str(root / 'benchmarks')]
    import soap
    import soap_reference
    import movement_retraction
    for module in (soap, soap_reference, movement_retraction):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError('Unexpected imported source')
    return soap, soap_reference, movement_retraction


def relative(actual, expected):
    a, b = actual.double(), expected.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def project(g, q):
    return q[0].T @ g @ q[1]


def world(g, q):
    return q[0] @ g @ q[1].T


def history_moments(history, q):
    """Direct gradient-history oracle, always in FP64."""
    q = [x.double() for x in q]
    m, v = torch.zeros_like(history[0]).double(), torch.zeros_like(history[0]).double()
    for g in history:
        z = project(g.double(), q)
        m = BETA1 * m + (1 - BETA1) * z
        v = BETA2 * v + (1 - BETA2) * z.square()
    return m, v


def full_second_moment(history):
    size = history[0].numel()
    s = torch.zeros(size, size, dtype=torch.float64)
    for g in history:
        z = g.double().reshape(-1)
        s = BETA2 * s + (1 - BETA2) * torch.outer(z, z)
    return s


def full_s_diagonal(s, q, shape):
    # Row-major vec(A G B.T) = (A tensor-product B) vec(G).
    t = torch.kron(q[0].double().T.contiguous(), q[1].double().T.contiguous())
    return (t @ s @ t.T).diagonal().reshape(shape)


def adaptive_update(m, v, g, q, age):
    m = BETA1 * m + (1 - BETA1) * project(g, q)
    v = BETA2 * v + (1 - BETA2) * project(g, q).square()
    scalar = math.sqrt(1 - BETA2 ** (age + 1)) / (1 - BETA1 ** (age + 1))
    return scalar * world(m / (v.sqrt() + EPS), q)


def update_record(actual, oracle):
    a, b = actual.double(), oracle.double()
    cosine = float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-24))
    cosine = max(-1., min(1., cosine))
    return dict(relative_error=relative(a, b), cosine=cosine,
                angle_degrees=math.degrees(math.acos(cosine)),
                norm_ratio=float(a.norm() / b.norm().clamp_min(1e-12)))


def geometry(cov, q):
    q = q.double()
    b = q.T @ cov.double() @ q
    e = q.T @ q - torch.eye(len(q), dtype=torch.float64)
    return dict(covariance_residual=float((b - torch.diag(b.diagonal())).norm() / cov.double().norm()),
                gram_normalized_frobenius=float(e.norm() / math.sqrt(len(q))),
                gram_spectral=float(torch.linalg.eigvalsh(e).abs().max()))


def rotation(n, angle):
    r = torch.eye(n, dtype=torch.float64)
    c, s = math.cos(angle), math.sin(angle)
    r[0, 0], r[0, 1], r[1, 0], r[1, 1] = c, -s, s, c
    return r


def histories(shape, kind, seed):
    m, n = shape
    length = 12 if shape == (2, 2) else 32
    gen = torch.Generator().manual_seed(seed)
    a, b = torch.linspace(.3, 2., m, dtype=torch.float64), torch.linspace(.2, 1.5, n, dtype=torch.float64)
    outer = torch.outer(a, b)
    values = []
    for i in range(length):
        if kind == 'correlated':
            g = outer * (1 + .2 * math.sin(i)) + .03 * torch.randn(m, n, generator=gen, dtype=torch.float64)
        else:
            g = torch.zeros(m, n, dtype=torch.float64)
            g.reshape(-1)[i % (m * n)] = outer.reshape(-1)[i % (m * n)] * (-1 if i % 2 else 1)
        values.append(g.float())
    future = (outer + .02 * torch.randn(m, n, generator=gen, dtype=torch.float64)).float()
    return values, future


def covariances(name):
    if name == 'equal_diagonal_2x2':
        c = torch.tensor([[1., .4], [.4, 1.]])
        return [c, c.clone()]
    if name == 'near_degenerate_2x2':
        c = torch.tensor([[1.0001, .00005], [.00005, .9999]])
        return [c, c.clone()]
    left = torch.diag(torch.tensor([4., 4., 2., 1.]))
    right = torch.diag(torch.tensor([3., 3., 2.5, 2., 1.5, 1., .8, .5]))
    left[0, 1] = left[1, 0] = .9
    right[0, 1] = right[1, 0] = .6
    return [left, right]


def source_transport(opt, group, m, v, old, new):
    state = dict(exp_avg=m.clone(), exp_avg_sq=v.clone())
    opt._transport_first_moment(state, old, new, group)
    opt._transport_second_moment_for_reset(state, old, new, group)
    return state['exp_avg'], state['exp_avg_sq']


@torch.no_grad()
def common_state(name, kind, seed, modules):
    soap, reference, movement = modules
    cov = covariances(name)
    shape = (len(cov[0]), len(cov[1]))
    old = [torch.eye(d) for d in shape]
    history, future = histories(shape, kind, seed)
    m64, v64 = history_moments(history, old)
    m, v = m64.float(), v64.float()
    s = full_second_moment(history)
    opt = reference.ResearchSOAP([torch.nn.Parameter(torch.zeros(shape))], basis_method='qr', transport_precision='highest')
    group = opt.param_groups[0]
    # Actual frozen QR refresh, including sorting, M transport and V permutation.
    qr_state = dict(Q=[q.clone() for q in old], GG=cov, exp_avg=m.clone(), exp_avg_sq=v.clone(), step=10)
    opt._research_refresh(qr_state, group)
    pair = [soap._largest_pair_jacobi_step(q, soap.SOAP._basis_covariance(c, q), math.pi / 4)
            for c, q in zip(cov, old)]
    eigen = []
    for c in cov:
        _, q = torch.linalg.eigh(c.double())
        eigen.append(q.flip(1).float())
    candidates = dict(hold=old,
        current_warm=[movement.warm(c, q, group, movement.POLICIES['fro_cubic2_cap05']) for c, q in zip(cov, old)],
        qr=qr_state['Q'], pair=pair, eigensolve=eigen)
    hold_u = adaptive_update(m64, v64, future.double(), [q.double() for q in old], len(history))
    rows = []
    for policy, new in candidates.items():
        q64 = [q.double() for q in new]
        oracle_m, oracle_v = history_moments(history, new)
        oracle_err = relative(full_s_diagonal(s, new, shape), oracle_v)
        assert oracle_err <= 1e-12, 'Independent history/full-S oracle disagreement'
        tm, ov = source_transport(opt, group, m, v, old, new)
        m_err = relative(world(tm.double(), q64), world(m64, [q.double() for q in old]))
        assert m_err <= 1e-5, 'Physical M preservation budget exceeded'
        geom = [geometry(c, q) for c, q in zip(cov, new)]
        assert max(x['gram_normalized_frobenius'] for x in geom) < .05
        oracle_u = adaptive_update(oracle_m, oracle_v, future.double(), q64, len(history))
        variants = {'carry': v, 'overlap': ov, 'exact_history': oracle_v}
        if policy == 'qr':
            assert torch.equal(qr_state['exp_avg'], tm)
            variants['qr_permutation'] = qr_state['exp_avg_sq']
        for variance_policy, variant_v in variants.items():
            # Exact-history row uses oracle M as well, isolating other rows' arithmetic/history.
            use_m = oracle_m if variance_policy == 'exact_history' else tm.double()
            u = adaptive_update(use_m, variant_v.double(), future.double(), q64, len(history))
            assert torch.isfinite(u).all() and torch.isfinite(variant_v).all()
            v_err = relative(variant_v, oracle_v)
            if kind == 'uncorrelated' and variance_policy == 'overlap':
                assert v_err <= 1e-5, 'Zero-cross-moment overlap formula mismatch'
            rows.append(dict(policy=policy, variance_policy=variance_policy, geometry=geom,
                world_m_relative_error=m_err, same_basis_v_relative_error=v_err,
                same_basis_next_update=update_record(u, oracle_u),
                candidate_oracle_update_vs_hold=update_record(oracle_u, hold_u),
                independent_oracle_relative_error=oracle_err))
    if name != 'near_degenerate_2x2':
        assert all(torch.equal(a, b) for a, b in zip(candidates['current_warm'], old)), 'Known stationary case changed'
    for c, q, initial in zip(cov, pair, old):
        before, after = geometry(c, initial)['covariance_residual'], geometry(c, q)['covariance_residual']
        assert after <= 1e-5 * before + 1e-7, 'Pair rotation did not diagonalize the single-block fixture'
    return dict(case=name, history_kind=kind, seed=seed, shape=list(shape), history_length=len(history), rows=rows)


@torch.no_grad()
def closed_loop(modules):
    _, reference, _ = modules
    history, _ = histories((4, 8), 'correlated', SEED + 99)
    old = [torch.eye(4), torch.eye(8)]
    new = [rotation(4, .35).float(), rotation(8, -.2).float()]
    m, v = [x.float() for x in history_moments(history, old)]
    opt = reference.ResearchSOAP([torch.nn.Parameter(torch.zeros(4, 8))])
    m_forward, v_forward = source_transport(opt, opt.param_groups[0], m, v, old, new)
    m_back, v_back = source_transport(opt, opt.param_groups[0], m_forward, v_forward, new, old)
    direct_m = new[0].T @ m @ new[1]
    direct_back = new[0] @ direct_m @ new[1].T
    assert relative(m_back, m) < 1e-5 and relative(direct_back, m) < 1e-5
    assert relative(v_back, v) > 1e-4, 'Expected closed-loop diagonal mixing absent'
    return dict(no_gradients_or_aging_between_rotations=True,
        round_trip_m_relative_error=relative(m_back, m),
        known_relative_m_round_trip_error=relative(direct_back, m),
        overlap_v_round_trip_error=relative(v_back, v), carry_v_round_trip_error=0.,
        v_sum_relative_change=float((v_back.sum() - v.sum()).abs() / v.sum()),
        note='Overlap conserves total variance approximately but is not reversible; no learning claim')


@torch.no_grad()
def missing_cross_moments():
    q = [rotation(2, .02), torch.eye(2, dtype=torch.float64)]
    b = torch.tensor([1., .3], dtype=torch.float64)
    outputs = []
    for a in [torch.tensor([1., 1.], dtype=torch.float64), torch.tensor([1., -1.], dtype=torch.float64)]:
        g = torch.outer(a, b)
        history = [g, -BETA1 * g]
        old_m, old_v = history_moments(history, [torch.eye(2, dtype=torch.float64)] * 2)
        _, exact_v = history_moments(history, q)
        approximate_v = q[0].square().T @ old_v @ q[1].square()
        outputs.append((old_m, old_v, exact_v, approximate_v))
    assert torch.equal(outputs[0][1], outputs[1][1])
    assert float(outputs[0][0].norm()) < 1e-14 and float(outputs[1][0].norm()) < 1e-14
    assert torch.equal(outputs[0][3], outputs[1][3])
    difference = float((outputs[0][2] - outputs[1][2]).norm())
    assert difference > 0.
    return dict(angle_radians=.02, same_stored_m=True, same_stored_diagonal_v=True,
        exact_v_difference_norm=difference, overlap_v_difference_norm=0.,
        positive_history_overlap_error=relative(outputs[0][3], outputs[0][2]),
        negative_history_overlap_error=relative(outputs[1][3], outputs[1][2]),
        note='Different discarded cross-moments; not a claim of identical complete SOAP covariance state')


@torch.no_grad()
def inherited_error(modules):
    _, reference, _ = modules
    q = [torch.diag(torch.tensor([1.1, 1.])), torch.eye(2)]
    r = rotation(2, .3).float()
    rotated = [q[0] @ r, q[1]]
    identity = torch.eye(2)
    before, after = geometry(identity, q[0]), geometry(identity, rotated[0])
    m = torch.tensor([[.3, .7], [-.4, 1.]])
    opt = reference.ResearchSOAP([torch.nn.Parameter(torch.zeros(2, 2))])
    state = {'exp_avg': m.clone()}
    opt._transport_first_moment(state, q, q, opt.param_groups[0])
    err = relative(world(state['exp_avg'], q), world(m, q))
    assert before['gram_normalized_frobenius'] > .05 and after['gram_normalized_frobenius'] > .05
    assert abs(before['gram_spectral'] - after['gram_spectral']) < 1e-6 and err > .01
    return dict(deliberately_invalid_incoming_basis=True, before=before, after_relative_rotation=after,
        unchanged_basis_source_m_round_trip_error=err,
        note='Orthogonal relative motion preserves inherited drift; retaining invalid input does not repair it')


def run(root):
    cpu_guard()
    started = time.monotonic()
    modules = load_frozen(root)
    cases = []
    for i, name in enumerate(['equal_diagonal_2x2', 'near_degenerate_2x2', 'rectangular_4x8']):
        for kind in ['correlated', 'uncorrelated']:
            if time.monotonic() - started > 110:
                raise TimeoutError('CPU fixture exceeded internal110s limit')
            cases.append(common_state(name, kind, SEED + i, modules))
    result = dict(status='PASS_CPU_DIRECTION_HISTORY_SCOPE', cases=cases,
        closed_loop=closed_loop(modules), missing_cross_moments=missing_cross_moments(),
        inherited_error=inherited_error(modules), seconds=time.monotonic() - started,
        seed=SEED, beta1=BETA1, beta2=BETA2, epsilon=EPS,
        covariance_is_prescribed_not_history_reconstruction=True,
        conservative_deliberate_tensor_bytes=128 * 32 * 32 * 8,
        tensor_budget_excludes_native_workspace_and_import_rss=True,
        torch_version=torch.__version__, cuda_initialized=torch.cuda.is_initialized(), gpu_launches=0,
        source_sha256=EXPECTED,
        interpretation='Synthetic mechanisms only; no learning, large-dimension, speed or production qualification')
    assert not result['cuda_initialized'] and result['seconds'] < 110
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Never overwrite completed evidence')
    try:
        result = run(args.source_root)
    except Exception as exc:
        result = dict(status='FAILED_CPU_DIRECTION_HISTORY_SCOPE', error=repr(exc), traceback=traceback.format_exc())
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        raise
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if len(text.encode()) > 10 * 2 ** 20:
        raise RuntimeError('Compact evidence budget exceeded')
    args.output.write_text(text)
    print(json.dumps({k: result[k] for k in ['status', 'seconds', 'cuda_initialized', 'conservative_deliberate_tensor_bytes']}))


if __name__ == '__main__':
    main()
