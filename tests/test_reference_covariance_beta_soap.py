"""Independent leaf oracle and exact baseline regression for covariance-only decay."""
import math

import pytest
import torch

from fast_soap import fast_warm_refresh
from reference_blocked_soap import ReferenceBlockedSOAP, projected_update
from reference_sparse_soap import SparseReferenceBlockedSOAP
from reference_covariance_beta_soap import CovarianceBetaSOAP, covariance_with_beta
from test_reference_blocked_soap import oracle_init


def oracle_transition(s, g, call, arm, beta, embedding):
    # Explicit leaf/block loops. Adaptive variance and bias correction remain .999.
    s['momentum'] = .9 * s['momentum'] + .1 * g
    full = torch.zeros_like(g)
    event = 'eigh' if call <= 50 else ('qr' if call % 50 == 0 else 'ordinary') if arm == 'sparse' else 'ordinary' if call % 10 else 'qr' if arm == 'qr' or call % 50 == 0 else 'warm'
    for block in s['blocks']:
        a, b, x, y = block['slice']
        v = block['v']
        gg, mm = torch.zeros_like(v), torch.zeros_like(v)
        gg[:b-a, :y-x] = g[a:b, x:y]
        mm[:b-a, :y-x] = s['momentum'][a:b, x:y]
        for side in ('l', 'r'):
            cov = block[side]
            if cov is None:
                continue
            cov = beta * cov + (1 - beta) * (gg @ gg.T if side == 'l' else gg.T @ gg)
            block[side] = cov
            q = block['q' + side]
            if event != 'ordinary':
                regular = cov + 1e-10 * torch.eye(len(cov))
                if event == 'eigh':
                    q = torch.linalg.eigh(regular).eigenvectors[:, torch.arange(len(cov)-1, -1, -1)]
                elif event == 'qr':
                    estimates = torch.diag(q.T @ regular @ q)
                    order = estimates.argsort(descending=True)
                    q = torch.linalg.qr(regular @ q[:, order]).Q
                    v = v[order, :] if side == 'l' else v[:, order]
                else:
                    q = fast_warm_refresh(regular, q)
                block['q' + side] = q
        gp, mp = gg, mm
        if block['ql'] is not None:
            gp, mp = block['ql'].T @ gp, block['ql'].T @ mp
        if block['qr'] is not None:
            gp, mp = gp @ block['qr'], mp @ block['qr']
        v = .999 * v + .001 * gp.square()
        block['v'] = v
        u = mp / (v.sqrt() + 1e-8)
        if block['qr'] is not None:
            u = u @ block['qr'].T
        if block['ql'] is not None:
            u = block['ql'] @ u
        full[a:b, x:y] = u[:b-a, :y-x]
    step = torch.tensor(float(call))
    full *= torch.sqrt(1 - .999**step) / (1 - .9**step)
    vector = s['vector']
    for _ in range(5 if call <= 50 else 1):
        yy = full @ vector
        sigma = yy.norm()
        z = full.T @ yy
        candidate = z / (z.norm() + 1e-10)
        if candidate.norm() > .5:
            vector = candidate
    s['vector'] = vector
    if embedding:
        full = full / (full.square().mean().sqrt() + 1e-10)
    else:
        full = full * math.sqrt(g.shape[1] / g.shape[0]) / (sigma + 1e-10)
    t = call - 1
    schedule = t / 177 if t < 177 else max(0, 1 - (t - 177) / (28332 - 177))
    s['p'] = s['p'] - schedule * (.032 * full + .0002 * s['p'])
    return event


@pytest.mark.parametrize('beta', [.99])
@pytest.mark.parametrize('arm', ['qr', 'warm', 'sparse'])
@pytest.mark.parametrize('shape,name', [((5, 7), 'weight'), ((11, 4), 'embed'), ((4, 11), 'weight')])
@pytest.mark.parametrize('resident', [False, True])
def test_covariance_only_change_matches_independent_complete_oracle(beta, arm, shape, name, resident):
    generator = torch.Generator().manual_seed(991)
    p = torch.nn.Parameter(torch.randn(*shape, generator=generator) * .01)
    opt = CovarianceBetaSOAP([(name, p)], arm=arm, shampoo_beta=beta,
                             block=4, maximum=8, compile_kernels=False)
    assert opt.projected is projected_update
    assert CovarianceBetaSOAP.step is SparseReferenceBlockedSOAP.step
    state = oracle_init(p.detach(), opt.groups[0]['state']['vector'][0], 4, 8)
    if resident:
        actual = opt.groups[0]['state']
        state['momentum'] = .02 * torch.randn(*shape, generator=generator)
        actual['momentum'][0].copy_(state['momentum'])
        for i, block in enumerate(state['blocks']):
            block['v'] = .003 + .002 * torch.rand(block['v'].shape, generator=generator)
            actual['nu'][i].copy_(block['v'])
            for side in ('l', 'r'):
                if block[side] is not None:
                    block[side] = torch.diag(.003 * torch.arange(1, len(block[side])+1))
                    actual[side+'c'][i].copy_(block[side])
    for call in range(1, 112):
        grad = .03 * torch.randn(*shape, generator=generator)
        p.grad = grad.clone()
        event = oracle_transition(state, grad, call, arm, beta, name == 'embed')
        assert opt.step() == event
        torch.testing.assert_close(p, state['p'], atol=3e-5, rtol=3e-4)
        actual = opt.groups[0]['state']
        torch.testing.assert_close(actual['momentum'][0], state['momentum'], atol=1e-8, rtol=1e-6)
        # At zero variance the first preconditioned matrix has near-equal singular
        # values; power-iteration directions are numerically unidentifiable (also
        # in the unmodified .999 baseline). Check that direction on the resident
        # nondegenerate state; parameters/M/C/V are checked in both cases.
        if resident:
            torch.testing.assert_close(torch.outer(actual['vector'][0], actual['vector'][0]),
                                       torch.outer(state['vector'], state['vector']), atol=3e-4, rtol=3e-3)
        for i, block in enumerate(state['blocks']):
            torch.testing.assert_close(actual['nu'][i], block['v'], atol=2e-8, rtol=2e-3)
            for side in ('l', 'r'):
                if block[side] is not None:
                    torch.testing.assert_close(actual[side+'c'][i], block[side], atol=1e-9, rtol=1e-5)


@pytest.mark.parametrize('arm', ['qr', 'warm', 'sparse'])
def test_baseline_beta_reproduces_all_qualified_state_exactly(arm):
    generator = torch.Generator().manual_seed(212)
    p = torch.nn.Parameter(torch.randn(5, 7, generator=generator) * .01)
    q = torch.nn.Parameter(p.detach().clone())
    kwargs = dict(block=4, maximum=8, compile_kernels=False)
    control = SparseReferenceBlockedSOAP([('weight', p)], **kwargs) if arm == 'sparse' else ReferenceBlockedSOAP([('weight', p)], arm=arm, **kwargs)
    candidate = CovarianceBetaSOAP([('weight', q)], arm=arm, shampoo_beta=.999, **kwargs)
    for _ in range(111):
        p.grad = torch.randn(p.shape, generator=generator) * .03
        q.grad = p.grad.clone()
        assert control.step() == candidate.step()
        torch.testing.assert_close(p, q, atol=0, rtol=0)
        for key, value in control.groups[0]['state'].items():
            torch.testing.assert_close(value, candidate.groups[0]['state'][key], atol=0, rtol=0)
        torch.testing.assert_close(control.step_tensor, candidate.step_tensor, atol=0, rtol=0)


@pytest.mark.parametrize('beta', [-1, 1, float('nan'), float('inf'), True, '.99'])
def test_bad_covariance_decay_rejected_before_allocating(beta):
    with pytest.raises(ValueError):
        CovarianceBetaSOAP([], shampoo_beta=beta)


def test_compiled_covariance_has_unchanged_raw_momentum_and_independent_decay():
    g = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) * .01
    meta = (3, 4, 1, 1, 3, 4, True, True)
    momentum, left, right = torch.ones_like(g), torch.eye(3)[None], torch.eye(4)[None]
    compiled = torch.compile(covariance_with_beta, backend='eager', fullgraph=True)
    for beta in [0., .99, .999]:
        m, l, r = compiled(g, momentum, left, right, meta, beta)
        torch.testing.assert_close(m, .9 * momentum + .1 * g, atol=0, rtol=0)
        torch.testing.assert_close(l, beta * left + (1-beta) * (g @ g.transpose(-1, -2)), atol=0, rtol=0)
        torch.testing.assert_close(r, beta * right + (1-beta) * (g.transpose(-1, -2) @ g), atol=0, rtol=0)
