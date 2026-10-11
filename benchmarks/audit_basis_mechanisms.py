"""Bounded CPU mechanism audit; never a trainer or optimizer repair."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

import torch

PIN = 'fff531c00b321531d07121f60cd25d4c7b46b9033d8617c5d03e11b28ac93b5d'
LOCAL_SOURCE = Path(__file__).resolve().parent/'reference/soap_may_930a5a3.py'
DEFAULT_SOURCE = LOCAL_SOURCE if LOCAL_SOURCE.exists() else Path('/workspace/optimizer_replay_results/cifar_may_refresh_frozen_20261010/source/benchmarks/reference/soap_may_930a5a3.py')
DIMENSIONS = (8, 32, 128)
SEED = 20261010


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(path=DEFAULT_SOURCE):
    if sha(path) != PIN:
        raise ValueError('Literal source digest mismatch')
    spec = importlib.util.spec_from_file_location('pinned_may_mechanisms', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def block_cuda():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('CPU audit requires hidden, uninitialized CUDA')
    def blocked(*args, **kwargs):
        raise RuntimeError('CUDA initialization prohibited by CPU audit')
    torch.cuda._lazy_init = blocked


def digest(value):
    """Fingerprint tiny CPU state only; no tensor data is retained."""
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            t = v.detach().contiguous()
            h.update(str((list(t.shape), str(t.dtype))).encode())
            h.update(t.view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v):
                h.update(str(k).encode()); add(v[k])
        elif isinstance(v, (list, tuple)):
            for x in v: add(x)
        else:
            h.update(repr(v).encode())
    add(value)
    return h.hexdigest()


def matrix_record(t):
    a = t.detach().double()
    col = torch.linalg.vector_norm(a, dim=0)
    result = dict(shape=list(t.shape), dtype=str(t.dtype), finite=bool(torch.isfinite(t).all()),
        sha256=digest(t), frobenius_norm=float(a.norm()),
        column_norm_min=float(col.min()), column_norm_max=float(col.max()))
    if t.ndim == 2 and t.shape[0] == t.shape[1]:
        result['gram_error_fp32'] = float((t.float().T @ t.float()-torch.eye(len(t))).norm()/len(t)**.5)
        result['gram_error_fp64_reference'] = float((a.T @ a-torch.eye(len(t), dtype=torch.float64)).norm()/len(t)**.5)
    return result


def moment_record(m):
    return dict(step=m.get('col_step', 0), sha256=digest(m),
        column_variance_min=float(m['col'].min()) if 'col' in m else None,
        column_variance_max=float(m['col'].max()) if 'col' in m else None,
        finite=all(bool(torch.isfinite(t).all()) for t in m.values() if torch.is_tensor(t)))


def retraction_record(module, x):
    gram = x.T @ x
    upper = gram.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(0)
    scale = (module._SOAP_NS_MAX_SINGULAR/(upper.sqrt()+1e-7)).clamp(max=1.)
    sv = torch.linalg.svdvals(x.double())
    y = module._soap_ns_retract(x)
    # Analytic spectral polynomial, with the actual scalar scale, is independent
    # of the matrix evaluation of that polynomial and reveals rounding effects.
    z = sv * float(scale)
    predicted_sv = (15*z-10*z**3+3*z**5).abs()/8
    predicted_gram = ((predicted_sv.square()-1).square().mean()).sqrt()
    y32 = module._soap_ns_retract(x.float())
    return y, dict(input=matrix_record(x), output=matrix_record(y),
        upper_squared_compute_dtype=float(upper),
        upper_squared_fp64=float((x.double().T @ x.double()).abs().sum(dim=-1).max()),
        input_singular_min=float(sv.min()), input_singular_max=float(sv.max()),
        actual_scale=float(scale), scaled_singular_min=float(z.min()), scaled_singular_max=float(z.max()),
        spectral_polynomial_gram_fp64=float(predicted_gram),
        fp32_arithmetic_control_on_same_input=matrix_record(y32),
        scope='One literal retraction on synthetic input; FP32 control changes arithmetic only')


def synthetic_input(n):
    gen = torch.Generator().manual_seed(SEED+n)
    u = torch.randn(n, n, generator=gen, dtype=torch.float64)
    cov = u @ u.T/n + torch.diag(torch.linspace(.5, 1.5, n, dtype=torch.float64))
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    return cov.float(), q.float()


def make_optimizer(module, n, dtype):
    dummy = torch.nn.Parameter(torch.zeros(n, n))
    opt = module.SOAP([dummy], basis_compute_dtype=dtype, basis_storage_dtype=dtype,
        covariance_compute_dtype=dtype, basis_compile=False)
    return opt, opt.param_groups[0]


@torch.no_grad()
def tracker_case(module, n, dtype, use_metrics):
    c, q32 = synthetic_input(n)
    q = q32.to(torch.bfloat16 if dtype == 'bfloat16' else torch.float32)
    opt, group = make_optimizer(module, n, dtype)
    # Synthetic incoming state at scale .7; empty direction EMA is deliberate
    # and NOT a reconstruction of the actual failed step-2 moments.
    state = dict(GG=[c], Q=[q], basis_scales=[.7], basis_direction_moments=[{}])
    initial = digest(state)
    traces = []
    def traced_step(C, Q, eta, normalizer, g, moment):
        cloned = copy.deepcopy(moment)
        cq = C @ Q
        raw = module._soap_tangent_direction(cq, Q)
        pre = module._soap_maybe_precondition_direction(cq, Q, raw,
            g['basis_direction_preconditioner'], g['basis_eigengap_floor'])
        direction = module._soap_normalize_direction_by_mode(pre, normalizer,
            moment=cloned, beta=opt._direction_normalizer_beta(g))
        x = Q + float(eta)*direction
        expected, ret = retraction_record(module, x)
        actual = opt._basis_step_tangent(C, Q, eta, normalizer, g, moment)
        assert torch.equal(expected, actual), 'Trace altered candidate arithmetic'
        assert digest(cloned) == digest(moment), 'Trace changed EMA semantics'
        traces.append(dict(covariance_work=matrix_record(C), basis_input=matrix_record(Q),
            raw_direction=matrix_record(raw), preconditioned_direction=matrix_record(pre),
            normalized_direction=matrix_record(direction), eta=float(eta),
            candidate=ret, moment_after=moment_record(moment),
            candidate_diagonal_error=opt._basis_metrics(C.float(), actual.float(), Q.float())['diag_err']))
        return actual
    result = dict(n=n, dtype=dtype, use_metrics=use_metrics, initial_state_sha256=initial,
        seed=SEED+n, input_covariance=matrix_record(c), input_basis=matrix_record(q),
        input_construction='SPD random U@U.T/n+positive diagonal; independent random QR basis, then cast',
        incoming_scale=.7, incoming_eta=.007, incoming_moment='empty synthetic EMA; not saved failure moments')
    try:
        out = opt._run_stiefel_tracker(state, [c], [q], state['basis_scales'], group, traced_step, use_metrics)[0]
        result.update(status='completed', output=matrix_record(out),
            unchanged_basis=torch.equal(q,out), stats=copy.deepcopy(getattr(opt,'_basis_stats_accum',None)))
    except Exception as exc:
        result.update(status='unsupported_or_failed', error=repr(exc), traceback=traceback.format_exc(),
            qualification='CPU limitation retained; no substitute qualifies literal BF16 fast path')
    result.update(traces=traces, scale_after=state['basis_scales'][0],
        moment_after=moment_record(state['basis_direction_moments'][0]))
    return result


@torch.no_grad()
def dense_control(module, n, dtype):
    gen = torch.Generator().manual_seed(SEED+1000+n)
    w = torch.randn(n,n,generator=gen,dtype=torch.float64)
    a = (w-w.T)/2
    # Single common scaling keeps the generator skew, independent of objective.
    a = a / (a.norm()/n)
    q = torch.eye(n,dtype=torch.bfloat16 if dtype=='bfloat16' else torch.float32)
    direction = a.to(q.dtype)
    x = q + .007*direction
    _, r = retraction_record(module,x)
    return dict(n=n,dtype=dtype,seed=SEED+1000+n,
        construction='Identity basis + .007 times random dense skew, common RMS=1; no covariance objective',
        direction=matrix_record(direction), direction_spectral_norm=float(torch.linalg.svdvals(direction.double()).max()),
        rotation_frobenius=float((.007*direction.double()).norm()), retraction=r)


@torch.no_grad()
def domain_control(module,n,dtype,scalar):
    x = torch.eye(n,dtype=torch.bfloat16 if dtype=='bfloat16' else torch.float32)*scalar
    _, r = retraction_record(module,x)
    return dict(n=n,dtype=dtype,input_scalar=scalar,
        construction='Scalar identity: contraction/near-orthogonal/declared scaling-limit stress; no direction objective',
        retraction=r, strict_gram_lt_005=r['output']['gram_error_fp32']<.05)


def run_audit(source=DEFAULT_SOURCE):
    block_cuda(); started=time.monotonic(); module=load_source(source)
    rows=[]; dense=[]; domain=[]
    for n in DIMENSIONS:
        for dtype in ['float32','bfloat16']:
            for metrics in [True,False]:
                if time.monotonic()-started>110:raise TimeoutError('Bounded CPU audit exceeded110s')
                rows.append(tracker_case(module,n,dtype,metrics))
            dense.append(dense_control(module,n,dtype))
            for scalar in [.5,1.,1.25]:domain.append(domain_control(module,n,dtype,scalar))
    assert not torch.cuda.is_initialized()
    # <=128 dimension, processed one case at a time; retained results are JSON
    # scalars. Generous 256 simultaneous FP64 matrix allowance is32MiB, including
    # explicit trace/reference copies. Native-library workspace/RSS is separate.
    conservative=256*max(DIMENSIONS)**2*8
    assert conservative<=64*2**20
    return dict(status='COMPLETED_CPU_MECHANISM_AUDIT',source_sha256=sha(source),
        source_path=str(source),dimensions=list(DIMENSIONS),seed=SEED,eta=.007,
        tracker=rows,dense_objective_independent=dense,retraction_domain=domain,
        seconds=time.monotonic()-started,cuda_initialized=False,gpu_launches=0,
        conservative_explicit_tensor_budget_bytes=conservative,
        tensor_budget_scope='Conservative explicit diagnostic arrays; not whole-process RSS or native workspace',
        interpretation='Synthetic mechanism controls only; no4608 replay, learning result, repair or GPU qualification')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise RuntimeError('Do not overwrite completed CPU evidence')
    result=run_audit(args.source)
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if len(text.encode())>10*2**20:raise RuntimeError('Compact evidence budget exceeded')
    args.output.write_text(text)
    print(json.dumps({k:result[k] for k in ['status','seconds','cuda_initialized','gpu_launches','conservative_explicit_tensor_budget_bytes']}))


if __name__=='__main__':main()
