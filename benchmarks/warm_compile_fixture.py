"""Bounded synthetic common-input cost/fidelity probe; no trainer or weights."""
import argparse
from contextlib import contextmanager
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

PROCESS_STARTED = time.monotonic()
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import torch
from warm_compile_kernel import warm_kernel, classify, LIMITS

PROTOCOL = dict(dimensions=[1536, 4608, 6144], spectra=['broad', 'clustered'], seed=101017,
    beta=.99, rank=32, perturbation=2., basis_lr=.5, damping=.01, cap=.5, cubic_iterations=2,
    variants=['eager_fp32', 'compiled_fp32', 'compiled_mixed'], compile_mode='default',
    fullgraph=True, dynamic=False, first_setup_calls=2, warmup_calls=3, samples=9,
    recurrent_updates=12, recurrent_observe=[1, 4, 8, 12],
    internal_seconds=900, external_seconds=960, allocated_bytes=8*2**30,
    evidence_bytes=25*2**20, cpu_hash_chunk_bytes=2**20, limits=LIMITS,
    basis_precision='high', moment_probe_precision='highest',
    bf16_reduced_precision_reduction=False,
    scope='synthetic fixed-covariance basis fixture; skinny highest-FP32 momentum probes; no full SOAP timing or learning',
    artifacts='compact JSON/logs/source only; no checkpoints, tensor files, weights or profiler traces',
    retention='retain compact evidence; new isolated compiler cache retained for provenance, not model state',
    expected_evidence_bytes=2*2**20, expected_compiler_cache_bytes=100*2**20,
    compiler_cache_limit_bytes=512*2**20)


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def verify(bundle):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert json.loads((bundle / 'protocol.json').read_text()) == PROTOCOL
    assert sha(bundle / 'protocol.json') == manifest['protocol_sha256']
    for name, digest in manifest['source_sha256'].items():
        assert sha(bundle / 'source' / name) == digest, name
    return manifest


@contextmanager
def highest():
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try: yield
    finally: torch.set_float32_matmul_precision(old)


def scalar(x): return float(x.item())


def relative(a, b): return scalar((a - b).norm() / b.norm().clamp_min(1e-12))


def tensor_hash(x):
    h = hashlib.sha256()
    for part in x.detach().reshape(-1).split(PROTOCOL['cpu_hash_chunk_bytes'] // x.element_size()):
        h.update(part.cpu().numpy().tobytes())
    return h.hexdigest()


def graph_counts():
    from torch._dynamo.utils import counters
    return {group: dict(counters[group]) for group in ('stats', 'frames', 'graph_break', 'inductor')}


def metrics(actual, reference, initial, cov, world_m):
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all(), 'nonfinite or non-FP32 output'
    with highest():
        delta, target = actual - initial, reference - initial
        error = delta - target
        denom = target.norm().clamp_min(1e-12)
        gram = actual.T @ actual - torch.eye(actual.shape[0], device=actual.device)
        norm_gram = scalar(gram.norm() / actual.shape[0]**.5)
        diag_mean = scalar(gram.diagonal().mean())
        diag_max = scalar(gram.diagonal().abs().max())
        movement_relative = scalar(error.norm() / denom)
        cosine = scalar((delta * target).sum() / (delta.norm() * denom).clamp_min(1e-24))
        signed_parallel = scalar((error * target).sum() / denom.square())
        pm = actual.T @ world_m
        reference_pm = reference.T @ world_m
        reconstructed = actual @ pm
        def offdiag(q):
            b = q.T @ cov @ q
            return (b - torch.diag_embed(b.diagonal())).norm()
        actual_residual = offdiag(actual)
        reference_residual = offdiag(reference)
        r = dict(basis_relative=relative(actual, reference), movement_relative=movement_relative,
                 movement_cosine=cosine, signed_parallel_movement_error=signed_parallel,
                 gram=norm_gram, signed_diagonal_mean=diag_mean, max_abs_diagonal_error=diag_max,
                 world_m_relative=relative(reconstructed, world_m),
                 projected_m_relative=relative(pm, reference_pm),
                 tracking_ratio=scalar(actual_residual / reference_residual.clamp_min(1e-12)),
                 tracking_offdiag=scalar(actual_residual), reference_tracking_offdiag=scalar(reference_residual))
    return r


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@torch.no_grad()
def run(bundle):
    started = PROCESS_STARTED
    manifest = verify(bundle)
    assert os.environ.get('GPU_CLAIM_INDICES') and torch.cuda.device_count() == 1
    rows, setup = [], []
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    assert not torch.is_autocast_enabled()
    generator = torch.Generator(device='cuda').manual_seed(PROTOCOL['seed'])
    torch.cuda.reset_peak_memory_stats()
    initial_counts = graph_counts()
    info = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                uuid=os.environ['CUDA_VISIBLE_DEVICES'], claim_indices=os.environ['GPU_CLAIM_INDICES'],
                threads=torch.get_num_threads(), precision=torch.get_float32_matmul_precision(),
                allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                compile_cache=os.environ.get('TORCHINDUCTOR_CACHE_DIR'),
                environment={k: os.environ.get(k) for k in ('CUDA_HOME', 'CUDA_VERSION', 'OMP_NUM_THREADS',
                    'MKL_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS', 'PYTHONDONTWRITEBYTECODE')},
                cpu_affinity=sorted(os.sched_getaffinity(0)), protocol_sha256=manifest['protocol_sha256'])
    write(bundle / 'actual_environment.json', info)
    def guard():
        assert time.monotonic() - started < PROTOCOL['internal_seconds'], 'internal time budget'
        assert torch.cuda.max_memory_allocated() < PROTOCOL['allocated_bytes'], 'allocated memory budget'
        assert sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file()) < PROTOCOL['evidence_bytes'], 'evidence budget'
        cache = Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])
        assert sum(p.stat().st_size for p in cache.rglob('*') if p.is_file()) < PROTOCOL['compiler_cache_limit_bytes'], 'compiler cache budget'
    def save():
        write(bundle / 'results.json', dict(status='running', rows=rows, setup=setup,
            all_in_seconds=time.monotonic() - started, environment=info,
            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    for dimension in PROTOCOL['dimensions']:
        generation_started = time.monotonic()
        with highest():
            q0 = torch.linalg.qr(torch.randn(dimension, dimension, device='cuda', generator=generator)).Q.contiguous()
            h = torch.randn(dimension, PROTOCOL['rank'], device='cuda', generator=generator) / PROTOCOL['rank']**.5
            noise = h @ h.T
            old_m = torch.randn(dimension, 16, device='cuda', generator=generator) / dimension**.5
            world_m = q0 @ old_m
        functions = dict(eager_fp32=functools.partial(warm_kernel, mixed=False),
            compiled_fp32=torch.compile(functools.partial(warm_kernel, mixed=False), mode='default', fullgraph=True, dynamic=False),
            compiled_mixed=torch.compile(functools.partial(warm_kernel, mixed=True), mode='default', fullgraph=True, dynamic=False))
        for spectrum in PROTOCOL['spectra']:
            guard()
            with highest():
                d = torch.linspace(.5, 2., dimension, device='cuda') if spectrum == 'broad' else torch.linspace(1., 1.001, dimension, device='cuda')
                base = (q0 * d) @ q0.T
                # One explicit FP32 beta=.99 covariance update; no EMA/storage quantization.
                cov = .99 * base + .01 * (base + PROTOCOL['perturbation'] * noise)
                cov = .5 * (cov + cov.T)
            del base
            hashes_before = dict(cov=tensor_hash(cov), q=tensor_hash(q0))
            print('INPUT ' + json.dumps(dict(dimension=dimension, spectrum=spectrum, hashes=hashes_before)), flush=True)
            reference = warm_kernel(cov, q0)
            with highest(): oracle = warm_kernel(cov, q0)
            oracle_difference = metrics(reference, oracle, q0, cov, world_m)
            preflight = {}
            for name, fn in functions.items():
                torch.cuda.synchronize(); began = time.monotonic()
                before = graph_counts()
                for _ in range(PROTOCOL['first_setup_calls']): actual = fn(cov, q0)
                torch.cuda.synchronize()
                setup_row = dict(dimension=dimension, spectrum=spectrum, variant=name,
                    first_two_calls_seconds=time.monotonic() - began, counts_before=before, counts_after=graph_counts())
                eager_same = warm_kernel(cov, q0, mixed=name.endswith('mixed'))
                setup_row['compiled_vs_own_eager_relative'] = relative(actual, eager_same)
                setup_row['compiled_vs_own_eager_pass'] = setup_row['compiled_vs_own_eager_relative'] <= 1e-4
                setup.append(setup_row)
                preflight[name] = metrics(actual, reference, q0, cov, world_m)
                preflight[name]['gate'] = classify(preflight[name], name.endswith('mixed'))
                for _ in range(PROTOCOL['warmup_calls']): actual = fn(cov, q0)
                del actual, eager_same
                guard()
            torch.cuda.synchronize()
            counts_before_steady = graph_counts()
            samples = {name: [] for name in functions}
            order_rng = random.Random(PROTOCOL['seed'] + dimension + (spectrum == 'clustered'))
            order_log = []
            for repeat in range(PROTOCOL['samples']):
                order = list(functions); order_rng.shuffle(order); order_log.append(order)
                for name in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(); began = time.perf_counter(); start.record()
                    actual = functions[name](cov, q0)
                    end.record(); end.synchronize()
                    samples[name].append(dict(device_ms=start.elapsed_time(end), wall_ms=1000 * (time.perf_counter() - began)))
                    del actual
                guard()
            counts_after_steady = graph_counts()
            assert counts_before_steady == counts_after_steady, 'compilation or graph break during steady samples'
            recurrent = {name: [] for name in functions}
            reference_q = q0
            qs = dict.fromkeys(functions, q0)
            for step in range(1, PROTOCOL['recurrent_updates'] + 1):
                reference_q = warm_kernel(cov, reference_q)
                for name in functions: qs[name] = functions[name](cov, qs[name])
                if step in PROTOCOL['recurrent_observe']:
                    for name in functions:
                        m = metrics(qs[name], reference_q, q0, cov, world_m)
                        m.update(step=step, gate=classify(m, name.endswith('mixed')))
                        recurrent[name].append(m)
                guard()
            hashes_after = dict(cov=tensor_hash(cov), q=tensor_hash(q0))
            assert hashes_after == hashes_before, 'common inputs mutated'
            row = dict(dimension=dimension, spectrum=spectrum, hashes=hashes_before,
                preflight=preflight, high_vs_highest_oracle=oracle_difference, recurrent=recurrent,
                common_inputs_unchanged=True, samples=samples, timing_order=order_log,
                steady_counts_before=counts_before_steady, steady_counts_after=counts_after_steady,
                medians={name: {k: statistics.median(s[k] for s in values) for k in ('device_ms', 'wall_ms')}
                    for name, values in samples.items()},
                eligible={name: preflight[name]['gate']['passed'] and all(m['gate']['passed'] for m in recurrent[name])
                    and next(s for s in setup if s['dimension']==dimension and s['spectrum']==spectrum and s['variant']==name)['compiled_vs_own_eager_pass']
                    for name in functions})
            rows.append(row)
            print('CASE ' + json.dumps({k: row[k] for k in ('dimension', 'spectrum', 'medians', 'eligible')}), flush=True)
            save()
            del cov, reference, oracle, qs, reference_q, d
        del functions, q0, h, noise, old_m, world_m
        torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
        setup.append(dict(dimension=dimension, generation_and_two_spectra_all_in_seconds=time.monotonic()-generation_started))
    guard()
    verify(bundle)
    summary = dict(status='completed', rows=rows, setup=setup, environment=info,
        all_in_seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        compiler_counts_initial=initial_counts, compiler_counts_final=graph_counts(),
        all_common_inputs_unchanged=True, no_steady_recompilation=True,
        scope=PROTOCOL['scope'], timing_qualification=False, learning_qualification=False)
    write(bundle / 'results.json', summary)
    print('COMPLETE ' + json.dumps({k: summary[k] for k in ('all_in_seconds', 'peak_allocated_bytes')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--bundle', type=Path, required=True)
    bundle = parser.parse_args().bundle
    assert not (bundle / 'execution_status.json').exists(), 'spent bundle: no retry'
    write(bundle / 'execution_status.json', dict(status='running', pid=os.getpid(), utc=time.time()))
    try:
        run(bundle)
    except BaseException as exc:
        write(bundle / 'execution_status.json', dict(status='failed', error=repr(exc), traceback=traceback.format_exc()))
        raise
    else:
        write(bundle / 'execution_status.json', dict(status='completed', utc=time.time()))
