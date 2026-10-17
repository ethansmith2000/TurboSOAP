"""Prospective common-Y residual retraction screen; no optimizer/default edits."""
import argparse
from collections import Counter
import functools
import gc
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
import weakref

STARTED = time.monotonic()
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import torch
from common_y_kernel import warm_prefix, retract_two
from residual_retraction import residual_retract_two
from warm_compile_fixture import highest, metrics, relative, tensor_hash, graph_counts, sha
from warm_compile_kernel import LIMITS, classify

PROTOCOL = dict(
    seed=147, dimensions=[2048, 6144, 8192], spectra=['broad', 'clustered'],
    layouts=['row', 'column'], atom_column_scale_defects=[0., .0007], perturbation_relative=1e-6,
    beta=.99, covariance_noise_rank=32, covariance_noise_weight=.02,
    prefix='unchanged high FP32 covariance products/generator/cap; BF16 Q@A only',
    gram_precision='highest', stored_dtype='float32', correction_variants=['old', 'residual_fp32', 'residual_bf16'],
    implementations=['eager', 'compiled'], compile_mode='default', fullgraph=True, dynamic=False,
    first_setup_calls=2, warmup_calls=3, timing_samples=7, recurrent_updates=12,
    recurrent_observe=[1, 4, 12], limits=LIMITS, common_strict_basis_limit=1e-4,
    own_eager_basis_limit=1e-4, observer_endpoint_limit=1e-6,
    reference='same-Y old highest eager atom; whole current compiled BF16-candidate/highest-old-retraction bundle',
    perturbation_policy='same deterministic skinny perturbation direction and relative norm for all atoms; old eager reference at each perturbed Y',
    physical_m_policy='independent highest-FP32 skinny world/projected reconstruction, not full moment/V history',
    internal_seconds=1800, external_seconds=1860, allocated_bytes=24*2**30,
    evidence_bytes=40*2**20, expected_evidence_bytes=4*2**20,
    compiler_cache_limit_bytes=512*2**20, expected_compiler_cache_bytes=100*2**20,
    whole_call_limit=2400, prefix_call_limit=3000, retraction_call_limit=4000, metric_call_limit=1000,
    bf16_reduced_precision_reduction=False,
    artifacts='RAM-only Q/C/Y/skinny M; compact configs/seeds/source/hash/metrics/logs; no tensors/weights/checkpoints/traces',
    retention='retain compact evidence and isolated generated-code cache through next arithmetic review; no model recovery artifacts',
    scope='synthetic common-input atom and complete warm basis update; no full SOAP/QR equivalence/training/learning qualification')


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify(bundle):
    manifest = json.loads((bundle/'manifest.json').read_text())
    assert json.loads((bundle/'protocol.json').read_text()) == PROTOCOL
    assert sha(bundle/'protocol.json') == manifest['protocol_sha256']
    for name, digest in manifest['source_sha256'].items():
        assert sha(bundle/'source'/name) == digest, name
    for name, digest in manifest['prepared_sha256'].items():
        assert sha(bundle/name) == digest, name
    return manifest


def observer_prefix(cov, q):
    y = warm_prefix(cov, q, cov_mixed=False, candidate_mixed=True)
    return y, y.norm()


def make_functions(counts):
    def counted(fn, kind):
        def call(*args):
            counts[kind] += 1
            return fn(*args)
        return call
    def compiled(fn):
        return torch.compile(fn, mode='default', fullgraph=True, dynamic=False)
    prefix = functools.partial(warm_prefix, cov_mixed=False, candidate_mixed=True)
    prefixes = {impl: counted(compiled(prefix) if impl == 'compiled' else prefix, 'prefix')
                for impl in PROTOCOL['implementations']}
    atoms = dict(old=retract_two,
                 residual_fp32=functools.partial(residual_retract_two, correction_mixed=False),
                 residual_bf16=functools.partial(residual_retract_two, correction_mixed=True))
    rets = {impl+'_'+name: counted(compiled(fn) if impl == 'compiled' else fn, 'retraction')
            for impl in PROTOCOL['implementations'] for name, fn in atoms.items()}
    def pipeline(prefix_fn, ret_fn):
        def call(cov, q):
            torch.set_float32_matmul_precision('high')
            y = prefix_fn(cov, q)
            with highest():
                return ret_fn(y)
        return counted(call, 'whole')
    whole = {name: pipeline(prefixes[name.split('_')[0]], fn) for name, fn in rets.items()}
    return prefixes, rets, whole, compiled(observer_prefix)


def column_layout(x):
    return x.T.contiguous().T


def checked_metrics(actual, reference, initial, cov, world_m, counts):
    counts['metric'] += 1
    result = metrics(actual, reference, initial, cov, world_m)
    result['mixed_gate'] = classify(result, True)
    result['strict_basis_pass'] = result['basis_relative'] <= PROTOCOL['common_strict_basis_limit']
    return result


def timed(fn, args):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    began = time.perf_counter()
    start.record()
    output = fn(*args)
    end.record()
    end.synchronize()
    return output, dict(device_ms=start.elapsed_time(end), wall_ms=1000*(time.perf_counter()-began))


def atom_call(fn, y):
    with highest():
        return fn(y)


@torch.no_grad()
def run_dimension(n, generator, counts, guard, owners):
    # Each registered tensor owner is confined to this function. No tensor is serialized.
    def own(x):
        owners.append(weakref.ref(x))
        return x
    with highest():
        qbase = own(torch.linalg.qr(torch.randn(n, n, device='cuda', generator=generator)).Q.contiguous())
        counts['fixture_qr'] += 1
        h = torch.randn(n, 32, device='cuda', generator=generator)/32**.5
        noise = h@h.T
        world_m = own(qbase@(torch.randn(n, 16, device='cuda', generator=generator)/n**.5))
        perturbation = h@torch.randn(32, n, device='cuda', generator=generator)
        perturbation /= perturbation.norm()
    prefixes, rets, whole, observer = make_functions(counts)
    rows = []
    for spectrum in PROTOCOL['spectra']:
        with highest():
            d = torch.linspace(.5, 2., n, device='cuda') if spectrum == 'broad' else torch.linspace(1., 1.001, n, device='cuda')
            base = (qbase*d)@qbase.T
            cov = base+.02*noise
            cov = own(.5*(cov+cov.T))
        for layout in PROTOCOL['layouts']:
            q = own(qbase.clone() if layout == 'row' else column_layout(qbase))
            input_hashes = dict(q=tensor_hash(q), cov=tensor_hash(cov))
            setup = []
            for name, fn in whole.items():
                before = graph_counts()
                torch.cuda.synchronize(); began = time.monotonic()
                for _ in range(2): output = fn(cov, q)
                torch.cuda.synchronize()
                setup.append(dict(variant=name, first_two_seconds=time.monotonic()-began,
                                  compiler_before=before, compiler_after=graph_counts()))
                for _ in range(3): output = fn(cov, q)
                del output
                guard()
            torch.set_float32_matmul_precision('high')
            y = own(prefixes['compiled'](cov, q))
            observed_y, observed_norm = observer(cov, q)
            with highest(): observer_error = relative(observed_y, y)
            assert observer_error <= PROTOCOL['observer_endpoint_limit'], 'observer changed candidate'
            del observed_y, observed_norm
            atom_rows = []
            for defect in PROTOCOL['atom_column_scale_defects']:
                with highest():
                    candidate = y*(1+defect*torch.linspace(-1, 1, n, device='cuda'))
                candidate = own(candidate.contiguous() if layout == 'row' else column_layout(candidate))
                candidate_hash = tensor_hash(candidate)
                # Compile column-layout retractions prospectively, before their timing windows.
                for name, fn in rets.items():
                    torch.cuda.synchronize(); began = time.monotonic(); before = graph_counts()
                    for _ in range(2): output = atom_call(fn, candidate)
                    torch.cuda.synchronize()
                    setup.append(dict(variant=name, atom_defect=defect, first_two_seconds=time.monotonic()-began,
                                      compiler_before=before, compiler_after=graph_counts()))
                    for _ in range(3): output = atom_call(fn, candidate)
                    del output
                reference = own(atom_call(rets['eager_old'], candidate))
                fidelity, sensitivity = {}, {}
                with highest():
                    perturbed = own(candidate+perturbation*(candidate.norm()*1e-6))
                    input_distance = float((perturbed-candidate).norm())
                perturb_reference = own(atom_call(rets['eager_old'], perturbed))
                for name, fn in rets.items():
                    actual = atom_call(fn, candidate)
                    own_eager = atom_call(rets['eager_'+name.split('_', 1)[1]], candidate)
                    mm = checked_metrics(actual, reference, q, cov, world_m, counts)
                    with highest(): mm['own_eager_relative'] = relative(actual, own_eager)
                    mm['own_eager_pass'] = mm['own_eager_relative'] <= PROTOCOL['own_eager_basis_limit']
                    fidelity[name] = mm
                    pactual = atom_call(fn, perturbed)
                    pmm = checked_metrics(pactual, perturb_reference, q, cov, world_m, counts)
                    with highest(): output_distance = float((pactual-actual).norm())
                    sensitivity[name] = dict(input_distance=input_distance, output_distance=output_distance,
                                             amplification=output_distance/input_distance, perturbed_fidelity=pmm)
                    del actual, own_eager, pactual
                assert candidate_hash == tensor_hash(candidate), 'common Y mutated'
                before = graph_counts(); samples = {name: [] for name in rets}; orders = []
                order_rng = random.Random(PROTOCOL['seed']+n+int(defect*1e6)+(layout == 'column'))
                for _ in range(7):
                    order = list(rets); order_rng.shuffle(order); orders.append(order)
                    for name in order:
                        output, sample = timed(functools.partial(atom_call, rets[name]), (candidate,))
                        samples[name].append(sample); del output
                    guard()
                assert before == graph_counts(), 'steady atom compilation/graphbreak'
                atom_rows.append(dict(defect=defect, candidate_stride=list(candidate.stride()), candidate_hash=candidate_hash,
                                      candidate_unchanged=True, fidelity=fidelity, sensitivity=sensitivity,
                                      samples=samples, timing_order=orders, steady_compiler_before=before,
                                      steady_compiler_after=graph_counts()))
                del candidate, reference, perturbed, perturb_reference
            before = graph_counts(); samples = {name: [] for name in whole}; orders = []
            order_rng = random.Random(PROTOCOL['seed']+n+(spectrum == 'clustered')+(layout == 'column'))
            for _ in range(7):
                order = list(whole); order_rng.shuffle(order); orders.append(order)
                for name in order:
                    output, sample = timed(whole[name], (cov, q))
                    samples[name].append(sample); del output
                guard()
            assert before == graph_counts(), 'steady whole compilation/graphbreak'
            qs = dict.fromkeys(whole, q); recurrent = {name: [] for name in whole}
            before_recurrent = graph_counts()
            for step in range(1, 13):
                qs = {name: own(whole[name](cov, value)) for name, value in qs.items()}
                if step in PROTOCOL['recurrent_observe']:
                    for name, actual in qs.items():
                        mm = checked_metrics(actual, qs['compiled_old'], q, cov, world_m, counts)
                        with highest(): mm['own_eager_relative'] = relative(actual, qs['eager_'+name.split('_', 1)[1]])
                        mm.update(step=step, own_eager_pass=mm['own_eager_relative'] <= PROTOCOL['own_eager_basis_limit'])
                        recurrent[name].append(mm)
                    del actual
                guard()
            # A column-layout initial Q legitimately produces row-layout next Q: row kernels were set up first.
            assert before_recurrent == graph_counts(), 'recurrent compilation/graphbreak'
            assert input_hashes == dict(q=tensor_hash(q), cov=tensor_hash(cov)), 'common Q/C mutated'
            row = dict(dimension=n, spectrum=spectrum, layout=layout, q_stride=list(q.stride()),
                       input_hashes=input_hashes, inputs_unchanged=True, observer_relative=observer_error,
                       setup=setup, atoms=atom_rows, recurrent=recurrent, samples=samples, timing_order=orders,
                       steady_compiler_before=before, steady_compiler_after=before_recurrent,
                       recurrent_compiler_before=before_recurrent, recurrent_compiler_after=graph_counts())
            row['medians'] = {name: {key: statistics.median(s[key] for s in values)
                                    for key in ('device_ms', 'wall_ms')} for name, values in samples.items()}
            row['eligible'] = {name: all(x['mixed_gate']['passed'] and x['own_eager_pass'] for x in recurrent[name])
                               and all(a['fidelity'][name]['mixed_gate']['passed'] and a['fidelity'][name]['own_eager_pass']
                                       and a['sensitivity'][name]['perturbed_fidelity']['mixed_gate']['passed'] for a in atom_rows)
                               for name in whole}
            rows.append(row)
            print('CASE '+json.dumps({key: row[key] for key in ('dimension', 'spectrum', 'layout', 'medians', 'eligible')}), flush=True)
            del qs, y, q
        del cov, base, d
    return rows


@torch.no_grad()
def run(bundle):
    manifest = verify(bundle)
    launch = json.loads((bundle/'launch_manifest.json').read_text())
    assert os.environ.get('GPU_CLAIM_INDICES') == str(launch['gpu']['index'])
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == launch['gpu']['uuid']
    assert torch.cuda.device_count() == 1
    prefixes = ('TORCH', 'TRITON', 'CUDA', 'CUBLAS', 'CUDNN', 'NVIDIA_TF32', 'OMP', 'MKL', 'PYTHONDONTWRITEBYTECODE')
    observed = {k: v for k, v in os.environ.items() if k.startswith(prefixes)}
    assert observed == dict(launch['environment'], CUDA_VISIBLE_DEVICES=launch['gpu']['uuid']), observed
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    assert not torch.is_autocast_enabled()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device='cuda').manual_seed(PROTOCOL['seed'])
    counts, rows, releases = Counter(), [], []
    info = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                uuid=os.environ['CUDA_VISIBLE_DEVICES'], claim_indices=os.environ['GPU_CLAIM_INDICES'],
                environment=observed, affinity=sorted(os.sched_getaffinity(0)), protocol_sha256=manifest['protocol_sha256'],
                bf16_reduced_precision_reduction=False, pid=os.getpid())
    assert info['torch'] == '2.11.0+cu128' and info['cuda'] == '12.8'
    write(bundle/'actual_environment.json', info)
    initial = graph_counts()
    def guard():
        assert time.monotonic()-STARTED < PROTOCOL['internal_seconds'], 'runtime bound'
        assert torch.cuda.max_memory_allocated() < PROTOCOL['allocated_bytes'], 'GPU memory bound'
        for name in ('whole', 'prefix', 'retraction', 'metric'):
            assert counts[name] <= PROTOCOL[name+'_call_limit'], name+' call bound'
        cache = bundle/'compile_cache'
        assert sum(p.stat().st_size for p in cache.rglob('*') if p.is_file()) < PROTOCOL['compiler_cache_limit_bytes']
        assert sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file() and cache not in p.parents) < PROTOCOL['evidence_bytes']
    def save(status):
        write(bundle/'results.json', dict(status=status, rows=rows, counts=dict(counts), releases=releases,
              environment=info, all_in_seconds=time.monotonic()-STARTED, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              compiler_initial=initial, compiler_final=graph_counts(), scope=PROTOCOL['scope']))
    save('running')
    for n in PROTOCOL['dimensions']:
        owners = []
        rows.extend(run_dimension(n, generator, counts, guard, owners))
        torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
        alive = sum(ref() is not None for ref in owners)
        assert alive == 0, 'registered tensor owner survived dimension release'
        releases.append(dict(dimension=n, registered_owners=len(owners), alive_after_release=alive,
                             allocated_after_release=torch.cuda.memory_allocated()))
        guard(); save('running')
    verify(bundle); guard(); save('completed')
    print('COMPLETE '+json.dumps(dict(counts=dict(counts), seconds=time.monotonic()-STARTED,
                                     peak_allocated_bytes=torch.cuda.max_memory_allocated())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--bundle', type=Path, required=True)
    b = parser.parse_args().bundle
    assert Path(__file__).resolve() == (b/'source/benchmarks/residual_gpu_fixture.py').resolve()
    assert not (b/'execution_status.json').exists(), 'spent bundle, no retry'
    verify(b)
    write(b/'execution_status.json', dict(status='running', pid=os.getpid(), utc=time.time()))
    try:
        run(b)
    except BaseException as exc:
        write(b/'execution_status.json', dict(status='failed', pid=os.getpid(), error=repr(exc), traceback=traceback.format_exc()))
        raise
    else:
        write(b/'execution_status.json', dict(status='completed', pid=os.getpid(), utc=time.time()))
