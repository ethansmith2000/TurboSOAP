"""One bounded CUDA qualification/cost diagnostic; synthetic histories, no LM."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import weakref

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE/'cifar_support')]
import torch
from fast_soap import FastSOAP, fast_warm_refresh
from soap_reference import ResearchSOAP
from hybrid_training_kernel import configure_hybrid, precision
from movement_retraction import configure_matrix
from native_residual_named_kernel import PolicyKernel
from tensor_ownership import TensorOwnership

SHAPES = ((2304, 768), (768, 768), (3072, 768), (768, 1536))
DIMENSIONS = (768, 1536, 2304, 3072)
ARMS = ('qr20', 'hold200', 'j1', 'fast')


def kind(arm, age):
    if arm not in ARMS:
        raise ValueError('Undeclared diagnostic arm')
    if age % (20 if arm == 'qr20' else 200) == 0:
        return 'qr'
    if arm in ('j1', 'fast') and age % 20 == 0:
        return 'warm'
    return 'ordinary'


def weighted(arm, means):
    if arm == 'qr20':
        return .95*means['ordinary'] + .05*means['qr']
    if arm == 'hold200':
        return .995*means['ordinary'] + .005*means['qr']
    return .95*means['ordinary'] + .045*means['warm'] + .005*means['qr']


def config():
    return dict(trial=197, seed=196, shapes=[list(s) for s in SHAPES],
                dimensions=list(DIMENSIONS), arms=list(ARMS), blocks=3, calls=425,
                factor_steps=10, factor_timing_repeats=10, factor_timing_blocks=3,
                wall_seconds=1140, external_seconds=1200, gpu_bytes=24*2**30,
                rss_bytes=12*2**30, cache_bytes=512*2**20, evidence_bytes=100*2**20,
                minimum_free_disk_bytes=5*2**30, absolute_gram_max=.05,
                normalized_gram_max=.005, physical_m_relative_max=.01,
                scope='Synthetic PSD factors and one real-shaped four-matrix block; no model, data or learning claim',
                artifacts='Compact JSON/logs/source/code-cache only; no weights/checkpoints/tensors/traces',
                tensor_purpose='Transient C/Q/m/v/parameter/gradient copies for geometry, physical-M and cost; RAM/GPU only, released after each block')


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify(bundle):
    manifest = json.loads((bundle/'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen input changed: '+name)
    if json.loads((bundle/'config.json').read_text()) != config():
        raise ValueError('Undeclared diagnostic recipe change')


class Run:
    def __init__(self, bundle):
        self.bundle = bundle
        self.started = time.monotonic()
        self.factor_rows = []
        self.factor_times = []
        self.blocks = []
        self.checks = []

    def emit(self, event, **row):
        record = dict(event=event, elapsed_seconds=time.monotonic()-self.started, **row)
        with (self.bundle/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        if event != 'update':
            print(json.dumps(record, allow_nan=False), flush=True)

    def budget(self):
        import psutil
        import shutil
        if time.monotonic()-self.started > config()['wall_seconds']:
            raise TimeoutError('Internal diagnostic wall limit')
        if torch.cuda.max_memory_allocated() > config()['gpu_bytes']:
            raise MemoryError('GPU allocation limit')
        if psutil.Process().memory_info().rss > config()['rss_bytes']:
            raise MemoryError('Process RSS limit')
        if shutil.disk_usage(self.bundle).free < config()['minimum_free_disk_bytes']:
            raise OSError('Shared free disk limit')
        cache = sum(p.stat().st_size for p in (self.bundle/'compile_cache').rglob('*') if p.is_file())
        compact = sum(p.stat().st_size for p in self.bundle.rglob('*') if p.is_file() and 'compile_cache' not in p.parts)
        if cache > config()['cache_bytes'] or compact > config()['evidence_bytes']:
            raise OSError('Declared code-cache/evidence limit')

    def guard(self, q):
        with precision('highest'):
            gram = q.T @ q
            gram.diagonal().sub_(1.)
            absolute = float(gram.norm())
        normalized = absolute/math.sqrt(len(q))
        row = dict(absolute_gram=absolute, normalized_gram=normalized,
                   passed=math.isfinite(absolute) and absolute <= .05 and normalized <= .005)
        return row

    def factor(self, dimension):
        generator = torch.Generator(device='cuda').manual_seed(196+dimension)
        x = torch.randn(dimension, 32, device='cuda', generator=generator)
        covariance = torch.diag(torch.linspace(.5, 1.5, dimension, device='cuda')) + .02*(x@x.T)/32
        q = torch.eye(dimension, device='cuda')
        owner = TensorOwnership()
        owner.watch(covariance, 'factor.covariance')
        owner.watch(q, 'factor.initial_basis')
        del x
        # Both input basis layouts occur before/after scheduled QR repairs.
        for arm, kernel in [('j1', self.j1.evaluate), ('fast', self.fast)]:
            start = time.monotonic()
            for basis in (q, q.T):
                output = kernel(covariance, basis)
                del output
            torch.cuda.synchronize()
            self.emit('compile_warmup', arm=arm, dimension=dimension,
                      seconds=time.monotonic()-start, row_and_column_layouts=True)
        for arm, kernel in [('j1', self.j1.evaluate), ('fast', self.fast)]:
            current = q
            for step in range(1, 11):
                rng = torch.cuda.get_rng_state().clone()
                versions = (covariance._version, current._version)
                output = kernel(covariance, current)
                owner.watch(output, 'factor.output')
                row = dict(arm=arm, dimension=dimension, step=step, **self.guard(output))
                if arm == 'fast':
                    reference = fast_warm_refresh(covariance, current)
                    with precision('highest'):
                        row['own_eager_basis_relative'] = float((output-reference).norm()/reference.norm())
                        target = reference-current
                        row['own_eager_movement_relative'] = float(((output-current)-target).norm()/target.norm().clamp_min(1e-12))
                    del reference, target
                row['source_versions_unchanged'] = versions == (covariance._version, current._version)
                row['cuda_rng_unchanged'] = torch.equal(rng, torch.cuda.get_rng_state())
                self.factor_rows.append(row)
                self.emit('factor_guard_before_gate', **row)
                if not row['passed'] or not row['source_versions_unchanged'] or not row['cuda_rng_unchanged']:
                    raise FloatingPointError('Factor geometry/source guard failed: '+str(row))
                current = output
                self.budget()
            del current, output
        for block in range(3):
            for arm in (('fast', 'j1') if block % 2 == 0 else ('j1', 'fast')):
                kernel = self.fast if arm == 'fast' else self.j1.evaluate
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                begin.record()
                for _ in range(10):
                    output = kernel(covariance, q)
                end.record()
                torch.cuda.synchronize()
                row = dict(block=block, arm=arm, dimension=dimension,
                           host_ms=(time.perf_counter()-start)*1000/10,
                           event_ms=begin.elapsed_time(end)/10)
                self.factor_times.append(row)
                self.emit('factor_timing', **row)
                del output
        del q, covariance, basis
        gc.collect()
        release = owner.report()
        owner.close()
        self.emit('factor_release', dimension=dimension, **release)
        if not release['all_registered_released']:
            raise RuntimeError('Registered factor tensors retained')
        self.budget()

    def optimizer(self, parameters, arm):
        common = dict(lr=.0005, shampoo_beta=.99, weight_decay=.05)
        if arm == 'fast':
            return FastSOAP(parameters, **common)
        opt = ResearchSOAP(parameters, basis_method='warm' if arm == 'j1' else 'qr',
                           precondition_frequency=200 if arm == 'hold200' else 20,
                           hard_reset_interval=200 if arm == 'j1' else 0,
                           transport_precision='highest', **common)
        if arm == 'j1':
            configure_matrix(opt, 'fro_cubic2_cap05')
            configure_hybrid(opt, self.j1)
        return opt

    def block(self, arm, block):
        generator = torch.Generator(device='cuda').manual_seed(196)
        parameters = [torch.nn.Parameter(.01*torch.randn(*shape, device='cuda', generator=generator))
                      for shape in SHAPES]
        opt = self.optimizer(parameters, arm)
        group = opt.param_groups[0]
        method = type(opt)._research_refresh
        optimizer_ref = weakref.ref(opt)

        def refresh(state, active_group):
            obj = optimizer_ref()
            if state['step'] not in (20, 200):
                return method(obj, state, active_group)
            rng = torch.cuda.get_rng_state().clone()
            with precision('highest'):
                world = obj._project_back_with_basis(state['exp_avg'], state['Q'], active_group)
            method(obj, state, active_group)
            with precision('highest'):
                after = obj._project_back_with_basis(state['exp_avg'], state['Q'], active_group)
                error = float((after-world).norm()/world.norm().clamp_min(1e-12))
            row = dict(arm=arm, block=block, age=state['step'], shape=list(world.shape),
                       physical_m_relative=error, rng_unchanged=torch.equal(rng, torch.cuda.get_rng_state()),
                       passed=math.isfinite(error) and error <= .01)
            self.checks.append(row)
            self.emit('actual_m_before_gate', **row)
            if not row['passed'] or not row['rng_unchanged']:
                raise FloatingPointError('Actual physical-M preservation failed')

        opt._research_refresh = refresh
        # Deterministic common synthetic state; no cold-start learning claim.
        for p in parameters:
            p.grad = .01*torch.randn(p.shape, device='cuda', generator=generator)
            state = opt.state[p]
            opt._init_state(p.grad, state, group, p)
            state['GG'] = [torch.diag(torch.linspace(.5, 1.5, size, device='cuda')) for size in p.shape]
            state['Q'] = [torch.eye(size, device='cuda') for size in p.shape]
        samples = []
        for age in range(1, 426):
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            begin.record()
            opt.step()
            end.record()
            torch.cuda.synchronize()
            row = dict(arm=arm, block=block, age=age, kind=kind(arm, age),
                       diagnostic=age in (20, 200) and kind(arm, age) != 'ordinary',
                       host_ms=(time.perf_counter()-start)*1000,
                       event_ms=begin.elapsed_time(end))
            samples.append(row)
            self.emit('update', **row)
            if age % 20 == 0:
                for p in parameters:
                    state = opt.state[p]
                    for q in state['Q']:
                        metric = self.guard(q)
                        self.emit('block_geometry_before_gate', arm=arm, block=block, age=age, **metric)
                        if not metric['passed']:
                            raise FloatingPointError('Own-history block geometry failed')
                    if not all(bool(torch.isfinite(t).all()) for t in
                               [p, state['exp_avg'], state['exp_avg_sq'], *state['GG']]):
                        raise FloatingPointError('Nonfinite block state')
                self.budget()
        clean = [r for r in samples if r['age'] > 25 and not r['diagnostic']]
        means = {event: statistics.mean(r['host_ms'] for r in clean if r['kind'] == event)
                 for event in set(r['kind'] for r in clean)}
        expected_qr = 21 if arm == 'qr20' else 2
        expected_warm = 19 if arm in ('j1', 'fast') else 0
        if any(s.get('research_qr_refreshes', 0) != expected_qr or
               s.get('research_warm_refreshes', 0) != expected_warm for s in opt.state.values()):
            raise RuntimeError('Own-history refresh counts differ')
        row = dict(arm=arm, block=block, clean_means_host_ms=means,
                   weighted_optimizer_ms=weighted(arm, means), calls=425,
                   qr_per_matrix=expected_qr, warm_per_matrix=expected_warm)
        self.blocks.append(row)
        self.emit('block_complete', **row)
        owner = TensorOwnership()
        owner.watch_tree(opt.state_dict(), 'block.optimizer')
        owner.watch_tree(parameters, 'block.parameters')
        owner.watch_tree([p.grad for p in parameters], 'block.gradients')
        del opt, parameters, p, state, q, group
        gc.collect()
        release = owner.report()
        owner.close()
        self.emit('block_release', arm=arm, block=block, **release)
        if not release['all_registered_released']:
            raise RuntimeError('Registered block tensors retained')
        self.budget()

    @torch.no_grad()
    def run(self):
        verify(self.bundle)
        if os.environ.get('GPU_CLAIM_INDICES') != '7' or torch.cuda.device_count() != 1:
            raise RuntimeError('One lifetime-claimed physical GPU7 required')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != 'GPU-866cad81-dc9b-cc40-afb8-c387d3ffba5f':
            raise RuntimeError('Unexpected GPU UUID')
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.cuda.set_per_process_memory_fraction(config()['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        router = SimpleNamespace(calls=Counter(), probe=False)
        self.j1 = PolicyKernel('highest_residual20', router)
        placeholder = torch.nn.Parameter(torch.zeros(1, 1, device='cuda'))
        self.fast = FastSOAP([placeholder])._warm_kernel
        self.emit('started', torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), config=config())
        for dimension in DIMENSIONS:
            self.factor(dimension)
        for block in range(3):
            order = ARMS[block:]+ARMS[:block]
            for arm in order:
                self.block(arm, block)
        verify(self.bundle)
        if len(self.checks) != 84:
            raise RuntimeError('Actual-M coverage differs')
        summary = dict(status='completed', factor_guards=self.factor_rows,
                       actual_m_checks=self.checks,
                       factor_timings=self.factor_times, blocks=self.blocks,
                       elapsed_seconds=time.monotonic()-self.started,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       scope=config()['scope'], no_training_or_loss_claim=True)
        write(self.bundle/'summary.json', summary)
        self.emit('finished', elapsed=summary['elapsed_seconds'])
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    if HERE.parent != args.bundle.resolve()/'source':
        raise RuntimeError('Frozen entry only')
    if (args.bundle/'execution_status.json').exists():
        raise RuntimeError('One attempt only; never relaunch')
    write(args.bundle/'execution_status.json', dict(status='running', pid=os.getpid(), trial=197))
    run = Run(args.bundle)
    try:
        run.run()
        write(args.bundle/'execution_status.json', dict(status='completed', pid=os.getpid(), trial=197))
    except BaseException as exc:
        import traceback
        write(args.bundle/'execution_status.json', dict(status='failed', pid=os.getpid(), trial=197,
              error=repr(exc), traceback=traceback.format_exc()))
        raise
