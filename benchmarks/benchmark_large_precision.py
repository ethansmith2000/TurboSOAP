"""Natural-cycle, shallow large-factor precision screen; no saved tensors."""
import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soap import SOAP
from soap_reference import ResearchSOAP
from benchmarks.benchmark_synthetic_scaling import inspect_state, StateGuardError, tensor_bytes


def arms():
    base = dict(method='qr', frequency=10, precision='high', covariance_dtype='float32',
                ns_iterations=2, compile_gauge=False, inference=False, bf16_gauge=False)
    variants = [
        ('qr10_fp32', dict(precision='highest')),
        ('qr10_high', {}), ('qr20_high', dict(frequency=20)),
        ('qr10_bf16_cov', dict(covariance_dtype='bfloat16')),
        ('warm2_high', dict(method='warm')),
        ('warm6_fp32', dict(method='warm', ns_iterations=6, precision='highest')),
        ('warm6_high', dict(method='warm', ns_iterations=6)),
        ('warm6_bf16_cov', dict(method='warm', ns_iterations=6, covariance_dtype='bfloat16')),
        ('warm6_bf16_gauge', dict(method='warm', ns_iterations=6, bf16_gauge=True)),
        ('warm6_compile', dict(method='warm', ns_iterations=6, compile_gauge=True)),
        ('warm6_inference', dict(method='warm', ns_iterations=6, inference=True)),
    ]
    return [dict(base, **{'name': name, **changes}) for name, changes in variants]


def event_name(step, arm):
    if step == 0: return 'initialization'
    if step % arm['frequency']: return 'ordinary'
    if arm['method'] == 'qr': return 'qr_refresh'
    return 'qr_reset' if step % 200 == 0 else 'warm_refresh'


def warm_six(cov, q):
    # Compile exactly the existing tensor path, with explicit experimental NS6.
    return SOAP._gauge_step_one(cov, q, dict(basis_jacobi_damping=.01,
        basis_lr=.5, basis_rotation_cap=.1, basis_rotation_cap_mode='average',
        basis_stall_pair_threshold=0., basis_ns_iterations=6))


def state_tensors(value):
    if isinstance(value, torch.Tensor): yield value
    elif isinstance(value, dict):
        for item in value.values(): yield from state_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value: yield from state_tensors(item)


def summarize_case(result):
    if result['status'] != 'valid': return None
    samples = [s for s in result['samples'] if s['step'] > 0]
    return dict(event_counts=dict(Counter(s['event'] for s in samples)),
        mean_ms={key: statistics.mean(s[key] for s in samples) for key in
            ['model_cuda_ms', 'optimizer_cuda_ms', 'step_cuda_ms', 'step_wall_ms']},
        event_optimizer_ms={event: statistics.mean(s['optimizer_cuda_ms'] for s in samples if s['event'] == event)
                            for event in sorted(set(s['event'] for s in samples))})


def measure(width, arm, round_index, config, budget):
    torch.set_float32_matmul_precision(arm['precision'])
    torch.manual_seed(config['seed'] + width)
    model = nn.Linear(width, width, bias=False).cuda().train()
    generator = torch.Generator(device='cuda').manual_seed(config['seed'])
    batches = [(torch.randn(config['tokens'], width, device='cuda', generator=generator),
                torch.randn(config['tokens'], width, device='cuda', generator=generator)) for _ in range(4)]
    opt = ResearchSOAP(model.parameters(), lr=.0005, betas=(.9, .999), shampoo_beta=.999,
        precondition_mode='all', basis_method=arm['method'], precondition_frequency=arm['frequency'],
        covariance_compute_dtype=arm['covariance_dtype'], hard_reset_interval=200)
    opt.param_groups[0]['basis_ns_iterations'] = arm['ns_iterations']
    if arm['bf16_gauge']:
        def bf16_gauge(cov, q, group):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return SOAP._gauge_step_one(cov, q, group).float()
        opt._gauge_step_one = bf16_gauge

    def backward(index):
        opt.zero_grad(set_to_none=True)
        x, target = batches[index % 4]
        with torch.autocast('cuda', dtype=torch.bfloat16): output = F.gelu(model(x))
        loss = F.mse_loss(output.float(), target)
        loss.backward()
        return loss

    for index in range(3): backward(index)
    torch.cuda.synchronize()
    result = dict(width=width, round=round_index, arm=arm, status='running', samples=[], geometry=[],
        parameter_count=width**2, failure=None, compile_preflight=None,
        effective_optimizer_group={k: v for k, v in opt.param_groups[0].items() if k != 'params'})
    peak = 0
    for step in range(config['updates'] + 1):
        budget()
        torch.cuda.reset_peak_memory_stats()
        start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        torch.cuda.synchronize(); began = time.perf_counter(); start.record()
        loss = backward(step); middle.record()
        with torch.inference_mode() if arm['inference'] else nullcontext(): opt.step()
        end.record(); end.synchronize(); wall_ms = 1000 * (time.perf_counter() - began)
        peak = max(peak, torch.cuda.max_memory_allocated())
        event = event_name(step, arm)
        expected_refresh = int(event not in ('initialization', 'ordinary'))
        expected_reset = int(event in ('qr_refresh', 'qr_reset'))
        if (opt.last_basis_refreshes, opt.last_hard_reset_events) != (expected_refresh, expected_reset):
            raise RuntimeError('Unexpected optimizer event')
        value = float(loss.detach())
        result['samples'].append(dict(step=step, event=event, loss_sanity_only=value,
            model_cuda_ms=start.elapsed_time(middle), optimizer_cuda_ms=middle.elapsed_time(end),
            step_cuda_ms=start.elapsed_time(end), step_wall_ms=wall_ms))
        if not math.isfinite(value):
            result.update(status='guard_rejected', failure=dict(step=step, reason='nonfinite_loss')); break
        if step == 0 or expected_refresh or step == config['updates']:
            try: diagnostic = inspect_state(opt, 'all')
            except StateGuardError as error:
                result['geometry'].append(dict(step=step, **error.diagnostics))
                result.update(status='guard_rejected', failure=dict(step=step, reason='state_guard')); break
            result['geometry'].append(dict(step=step, **diagnostic))
        if step == 0 and arm['compile_gauge']:
            # Compile/validate on separate, noncommuting inputs. No mutation of the
            # executed trajectory and no compilation hidden in steady event costs.
            budget(); torch.cuda.synchronize(); began = time.perf_counter()
            compiled = torch.compile(warm_six, fullgraph=True, dynamic=False)
            state = next(iter(opt.state.values())); q = state['Q'][0]
            cov = state['GG'][0]
            probe = cov + torch.diag(torch.linspace(0, float(cov.diagonal().abs().max()), width, device='cuda'))
            with torch.no_grad():
                expected = warm_six(probe, q); actual = compiled(probe, q)
            torch.cuda.synchronize()
            relative_error = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
            result['compile_preflight'] = dict(wall_seconds=time.perf_counter() - began,
                relative_output_error=relative_error, fullgraph=True, backend='inductor', mode='default')
            if not math.isfinite(relative_error) or relative_error > .01:
                result.update(status='guard_rejected', failure=dict(step=step, reason='compiled_output_parity')); break
            opt._gauge_step_one = lambda cov, q, group: compiled(cov, q)
            del expected, actual, probe
    if result['status'] == 'running': result['status'] = 'valid'
    result.update(state_bytes=tensor_bytes(opt.state), peak_allocated_bytes=peak,
        inference_state_tensor_count=sum(torch.is_inference(t) for t in state_tensors(opt.state)),
        maximum_gram_error=max((g['maximum_gram_error'] for g in result['geometry']), default=None))
    if arm['inference']:
        # Compatibility probe outside all timing. Do not subsequently reuse this
        # state: a rejected in-place operation can partially change it.
        loss = backward(config['updates'] + 1)
        result['inference_exit_probe'] = dict(subsequent_backward_finite=bool(torch.isfinite(loss)))
        try:
            opt.step()
            result['inference_exit_probe']['ordinary_step_succeeded'] = True
        except RuntimeError as error:
            result['inference_exit_probe'].update(ordinary_step_succeeded=False, error=str(error))
    result['summary'] = summarize_case(result)
    del opt, model, batches
    gc.collect(); torch.cuda.empty_cache()
    return result


def prepare(bundle, widths, rounds):
    if bundle.exists(): raise ValueError('Use a new output directory')
    if not widths or min(widths) < 1 or max(widths) > 10000 or len(set(widths)) != len(widths):
        raise ValueError('Every square factor must fit the explicit 10000 cap')
    if rounds < 2: raise ValueError('At least two rounds')
    if shutil.disk_usage(bundle.parent).free < 5 * 2**30: raise RuntimeError('Require 5 GiB free')
    config = dict(widths=widths, tokens=64, rounds=rounds, updates=200, seed=20261002,
        arms=arms(), maximum_seconds=900, checkpoint=False, artifact_budget_bytes=20*2**20,
        architecture='One square bias-free linear layer followed by GELU; both factors active',
        compiler_cache_policy='Disposable local code cache; not training weights; at most 2 GiB expected')
    root = Path(__file__).resolve().parents[1]
    (bundle/'source/benchmarks').mkdir(parents=True)
    for name in ['soap.py', 'soap_reference.py', 'benchmarks/benchmark_synthetic_scaling.py',
                 'benchmarks/benchmark_large_precision.py']:
        shutil.copy2(root/name, bundle/'source'/name)
    (bundle/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    shutil.copy2(root/'benchmarks/LARGE_PRECISION.md', bundle/'PROTOCOL.md')
    manifest = dict(sha256={str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in bundle.rglob('*') if p.is_file()},
                    checkpoint_policy='No weights, resume states or diagnostic tensors; keep compact scalar evidence')
    (bundle/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--widths', type=int, nargs='+', default=[2048, 4096])
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args(); bundle = args.bundle.resolve()
    if args.prepare:
        prepare(bundle, args.widths, args.rounds); print(bundle); return
    if not os.environ.get('GPU_CLAIM_INDICES') or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one lifetime GPU claim required')
    for name, digest in json.loads((bundle/'manifest.json').read_text())['sha256'].items():
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest() != digest: raise RuntimeError('Source changed: '+name)
    config = json.loads((bundle/'config.json').read_text()); torch.set_num_threads(4)
    began = time.monotonic()
    def budget():
        if time.monotonic() - began > config['maximum_seconds']: raise RuntimeError('Declared budget exceeded')
    report = dict(status='running', config=config, results=[], torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        limitations=['Synthetic repeated batches are not learning evidence.',
            'Synchronized natural 200-update cycle; geometry checks and compiler preflight excluded from step times.',
            'One QR reset per cycle gives sparse reset timing; cold initialization reported separately.',
            'Compile arm only compiles warm basis tensor math, not the entire optimizer or model.',
            'Two timing repeats use identical seeds, not independent learning seeds.',
            'High permits reduced precision; the actual dispatched matmul kernel is not identified.',
            'NS6 is an experimental candidate, not a default fix; guard pass is not accuracy equivalence.'])
    def save():
        path = bundle/'results.tmp'; path.write_text(json.dumps(report, indent=2)+'\n'); path.replace(bundle/'results.json')
    save()
    for width in config['widths']:
        for round_index in range(config['rounds']):
            for arm in config['arms'][::1 if round_index % 2 == 0 else -1]:
                result = measure(width, arm, round_index, config, budget)
                report['results'].append(result); save()
                print(json.dumps(dict(width=width, arm=arm['name'], round=round_index,
                    status=result['status'], failure=result['failure'], gram=result['maximum_gram_error'],
                    summary=result['summary'], compile=result['compile_preflight'])), flush=True)
    report.update(status='completed_with_rejections' if any(r['status'] != 'valid' for r in report['results']) else 'completed',
                  all_in_seconds=time.monotonic() - began)
    save(); print('COMPLETE', report['all_in_seconds'], flush=True)


if __name__ == '__main__': main()
