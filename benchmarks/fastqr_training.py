"""Fast-QR study (2026-11-20): matched 109M training arms for the refresh candidates.

Same conventions as benchmarks/bf16_refresh_controls.py (the paper's QR20/QR50 controls): the 109,396,992-
parameter tied-embedding OWT Transformer, model seed 204, the paper's frozen train/validation rows (sha256-matched
copies in /workspace/data/untied_head_20261120), the full 10,000-call LR horizon, matrix peak LR .0015, stock
betas (.95, .95), covariance .95, auxiliary AdamW .0006 / (.9, .999), decay .05, clip 1, batch 8 x accumulation 4,
BF16 M/V/C/Q, FP32 master weights, strict deterministic algorithms, validation on all 1,024 rows. Startup
(initial weights, gradient, loss, norm and validation CE) must match the paper's verified seed-204 reference.

The training loop is ``bf16_storage_screen.run`` itself; this script only swaps in the optimizer builder,
the event-kind table and the eager/compiled parity screen for the new arms (module attributes of the frozen
copy), and adds a spectral Gram-error diagnostic, CholQR fallback logging and a final weight save.

Arms (all BF16, compiled):
  bf16_hh_qr10       production CompiledSOAP, Householder QR every 10 ages (no warm) - the matched QR10 control
  bf16_cq_qr10       FastRefreshSOAP qr_kind='cholqr', QR every 10 ages (C3)
  bf16_cq_qr20       FastRefreshSOAP qr_kind='cholqr', QR every 20 ages, no warm (C1-20; CANDIDATES correction)
  bf16_w10_cq50_ns1  FastRefreshSOAP warm every 10 with one Newton-Schulz step (C2), CholQR every 50 (C1)
  bf16_w10_qr50      production CompiledSOAP warm10/QR50 (fresh, for paired timing and reproducibility)
  (bf16_w10_cq50, bf16_w10_qr50_ns1: optional single-change arms)

  prepare   --bundle DIR [--calls N] [--seed S] --arms ...   freeze sources, inputs and protocol
  campaign  --bundle DIR                                      run every arm sequentially (one GPU claim)
  worker    --bundle DIR --arm A                              one arm (called by campaign)
  summarize --bundle DIR                                      comparison.json from finished arms
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

# The paper's recorded launch environment (paper/REPRODUCIBILITY.md, "Execution setting"); set before torch loads.
# Without CUBLAS_WORKSPACE_CONFIG the initial gradient differed from the paper's seed-204 reference.
LAUNCH_ENV = dict(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='4', TORCHINDUCTOR_COMPILE_THREADS='2', HF_HUB_OFFLINE='1',
                  PYTHONDONTWRITEBYTECODE='1')
for _key, _value in LAUNCH_ENV.items():
    os.environ.setdefault(_key, _value)

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

REPO = Path('/workspace/TurboSOAP')
DATA = Path('/workspace/data/untied_head_20261120')
CONTROLS = REPO / 'reports/bf16_refresh_controls_20261027'
ARMS = {
    'bf16_hh_qr10': dict(cls='production', frequency=10, repair=10),
    'bf16_cq_qr10': dict(cls='fast', frequency=10, repair=10, qr_kind='cholqr', warm_ns_steps=2),
    'bf16_cq_qr20': dict(cls='fast', frequency=20, repair=20, qr_kind='cholqr', warm_ns_steps=2),
    'bf16_w10_cq50_ns1': dict(cls='fast', frequency=10, repair=50, qr_kind='cholqr', warm_ns_steps=1),
    'bf16_w10_qr50': dict(cls='production', frequency=10, repair=50),
    'bf16_w10_cq50': dict(cls='fast', frequency=10, repair=50, qr_kind='cholqr', warm_ns_steps=2),
    'bf16_w10_qr50_ns1': dict(cls='fast', frequency=10, repair=50, qr_kind='householder', warm_ns_steps=1),
    # stage 3: the paper's bf16_qr50 control cadence (no warm events), and Householder QR10 with FP32 storage
    # (the cached compiled_fp32 configuration of bf16_learning_20261026: FP32 state and basis, BF16 compute)
    'bf16_hh_qr50': dict(cls='production', frequency=50, repair=50),
    'fp32s_hh_qr10': dict(cls='production', frequency=10, repair=10, state_dtype='float32', basis_dtype='float32'),
    # stage 4: C6 front-loaded schedule, QR every 10 ages through age qr_until, then production warm10/QR50 on the
    # global age grid; pb8 logs the settling statistic p99.9 |M/(sqrt(V)+eps)| around every refresh
    'bf16_fl300_w10_qr50': dict(cls='frontloaded', frequency=10, repair=50, qr_until=300, pb8=True),
    'bf16_fl1000_w10_qr50': dict(cls='frontloaded', frequency=10, repair=50, qr_until=1000, pb8=True),
}
HOLDER = {}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def event_kind_for(arm, call):
    age = call - 1
    spec = ARMS[arm]
    if not age:
        return 'cold'
    if age % spec['frequency']:
        return 'ordinary'
    if age <= spec.get('qr_until', -1):
        return 'qr'
    return 'qr' if age % spec['repair'] == 0 else 'warm'


def protocol_for(calls, seed, startup):
    # ages >= 2000 (late-training geometry for the 10k stage) are filtered out of 2k bundles, so stage-1 protocols
    # are unchanged
    ages = [a for a in (50, 100, 200, 500, 540, 1000, 1040, 1500, 1540, 1950, 1990, 5000, 5040, 9950, 9990)
            if a < calls]
    return dict(model_seed=seed, validation_rows=1024,
                validation_calls=[c for c in (50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000,
                                              7000, 7500, 8000, 9000, 10000) if c <= calls],
                evaluation_hash_calls=[0, calls // 2, calls], diagnostic_calls=[a + 1 for a in ages],
                residual_calls=[a + 1 for a in ages if a % 500 == 0 or a in (1950, 9950)],
                capture_calls=[c for c in (101, 201, 501, 1001, 1501, 2000, 5000, 10000) if c <= calls],
                repeat_initial_backward=True, expected_startup=startup,
                startup_reference=None if not startup else
                'paper verified full BF16 seed204 replication (bf16_refresh_controls manifest)' if seed == 204 else
                f'paper BF16 full-horizon seed{seed} run (bf16_full_horizon_20261026_seed{seed}/runs/bf16)',
                check_final_state_finite=True, arm_timeout_seconds=max(3600, int(calls * 0.6) + 1200))


# ----------------------------------------------------------------------------- patched pieces of bss.run
def install(bss, arm):
    import torch
    from bf16_soap import CompiledSOAP
    import bf16_soap_fastqr as fq
    spec = ARMS[arm]
    HOLDER['arm'] = arm

    def build(model, case, compile_kernels=True):
        assert case == arm
        groups = bss.grouped_parameters(model, dict(matrix_lr=.0015, lr=.0006, weight_decay=.05))
        matrix_groups = [g for g in groups if g['role'] == 'matrix']
        common = dict(state_dtype=getattr(torch, spec.get('state_dtype', 'bfloat16')),
                      basis_dtype=getattr(torch, spec.get('basis_dtype', 'bfloat16')),
                      precondition_frequency=spec['frequency'], hard_reset_interval=spec['repair'],
                      compile_kernels=compile_kernels)
        if spec['cls'] == 'production':
            matrix = CompiledSOAP(matrix_groups, **common)
        elif spec['cls'] == 'frontloaded':
            matrix = fq.FrontLoadedSOAP(matrix_groups, qr_until=spec['qr_until'], **common)
        else:
            matrix = fq.FastRefreshSOAP(matrix_groups, qr_kind=spec['qr_kind'],
                                        warm_ns_steps=spec['warm_ns_steps'], **common)
        auxiliary = torch.optim.AdamW([g for g in groups if g['role'] == 'auxiliary'],
                                      betas=(.9, .999), eps=1e-8, fused=next(model.parameters()).is_cuda)
        HOLDER.update(model=model, matrix=matrix)
        return bss.SplitOptimizer(matrix, auxiliary)

    @torch.no_grad()
    def parity(case):
        """Eager vs compiled for each event this arm uses, on a small random state (CUDA)."""
        from functools import partial
        torch.manual_seed(944)
        p = torch.randn(2, 48, 32, device='cuda')
        g = torch.randn_like(p)
        state_dtype = getattr(torch, spec.get('state_dtype', 'bfloat16'))
        basis_dtype = getattr(torch, spec.get('basis_dtype', 'bfloat16'))
        ql = torch.linalg.qr(torch.randn(2, 48, 48, device='cuda')).Q.to(basis_dtype).contiguous()
        qr = torch.linalg.qr(torch.randn(2, 32, 32, device='cuda')).Q.to(basis_dtype).contiguous()
        xl, xr = torch.randn(2, 48, 200, device='cuda'), torch.randn(2, 32, 200, device='cuda')
        state = (torch.randn_like(p).to(state_dtype), torch.rand_like(p).to(state_dtype),
                 (xl @ xl.mT / 200).to(state_dtype), (xr @ xr.mT / 200).to(state_dtype), ql, qr)
        config = dict(beta1=.95, beta2=.95, cov_beta=.95, eps=1e-8, weight_decay=.05,
                      compute_dtype=torch.bfloat16, basis_dtype=basis_dtype)
        args = (p, g, state, torch.tensor(10., device='cuda'), torch.tensor(.001, device='cuda'))
        events = ['ordinary']
        if spec['repair'] != spec['frequency']:
            events.append('warm')
        events.append('qr')
        rows = []
        for event in events:
            fast = spec['cls'] == 'fast' and ((event == 'qr' and spec['qr_kind'] == 'cholqr') or
                                              (event == 'warm' and spec['warm_ns_steps'] != 2))
            if fast:
                kind = 'cholqr' if event == 'qr' else 'warm'
                fn = partial(fq.fast_matrix_step, event=kind, warm_ns_steps=spec['warm_ns_steps'],
                             **fq.CHOLQR_DEFAULTS, **config)
            else:
                fn = partial(bss.matrix_step, event=event, **config)
            compiled = torch.compile(fn, fullgraph=True, dynamic=False, options={'emulate_precision_casts': True})
            ieee = fast and event == 'qr'
            previous = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = not ieee and previous
            try:
                eager, actual = fn(*args), compiled(*args)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous
            row = dict(event=event, fast=fast, parameter_max_absolute=float((actual[0] - eager[0]).abs().max()),
                       state_relative_errors={k: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
                                              for k, a, b in zip(bss.STATE_KEYS, actual[1], eager[1])})
            if fast and event == 'qr':
                row['ok_flags'] = [bool(actual[2][3].all()), bool(actual[2][4].all())]
            assert row['parameter_max_absolute'] < 2e-5, row
            assert max(row['state_relative_errors'].values()) < .025, row
            rows.append(row)
        torch._dynamo.reset()
        return dict(eager_compiled=rows, arm=arm, spec=spec)

    original_diagnostics = bss.diagnostics

    def diagnostics(matrix, before=None, rounded_floor=False):
        result = original_diagnostics(matrix, before, rounded_floor)
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision('highest')
        try:
            spectral = []
            for name, p, s in bss.states(matrix):
                for axis, key in enumerate(('ql', 'qr')):
                    q = s[key].float()
                    e = q.T @ q - torch.eye(q.shape[-1], device=q.device)
                    spectral.append(float(torch.linalg.eigvalsh(e).abs().max()))
        finally:
            torch.set_float32_matmul_precision(previous)
        for row, value in zip(result['factors'], spectral):
            row['spectral_gram'] = value
        result['max_spectral_gram'] = max(spectral)
        return result

    if spec.get('pb8'):
        install_pb8(bss, spec)
    bss.build_storage_optimizer = build
    bss.event_kind = event_kind_for
    bss.cuda_parity = parity
    bss.diagnostics = diagnostics
    bss.CONTROL_CADENCES = {**bss.CONTROL_CADENCES, arm: spec['repair']}  # activates the per-call event assert


def install_pb8(bss, spec):
    """PB8 settling log (observation only, outside the timed region): at the start of a call, before backward,
    the p99.9 of |M/(sqrt(V)+eps)| over every matrix-state coordinate. Samples: 'pre' (state after 9 ordinary steps,
    just before a refresh), 'post' (state right after the refresh step) and 'mid' (after the refresh and 4 ordinary
    steps). Bias correction is omitted; the new gradient is not yet mixed in. Read-only, no RNG."""
    import torch
    original = bss.apply_lr_multiplier
    HOLDER['pb8'] = []
    f = spec['frequency']

    @torch.no_grad()
    def settle(matrix):
        parts = []
        for group in matrix.param_groups:
            s = matrix.state[group['params'][0]]
            parts.append((s['m'].float().abs() / (s['v'].float().sqrt() + group['eps'])).flatten())
        x = torch.sort(torch.cat(parts)).values
        n = x.numel()
        pick = lambda q: float(x[min(n - 1, int(q * (n - 1)))])
        return dict(p50=pick(.5), p99=pick(.99), p999=pick(.999), max=float(x[-1]), n=n)

    def apply_lr_multiplier(optimizer, multiplier):
        original(optimizer, multiplier)
        matrix = HOLDER.get('matrix')
        if matrix is None or matrix.call < 1 or 'pb8_error' in HOLDER:
            return
        upcoming, last = matrix.call, matrix.call - 1  # ages of the step about to run / just run
        if last > 0 and last % f == 0:
            kind, event = 'post', matrix.last_event
        elif upcoming % f == 0:
            kind, event = 'pre', event_kind_for(HOLDER['arm'], upcoming + 1)
        elif upcoming % f == 5:
            kind, event = 'mid', None
        else:
            return
        try:
            HOLDER['pb8'].append(dict(call=matrix.call + 1, kind=kind, refresh_event=event,
                                      refresh_age=upcoming if kind == 'pre' else (last if kind == 'post' else upcoming - 5),
                                      **settle(matrix)))
        except Exception as error:  # never let the diagnostic stop training
            HOLDER['pb8_error'] = repr(error)

    bss.apply_lr_multiplier = apply_lr_multiplier


# ----------------------------------------------------------------------------- bundle
def source_modules():
    """Repo files the worker imports (resolved by importing in a subprocess)."""
    code = ('import sys,json;sys.path[:0]=["%s","%s/benchmarks"];import bf16_storage_screen,bf16_soap_fastqr;'
            'print(json.dumps(sorted(m.__file__ for m in list(sys.modules.values()) '
            'if getattr(m,"__file__",None) and m.__file__.startswith("%s/"))))') % (REPO, REPO, REPO)
    out = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True).stdout
    names = [str(Path(f).relative_to(REPO)) for f in json.loads(out.strip().splitlines()[-1])]
    return sorted(set(names) | {'benchmarks/fastqr_training.py'})


def startup_reference(seed, controls):
    """The paper's verified startup for a seed: 204 from the QR20/QR50 controls manifest, 205 from the paper's
    seed-205 full-horizon BF16 run (same model, data, recipe and launch environment)."""
    if seed == 204:
        return controls['reference_startup']
    if seed == 205:
        folder = REPO / 'reports/bf16_full_horizon_20261026_seed205/runs/bf16'
        r = json.loads((folder / 'result.json').read_text())
        g = next(json.loads(s) for s in (folder / 'events.jsonl').read_text().splitlines()
                 if json.loads(s)['event'] == 'initial_gradient')
        assert r['seed'] == 205 and r['initial_gradient_sha256'] == g['sha256']
        return dict(initial_parameters_sha256=r['initial_parameters_sha256'],
                    initial_gradient_sha256=r['initial_gradient_sha256'], initial_loss=g['loss'],
                    initial_norm=g['norm'], initial_validation_ce=r['validation'][0]['ce'])
    return None


def prepare(bundle, arms, calls, seed):
    assert not bundle.exists(), 'fresh bundle required'
    for arm in arms:
        assert arm in ARMS, arm
    controls = json.loads((CONTROLS / 'manifest.json').read_text())
    startup = startup_reference(seed, controls)
    bundle.mkdir(parents=True)
    (bundle / 'source').mkdir()
    names = source_modules()
    for name in names:
        target = bundle / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    for name in ('train.npy', 'validation.npy'):
        (bundle / name).symlink_to(DATA / name)
        assert sha(bundle / name) == controls['inputs'][name], name
    checks = {p.name: sha(p) for p in sorted((REPO / 'reports/fastqr_20261120').glob('unit_*.json'))}
    m = dict(arms=list(arms), calls=calls, seed=seed, schedule_horizon=10000, specs={a: ARMS[a] for a in arms},
             protocols={a: protocol_for(calls, seed, startup) for a in arms},
             source_files={n: sha(bundle / 'source' / n) for n in names},
             inputs={n: sha(bundle / n) for n in ('train.npy', 'validation.npy')},
             production_bf16_soap_sha256=sha(REPO / 'bf16_soap.py'),
             paper_bf16_soap_sha256=controls['production_source_sha256']['bf16_soap.py'],
             unit_check_receipts=checks, created_at_utc=datetime.now(timezone.utc).isoformat(),
             save_final_weights=True, launch_environment=LAUNCH_ENV,
             compiler_cache='<bundle>/compiler_cache, shared sequentially by the campaign')
    assert m['production_bf16_soap_sha256'] == m['paper_bf16_soap_sha256']
    write(bundle / 'manifest.json', m)
    print(json.dumps(dict(status='prepared', bundle=str(bundle), arms=arms, sources=len(names))), flush=True)


def verify(bundle):
    m = json.loads((bundle / 'manifest.json').read_text())
    for name, digest in m['source_files'].items():
        assert sha(bundle / 'source' / name) == digest, name
    for name, digest in m['inputs'].items():
        assert sha(bundle / name) == digest, name
    return m


def worker(bundle, arm):
    import torch
    m = verify(bundle)
    for key, value in m.get('launch_environment', {}).items():
        assert os.environ.get(key) == value, (key, os.environ.get(key), value)
    assert Path(__file__).resolve() == bundle / 'source/benchmarks/fastqr_training.py', 'run the frozen copy'
    import bf16_storage_screen as bss
    install(bss, arm)
    result = bss.run(bundle, arm, m['calls'], protocol=m['protocols'][arm])
    out = bundle / 'runs' / arm
    matrix = HOLDER['matrix']
    stats = dict(arm=arm, spec=ARMS[arm], last_call=matrix.call)
    if hasattr(matrix, 'refresh_log'):
        stats.update(refresh_counts=matrix.refresh_counts(), refresh_log=matrix.refresh_log,
                     fallback_log=matrix.fallback_log)
    if 'pb8' in HOLDER:
        stats.update(pb8=HOLDER['pb8'], pb8_error=HOLDER.get('pb8_error'))
    write(out / 'fastqr_stats.json', stats)
    if m.get('save_final_weights'):
        torch.save({k: v.detach().cpu() for k, v in HOLDER['model'].state_dict().items()}, out / 'final_model.pt')
    return result


def audit(bundle, arm, m):
    folder = bundle / 'runs' / arm
    r = json.loads((folder / 'result.json').read_text())
    events = [json.loads(s) for s in (folder / 'events.jsonl').read_text().splitlines()]
    updates = [e for e in events if e['event'] == 'update']
    calls = m['calls']
    assert [e['call'] for e in updates] == list(range(1, calls + 1))
    assert all(e['kind'] == event_kind_for(arm, e['call']) for e in updates)
    assert all(math.isfinite(e[k]) for e in updates for k in ('loss', 'fb_ms', 'optimizer_ms', 'step_ms'))
    for name in ('final_state_finite', 'initial_backward_repeat'):
        assert any(e['event'] == name and e['passed'] for e in events), name
    if m['protocols'][arm]['expected_startup']:
        assert any(e['event'] == 'startup_match' and e['passed'] for e in events)
        assert any(e['event'] == 'initial_validation_match' and e['passed'] for e in events)
    assert not r['dynamo_counters'].get('graph_break', {})
    captures = [e for e in events if e['event'] == 'capture_counters']
    graphs = [e['stats'].get('unique_graphs') for e in captures]
    stats = json.loads((folder / 'fastqr_stats.json').read_text())
    return dict(status='passed', arm=arm, event_counts=dict(Counter(e['kind'] for e in updates)),
                unique_graphs_at_captures=graphs, refresh_counts=stats.get('refresh_counts'))


def summarize(bundle):
    m = json.loads((bundle / 'manifest.json').read_text())
    rows = []
    for arm in m['arms']:
        folder = bundle / 'runs' / arm
        if not (folder / 'result.json').exists():
            continue
        r = json.loads((folder / 'result.json').read_text())
        events = [json.loads(s) for s in (folder / 'events.jsonl').read_text().splitlines()]
        updates = [e for e in events if e['event'] == 'update' and 101 <= e['call'] <= m['calls']]
        geometry = json.loads((folder / 'geometry.json').read_text())
        stats = json.loads((folder / 'fastqr_stats.json').read_text())
        by_kind = {}
        for k in ('ordinary', 'warm', 'qr'):
            xs = [e['optimizer_ms'] for e in updates if e['kind'] == k]
            if xs:
                by_kind[k] = dict(n=len(xs), mean_ms=sum(xs) / len(xs), median_ms=sorted(xs)[len(xs) // 2])
        rows.append(dict(arm=arm, spec=m['specs'][arm], validation=r['validation'],
                         mean_step_ms=r['mean_step_ms'], mean_optimizer_ms=r['mean_optimizer_ms'],
                         mean_fb_ms=sum(e['fb_ms'] for e in updates) / len(updates), clean_calls=len(updates),
                         optimizer_by_kind=by_kind, peak_gpu_bytes=r['peak_gpu_bytes'],
                         matrix_state_bytes=r['matrix_state_bytes'], elapsed_seconds=r['elapsed_seconds'],
                         max_normalized_gram=r['max_normalized_gram'], max_physical_m_error=r['max_physical_m_error'],
                         max_spectral_gram=max(g.get('max_spectral_gram', 0) for g in geometry),
                         spectral_gram_by_call={g['call']: g.get('max_spectral_gram') for g in geometry},
                         refresh_counts=stats.get('refresh_counts'), fallback_log=stats.get('fallback_log', [])[:200]))
    cached = json.loads((CONTROLS / 'comparison.json').read_text())
    write(bundle / 'comparison.json', dict(manifest=dict(calls=m['calls'], seed=m['seed'], arms=m['arms']),
                                           results=rows, paper_cached_warm_prefix=cached['cached_warm_prefix'],
                                           paper_qr20_qr50=[dict(case=r['case'], validation=r['validation'],
                                                                 mean_step_ms=r['mean_step_ms'],
                                                                 mean_optimizer_ms=r['mean_optimizer_ms'])
                                                            for r in cached['results']]))
    return rows


def campaign(bundle):
    m = verify(bundle)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(bundle / 'compiler_cache')  # the paper's per-campaign cache
    progress = dict(status='running', arms=m['arms'], completed=[], active=None, started=time.time())
    write(bundle / 'progress.json', progress)
    audits = []
    try:
        for arm in m['arms']:
            if (bundle / 'runs' / arm / 'result.json').exists():
                audits.append(audit(bundle, arm, m))
                progress['completed'].append(arm)
                continue
            progress['active'] = arm
            write(bundle / 'progress.json', progress)
            subprocess.run([sys.executable, str(bundle / 'source/benchmarks/fastqr_training.py'), 'worker',
                            '--bundle', str(bundle), '--arm', arm], check=True,
                           timeout=m['protocols'][arm]['arm_timeout_seconds'] + 600)
            audits.append(audit(bundle, arm, m))
            progress['completed'].append(arm)
            write(bundle / 'progress.json', progress)
        summarize(bundle)
        write(bundle / 'completion_audit.json', dict(status='passed', arms=audits))
        progress.update(status='completed', active=None, elapsed_seconds=time.time() - progress['started'])
        write(bundle / 'progress.json', progress)
    except BaseException as error:
        progress.update(status='failed', error=repr(error), elapsed_seconds=time.time() - progress['started'])
        write(bundle / 'progress.json', progress)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('prepare', 'campaign', 'worker', 'summarize'))
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--arms', nargs='*')
    p.add_argument('--arm')
    p.add_argument('--calls', type=int, default=2000)
    p.add_argument('--seed', type=int, default=204)
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a.bundle, a.arms, a.calls, a.seed)
        return
    if a.mode == 'summarize':
        summarize(a.bundle)
        return
    if not (os.environ.get('GPU_CLAIM_UUIDS') or os.environ.get('GPU_CLAIM_INDICES')):
        # gpu-claim exports only CUDA_VISIBLE_DEVICES; bss.run records GPU_CLAIM_INDICES
        assert os.environ.get('CUDA_VISIBLE_DEVICES'), 'run under gpu-claim (it sets CUDA_VISIBLE_DEVICES)'
        os.environ['GPU_CLAIM_INDICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    if a.mode == 'worker':
        worker(a.bundle, a.arm)
    else:
        campaign(a.bundle)


if __name__ == '__main__':
    main()
