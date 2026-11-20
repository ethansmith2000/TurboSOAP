"""Untied LM head / input embedding: one-sided SOAP vs AdamW (side study, not in the paper).

Model: the paper's 109M Transformer with ``tie_embeddings=False`` (147,994,368 parameters).
Data: the paper's frozen OWT rows, regenerated and sha256-matched by untied_head_prepare.py.
Hidden block matrices: production TurboSOAP (bf16_soap.CompiledSOAP, BF16 state, warm10/QR50,
betas .95/.95, covariance .95, peak LR 1.5e-3, wd .05), identical in every arm.
Norm gains: AdamW (.9, .999), peak LR 6e-4, wd 0, as in the paper.
Input embedding E (wd 0) and LM head H (wd .05; the repo's grouping rule for an untied
nn.Linear weight): AdamW or one-sided SOAP, each with its own peak LR and beta2, beta1 .9.
One-sided SOAP = soap.SOAP(precondition_mode='smaller_side'): one 768x768 covariance factor
and basis on the hidden side only; per-element Adam moments on the 50257x768 rotated gradient.
The vocab side gets no factor (it also exceeds max_precond_dim). Basis: QR (one power step +
QR) every 10 calls, i.e. the paper's QR10 control, not the warm tracker.

Subcommands
  selftest   CPU checks: AdamW equivalence, one-sided structure/math, tiny end-to-end arms
  run        one training run from a JSON config
  probe      sequential GPU timing/memory probe of the four arms at base LRs (one claim)
  plan       print a stage's pre-registered run list
  launch     submit runs through gpu-claim, deferring to refresh-study; --auto chains stages
  summarize  apply the pre-registered selection and write tables to reports/untied_head_20261120
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '8')
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO), str(HERE)]

import numpy as np
import torch
from torch.nn import functional as F

from bf16_soap import CompiledSOAP
from cifar_optimizers import grouped_parameters
from soap import SOAP
from transformer import Transformer

DATA_DIR = Path('/workspace/data/untied_head_20261120')
RUNS_DIR = DATA_DIR / 'runs'
REPORT_DIR = REPO / 'reports/untied_head_20261120'
PYTHON = '/venv/main/bin/python'

HORIZON, WARMUP, BATCH, MICRO = 10000, 200, 32, 8
EVAL_CALLS = (50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 7500,
              8000, 9000, 10000)
MATRIX = dict(lr=.0015, betas=(.95, .95), shampoo_beta=.95, weight_decay=.05,
              precondition_frequency=10, hard_reset_interval=50)
AUX_LR, AUX_BETA1, AUX_BETA2, EPS = .0006, .9, .999, 1e-8
EH_SOAP = dict(shampoo_beta=.95, precondition_frequency=10, max_precond_dim=10000,
               precondition_mode='smaller_side', normalize_grads=False, basis_reset_frequency=1,
               basis_reset_method='qr', covariance_compute_dtype='float32', basis_track_stats=False)
BUCKET_EDGES = (1000, 10000, 30000)  # frequency-rank edges in the 320k training rows
BUCKETS = ('top1k', '1k-10k', '10k-30k', '30k+', 'unseen')

BASE = dict(opt='adamw', lr=6e-4, beta2=.999)  # the paper's auxiliary AdamW
BETA2S = (.95, .999)
GRID = dict(H=(3e-4, 6e-4, 1.2e-3, 2.4e-3), E=(6e-4, 1.2e-3, 2.4e-3, 4.8e-3))
MAX_EDGE_STEPS = 2          # extra factor-2 points allowed past each original grid edge
TUNE_SEED, CONFIRM_SEEDS = 204, (205, 206)
ARMS = dict(A0=('adamw', 'adamw'), A1=('adamw', 'soap'), A2=('soap', 'soap'), A3=('soap', 'adamw'))
VERDICT_THRESHOLD = .005    # CE; paper's same-config seed spread at 10k is ~.003


# ----------------------------------------------------------------------------- helpers

def multiplier(call):
    """The paper's schedule (owt_restart.multiplier): 200-call warmup, cosine to 10% at 10k."""
    if call <= WARMUP:
        return call / WARMUP
    return .1 + .9 * .5 * (1 + math.cos(math.pi * (call - WARMUP) / (HORIZON - WARMUP)))


def token_loss(logits, labels):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))


def token_ce(logits, labels):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           reduction='none')


def tensor_digest(items):
    h = hashlib.sha256()
    for name, t in items:
        h.update(str((name, str(t.dtype), tuple(t.shape))).encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=True) + '\n')
    temp.replace(path)


def lr_tag(lr):
    return f'{lr:.1e}'.replace('e-0', 'e-')


def spec_tag(spec):
    return f"{spec['opt']}{lr_tag(spec['lr'])}b{str(spec['beta2'])[2:]}"


def run_name(E, H, seed, calls=HORIZON):
    name = f'E-{spec_tag(E)}_H-{spec_tag(H)}_s{seed}'
    return name if calls == HORIZON else f'{name}_c{calls}'


def make_config(E, H, seed, calls=HORIZON, **extra):
    name = run_name(E, H, seed, calls) + extra.pop('suffix', '')
    return dict(name=name, seed=seed, calls=calls, E=dict(E), H=dict(H), tiny=False, device='cuda',
                compile=True, val_rows=1024, out=str(RUNS_DIR / name), **extra)


def bucket_of_tokens(counts):
    order = np.argsort(-counts, kind='stable')
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    bucket = np.searchsorted(np.asarray(BUCKET_EDGES), rank, side='right')
    bucket[counts == 0] = len(BUCKET_EDGES) + 1
    return bucket


# ------------------------------------------------------------------- model and optimizers

def build_model(cfg):
    torch.manual_seed(cfg['seed'])
    common = dict(vocab_size=50257, tie_embeddings=False, use_rope=True, qk_norm=True, bias=False,
                  dropout=0., fused_qkv=True, fused_swiglu=True, gradient_checkpointing=False)
    if cfg.get('tiny'):
        return Transformer(dim=64, depth=2, heads=2, ff_mult=2., ff_hidden_dim=128, max_seq_len=64,
                           **common)
    return Transformer(dim=768, depth=12, heads=12, ff_mult=2., ff_hidden_dim=1536, max_seq_len=1024,
                       **common)


def make_group_optimizer(parameter, weight_decay, spec, fused):
    group = dict(params=[parameter], lr=spec['lr'], peak_lr=spec['lr'], weight_decay=weight_decay)
    betas = (AUX_BETA1, spec['beta2'])
    if spec['opt'] == 'adamw':
        return torch.optim.AdamW([group], betas=betas, eps=EPS, fused=fused)
    if spec['opt'] == 'soap':
        return SOAP([group], betas=betas, eps=EPS, **{**EH_SOAP, **spec.get('soap', {})})
    raise ValueError(spec['opt'])


def build_optimizers(model, cfg):
    """Returns ({part: optimizer}, {'E': param, 'H': param}). Parts step independently."""
    fused = next(model.parameters()).is_cuda
    groups = grouped_parameters(model, dict(matrix_lr=MATRIX['lr'], lr=AUX_LR,
                                            weight_decay=MATRIX['weight_decay']))
    matrix = CompiledSOAP([g for g in groups if g['role'] == 'matrix'], lr=MATRIX['lr'],
                          betas=MATRIX['betas'], shampoo_beta=MATRIX['shampoo_beta'],
                          weight_decay=MATRIX['weight_decay'],
                          precondition_frequency=MATRIX['precondition_frequency'],
                          hard_reset_interval=MATRIX['hard_reset_interval'],
                          compile_kernels=cfg.get('compile', True))
    auxiliary = {name: (p, g['weight_decay']) for g in groups if g['role'] == 'auxiliary'
                 for name, p in zip(g['param_names'], g['params'])}
    E, wd_E = auxiliary.pop('token_embedding.weight')
    H, wd_H = auxiliary.pop('lm_head.weight')
    assert E is not H and wd_E == 0. and wd_H == MATRIX['weight_decay'], (wd_E, wd_H)
    assert all(p.ndim == 1 and wd == 0. for p, wd in auxiliary.values()), 'rest must be norm gains'
    rest = torch.optim.AdamW([dict(params=[p for p, _ in auxiliary.values()], lr=AUX_LR,
                                   peak_lr=AUX_LR, weight_decay=0.)],
                             betas=(AUX_BETA1, AUX_BETA2), eps=EPS, fused=fused)
    parts = dict(matrix=matrix, rest=rest,
                 E=make_group_optimizer(E, wd_E, cfg['E'], fused),
                 H=make_group_optimizer(H, wd_H, cfg['H'], fused))
    return parts, dict(E=E, H=H)


def set_lr(parts, scale):
    for optimizer in parts.values():
        for group in optimizer.param_groups:
            group['lr'] = group['peak_lr'] * scale


def state_bytes(optimizer):
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            for t in value if isinstance(value, list) else [value]:
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
    return total


# ----------------------------------------------------------------------------- diagnostics

def spectrum_stats(matrix):
    """Eigen-summary of the 768-side Gram matrix X^T X (fp64)."""
    gram = matrix.double().T @ matrix.double()
    values = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    if not float(values[0]) > 0:  # e.g. SOAP's observe-only first call on an undecayed group
        return None
    total = values.sum().clamp_min(1e-300)
    p = values / total
    entropy = -(p * p.clamp_min(1e-300).log()).sum()
    positive = values[values > values[0] * 1e-12]
    return dict(top1_fraction=float(p[0]), top10_fraction=float(p[:10].sum()),
                effective_rank=float(entropy.exp()), max_over_median=float(values[0] / values.median().clamp_min(1e-300)),
                condition_positive=float(positive[0] / positive[-1]))


@torch.no_grad()
def group_diagnostics(before, after, grad, lr, weight_decay, optimizer, parameter, bucket):
    """Unit-LR optimizer direction u = (W0 (1 - lr wd) - W1) / lr, i.e. decoupled decay removed."""
    W0, W1, g = before.float(), after.float(), grad.float()
    u = (W0 * (1 - lr * weight_decay) - W1) / lr
    row_u = u.square().mean(1).sqrt()
    row_norm = W1.norm(dim=1)
    active = g.abs().amax(1) > 0
    out = dict(lr=lr, update_rms=float(u.square().mean().sqrt()), grad_rms=float(g.square().mean().sqrt()),
               active_rows=int(active.sum()), buckets={})
    for index, name in enumerate(BUCKETS):
        mask = bucket == index
        if not bool(mask.any()):
            continue
        norms = row_norm[mask]
        out['buckets'][name] = dict(
            rows=int(mask.sum()), active_fraction=float(active[mask].float().mean()),
            update_row_rms_mean=float(row_u[mask].mean()), update_rms=float(u[mask].square().mean().sqrt()),
            update_row_rms_active=float(row_u[mask & active].mean()) if bool((mask & active).any()) else None,
            row_norm_mean=float(norms.mean()), row_norm_cv=float(norms.std() / norms.mean().clamp_min(1e-30)))
    seen = bucket < len(BUCKETS) - 1
    norms = row_norm[seen]
    quantiles = torch.quantile(norms.float(), torch.tensor([.1, .5, .9], device=norms.device))
    out['row_norm_seen'] = dict(mean=float(norms.mean()), cv=float(norms.std() / norms.mean()),
                                p90_over_p10=float(quantiles[2] / quantiles[0]),
                                max_over_median=float(norms.max() / quantiles[1]))
    out['grad_768_spectrum'] = spectrum_stats(g)
    out['update_768_spectrum'] = spectrum_stats(u)
    out['weight_768_spectrum'] = spectrum_stats(W1)
    if isinstance(optimizer, SOAP):
        state = optimizer.state[parameter]
        assert state['GG'][0] is None and state['Q'][0] is None
        C, Q = state['GG'][1].double(), state['Q'][1].double()
        rotated = Q.T @ C @ Q
        off = rotated - torch.diag(rotated.diagonal())
        eig = torch.linalg.eigvalsh(C).clamp_min(0).flip(0)
        out['soap_factor'] = dict(
            basis_offdiag_ratio=float(off.norm() / rotated.norm().clamp_min(1e-300)),
            orthogonality_error=float((Q.T @ Q - torch.eye(Q.shape[0], dtype=Q.dtype, device=Q.device)).norm()),
            eig_top1_fraction=float(eig[0] / eig.sum()), eig_max_over_median=float(eig[0] / eig.median().clamp_min(1e-300)))
    return out


# ---------------------------------------------------------------------------------- run

def run(cfg):
    out = Path(cfg['out'])
    out.mkdir(parents=True, exist_ok=True)
    previous = out / 'result.json'
    if previous.exists() and json.loads(previous.read_text()).get('status') in ('completed', 'diverged'):
        print(f'{cfg["name"]}: already finished', flush=True)
        return json.loads(previous.read_text())
    write_json(out / 'config.json', cfg)
    start = time.monotonic()
    device = torch.device(cfg['device'])
    cuda = device.type == 'cuda'
    torch.set_num_threads(4)
    # Strict, as in the paper: warn_only=True leaves FlashAttention's backward nondeterministic.
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    events = (out / 'events.jsonl').open('w')

    def emit(event, **values):
        row = dict(event=event, elapsed_seconds=round(time.monotonic() - start, 3), **values)
        events.write(json.dumps(row) + '\n')
        events.flush()
        if event != 'update' or values['call'] % 100 == 0:
            print(json.dumps(row), flush=True)

    model = build_model(cfg).to(device)
    initial_hash = tensor_digest(model.named_parameters())
    parts, eh = build_optimizers(model, cfg)
    compile_options = {} if cuda else {'pattern_matcher': False}
    if cfg.get('compile', True):
        train_loss = torch.compile(token_loss, fullgraph=True, dynamic=False,
                                   options={'emulate_precision_casts': True, **compile_options})
        eval_ce = torch.compile(token_ce, fullgraph=True, dynamic=False,
                                options={'emulate_precision_casts': True, **compile_options})
    else:
        train_loss, eval_ce = token_loss, token_ce
    sequence = model.max_seq_len + 1
    train = np.load(DATA_DIR / 'train.npy', mmap_mode='r')
    val = np.load(DATA_DIR / 'validation.npy', mmap_mode='r')[:cfg.get('val_rows', 1024)]
    bucket = torch.from_numpy(bucket_of_tokens(np.load(DATA_DIR / 'train_token_counts.npy'))).to(device)
    accumulation = BATCH // MICRO

    def rows(array):
        return torch.from_numpy(np.array(array[:, :sequence], dtype=np.int64)).to(device)

    def forward_context():
        if not cuda:
            return contextlib.nullcontext()
        from torch.nn.attention import SDPBackend, sdpa_kernel
        stack = contextlib.ExitStack()
        stack.enter_context(sdpa_kernel(SDPBackend.FLASH_ATTENTION))
        stack.enter_context(torch.autocast('cuda', dtype=torch.bfloat16))
        return stack

    def backward(call):
        model.zero_grad(set_to_none=True)
        tokens = rows(train[(call - 1) * BATCH:call * BATCH])
        losses = []
        for t in tokens.split(MICRO):
            with forward_context():
                loss = train_loss(model(t[:, :-1]), t[:, 1:]) / accumulation
            loss.backward()
            losses.append(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
        return float(torch.stack(losses).sum()), float(norm)

    val_tokens = rows(val)
    val_bucket = bucket[val_tokens[:, 1:]]
    val_counts = [int((val_bucket == i).sum()) for i in range(len(BUCKETS))]
    validation = []

    def evaluate(call):
        model.eval()
        sums = torch.zeros(len(BUCKETS) + 1, dtype=torch.float64, device=device)
        with torch.no_grad():
            for begin in range(0, len(val_tokens), MICRO):
                t = val_tokens[begin:begin + MICRO]
                with forward_context():
                    ce = eval_ce(model(t[:, :-1]), t[:, 1:]).double()
                b = val_bucket[begin:begin + MICRO].reshape(-1)
                sums[0] += ce.sum()
                for i in range(len(BUCKETS)):
                    sums[i + 1] += (ce * (b == i)).sum()
        model.train()
        sums = sums.tolist()
        targets = int(val_tokens[:, 1:].numel())  # 1,024 rows x 1,023 targets
        row = dict(call=call, ce=sums[0] / targets, targets=targets,
                   ce_by_bucket={name: (sums[i + 1] / val_counts[i] if val_counts[i] else None)
                                 for i, name in enumerate(BUCKETS)},
                   targets_by_bucket=dict(zip(BUCKETS, val_counts)))
        validation.append(row)
        emit('validation', **row)
        return row

    emit('started', config=cfg, initial_parameters_sha256=initial_hash, torch=torch.__version__,
         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if cuda else 'cpu',
         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
         parameters=sum(p.numel() for p in model.parameters()))
    evaluate(0)
    calls = cfg['calls']
    eval_calls = {c for c in (*EVAL_CALLS, calls) if c <= calls} | set(cfg.get('extra_eval_calls', ()))
    diag_calls = eval_calls | {1, 2, 11}
    updates, diagnostics = [], []
    status, initial_gradient_hash = 'completed', None
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    def sync():
        if cuda:
            torch.cuda.synchronize()

    for call in range(1, calls + 1):
        set_lr(parts, multiplier(call))
        sync()
        begin = time.perf_counter()
        loss, norm = backward(call)
        sync()
        middle = time.perf_counter()
        if not (math.isfinite(loss) and math.isfinite(norm)):
            status = 'diverged'
            emit('diverged', call=call, loss=loss, norm=norm)
            break
        if call == 1:
            initial_gradient_hash = tensor_digest((n, p.grad) for n, p in model.named_parameters())
            emit('initial_gradient', loss=loss, norm=norm, sha256=initial_gradient_hash)
        checked = call in diag_calls
        if checked:  # diagnostic calls are excluded from the timing summary
            snapshot = {k: (p.detach().clone(), p.grad.detach().clone()) for k, p in eh.items()}
        part_ms = {}
        for key, optimizer in parts.items():
            t0 = time.perf_counter()
            optimizer.step()
            sync()
            part_ms[key] = (time.perf_counter() - t0) * 1000
        finish = time.perf_counter()
        row = dict(call=call, loss=loss, norm=norm, fb_ms=(middle - begin) * 1000,
                   optimizer_ms=sum(part_ms.values()), part_ms=part_ms, diagnostic=checked)
        updates.append(row)
        emit('update', **row)
        if checked:
            entry = dict(call=call)
            for key, p in eh.items():
                group = parts[key].param_groups[0]
                entry[key] = group_diagnostics(snapshot[key][0], p.detach(), snapshot[key][1], group['lr'],
                                               group['weight_decay'], parts[key], p, bucket)
            diagnostics.append(entry)
            write_json(out / 'diagnostics.json', diagnostics)
            del snapshot
        if not all(bool(torch.isfinite(p).all()) for p in eh.values()):
            status = 'diverged'
            emit('diverged', call=call, reason='nonfinite E/H parameters')
            break
        if call in eval_calls:
            row = evaluate(call)
            if not math.isfinite(row['ce']):
                status = 'diverged'
                break
    clean = [r for r in updates if r['call'] > 100 and not r['diagnostic']] or updates[1:] or updates

    def mean(key, sub=None):
        values = [r[key][sub] if sub else r[key] for r in clean]
        return statistics.mean(values) if values else None

    result = dict(
        status=status, name=cfg['name'], config=cfg, seed=cfg['seed'], calls_requested=calls,
        calls_completed=len(updates), initial_parameters_sha256=initial_hash,
        initial_gradient_sha256=initial_gradient_hash, validation=validation,
        final_ce=validation[-1]['ce'] if status == 'completed' and validation[-1]['call'] == calls else math.inf,
        timing=dict(calls_averaged=len(clean), step_ms=mean('fb_ms') + mean('optimizer_ms') if clean else None,
                    step_ms_median=statistics.median(r['fb_ms'] + r['optimizer_ms'] for r in clean) if clean else None,
                    fb_ms=mean('fb_ms'), optimizer_ms=mean('optimizer_ms'),
                    part_ms={k: mean('part_ms', k) for k in parts}),
        state_bytes={k: state_bytes(o) for k, o in parts.items()},
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if cuda else None,
        parameters=sum(p.numel() for p in model.parameters()),
        elapsed_seconds=time.monotonic() - start, torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if cuda else 'cpu')
    if status == 'completed' and cfg.get('save_weights', not cfg.get('tiny') and not cfg.get('probe')):
        # Final FP32 model weights (no optimizer state) for later per-token / row analyses.
        weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(weights, out / 'final_weights.tmp')
        (out / 'final_weights.tmp').replace(out / 'final_weights.pt')
        result['final_weights'] = str(out / 'final_weights.pt')
    write_json(out / 'result.json', result)
    emit('finished', status=status, final_ce=result['final_ce'], step_ms=result['timing']['step_ms'])
    events.close()
    return result


# ------------------------------------------------------------------------- run lists

def tuning_configs():
    configs = {}
    for group in ('H', 'E'):
        for opt in ('adamw', 'soap'):
            for beta2 in BETA2S:
                for lr in GRID[group]:
                    spec = dict(opt=opt, lr=lr, beta2=beta2)
                    E, H = (BASE, spec) if group == 'H' else (spec, BASE)
                    cfg = make_config(E, H, TUNE_SEED)
                    configs[cfg['name']] = cfg
    return list(configs.values())


def load_results():
    results = {}
    if RUNS_DIR.exists():
        for path in RUNS_DIR.glob('*/result.json'):
            r = json.loads(path.read_text())
            results[r['name']] = r
    return results


def score(result):
    if result is None or result['status'] != 'completed':
        return math.inf
    return result['final_ce']


def family_cells(results, group, opt):
    """Seed-204 full-horizon cells varying `group` with the other group at BASE."""
    other = 'H' if group == 'E' else 'E'
    cells = []
    for r in results.values():
        c = r['config']
        if (r['seed'] == TUNE_SEED and c['calls'] == HORIZON and c[other] == BASE
                and c[group]['opt'] == opt and not c.get('probe')):
            cells.append(r)
    return cells


def best_specs(results):
    best = {}
    for group in ('E', 'H'):
        for opt in ('adamw', 'soap'):
            cells = family_cells(results, group, opt)
            if cells:
                winner = min(cells, key=score)
                best[(group, opt)] = dict(spec=winner['config'][group], ce=score(winner), name=winner['name'])
    return best


def edge_configs(results):
    """Pre-registered edge rule: if a (group, optimizer)'s best LR is the largest (smallest) LR
    tried, add the next factor-2 point outward at both beta2 values, at most MAX_EDGE_STEPS points
    past the original grid. (A diverged LR above the best is itself 'tried', so no extension
    ever goes past a diverged LR.)"""
    configs = []
    for (group, opt), best in best_specs(results).items():
        cells = family_cells(results, group, opt)
        tried = sorted({c['config'][group]['lr'] for c in cells})
        lr = best['spec']['lr']
        candidates = []
        if lr == tried[-1] and lr * 2 <= GRID[group][-1] * 2 ** MAX_EDGE_STEPS * 1.001:
            candidates.append(lr * 2)
        if lr == tried[0] and lr / 2 >= GRID[group][0] / 2 ** MAX_EDGE_STEPS * .999:
            candidates.append(lr / 2)
        for new_lr in candidates:
            for beta2 in BETA2S:
                spec = dict(opt=opt, lr=float(f'{new_lr:.4g}'), beta2=beta2)
                E, H = (BASE, spec) if group == 'H' else (spec, BASE)
                configs.append(make_config(E, H, TUNE_SEED))
    return [c for c in configs if c['name'] not in results]


def require_families(results):
    best = best_specs(results)
    missing = [f'{g}/{o}' for g in ('E', 'H') for o in ('adamw', 'soap')
               if (g, o) not in best or not math.isfinite(best[(g, o)]['ce'])]
    if missing:
        raise RuntimeError(f'no finished finite tuning cell for {missing}; fix before combining')
    return best


def combine_configs(results):
    """Seed-204 run of each arm's combination: best E setting for its E optimizer with the best H
    setting for its H optimizer (each tuned with the other group at base AdamW)."""
    best = require_families(results)
    configs = []
    for arm, (opt_E, opt_H) in ARMS.items():
        cfg = make_config(best[('E', opt_E)]['spec'], best[('H', opt_H)]['spec'], TUNE_SEED)
        cfg['arm'] = arm
        configs.append(cfg)
    return configs


def lr_cap_flags(results, spec, group):
    """'high'/'low' if spec's LR is the largest/smallest LR tried in its (group, optimizer) family."""
    tried = sorted({c['config'][group]['lr'] for c in family_cells(results, group, spec['opt'])})
    if not tried or len(tried) < 2:
        return None
    if spec['lr'] >= tried[-1]:
        return 'high'
    if spec['lr'] <= tried[0]:
        return 'low'
    return None


def arm_candidates(results, arm):
    """Every finished seed-204 full-horizon cell whose (E optimizer, H optimizer) pair is the arm's:
    tuning cells of both families plus the combination."""
    opt_E, opt_H = ARMS[arm]
    return [r for r in results.values()
            if r['seed'] == TUNE_SEED and r['config']['calls'] == HORIZON and not r['config'].get('probe')
            and r['config']['E']['opt'] == opt_E and r['config']['H']['opt'] == opt_H]


def arm_choice(results):
    """Pre-registered: each arm is confirmed at its seed-204 argmin over arm_candidates."""
    choice = {}
    for arm in ARMS:
        candidates = arm_candidates(results, arm)
        if candidates:
            choice[arm] = min(candidates, key=score)
    return choice


def confirm_configs(results):
    choice = arm_choice(results)
    missing = [arm for arm in ARMS if arm not in choice or not math.isfinite(score(choice[arm]))]
    if missing:
        raise RuntimeError(f'no finite seed-{TUNE_SEED} cell for arms {missing}')
    configs = []
    for arm, winner in choice.items():
        for seed in CONFIRM_SEEDS:
            cfg = make_config(winner['config']['E'], winner['config']['H'], seed)
            cfg['arm'] = arm
            configs.append(cfg)
    return configs


def probe_configs(calls=150):
    soap = dict(opt='soap', lr=6e-4, beta2=.999)
    configs = []
    for arm in ('A0', 'A2'):  # A0/A2 cover all four E/H optimizer parts
        opt_E, opt_H = ARMS[arm]
        E = soap if opt_E == 'soap' else BASE
        H = soap if opt_H == 'soap' else BASE
        cfg = make_config(E, H, TUNE_SEED, calls=calls, suffix='_probe')
        cfg.update(arm=arm, probe=True, extra_eval_calls=[])
        configs.append(cfg)
    return configs


def stage_configs(stage, results=None):
    results = load_results() if results is None else results
    if stage == 'probe':
        return probe_configs()
    if stage == 'tune':
        return tuning_configs()
    if stage == 'edge':
        return edge_configs(results)
    if stage == 'combine':
        return combine_configs(results)
    if stage == 'confirm':
        return confirm_configs(results)
    raise ValueError(stage)


# ---------------------------------------------------------------------------- launcher

def gpu_claim_processes():
    """[(pid, owner, exclusive, holding)] for every live `gpu-claim run` process."""
    holders = {}
    lock_dir = Path(os.environ.get('GPU_CLAIM_LOCKS', '/var/lock/gpu-claim'))
    for path in lock_dir.glob('gpu*.lock'):
        try:
            text = path.read_text().strip()
            record = json.loads(text) if text else None
        except (OSError, json.JSONDecodeError):
            record = None
        if record and 'pid' in record:
            holders[int(record['pid'])] = record
    output = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True).stdout
    rows = []
    for line in output.splitlines():
        pid_text, _, args = line.strip().partition(' ')
        words = args.split()
        # Only the gpu-claim program itself (`gpu-claim run ...` or `python3 /path/gpu-claim run ...`),
        # not shells or wrappers whose command line merely mentions it.
        index = next((i for i, w in enumerate(words[:2]) if w == 'gpu-claim' or w.endswith('/gpu-claim')), None)
        if index is None or len(words) <= index + 1 or words[index + 1] != 'run':
            continue
        owner = None
        for i, word in enumerate(words):
            if word == '--owner' and i + 1 < len(words):
                owner = words[i + 1]
            elif word.startswith('--owner='):
                owner = word.split('=', 1)[1]
        pid = int(pid_text)
        rows.append((pid, owner, '--share' not in args, pid in holders))
    return rows


def must_defer(defer_owner):
    """Defer while the other owner has a waiter, or holds a GPU exclusively (timing runs)."""
    if not defer_owner:
        return False, ''
    for pid, owner, exclusive, holding in gpu_claim_processes():
        if owner == defer_owner and not holding:
            return True, f'{defer_owner} waiter pid {pid}'
        if owner == defer_owner and holding and exclusive:
            return True, f'{defer_owner} exclusive holder pid {pid}'
    return False, ''


def launch(args):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log = (REPORT_DIR / 'launcher.log').open('a')
    stages = (args.stages.split(',') if args.stages else
              ['tune', 'edge', 'combine', 'confirm'] if args.auto else [args.stage])
    assert all(st in ('probe', 'tune', 'edge', 'combine', 'confirm') for st in stages), stages
    attempts, failed = {}, set()
    running = {}  # name -> (Popen, cfg)

    def say(message):
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {message}'
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()

    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    stage_index, edge_rounds, deferring = 0, 0, False
    say(f'launcher start stages={stages} max_concurrent={args.max_concurrent} defer={args.defer_to}')
    if stages == ['probe']:  # one claim, four arms run sequentially on the same GPU
        while True:
            defer, reason = must_defer(args.defer_to)
            if not defer:
                break
            say(f'probe deferred: {reason}')
            time.sleep(args.poll)
        command = ['gpu-claim', 'run', '--owner', 'untied-head', '--job', 'probe', '--wait',
                   '--timeout', str(args.claim_timeout), '--', PYTHON, str(Path(__file__).resolve()), 'probe']
        code = subprocess.run(command, env=dict(os.environ, OMP_NUM_THREADS='4')).returncode
        say(f'probe exit {code}')
        return
    while stage_index < len(stages) and not stopping:
        stage = stages[stage_index]
        results = load_results()
        try:
            configs = stage_configs(stage, results)
        except RuntimeError as error:
            say(f'HALT before stage {stage}: {error}')
            summarize(argparse.Namespace())
            break
        pending = [c for c in configs if score(results.get(c['name'])) == math.inf
                   and (results.get(c['name']) or {}).get('status') != 'diverged'
                   and c['name'] not in running and c['name'] not in failed]
        for name, (process, cfg) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            del running[name]
            result = load_results().get(name)
            if result is None:
                attempts[name] = attempts.get(name, 0) + 1
                say(f'{name}: exit {code} without result (attempt {attempts[name]})')
                if attempts[name] >= 2:
                    failed.add(name)
            else:
                say(f'{name}: {result["status"]} final_ce={result["final_ce"]:.4f}')
        defer, reason = must_defer(args.defer_to)
        if defer != deferring:
            say(f'deferring to {args.defer_to}: {reason}' if defer else f'resuming (no {args.defer_to} waiter/exclusive holder)')
            deferring = defer
        if defer:
            # Withdraw our own not-yet-holding waiters so the deferred owner claims first.
            for pid, owner, _exclusive, holding in gpu_claim_processes():
                if owner == 'untied-head' and not holding:
                    for name, (process, _cfg) in list(running.items()):
                        if process.pid == pid:
                            process.terminate()
                            process.wait()
                            del running[name]
                            say(f'{name}: waiter withdrawn ({reason})')
        else:
            waiting = sum(1 for pid, owner, _e, holding in gpu_claim_processes()
                          if owner == 'untied-head' and not holding)
            if pending and len(running) < args.max_concurrent and waiting == 0:
                cfg = pending[0]
                path = Path(cfg['out'])
                path.mkdir(parents=True, exist_ok=True)
                write_json(path / 'config.json', cfg)
                stdout = (path / 'stdout.log').open('a')
                command = ['gpu-claim', 'run', '--owner', 'untied-head', '--job', cfg['name'][:60], '--wait',
                           '--timeout', str(args.claim_timeout), '--', PYTHON, str(Path(__file__).resolve()),
                           'run', '--config', str(path / 'config.json')]
                process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT,
                                           env=dict(os.environ, OMP_NUM_THREADS='4'))
                running[cfg['name']] = (process, cfg)
                say(f'{stage}: submitted {cfg["name"]} ({len(pending) - 1} more pending)')
        if not pending and not running:
            unfinished = [c['name'] for c in configs if c['name'] not in load_results()]
            if unfinished:
                say(f'HALT: stage {stage} has runs without a result: {unfinished}')
                summarize(argparse.Namespace())
                break
            if stage == 'edge' and edge_configs(load_results()) and edge_rounds < 2 * MAX_EDGE_STEPS:
                edge_rounds += 1  # loop guard only; edge_configs itself caps the LR steps
                continue
            say(f'stage {stage} complete')
            summarize(argparse.Namespace())
            stage_index += 1
            continue
        time.sleep(args.poll)
    for name, (process, _cfg) in running.items():
        if stopping:
            process.terminate()
    say(f'launcher exit failed={sorted(failed)} stopping={bool(stopping)}')


# ------------------------------------------------------------------------- summarize

def fmt(value, digits=4):
    return '—' if value is None or (isinstance(value, float) and not math.isfinite(value)) else f'{value:.{digits}f}'


def summarize(_args):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    lines = ['# Untied head/embedding study — generated tables', '',
             f'Generated {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}; selection metric: '
             f'validation CE at call {HORIZON}, seed {TUNE_SEED}.', '']
    best = best_specs(results)
    for group in ('H', 'E'):
        for opt in ('adamw', 'soap'):
            cells = family_cells(results, group, opt)
            if not cells:
                continue
            lrs = sorted({c['config'][group]['lr'] for c in cells})
            lines += [f'## Tuning: {group} with {opt} (other group at base AdamW 6e-4/0.999)', '',
                      '| beta2 | ' + ' | '.join(lr_tag(lr) for lr in lrs) + ' |',
                      '|---|' + '---|' * len(lrs)]
            for beta2 in BETA2S:
                row = []
                for lr in lrs:
                    match = [c for c in cells if c['config'][group]['lr'] == lr and c['config'][group]['beta2'] == beta2]
                    if not match:
                        row.append('')
                        continue
                    r = match[0]
                    at2k = next((v['ce'] for v in r['validation'] if v['call'] == 2000), None)
                    text = 'diverged' if r['status'] == 'diverged' else f"{fmt(r['final_ce'])} ({fmt(at2k, 3)}@2k)"
                    row.append(text)
                lines.append(f'| {beta2} | ' + ' | '.join(row) + ' |')
            if (group, opt) in best:
                lines += ['', f"Best: {spec_tag(best[(group, opt)]['spec'])} → {fmt(best[(group, opt)]['ce'])}"]
            lines.append('')
    # Arms are matched by (E, H) config, not by a stored tag: the seed-204 run of an arm can be a
    # tuning cell. Seed 204 = pre-registered argmin over arm_candidates; fresh seeds reuse its config.
    arms = {}
    for arm, winner in arm_choice(results).items():
        E, H = winner['config']['E'], winner['config']['H']
        arms[arm] = {seed: results.get(run_name(E, H, seed)) for seed in (TUNE_SEED, *CONFIRM_SEEDS)}
        arms[arm] = {seed: r for seed, r in arms[arm].items() if r is not None}
    summary = dict(best={f'{g}/{o}': v for (g, o), v in best.items()},
                   arm_configs={a: dict(E=w['config']['E'], H=w['config']['H'], seed204_ce=score(w),
                                        candidates=len(arm_candidates(results, a)))
                                for a, w in arm_choice(results).items()}, arms={})
    if arms:
        lines += ['## Arms at their best seed-204 cell (full 10k horizon)', '',
                  '| arm | E | H | seed | final CE | Δ vs A0 | step ms | E opt ms | H opt ms | peak GiB |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for arm in ARMS:
            for seed, r in sorted(arms.get(arm, {}).items()):
                base = arms.get('A0', {}).get(seed)
                delta = r['final_ce'] - base['final_ce'] if base and math.isfinite(score(base)) else None
                t = r['timing']
                tag = ' (selection)' if seed == TUNE_SEED else ''
                lines.append(f"| {arm} | {spec_tag(r['config']['E'])} | {spec_tag(r['config']['H'])} | {seed}{tag} | "
                             f"{fmt(r['final_ce'])} | {fmt(delta, 4) if delta is not None else '—'} | "
                             f"{fmt(t['step_ms'], 1)} | {fmt(t['part_ms']['E'], 2)} | {fmt(t['part_ms']['H'], 2)} | "
                             f"{fmt((r['peak_gpu_bytes'] or 0) / 2**30, 2)} |")
        lines += ['', '### Verdicts (fresh seeds only; threshold ±%.3f CE, both seeds same sign)' % VERDICT_THRESHOLD, '']
        fresh = {}
        for arm in ARMS:
            deltas = []
            for seed in CONFIRM_SEEDS:
                a, b = arms.get(arm, {}).get(seed), arms.get('A0', {}).get(seed)
                if a and b:
                    deltas.append(score(a) - score(b))
            if len(deltas) == len(CONFIRM_SEEDS):
                fresh[arm] = deltas
        for arm, deltas in fresh.items():
            if arm == 'A0':
                continue
            mean_delta = statistics.mean(deltas)
            if mean_delta <= -VERDICT_THRESHOLD and all(d < 0 for d in deltas):
                verdict = 'better than best-tuned AdamW'
            elif mean_delta >= VERDICT_THRESHOLD and all(d > 0 for d in deltas):
                verdict = 'worse than best-tuned AdamW'
            else:
                verdict = 'no resolvable difference'
            caps = {g: lr_cap_flags(results, arms[arm][CONFIRM_SEEDS[0]]['config'][g], g) for g in ('E', 'H')}
            caps_A0 = {g: lr_cap_flags(results, arms['A0'][CONFIRM_SEEDS[0]]['config'][g], g) for g in ('E', 'H')}
            capped = [f'{arm} {g} at {c} edge' for g, c in caps.items() if c] + \
                     [f'A0 {g} at {c} edge' for g, c in caps_A0.items() if c]
            if capped:
                verdict += f' (LR-capped, provisional: {", ".join(capped)})'
            summary['arms'][arm] = dict(deltas=deltas, mean_delta=mean_delta, verdict=verdict, lr_capped=capped)
            lines.append(f'- {arm}: Δ = {", ".join(fmt(d) for d in deltas)} (mean {fmt(mean_delta)}) → **{verdict}**')
        if all(a in fresh for a in ('A1', 'A2', 'A3')):
            mean = {a: statistics.mean(fresh[a]) for a in ('A1', 'A2', 'A3')}
            summary['factorial'] = dict(H_effect=mean['A1'], E_effect=mean['A3'],
                                        interaction=mean['A2'] - mean['A1'] - mean['A3'])
            lines.append(f"- 2×2 (fresh-seed means): H effect {fmt(mean['A1'])}, E effect {fmt(mean['A3'])}, "
                         f"interaction {fmt(mean['A2'] - mean['A1'] - mean['A3'])}")
        a0 = [score(r) for s, r in arms.get('A0', {}).items()]
        if len(a0) >= 2:
            lines.append(f'- A0 seed spread (max−min over {len(a0)} seeds): {fmt(max(a0) - min(a0))}')
        lines.append('')
        calls_list = [0, *EVAL_CALLS]
        lines += ['### Validation CE at every eval call', '',
                  '| call | ' + ' | '.join(f'{a} s{s}' for a in ARMS for s in sorted(arms.get(a, {}))) + ' |',
                  '|---|' + '---|' * sum(len(arms.get(a, {})) for a in ARMS)]
        for call in calls_list:
            cells = []
            for a in ARMS:
                for s in sorted(arms.get(a, {})):
                    v = next((x['ce'] for x in arms[a][s]['validation'] if x['call'] == call), None)
                    cells.append(fmt(v))
            lines.append(f'| {call} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    full = [r for r in results.values() if not r['config'].get('probe') and not r['config'].get('tiny')]
    hash_check = {}
    for seed in sorted({r['seed'] for r in full}):
        same = [r for r in full if r['seed'] == seed]
        hash_check[seed] = dict(runs=len(same),
                                initial_parameters=len({r['initial_parameters_sha256'] for r in same}) == 1,
                                initial_gradient=len({r['initial_gradient_sha256'] for r in same}) == 1)
    summary['hash_check'] = hash_check
    if hash_check:
        lines += ['Same-seed hash check (one initial-parameter / call-1 gradient hash per seed): ' +
                  ', '.join(f"s{s}: {v['runs']} runs, params {'ok' if v['initial_parameters'] else 'MISMATCH'}, "
                            f"grads {'ok' if v['initial_gradient'] else 'MISMATCH'}" for s, v in hash_check.items()), '']
    probes = [r for r in results.values() if r['config'].get('probe')]
    if probes:
        lines += ['## Timing probe (150 calls, same GPU, sequential)', '',
                  '| arm | step ms | fb ms | opt ms | matrix ms | E ms | H ms | rest ms | peak GiB | E+H state MiB |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for r in sorted(probes, key=lambda r: r['config']['arm']):
            t = r['timing']
            lines.append(f"| {r['config']['arm']} | {fmt(t['step_ms'], 1)} | {fmt(t['fb_ms'], 1)} | {fmt(t['optimizer_ms'], 1)} | "
                         f"{fmt(t['part_ms']['matrix'], 1)} | {fmt(t['part_ms']['E'], 2)} | {fmt(t['part_ms']['H'], 2)} | "
                         f"{fmt(t['part_ms']['rest'], 2)} | {fmt((r['peak_gpu_bytes'] or 0) / 2**30, 2)} | "
                         f"{fmt((r['state_bytes']['E'] + r['state_bytes']['H']) / 2**20, 0)} |")
        lines.append('')
    (REPORT_DIR / 'tables.md').write_text('\n'.join(lines) + '\n')
    write_json(REPORT_DIR / 'summary.json', summary)
    print('\n'.join(lines))


# --------------------------------------------------------------------------- selftest

def selftest(_args):
    torch.manual_seed(0)
    torch.set_num_threads(2)
    # 1. Without a factor, soap.SOAP is AdamW one call late (first call only observes).
    p_soap = torch.randn(40, 24)
    p_adam = p_soap.clone()
    p_soap.requires_grad_(True)
    p_adam.requires_grad_(True)
    soap = SOAP([dict(params=[p_soap], lr=1e-2, weight_decay=.05)], betas=(.9, .95), eps=1e-8,
                **{**EH_SOAP, 'max_precond_dim': 1})
    adam = torch.optim.AdamW([dict(params=[p_adam], lr=1e-2, weight_decay=.05)], betas=(.9, .95), eps=1e-8)
    grads = [torch.randn(40, 24) for _ in range(30)]
    p_soap.grad = torch.randn(40, 24)
    soap.step()
    assert soap.state[p_soap]['GG'] == [None, None]
    for i, g in enumerate(grads):
        scale = (i + 1) / 30
        for opt in (soap, adam):
            opt.param_groups[0]['lr'] = 1e-2 * scale
        p_soap.grad, p_adam.grad = g.clone(), g.clone()
        soap.step()
        adam.step()
    error = float((p_soap - p_adam).abs().max())
    assert error < 1e-5, error
    print(f'[ok] factorless soap.SOAP == AdamW one call late (max abs diff {error:.2e})')

    # 2. One-sided structure and math against a manual reference.
    V, d = 500, 32
    p = torch.randn(V, d, requires_grad=True)
    reference = p.detach().clone()
    soap = SOAP([dict(params=[p], lr=1e-3, weight_decay=.05)], betas=(.9, .999), eps=1e-8,
                **{**EH_SOAP, 'precondition_frequency': 10 ** 6})
    mix = torch.randn(d, d)
    sparse_rows = torch.rand(V) < .2
    gs = [(torch.randn(V, d) @ mix) * sparse_rows[:, None] for _ in range(8)]
    p.grad = gs[0].clone()
    soap.step()
    state = soap.state[p]
    assert state['GG'][0] is None and state['Q'][0] is None
    assert tuple(state['GG'][1].shape) == (d, d) and tuple(state['exp_avg_sq'].shape) == (V, d)
    C = (1 - .95) * gs[0].T @ gs[0]
    Q = torch.linalg.eigh(C).eigenvectors.flip(1)
    assert torch.allclose(state['Q'][1].abs(), Q.abs(), atol=1e-4)
    m = torch.zeros(V, d)
    v = torch.zeros(V, d)
    for step, g in enumerate(gs[1:], start=1):
        p.grad = g.clone()
        soap.step()
        rotated = g @ state['Q'][1]
        m = .9 * m + .1 * rotated
        v = .999 * v + .001 * rotated.square()
        u = (m / (1 - .9 ** step)) / ((v / (1 - .999 ** step)).sqrt() + 1e-8)
        reference = reference * (1 - 1e-3 * .05) - 1e-3 * (u @ state['Q'][1].T)
    error = float((p.detach() - reference).abs().max())
    assert error < 1e-5, error
    print(f'[ok] one-sided SOAP: GG=[None,{d}x{d}], V per element {V}x{d}, matches manual update ({error:.2e})')

    # 3. QR10 cadence: every refresh is a hard reset; basis stays orthogonal.
    p = torch.randn(V, d, requires_grad=True)
    soap = SOAP([dict(params=[p], lr=1e-3, weight_decay=0.)], betas=(.9, .95), eps=1e-8, **EH_SOAP)
    for _ in range(41):
        p.grad = torch.randn(V, d) @ mix
        soap.step()
    state = soap.state[p]
    assert state['basis_hard_reset_events'] == 4 and state['basis_refreshes'] == 4, state['basis_hard_reset_events']
    q = state['Q'][1]
    assert float((q.T @ q - torch.eye(d)).norm()) < 1e-4
    print('[ok] QR every 10 calls (4 hard resets in 40 updates), Q orthogonal')

    # 4. Grouping: tied paper model gives the shared matrix no decay; untied H decays, E does not.
    tied = Transformer(dim=64, depth=2, heads=2, ff_mult=2., ff_hidden_dim=128, vocab_size=1000,
                       max_seq_len=64, tie_embeddings=True)
    groups = grouped_parameters(tied, dict(matrix_lr=1., lr=1., weight_decay=.05))
    tied_wd = [g['weight_decay'] for g in groups for n in g['param_names'] if n == 'token_embedding.weight']
    assert tied_wd == [0.] and not any('lm_head' in n for g in groups for n in g['param_names'])
    print('[ok] tied model: shared embedding/head matrix is "token_embedding.weight" with wd 0')
    try:
        from owt_restart import multiplier as paper_multiplier
        assert all(multiplier(c) == paper_multiplier(c) for c in range(1, HORIZON + 1))
        print('[ok] schedule identical to owt_restart.multiplier')
    except ImportError as error:
        print(f'[skip] owt_restart import failed ({error}); schedule formula copied verbatim')

    # 5. Tiny end-to-end arms on CPU (real data rows, truncated to 64 tokens).
    hashes = set()
    soap_spec = dict(opt='soap', lr=1e-3, beta2=.95)
    root = Path(os.environ.get('UNTIED_SELFTEST_DIR', '/tmp/untied_head_selftest'))
    for arm, (opt_E, opt_H) in ARMS.items():
        E = soap_spec if opt_E == 'soap' else BASE
        H = soap_spec if opt_H == 'soap' else BASE
        cfg = make_config(E, H, TUNE_SEED, calls=24, suffix='_tiny')
        cfg.update(tiny=True, device='cpu', compile=False, val_rows=16, arm=arm,
                   out=str(root / f'{arm}_{cfg["name"]}'), extra_eval_calls=[12])
        result_path = Path(cfg['out']) / 'result.json'
        if result_path.exists():
            result_path.unlink()
        result = run(cfg)
        assert result['status'] == 'completed' and math.isfinite(result['final_ce']), result['status']
        hashes.add((result['initial_parameters_sha256'], result['initial_gradient_sha256']))
        diag = json.loads((Path(cfg['out']) / 'diagnostics.json').read_text())
        assert {d['call'] for d in diag} >= {1, 2, 11, 12, 24}
        for key, opt in (('E', opt_E), ('H', opt_H)):
            assert ('soap_factor' in diag[-1][key]) == (opt == 'soap')
        print(f"[ok] tiny {arm} E={opt_E} H={opt_H}: CE {result['validation'][0]['ce']:.3f} -> "
              f"{result['final_ce']:.3f}; E rare-row update rms "
              f"{diag[-1]['E']['buckets'].get('30k+', {}).get('update_row_rms_mean')}")
    assert len(hashes) == 1, 'arms must share initial parameters and call-1 gradients'
    print('[ok] identical initial-parameter and call-1 gradient hashes across the four arms')
    print('selftest passed')


# ------------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('selftest')
    p_run = sub.add_parser('run')
    p_run.add_argument('--config', type=Path, required=True)
    sub.add_parser('probe')
    p_plan = sub.add_parser('plan')
    p_plan.add_argument('--stage', choices=('probe', 'tune', 'edge', 'combine', 'confirm'), default='tune')
    p_launch = sub.add_parser('launch')
    p_launch.add_argument('--stage', choices=('probe', 'tune', 'edge', 'combine', 'confirm'), default='tune')
    p_launch.add_argument('--auto', action='store_true', help='tune -> edge rounds -> combine -> confirm')
    p_launch.add_argument('--stages', help='comma list, e.g. tune,edge (overrides --stage/--auto)')
    p_launch.add_argument('--max-concurrent', type=int, default=4)
    p_launch.add_argument('--defer-to', default='refresh-study')
    p_launch.add_argument('--claim-timeout', type=int, default=259200)
    p_launch.add_argument('--poll', type=float, default=20.)
    sub.add_parser('summarize')
    args = parser.parse_args()
    if args.command == 'selftest':
        selftest(args)
    elif args.command == 'run':
        run(json.loads(args.config.read_text()))
    elif args.command == 'probe':
        for cfg in probe_configs():
            path = Path(cfg['out'])
            path.mkdir(parents=True, exist_ok=True)
            write_json(path / 'config.json', cfg)
            subprocess.run([PYTHON, str(Path(__file__).resolve()), 'run', '--config', str(path / 'config.json')],
                           check=True)
        summarize(args)
    elif args.command == 'plan':
        configs = stage_configs(args.stage)
        for cfg in configs:
            print(cfg['name'])
        print(f'{len(configs)} runs in stage {args.stage}')
    elif args.command == 'launch':
        launch(args)
    elif args.command == 'summarize':
        summarize(args)


if __name__ == '__main__':
    main()
