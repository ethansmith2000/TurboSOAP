"""Open-loop refresh-quality replay + warm-step ablation (refresh study, 2026-11-20).

generate   Train a 1-layer Transformer (repo ``transformer.py``, fused QKV, SwiGLU,
           tied embeddings, RoPE, QK-norm) with the reference optimizer
           (``soap_reference.ResearchSOAP``, QR10, stock betas .95/.95/.95, via
           ``owt_restart.build_optimizer``) on cached OpenWebText, and save C_L/C_R
           of every matrix at age 0 and every 10th age.  C is the reference's FP32
           EMA rounded once to BF16 when saved (production stores a BF16 EMA).
           Optionally (--fp32) also saves FP32 C and the driver's FP32 Q.
replay     Chain refresh schedules over the saved factor trajectories (open loop:
           the covariances do not depend on the replayed bases) and record
           quality metrics before/after every refresh age.  Includes the
           one-at-a-time warm-step ablation (as warm10/QR50 chains).
align      For a --fp32 width: chain the driver's QR10 refresh offline on the
           FP32 trajectory and compare with the driver's saved Q (age alignment).
summarize  Merge per-width results into replay.json / ablation.json + figures.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import glob
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'benchmarks')]
from bf16_soap import warm_basis, qr_basis  # noqa: E402
from refresh_study_common import (warm_basis_param, eigh_basis, cholqr_basis, coordinate_matrix,  # noqa: E402
                                  residuals, gram_error, topk_overlap, highest, WARM_DEFAULTS)

BF16, FP32 = torch.bfloat16, torch.float32
DATA_ROOT = Path('/workspace/data/turbosoap_refresh_study_20261120')
REPORT = ROOT / 'reports/refresh_study_20261120'
OWT = Path('/workspace/data/tokenized/openwebtext_gpt2_u16_1024_v1')

SCHEDULES = ['qr10', 'warm10_qr50', 'qr20', 'qr50', 'eigh10', 'cholqr10', 'none']
VARIANTS = [('default', {})]
VARIANTS += [(f'damping={d}', dict(damping=d)) for d in (.001, .003, .03, .1)]
VARIANTS += [(f'cap={c}', dict(cap=c)) for c in (.25, 1.0, None)]
VARIANTS += [(f'ns={k}', dict(ns_steps=k)) for k in (0, 1, 3)]
VARIANTS += [('warm_steps=2', dict(warm_steps=2))]


def backend():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=True))
    tmp.replace(path)


# ============================================================================ generate
def load_rows(split, count, seed, shard=0):
    import pyarrow as pa
    files = sorted(glob.glob(str(OWT / split / '*.arrow')))
    table = pa.ipc.open_stream(pa.memory_map(files[shard])).read_all()
    col = table.column('input_ids').combine_chunks()
    arr = col.values.to_numpy(zero_copy_only=False).reshape(-1, 1024)
    idx = np.random.default_rng(seed).permutation(arr.shape[0])[:count]
    return np.ascontiguousarray(arr[np.sort(idx)][np.argsort(np.argsort(idx))]), dict(
        file=files[shard], rows_in_shard=int(arr.shape[0]), selected=int(count), seed=seed)


def lr_multiplier(call, warmup, horizon):
    if call <= warmup:
        return call / warmup
    return .1 + .9 * .5 * (1 + math.cos(math.pi * (call - warmup) / max(horizon - warmup, 1)))


def matrix_factors(opt_matrix):
    """[(param_name, side, dim, param)] in a fixed order."""
    out = []
    for group in opt_matrix.param_groups:
        for name, p in zip(group['param_names'], group['params']):
            out.append((name, 'L', p.shape[0], p))
            out.append((name, 'R', p.shape[1], p))
    return out


def generate(args):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformer import Transformer
    from owt_restart import build_optimizer, token_loss
    from cifar_optimizers import apply_lr_multiplier
    backend()
    d = args.width
    out = DATA_ROOT / f'd{d}'
    out.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    torch.manual_seed(args.seed)
    model = Transformer(dim=d, depth=1, heads=max(d // 64, 1), ff_mult=2., ff_hidden_dim=2 * d,
                        vocab_size=50257, max_seq_len=1024, tie_embeddings=True, use_rope=True,
                        qk_norm=True, bias=False, dropout=0., fused_qkv=True, fused_swiglu=True,
                        gradient_checkpointing=False).to(device)
    opt = build_optimizer(model, 'stock', 'qr', args.lr)
    calls = args.ages + 1
    seq = args.seq_len
    train, train_src = load_rows('train', calls * args.batch, seed=2205)
    val, val_src = load_rows('validation', args.val_rows, seed=4205)
    snaps = args.ages // 10 + 1
    factors = None
    mm = {}
    losses, times, vals = [], [], []

    def batch_tokens(rows):
        return torch.from_numpy(rows[:, :seq + 1].astype(np.int64)).to(device)

    def evaluate(age):
        model.eval()
        tot = []
        with torch.no_grad():
            for b in range(0, len(val), 8):
                t = batch_tokens(val[b:b + 8])
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION if device == 'cuda' else SDPBackend.MATH), \
                        torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                    tot.append(float(token_loss(model(t[:, :-1]), t[:, 1:])))
        model.train()
        vals.append(dict(age=age, ce=float(np.mean(tot))))
        print('val', vals[-1], flush=True)

    def snapshot(slot):
        for i, (name, side, dim, p) in enumerate(factors):
            st = opt.matrix.state[p]
            cov = st['GG'][0 if side == 'L' else 1]
            mm[('bf16', i)][slot] = cov.to(BF16).view(torch.int16).cpu().numpy().view(np.uint16)
            if args.fp32:
                mm[('fp32', i)][slot] = cov.float().cpu().numpy()
                mm[('q', i)][slot] = st['Q'][0 if side == 'L' else 1].float().cpu().numpy()

    t_start = time.time()
    for call in range(1, calls + 1):
        apply_lr_multiplier(opt, lr_multiplier(call, args.warmup, calls))
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        tokens = batch_tokens(train[(call - 1) * args.batch: call * args.batch])
        loss_sum = 0.
        for t in tokens.split(args.micro):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION if device == 'cuda' else SDPBackend.MATH), \
                    torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                loss = token_loss(model(t[:, :-1]), t[:, 1:]) * (len(t) / len(tokens))
            loss.backward()
            loss_sum += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=device == 'cuda')
        opt.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(loss_sum)
        assert math.isfinite(loss_sum)
        if factors is None:
            factors = matrix_factors(opt.matrix)
            for i, (name, side, dim, p) in enumerate(factors):
                mm[('bf16', i)] = np.lib.format.open_memmap(out / f'f{i}_bf16.npy', mode='w+', dtype=np.uint16,
                                                            shape=(snaps, dim, dim))
                if args.fp32:
                    mm[('fp32', i)] = np.lib.format.open_memmap(out / f'f{i}_fp32.npy', mode='w+', dtype=np.float32,
                                                                shape=(snaps, dim, dim))
                    mm[('q', i)] = np.lib.format.open_memmap(out / f'f{i}_driverQ_fp32.npy', mode='w+',
                                                             dtype=np.float32, shape=(snaps, dim, dim))
        ages = {opt.matrix.state[p]['step'] for _, _, _, p in factors}
        assert len(ages) == 1
        age = ages.pop()
        assert age == call - 1
        if age % 10 == 0:
            snapshot(age // 10)
        if age in args.eval_ages:
            evaluate(age)
        if call % 50 == 0 or call == 1:
            print(f'call {call} age {age} loss {loss_sum:.4f} step {times[-1]*1000:.0f}ms elapsed {time.time()-t_start:.0f}s',
                  flush=True)
    for m in mm.values():
        m.flush()
    meta = dict(width=d, ages=args.ages, snapshot_ages=list(range(0, args.ages + 1, 10)), seed=args.seed,
                factors=[dict(index=i, param=name, side=side, dim=dim) for i, (name, side, dim, _) in enumerate(factors)],
                optimizer='soap_reference.ResearchSOAP basis_method=qr (QR10) via owt_restart.build_optimizer(stock)',
                betas=(.95, .95), shampoo_beta=.95, matrix_lr=args.lr, aux_lr=.0006, weight_decay=.05,
                warmup=args.warmup, schedule='linear warmup then cosine to 0.1x over all calls',
                batch_rows=args.batch, micro=args.micro, seq_len=seq, clip=1.0, train_source=train_src,
                validation_source=val_src, dataset_manifest=str(OWT / 'MANIFEST.json'),
                losses=losses, step_seconds=times, validation=vals, elapsed_s=time.time() - t_start,
                saved_fp32=args.fp32, covariance_storage='FP32 EMA rounded once to BF16 at save',
                torch=torch.__version__, device=torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
                gpu_claim=os.environ.get('GPU_CLAIM_INDICES'), script_sha256=sha(__file__),
                model_params=sum(p.numel() for p in model.parameters()))
    write_json(out / 'meta.json', meta)
    print('done', out, f'{time.time()-t_start:.0f}s', flush=True)


# ============================================================================ replay
def refresh_kind(schedule, age):
    if age == 0:
        return 'init'
    if age % 10:
        return None
    if schedule == 'qr10':
        return 'qr'
    if schedule.startswith('warm10_qr50'):
        return 'qr' if age % 50 == 0 else 'warm'
    if schedule == 'qr20':
        return 'qr' if age % 20 == 0 else None
    if schedule == 'qr50':
        return 'qr' if age % 50 == 0 else None
    if schedule == 'eigh10':
        return 'eigh'
    if schedule == 'cholqr10':
        return 'cholqr'
    if schedule == 'none':
        return None
    raise ValueError(schedule)


def load_group(width_dir, indices, slot, device='cuda'):
    arrs = [np.load(width_dir / f'f{i}_bf16.npy', mmap_mode='r')[slot] for i in indices]
    x = torch.from_numpy(np.stack(arrs).view(np.int16)).to(device)
    return x.view(BF16)


def metric_block(c, q, evecs, ks):
    b = coordinate_matrix(c, q)
    rc, rcw, corr = residuals(b)
    over = topk_overlap(b, q, evecs, ks)
    ge = gram_error(q)
    fin = torch.isfinite(q.float()).flatten(1).all(1)
    nanify = lambda t: torch.where(fin, t, torch.full_like(t, float('nan')))
    out = dict(rC=nanify(rc), rCw=nanify(rcw), corr_rms=nanify(corr), gram=nanify(ge))
    for k, v in over.items():
        out[f'top{k}'] = nanify(v)
    return {k: v.double().cpu().numpy() for k, v in out.items()}


def replay(args):
    backend()
    torch.set_grad_enabled(False)
    width_dir = DATA_ROOT / f'd{args.width}'
    meta = json.loads((width_dir / 'meta.json').read_text())
    factors = meta['factors']
    ages = meta['snapshot_ages']
    if args.max_age is not None:
        ages = [a for a in ages if a <= args.max_age]
    groups = {}
    for f in factors:
        groups.setdefault(f['dim'], []).append(f['index'])
    chains = list(SCHEDULES) + [f'warm10_qr50|{name}' for name, _ in VARIANTS]
    variant_kw = {f'warm10_qr50|{name}': {**WARM_DEFAULTS, **kw} for name, kw in VARIANTS}
    result = dict(width=args.width, ages=ages, chains=chains, factors=factors, groups={}, open_loop=True,
                  script_sha256=sha(__file__), gpu_claim=os.environ.get('GPU_CLAIM_INDICES'))
    t_start = time.time()
    for dim, indices in sorted(groups.items()):
        ks = sorted({1, 8, max(8, dim // 8)})
        series = {ch: {ph: {} for ph in ('before', 'after')} for ch in chains}
        events = {ch: [] for ch in chains}
        chol_fail = []
        exact_default = []
        gaps = []
        qs = {}
        for slot, age in enumerate(ages):
            c = load_group(width_dir, indices, slot, device='cpu' if args.cpu else 'cuda')
            with highest():
                evals, evecs = torch.linalg.eigh(.5 * (c.float() + c.float().transpose(-1, -2)))
            evals, evecs = evals.flip(-1), evecs.flip(-1)
            lam1 = evals[:, :1].clamp_min(1e-30)
            gaps.append({str(k): (((evals[:, k - 1] - evals[:, k]) / lam1[:, 0]).double().cpu().numpy().tolist()
                                  if k < dim else None) for k in ks})
            for ch in chains:
                kind = refresh_kind(ch.split('|')[0], age)
                before = None
                if kind != 'init' and slot > 0:
                    before = metric_block(c, qs[ch], evecs, ks)
                    for key, v in before.items():
                        series[ch]['before'].setdefault(key, []).append(v)
                if kind == 'init':
                    qs[ch] = eigh_basis(c)
                elif kind == 'qr':
                    qs[ch] = qr_basis(c, qs[ch], BF16)[0]
                elif kind == 'eigh':
                    qs[ch] = eigh_basis(c)
                elif kind == 'cholqr':
                    new, _, info = cholqr_basis(c, qs[ch])
                    nonfinite = (~torch.isfinite(new.float()).flatten(1).all(1)).cpu().numpy()
                    chol_fail.append(dict(age=age, info_nonzero=(info != 0).cpu().numpy().tolist(),
                                          nonfinite=nonfinite.tolist()))
                    qs[ch] = new
                elif kind == 'warm':
                    if ch in variant_kw:
                        qs[ch] = warm_basis_param(c, qs[ch], BF16, BF16, **variant_kw[ch])
                    else:
                        qs[ch] = warm_basis(c, qs[ch], BF16, BF16)
                events[ch].append(kind)
                m = before if (kind is None and before is not None) else metric_block(c, qs[ch], evecs, ks)
                for key, v in m.items():
                    series[ch]['after'].setdefault(key, []).append(v)
                    if slot == 0:
                        series[ch]['before'].setdefault(key, []).append(np.full(len(indices), np.nan))
            # bit-exact replay check: param-copy default chain vs production chain
            exact_default.append(bool(torch.equal(qs['warm10_qr50'], qs['warm10_qr50|default'])))
            del c, evals, evecs
            if slot % 10 == 0:
                print(f'd{args.width} dim{dim} age {age} t={time.time()-t_start:.0f}s '
                      f'rC qr10={np.nanmean(series["qr10"]["after"]["rC"][-1]):.3f} '
                      f'warm={np.nanmean(series["warm10_qr50"]["after"]["rC"][-1]):.3f} '
                      f'qr50={np.nanmean(series["qr50"]["after"]["rC"][-1]):.3f}', flush=True)
        result['groups'][str(dim)] = dict(
            factor_indices=indices, ks=ks, events={ch: events[ch] for ch in chains},
            series={ch: {ph: {k: np.stack(v).tolist() for k, v in series[ch][ph].items()}
                         for ph in ('before', 'after')} for ch in chains},
            cholqr_failures=chol_fail, default_variant_bit_equal_every_age=all(exact_default),
            eig_gaps=gaps)
        del qs
        torch.cuda.empty_cache()
    result['elapsed_s'] = time.time() - t_start
    out = REPORT / 'replay_parts'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f'replay_d{args.width}.json', result)
    print('wrote', out / f'replay_d{args.width}.json', flush=True)


# ============================================================================ align
def align(args):
    """Offline FP32 QR10 chain (ResearchSOAP's 2-D refresh: argsort diag(Q^T C Q),
    QR of C Q[:, order]) from the driver's age-0 Q over the FP32 trajectory,
    compared with the driver's saved Q at every saved age."""
    backend()
    torch.set_grad_enabled(False)
    width_dir = DATA_ROOT / f'd{args.width}'
    meta = json.loads((width_dir / 'meta.json').read_text())
    dev = 'cpu' if meta['device'] == 'cpu' else 'cuda'
    rows = []
    for f in meta['factors']:
        cs = np.load(width_dir / f"f{f['index']}_fp32.npy", mmap_mode='r')
        qd = np.load(width_dir / f"f{f['index']}_driverQ_fp32.npy", mmap_mode='r')
        q = torch.from_numpy(np.array(qd[0])).to(dev)
        diffs, equal = [], []
        for slot in range(1, cs.shape[0]):
            cov = torch.from_numpy(np.array(cs[slot])).to(dev)
            order = torch.argsort((q.T @ cov @ q).diagonal(), descending=True)
            q, _ = torch.linalg.qr(cov @ q[:, order])
            ref = torch.from_numpy(np.array(qd[slot])).to(dev)
            diffs.append(float((q - ref).abs().max()))
            equal.append(bool(torch.equal(q, ref)))
        rows.append(dict(factor=f, max_abs_diff_per_age=diffs, bit_equal_all=all(equal),
                         max_abs_diff=max(diffs) if diffs else None))
        print(f['param'], f['side'], 'bit_equal_all', all(equal), 'max diff', max(diffs) if diffs else None, flush=True)
    out = REPORT / 'replay_parts'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f'align_d{args.width}.json', dict(width=args.width, rows=rows))


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--width', type=int, required=True)
    g.add_argument('--ages', type=int, default=1000)
    g.add_argument('--batch', type=int, default=32)
    g.add_argument('--micro', type=int, default=8)
    g.add_argument('--seq-len', type=int, default=1023)
    g.add_argument('--lr', type=float, default=.0015)
    g.add_argument('--warmup', type=int, default=100)
    g.add_argument('--seed', type=int, default=204)
    g.add_argument('--val-rows', type=int, default=64)
    g.add_argument('--eval-ages', type=lambda s: {int(x) for x in s.split(',')}, default={0, 250, 500, 750, 1000})
    g.add_argument('--fp32', action='store_true')
    g.add_argument('--cpu', action='store_true')
    g.add_argument('--data-root', default=None)
    r = sub.add_parser('replay')
    r.add_argument('--width', type=int, required=True)
    r.add_argument('--max-age', type=int, default=None)
    r.add_argument('--cpu', action='store_true')
    r.add_argument('--data-root', default=None)
    r.add_argument('--report-dir', default=None)
    a = sub.add_parser('align')
    a.add_argument('--width', type=int, required=True)
    a.add_argument('--data-root', default=None)
    a.add_argument('--report-dir', default=None)
    args = ap.parse_args()
    global DATA_ROOT, REPORT
    if args.data_root:
        DATA_ROOT = Path(args.data_root)
    if getattr(args, 'report_dir', None):
        REPORT = Path(args.report_dir)
    dict(generate=generate, replay=replay, align=align)[args.cmd](args)


if __name__ == '__main__':
    main()
