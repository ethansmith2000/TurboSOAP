"""CPU check for the C6 front-loaded arm: FrontLoadedSOAP(qr_until=S) equals CompiledSOAP(10, 10) bit for bit until
the first post-switch refresh, then follows warm10/QR50 on the global age grid; the runner's event_kind_for agrees
with the optimizer every call; the PB8 hook samples pre/post/mid without touching the trajectory."""
import sys, torch
sys.path[:0] = ['/workspace/TurboSOAP/benchmarks', '/workspace/TurboSOAP']
import bf16_storage_screen as bss
import fastqr_training as ft
from bf16_soap import CompiledSOAP
from transformer import Transformer

S = 30
ft.ARMS['t_fl30'] = dict(cls='frontloaded', frequency=10, repair=50, qr_until=S, pb8=True)


def model():
    torch.manual_seed(0)
    return Transformer(dim=64, depth=2, heads=4, ff_mult=2., ff_hidden_dim=128, vocab_size=97, max_seq_len=32,
                       tie_embeddings=True, use_rope=True, qk_norm=True, bias=False, dropout=0., fused_qkv=True,
                       fused_swiglu=True, gradient_checkpointing=False)


def reference(m, repair):
    groups = bss.grouped_parameters(m, dict(matrix_lr=.0015, lr=.0006, weight_decay=.05))
    matrix = CompiledSOAP([g for g in groups if g['role'] == 'matrix'], state_dtype=torch.bfloat16,
                          basis_dtype=torch.bfloat16, precondition_frequency=10, hard_reset_interval=repair,
                          compile_kernels=False)
    aux = torch.optim.AdamW([g for g in groups if g['role'] == 'auxiliary'], betas=(.9, .999), eps=1e-8)
    return bss.SplitOptimizer(matrix, aux)


def run(builder, steps, hook):
    m = model()
    opt = builder(m)
    g = torch.Generator().manual_seed(1)
    params, events = [], []
    for call in range(1, steps + 1):
        hook(opt, 1.0)
        x = torch.randint(0, 97, (4, 32), generator=g)
        loss = torch.nn.functional.cross_entropy(m(x)[:, :-1].reshape(-1, 97).float(), x[:, 1:].reshape(-1))
        loss.backward()
        opt.step(); opt.zero_grad()
        events.append(opt.matrix.last_event)
        params.append([p.detach().clone() for p in m.parameters()])
    return params, events


def first_diff(a, b):
    return next((i + 1 for i, (x, y) in enumerate(zip(a, b)) if not all(torch.equal(u, v) for u, v in zip(x, y))), None)


plain_hook = bss.apply_lr_multiplier
ft.install(bss, 't_fl30')
fl, ev = run(lambda m: bss.build_storage_optimizer(m, 't_fl30', compile_kernels=False), 130, bss.apply_lr_multiplier)
q10, ev10 = run(lambda m: reference(m, 10), 130, plain_hook)
w50, ev50 = run(lambda m: reference(m, 50), 130, plain_hook)
print('front-loaded refresh events:', [(i + 1, e) for i, e in enumerate(ev) if e not in ('ordinary', 'cold')][:12])
print('runner event_kind_for agrees every call:', all(ft.event_kind_for('t_fl30', c) == e for c, e in enumerate(ev, 1)))
print('events == QR10 through call', S + 1, ':', ev[:S + 1] == ev10[:S + 1],
      '| events == warm10/QR50 after call', S + 1, ':', ev[S + 1:] == ev50[S + 1:])
d = first_diff(fl, q10)
print('params bit-equal to CompiledSOAP(10,10) through call', (d or 131) - 1, '(expected', S + 10, 'i.e. first diff at the post-switch warm, call', S + 11, ')')
pb8 = ft.HOLDER['pb8']
print('pb8 samples', len(pb8), 'error', ft.HOLDER.get('pb8_error'), 'kinds', sorted({r['kind'] for r in pb8}))
for r in pb8[:7]:
    print('  ', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
# PB8 hook is read-only: rerun without the hook must give identical params
ft.HOLDER['pb8'] = []
fl2, _ = run(lambda m: bss.build_storage_optimizer(m, 't_fl30', compile_kernels=False), 130, plain_hook)
print('pb8 hook leaves trajectory unchanged:', first_diff(fl, fl2) is None)
