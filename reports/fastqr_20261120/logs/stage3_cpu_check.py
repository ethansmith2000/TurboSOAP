"""CPU check: the new runner arms build exactly the optimizers of the cached runs they are compared with.
bf16_hh_qr50 == bss 'bf16_qr50' (all steps); fp32s_hh_qr10 == bss 'compiled_fp32' up to the first refresh."""
import sys, copy, torch
sys.path[:0] = ['/workspace/TurboSOAP/benchmarks', '/workspace/TurboSOAP']
import bf16_storage_screen as bss
import fastqr_training as ft
from transformer import Transformer

orig_build = bss.build_storage_optimizer


def model():
    torch.manual_seed(0)
    return Transformer(dim=64, depth=2, heads=4, ff_mult=2., ff_hidden_dim=128, vocab_size=97, max_seq_len=32,
                       tie_embeddings=True, use_rope=True, qk_norm=True, bias=False, dropout=0., fused_qkv=True,
                       fused_swiglu=True, gradient_checkpointing=False)


def run(opt_builder, case, steps):
    m = model()
    opt = opt_builder(m, case, compile_kernels=False)
    g = torch.Generator().manual_seed(1)
    params, events = [], []
    for call in range(1, steps + 1):
        x = torch.randint(0, 97, (4, 32), generator=g)
        logits = m(x)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 97).float(), x[:, 1:].reshape(-1))
        loss.backward()
        opt.step(); opt.zero_grad()
        events.append(opt.matrix.last_event)
        params.append([p.detach().clone() for p in m.parameters()])
    st = next(iter(opt.matrix.state.values()))
    return params, events, {k: v.dtype for k, v in st.items() if torch.is_tensor(v) and v.ndim >= 2}


def equal_until(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if not all(torch.equal(u, v) for u, v in zip(x, y)):
            return i  # index of first differing call (0-based)
    return len(a)


ft.install(bss, 'bf16_hh_qr50')
new50, ev50, dt50 = run(bss.build_storage_optimizer, 'bf16_hh_qr50', 120)
old50, oev50, odt50 = run(orig_build, 'bf16_qr50', 120)
print('qr50: events', [i + 1 for i, e in enumerate(ev50) if e != 'ordinary'], 'dtypes', dt50)
print('qr50 == cached bf16_qr50 config over 120 calls:', equal_until(new50, old50) == 120, ev50 == oev50)

import importlib
importlib.reload(bss)
orig_build = bss.build_storage_optimizer
ft.install(bss, 'fp32s_hh_qr10')
new10, ev10, dt10 = run(bss.build_storage_optimizer, 'fp32s_hh_qr10', 30)
oldc, evc, dtc = run(orig_build, 'compiled_fp32', 30)
print('fp32s qr10: events', [i + 1 for i, e in enumerate(ev10) if e != 'ordinary'], 'dtypes', dt10)
print('compiled_fp32 dtypes', dtc, 'events', [i + 1 for i, e in enumerate(evc) if e != 'ordinary'])
print('fp32s_hh_qr10 == compiled_fp32 through call', equal_until(new10, oldc), '(first refresh age 10 = call 11)')
