import torch, torch._inductor.config as cfg
from torch._inductor.utils import run_and_get_code
def f(params, grads):
    p, g = torch.stack(params), torch.stack(grads)
    gb = g.to(torch.bfloat16)
    h = torch.bmm(gb, gb.transpose(-1, -2))
    new_p = p - 0.01 * h.float()
    for t, v in zip(params, new_p.unbind(0)): t.copy_(v)
    return h
for n in (8, 12):
    ps = [torch.randn(64, 64) for _ in range(n)]; gs = [torch.randn(64, 64) for _ in range(n)]
    _, code = run_and_get_code(torch.compile(f, fullgraph=True), ps, gs)
    src = code[0]; open(f"code_{n}.py","w").write(src)
    kernels = src.count('def kernel') + src.count('cpp_fused')//2
    print(n, 'inputs: cpp_fused kernels =', len(set(l.split('(')[0].strip() for l in src.splitlines() if l.strip().startswith('cpp_fused'))),
          '| buffers allocated =', src.count('empty_strided'), '| has cat/stack copy buf:', 'cat' in src.lower())
