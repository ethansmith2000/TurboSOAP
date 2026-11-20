"""Four consecutive warm retractions from a BF16 basis (as between QR repairs).
Products counted per refresh side, excluding the 2 for B = Q^T C Q."""
import json, torch
from ns_coeffs import apply_poly, errs
from ns_taylor import skew
STD = (1.5, -0.5); TAY = (15/8, -10/8, 3/8)
bf = lambda x: x.to(torch.bfloat16).double()

def retract(q, a, scheme):
    qd = q.double()
    if scheme.startswith('first-order'):           # production seed: Q + QA
        y = (qd + qd @ bf(a)).to(torch.bfloat16)
    else:                                          # exp seed: Q (I + A + A^2/2)
        a2 = bf(bf(a) @ bf(a))                     # one product, FP32 accum, BF16 operand
        p = torch.eye(a.shape[-1], dtype=torch.float64) + a + .5 * a2
        y = (qd @ bf(p)).to(torch.bfloat16)
    steps = {'first-order + 2 NS (production, 5 mm)': [STD, STD], 'first-order + 1 NS (C2, 3 mm)': [STD],
             'first-order + 3rd-order NS (4 mm)': [TAY], 'exp2 seed, no NS (2 mm)': [],
             'exp2 seed + 1 NS (4 mm)': [STD]}[scheme]
    for cf in steps: y = apply_poly(y, cf, True)
    return y

schemes = ['first-order + 2 NS (production, 5 mm)', 'first-order + 1 NS (C2, 3 mm)',
           'first-order + 3rd-order NS (4 mm)', 'exp2 seed, no NS (2 mm)', 'exp2 seed + 1 NS (4 mm)']
for cap in (0.5,):
    for shape in ('spread', 'decay', 'rank8', 'rank1'):
        g = torch.Generator().manual_seed(2)
        q0 = torch.linalg.qr(torch.randn(4, 512, 512, generator=g, dtype=torch.float64)).Q.to(torch.bfloat16)
        As = [skew(4, 512, cap, shape, g) for _ in range(4)]
        row = dict(cap=cap, shape=shape)
        for s in schemes:
            q = q0; trace = []
            for a in As:
                q = retract(q, a, s); trace.append(errs(q))
            row[s] = dict(frob_after4=round(trace[-1][0], 5), spec_max=round(max(t[1] for t in trace), 5))
        print(json.dumps(row))
