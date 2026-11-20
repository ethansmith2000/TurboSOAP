import json, torch
from ns_coeffs import apply_poly, errs, minimax
STD = (1.5, -0.5); TAY = (15/8, -10/8, 3/8)   # 3rd-order NS: y(I - E/2 + 3E^2/8)

def skew(batch, n, cap, shape, g):
    if shape == 'spread':
        a = torch.randn(batch, n, n, generator=g, dtype=torch.float64)
    elif shape.startswith('rank'):
        k = int(shape[4:]); u = torch.randn(batch, n, 2 * k, generator=g, dtype=torch.float64)
        a = u[..., :k] @ u[..., k:].mT
    else:  # 'decay': skew with geometrically decaying singular values
        u = torch.linalg.qr(torch.randn(batch, n, n, generator=g, dtype=torch.float64)).Q
        th = 0.8 ** torch.arange(n // 2, dtype=torch.float64)
        j = torch.zeros(n, n, dtype=torch.float64)
        for i, t in enumerate(th): j[2*i, 2*i+1], j[2*i+1, 2*i] = t, -t
        a = u @ j @ u.mT
    a = a - a.mT if shape != 'decay' else a
    return a * (cap / (a.norm(dim=(-2, -1), keepdim=True) / 2 ** .5))

out = []
for cap in (0.25, 0.5, 1.0):
    for shape in ('spread', 'decay', 'rank8', 'rank1'):
        g = torch.Generator().manual_seed(1)
        q = torch.linalg.qr(torch.randn(4, 512, 512, generator=g, dtype=torch.float64)).Q.to(torch.bfloat16)
        a = skew(4, 512, cap, shape, g)
        y = (q.double() + q.double() @ a.to(torch.bfloat16).double()).to(torch.bfloat16)
        row = dict(cap=cap, shape=shape, theta_max=round(torch.linalg.matrix_norm(a, ord=2).max().item(), 3),
                   input=errs(y))
        for name, steps in {'std x1 (2 mm)': [STD], 'std x2, production (4 mm)': [STD, STD],
                            '3rd-order x1 (3 mm)': [TAY], '3rd-order x2 (6 mm)': [TAY, TAY]}.items():
            z = y
            for cf in steps: z = apply_poly(z, cf, True)
            row[name] = [round(v, 5) for v in errs(z)]
        out.append(row)
        print(json.dumps(row))
# BF16 floor: an exactly orthogonal Q rounded to BF16
print('floor', errs(q))
