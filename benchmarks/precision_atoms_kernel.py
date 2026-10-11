"""Separately opt-in BF16 atoms and an explicitly observer-bearing stage oracle."""
import ast
import inspect
import math
import torch


def warm_split(cov, q, cov_mixed=False, candidate_mixed=False):
    scale = (cov.trace().abs() / cov.shape[0]).clamp_min(1e-12)
    c = cov.float() / scale
    if cov_mixed or candidate_mixed:
        qb = q.to(torch.bfloat16)
    if cov_mixed:
        cb = c.to(torch.bfloat16)
        b = (qb.T @ cb @ qb).float()
    else:
        b = q.float().T @ c @ q.float()
    b = .5 * (b + b.T)
    diagonal = b.diagonal()
    gap = diagonal[:, None] - diagonal[None, :]
    tau = diagonal.abs().amax().clamp_min(1e-12) * .01
    omega = -b * gap / (gap.square() + tau.square()).clamp_min(1e-24)
    raw = .5 * (.5 * (omega - omega.T))
    bound = raw.norm() / math.sqrt(2)
    a = raw * (.5 / bound.clamp_min(1e-12)).clamp(max=1)
    if candidate_mixed:
        delta = (qb @ a.to(torch.bfloat16)).float()
    else:
        delta = q.float() @ a
    z = q.float() + delta
    gram = z.T @ z
    z = 1.5 * z - .5 * (z @ gram)
    gram = z.T @ z
    return 1.5 * z - .5 * (z @ gram)


def build_stages():
    # Insert ONLY references to existing intermediate values; no arithmetic edits.
    tree = ast.parse(inspect.getsource(warm_split))
    fn = tree.body[0]; fn.name = 'warm_stages'
    body = []
    gram_count = 0
    for node in fn.body:
        body.append(node)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'gram':
                gram_count += 1
                body.extend(ast.parse(f'gram{gram_count} = gram').body)
            if node.targets[0].id == 'z':
                alias = 'candidate' if gram_count == 0 else 'first'
                body.extend(ast.parse(f'{alias} = z').body)
        if isinstance(node, ast.Return):
            body.pop()
            body.extend(ast.parse('qnew = 1.5 * z - .5 * (z @ gram)\nreturn dict(c=c, b=b, raw=raw, a=a, delta=delta, candidate=candidate, gram1=gram1, first=first, gram2=gram2, qnew=qnew)').body)
    fn.body = body
    namespace = dict(torch=torch, math=math)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(warm_split), 'exec'), namespace)
    return namespace['warm_stages']


warm_stages = build_stages()
