"""Research observer: aliases original prefix intermediates without extra arithmetic."""
import ast
import inspect
import math
import torch
from precision_atoms_kernel import warm_split


def build_prefix_stages():
    tree = ast.parse(inspect.getsource(warm_split))
    fn = tree.body[0]
    fn.name = 'prefix_stages'
    body = []
    for node in fn.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'gram':
            break
        body.append(node)
    body.extend(ast.parse('return dict(c=c,b=b,raw=raw,bound=bound,a=a,delta=delta,candidate=z,scale=scale)').body)
    fn.body = body
    namespace = dict(torch=torch, math=math)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(warm_split), 'exec'), namespace)
    return namespace['prefix_stages']


prefix_stages = build_prefix_stages()


def independent_direction(q, left, right):
    # Construct a skinny physical-space perturbation without a third covariance EMA.
    direction = left@right.T
    return direction/direction.norm().clamp_min(1e-24)


def scaled_perturbation(q, direction, relative_norm):
    return q+direction*(q.norm()*relative_norm/direction.norm().clamp_min(1e-24))
