"""Opt-in prefix/retraction split with original arithmetic; no production change."""
import ast
import inspect
import math
import torch
from precision_atoms_kernel import warm_split


def build_prefix():
    tree=ast.parse(inspect.getsource(warm_split));fn=tree.body[0];fn.name='warm_prefix'
    body=[]
    for node in fn.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='gram':
            break
        body.append(node)
    body.append(ast.Return(value=ast.Name(id='z',ctx=ast.Load())))
    fn.body=body
    namespace=dict(math=math,torch=torch)
    exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(warm_split),'exec'),namespace)
    return namespace['warm_prefix']


warm_prefix=build_prefix()


def retract_two(y):
    z=y.float()
    gram=z.T@z
    z=1.5*z-.5*(z@gram)
    gram=z.T@z
    return 1.5*z-.5*(z@gram)
