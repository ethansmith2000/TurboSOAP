"""Private profiler scopes around unchanged expressions; no persistent patches."""
import ast
from contextlib import contextmanager
import inspect
import textwrap
import types

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from soap import SOAP
from soap_reference import ResearchSOAP
import movement_retraction as movement
import warm_retraction as retraction


def scope(name):
    return torch.profiler.record_function('refresh/' + name)


class Scopes(ast.NodeTransformer):
    def __init__(self, prefix):
        self.prefix = prefix

    def wrap(self, node):
        node = self.generic_visit(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            return node
        target = ast.unparse(node.targets[0]) if isinstance(node, ast.Assign) else (
            ast.unparse(node.target) if isinstance(node, ast.AugAssign) else type(node).__name__)
        label = self.prefix + '/' + target + '/L' + str(node.lineno)
        return ast.With(items=[ast.withitem(context_expr=ast.Call(
            func=ast.Name(id='_profile_scope', ctx=ast.Load()),
            args=[ast.Constant(label)], keywords=[]))], body=[node])

    visit_Assign = wrap
    visit_AugAssign = wrap
    visit_Expr = wrap
    visit_Return = wrap

    def visit_For(self, node):
        node = self.generic_visit(node)
        if self.prefix == 'refresh_dispatch' and ast.unparse(node.target) == '(axis, (cov, q))':
            label = ast.parse("'factor/axis' + str(axis) + '/n' + str(q.shape[0] if q is not None else 0)", mode='eval').body
            node.body = [ast.With(items=[ast.withitem(context_expr=ast.Call(
                func=ast.Name(id='_profile_scope', ctx=ast.Load()), args=[label], keywords=[]))], body=node.body)]
        if self.prefix == 'retraction' and ast.unparse(node.target) == 'index':
            label = ast.parse("'retraction/pass' + str(index)", mode='eval').body
            node.body = [ast.With(items=[ast.withitem(context_expr=ast.Call(
                func=ast.Name(id='_profile_scope', ctx=ast.Load()), args=[label], keywords=[]))], body=node.body)]
        return node


class EraseScopes(ast.NodeTransformer):
    def visit_With(self, node):
        node = self.generic_visit(node)
        if len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Call):
            call = node.items[0].context_expr
            if isinstance(call.func, ast.Name) and call.func.id == '_profile_scope':
                return node.body
        return node


def annotated(fn, prefix, replacements=None):
    original = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    original.body[0].decorator_list = []
    transformed = ast.fix_missing_locations(Scopes(prefix).visit(ast.parse(ast.unparse(original))))
    erased = EraseScopes().visit(ast.parse(ast.unparse(transformed)))
    if ast.dump(original) != ast.dump(erased):
        raise ValueError('Profiler changed expression/control-flow AST: ' + prefix)
    namespace = {**fn.__globals__, '_profile_scope': scope, **(replacements or {})}
    exec(compile(ast.fix_missing_locations(transformed), '<refresh-profiler/' + prefix + '>', 'exec'), namespace)
    return namespace[fn.__name__]


class OperatorScopes(TorchDispatchMode):
    """Label operator shapes/dtypes using metadata only; perform each op once."""
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        tensors = [a for a in args if torch.is_tensor(a)]
        metadata = ';'.join(str(t.dtype).removeprefix('torch.') + ':' + 'x'.join(map(str, t.shape))
                            for t in tensors[:2])
        with torch.profiler.record_function('operator/' + str(func) + '/' + metadata):
            return func(*args, **(kwargs or {}))


def make_functions():
    basis = annotated(SOAP._basis_covariance, 'basis_covariance')
    private_soap = type('PrivateProfileSOAP', (SOAP,), {'_basis_covariance': staticmethod(basis)})
    retract = annotated(retraction.retract, 'retraction')
    warm = annotated(movement.warm, 'warm', {'SOAP': private_soap, 'retract': retract})
    functions = {name: annotated(getattr(ResearchSOAP, name), prefix) for name, prefix in (
        ('_research_refresh', 'refresh_dispatch'),
        ('_transport_first_moment', 'moment_transfer'),
        ('_accumulate_covariance', 'covariance_accumulation'),
        ('_project_with_basis', 'projection_forward'),
        ('_project_back_with_basis', 'projection_back'))}
    functions['_gauge_step_one'] = lambda cov, q, group: warm(
        cov, q, group, movement.POLICIES[group['research_movement_policy']])
    return functions


FUNCTIONS = make_functions()


@contextmanager
def instrument(matrix):
    """Patch only one instance during a profile call; always restore its attrs."""
    names = [n for n in FUNCTIONS if n != '_gauge_step_one' or any(
        g['basis_method'] == 'warm' for g in matrix.param_groups)]
    saved = {n: (n in matrix.__dict__, matrix.__dict__.get(n)) for n in names}
    try:
        for name in names:
            value = FUNCTIONS[name]
            setattr(matrix, name, value if name == '_gauge_step_one' else types.MethodType(value, matrix))
        with OperatorScopes():
            yield
    finally:
        for name, (present, value) in saved.items():
            if present:
                setattr(matrix, name, value)
            else:
                delattr(matrix, name)


def profiled_step(opt):
    with scope('optimizer'), instrument(opt.matrix):
        with scope('matrix'):
            opt.matrix.step()
        with scope('auxiliary'):
            opt.auxiliary.step()
