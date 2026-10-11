"""Audit one primary flag change and separately declared post-trajectory replay."""
import ast
try:
    from .benchmark_off_contract import validate_config,backend_delta,finite_records
except ImportError:
    from benchmark_off_contract import validate_config,backend_delta,finite_records


def validate_child_change(baseline,control):
    start='# INITIAL_STATE_REPLAY_START';end='# INITIAL_STATE_REPLAY_END'
    if control.count(start)!=1 or control.count(end)!=1:raise ValueError('Need one declared replay region')
    a=control.index(start);b=control.index(end)
    if not control.index("report['code_after_profile']")<a<b<control.index("report['peak_allocated_bytes']"):
        raise ValueError('Replay must follow all primary updates and existing final profile')
    stripped=control[:a]+control[b+len(end):]
    stripped=stripped.replace('from initial_state_replay import run_initial_replay\n','')
    tree=ast.parse(stripped);assignments=[]
    class RemoveFlag(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and ast.unparse(node.targets[0])=='torch.backends.cudnn.deterministic':
                assignments.append(node)
                return None
            return self.generic_visit(node)
    tree=RemoveFlag().visit(tree)
    if len(assignments)!=1 or not isinstance(assignments[0].value,ast.Constant) or assignments[0].value.value is not True:
        raise ValueError('Exactly one explicit cudnn.deterministic=True assignment')
    if ast.dump(ast.parse(baseline))!=ast.dump(tree):raise ValueError('Primary source changes beyond the single declared flag')
    return True
