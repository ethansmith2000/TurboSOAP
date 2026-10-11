"""Scope audit: unchanged primary and preceding replays, one later observer."""
import ast
import copy
try:
    from .benchmark_off_contract import validate_config as prior_config,backend_delta,finite_records
except ImportError:
    from benchmark_off_contract import validate_config as prior_config,backend_delta,finite_records


def validate_config(c):
    if c['max_wall_seconds']!=900 or c['execution_diagnostic']['per_process_seconds']!=210:
        raise ValueError('New task fixed900/210s limits')
    baseline=copy.deepcopy(c);baseline['max_wall_seconds']=600
    baseline['execution_diagnostic']['per_process_seconds']=150
    return prior_config(baseline)


def validate_child_change(baseline,control):
    start='# SAVED_OPERAND_REPLAY_START';end='# SAVED_OPERAND_REPLAY_END'
    if control.count(start)!=1 or control.count(end)!=1:raise ValueError('One saved-operand observer required')
    a=control.index(start);b=control.index(end)
    if not control.index('# INITIAL_STATE_REPLAY_END')<a<b<control.index("report['peak_allocated_bytes']"):
        raise ValueError('Observer must follow preceding complete replays')
    stripped=control[:a]+control[b+len(end):]
    stripped=stripped.replace('from saved_operand_replay import run_saved_operand_replay\n','')
    if ast.dump(ast.parse(baseline))!=ast.dump(ast.parse(stripped)):
        raise ValueError('Primary or preceding replay source changed')
    return True
