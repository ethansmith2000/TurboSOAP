"""Private parameterization of a frozen orchestrator; no old module globals changed."""
import hashlib
import importlib.util
from pathlib import Path

from cadence_training_contract import build_arm, execution_order, preupdate_gate, validate_counters

BASE_SHA = '11df8612a7c3f9a123fde81c86686bb761617efe7ed48b6d5e653876396ca2ac'


def run_cadence(runtime, protocol):
    order = execution_order()
    if tuple(protocol['execution_order']) != order:
        raise ValueError('Undeclared execution order')
    path = Path(__file__).with_name('matched_training_diagnostic_runtime.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('Pinned orchestration source changed')
    spec = importlib.util.spec_from_file_location('_cadence_private_orchestrator', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    base.ARMS = order
    base.all_arm_preupdate_gate = lambda snapshot, arms, observe: preupdate_gate(
        snapshot, arms, observe, order=order)
    old_check = base.check_state
    def checked(model, opt, maximum_gram_error=.05, *, external_call=None, phase='guard'):
        row = old_check(model, opt, maximum_gram_error, external_call=external_call, phase=phase)
        if external_call is not None:
            row['cadence_counters'] = validate_counters(opt, external_call)
        return row
    base.check_state = checked
    original_emit = runtime.emit
    def emit(event, payload):
        if event == 'finished':
            payload.update(paired_recipe_preflight_qualified=payload['status'] == 'completed',
                quality_qualified=False, statistical_quality_qualified=False,
                timing_qualified=False, qr_repeat_completed=False,
                interpretation='Single-seed paired exploratory deltas; no exact-trajectory repeat or timing winner')
        original_emit(event, payload)
    runtime.emit = emit
    try:
        return base.run_shared(runtime, protocol, factory=build_arm)
    finally:
        runtime.emit = original_emit
