"""Prospective longer horizon; no change to the completed pair's optimizer math."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

BASELINE = Path('/workspace/optimizer_replay_results/cifar_soap_replication_seed140_frozen_20261010')
ARMS = ('qr20', 'warm20')
HORIZON = 10000
OBSERVATIONS = (0, 100, 400, 875, 1000, 1750, 2625, 3500, 5000, 7500, 10000)
PAIR_SHA = '66ad868d2e1be402e20c84cd772772f75adf387c4f30ceca40c9a7c3d4eb1896'


def recipe(base):
    if base.get('seed') != 140 or base.get('matched_comparison', {}).get('steps_per_arm') != 3500:
        raise ValueError('Require completed seed140 pair recipe')
    result = copy.deepcopy(base)
    result['max_wall_seconds'] = 1200
    result['matched_comparison'].update(steps_per_arm=HORIZON,
        execution_order_policy='Same fixed pair order as seed140; longer-horizon follow-up')
    return result


def protocol(base=None):
    if base is None:
        base = json.loads((BASELINE / 'protocol.json').read_text())
    result = copy.deepcopy(base)
    result.update(scope='Seed140 longer fixed-recipe fidelity comparison', steps_per_arm=HORIZON,
        total_measured_updates=2 * HORIZON, maximum_optimizer_updates=2 * HORIZON + 25,
        observation_steps=list(OBSERVATIONS),
        execution_order_policy='Same fixed pair order as seed140; longer-horizon follow-up')
    result['cadence']['final_counts'] = {
        'qr20': dict(step=9999, research_qr_refreshes=499, research_warm_refreshes=0,
            research_strict_transports=499),
        'warm20': dict(step=9999, research_qr_refreshes=49, research_warm_refreshes=450,
            research_strict_transports=499)}
    result['bounds'].update(internal_wall_seconds=1200, external_watchdog_seconds=1260,
        maximum_evidence_bytes=200 * 2**20)
    result['artifact_policy']['expected_compact_evidence_bytes'] = 160 * 2**20
    result['implementation']['entry'] = 'source/benchmarks/cifar_longer_training.py'
    result['implementation']['private_pair_adapter'] = 'longer_training_contract.py'
    result['gate']['performance'] = 'Report fixed10000 CE and all prior endpoints; no equivalence or timing qualification'
    result['timing']['same_fixed_pair_order_not_counterbalanced'] = True
    result['lineage'] = dict(baseline_seed=140, seed=140,
        allowed_config_changes=['matched_comparison', 'max_wall_seconds'], unchanged_optimizer_math=True,
        reference='Completed seed1403500-call pair; new longer-horizon hypothesis, not context recovery',
        cross_process_prefix_exactness_required=False, within_pair_exactness_required=True)
    result['primary_endpoint'] = dict(step=HORIZON, metric='validation_cross_entropy',
        objective='Close QR learning gap with lower optimizer cost', statistical_qualification=False)
    return result


def validate_inputs(config, declared, baseline_config, baseline_protocol):
    if config != recipe(baseline_config):
        raise ValueError('Only declared horizon and resource metadata changes allowed')
    if declared != protocol(baseline_protocol):
        raise ValueError('Longer protocol differs from exact prospective contract')


def run_longer(runtime, declared):
    path = Path(__file__).with_name('replication_training_contract.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != PAIR_SHA:
        raise ValueError('Pinned pair orchestration changed')
    spec = importlib.util.spec_from_file_location('_longer_private_pair', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    emit = runtime.emit
    def annotated(event, payload):
        if event == 'finished':
            payload['interpretation'] = 'Longer fixed-recipe paired fidelity observations; no equivalence or timing winner'
        emit(event, payload)
    runtime.emit = annotated
    try:
        return base.run_replication(runtime, declared)
    finally:
        runtime.emit = emit
