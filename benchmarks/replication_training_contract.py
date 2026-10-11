"""Explicit seed140 pair; stdlib-only until runtime functions are called."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

BASELINE = Path('/workspace/optimizer_replay_results/cifar_soap_cadence_frozen_20261010')
ARMS = ('qr20', 'warm20')
SEED = 140
CONTRACT_SHA = '32903f3cde3c18c23cc810ac96fc83fd7d563e34380224b092ab46b05bd76926'
RUNTIME_SHA = '126f77143a4622cc9278092198f9e9744282d9e449c1835dad5e5e3ce68a293e'


def recipe(base):
    if base.get('seed') != 139:
        raise ValueError('Require the completed seed139 recipe as lineage')
    result = copy.deepcopy(base)
    result['seed'] = SEED
    result['matched_comparison'] = dict(arms=list(ARMS), steps_per_arm=3500,
        execution_order=list(ARMS), execution_order_policy='Reversed pair order relative to seed139',
        independent_seed=SEED)
    return result


def protocol(base=None):
    if base is None:
        base = json.loads((BASELINE / 'protocol.json').read_text())
    result = copy.deepcopy(base)
    result.update(scope='Independent seed140 QR20/warm20 confirmation', arms=list(ARMS),
        execution_order=list(ARMS), execution_order_seed=None,
        execution_order_policy='Reversed pair order relative to seed139', independent_seed=SEED,
        total_measured_updates=7000, all_arm_preflight_backward_calls=2,
        maximum_optimizer_updates=7025)
    result['cadence'].pop('warm10')
    result['cadence']['final_counts'].pop('warm10')
    result['implementation']['entry'] = 'source/benchmarks/cifar_replication_training.py'
    result['implementation']['private_pair_adapter'] = 'replication_training_contract.py'
    result['timing']['randomized_order_not_counterbalanced'] = False
    result['timing']['reversed_pair_order_not_counterbalanced'] = True
    result['gate']['performance'] = 'Report both independent seeds; no statistical or timing qualification'
    # Previous three-arm evidence was 75.83 MiB; budget two-arm scalar/guard evidence honestly.
    result['artifact_policy']['expected_compact_evidence_bytes'] = 70 * 2**20
    result['lineage'] = dict(baseline_seed=139, independent_seed=SEED,
        allowed_config_changes=['seed', 'matched_comparison'], unchanged_optimizer_math=True,
        reference='Completed seed139 QR20/warm20/warm10 screen; no old entry execution')
    return result


def validate_inputs(config, declared, baseline_config, baseline_protocol):
    if config != recipe(baseline_config):
        raise ValueError('Only declared seed and pair-study config changes allowed')
    if declared != protocol(baseline_protocol):
        raise ValueError('Replication protocol differs from exact prospective contract')


def private_module(filename, digest, name):
    path = Path(__file__).with_name(filename)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Pinned pair dependency changed: ' + filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair_preupdate_gate(snapshot, arms, observe, *, order):
    if tuple(order) != ARMS:
        raise ValueError('Only declared reversed pair order allowed')
    base = private_module('cadence_training_contract.py', CONTRACT_SHA, '_replication_private_contract')
    base.ARMS = ARMS
    return base.preupdate_gate(snapshot, arms, observe, order=order)


def run_replication(runtime, declared):
    if tuple(declared['execution_order']) != ARMS:
        raise ValueError('Undeclared replication execution order')
    base = private_module('cadence_training_runtime.py', RUNTIME_SHA, '_replication_private_runtime')
    base.execution_order = lambda: ARMS
    base.preupdate_gate = pair_preupdate_gate
    # The unchanged factory and per-parameter counter guard already support both arms.
    return base.run_cadence(runtime, declared)
