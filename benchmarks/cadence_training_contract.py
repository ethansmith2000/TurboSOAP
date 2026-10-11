"""Opt-in three-arm cadence contract; existing optimizer and study code stay intact."""
import copy
import random

from cifar_optimizers import build_optimizer
from movement_retraction import configure_matrix
from matched_training_contract import (
    assert_fresh_independent, digest, forbid_steps, optimizer_metadata, wire,
)

ARMS = ('qr20', 'warm20', 'warm10')
ORDER_SEED = 6140
HORIZON = 3500
OBSERVATIONS = (0, 100, 400, 875, 1000, 1750, 2625, 3500)


def execution_order():
    result = list(ARMS)
    random.Random(ORDER_SEED).shuffle(result)
    return tuple(result)


def expected_counts(calls, method, frequency, reset):
    if type(calls) is not int or calls < 1:
        raise ValueError('Calls include the cold initializer and must be positive')
    age = calls - 1
    events = age // frequency
    resets = events if method == 'qr' else (age // reset if reset else 0)
    return dict(step=age, research_qr_refreshes=resets,
        research_warm_refreshes=events - resets, research_strict_transports=events)


def build_arm(model, config, arm):
    if arm not in ARMS:
        raise ValueError('Only the three declared cadence arms are supported')
    opt = build_optimizer(model, {**config, 'transport_precision': 'highest'}, arm)
    if arm != 'qr20':
        configure_matrix(opt.matrix, 'fro_cubic2_cap05')
    return opt


def validate_counters(opt, calls):
    rows = []
    for group in opt.matrix.param_groups:
        expected = expected_counts(calls, group['basis_method'],
            group['precondition_frequency'], group['hard_reset_interval'])
        if group.get('transport_precision') != 'highest':
            # Disposable QR10 smoke retains the previously frozen inherited precision.
            expected.pop('research_strict_transports')
        for name, p in zip(group['param_names'], group['params']):
            state = opt.matrix.state[p]
            actual = {key: int(state.get(key, 0)) for key in expected}
            if actual != expected:
                raise RuntimeError('Cadence counter mismatch: ' + name)
            rows.append(dict(parameter=name, actual=actual))
    return dict(calls=calls, parameter_count=len(rows), parameters=rows)


def preupdate_gate(snapshot, arms, observe, *, order):
    """Same exact initial gate, with an explicit three-arm order contract."""
    if tuple(name for name, _ in arms) != tuple(order) or set(order) != set(ARMS):
        raise ValueError('Missing, duplicate or reordered declared cadence arm')
    opts = [opt for _, opt in arms]
    assert_fresh_independent(opts)
    policies = [optimizer_metadata(opt, arm) for arm, opt in arms]
    rows = []
    with forbid_steps(opts):
        for arm, opt in arms:
            with snapshot.probe_scope():
                record = wire(observe(arm, opt))
                required = {'indices', 'input', 'target', 'rng_before', 'rng_after_augmentation',
                    'output', 'loss', 'gradients_before_clip', 'clip_norm',
                    'gradients_after_clip', 'rng_after_backward'}
                if set(record) != required:
                    raise ValueError('Incomplete full pre-update observation')
                names = {name for name, _ in snapshot.model.named_parameters()}
                if any(set(record[key]) != names for key in ('gradients_before_clip', 'gradients_after_clip')):
                    raise ValueError('Incomplete gradient coverage')
                if digest(snapshot.model.state_dict()) != snapshot.initial_digest:
                    raise RuntimeError('Probe mutated model state')
                assert_fresh_independent(opts)
                if [optimizer_metadata(o, label) for label, o in arms] != policies:
                    raise RuntimeError('Probe mutated optimizer policy')
                if rows and record != rows[0]['observation']:
                    raise RuntimeError('Exact all-arm pre-update comparison failed: ' + arm)
                rows.append(dict(arm=arm, observation=record))
    return dict(status='PASS_PREUPDATE_ONLY', arms=rows, optimizer_steps=0,
        initial_model_sha256=snapshot.initial_digest, snapshot_bytes=snapshot.snapshot_bytes,
        purpose_budget_bytes=snapshot.budget_bytes)


def recipe(base):
    result = copy.deepcopy(base)
    result['matched_comparison'] = dict(arms=list(ARMS), steps_per_arm=HORIZON,
        execution_order=list(execution_order()), execution_order_seed=ORDER_SEED)
    return result


def protocol():
    return dict(status='PREPARED_NOT_LAUNCHED', scope='Single-seed matched-cadence wide screen',
        arms=list(ARMS), execution_order=list(execution_order()), execution_order_seed=ORDER_SEED,
        steps_per_arm=HORIZON, total_measured_updates=3 * HORIZON,
        disposable_qr10_smoke_updates=25, all_arm_preflight_backward_calls=3,
        maximum_optimizer_updates=3 * HORIZON + 25, observation_steps=list(OBSERVATIONS),
        schedule=dict(steps_per_epoch=175, original_total_updates=35000, warmup_updates=875),
        cadence=dict(qr20=dict(frequency=20, repair=0),
            warm20=dict(frequency=20, repair=200), warm10=dict(frequency=10, repair=200),
            coincident_repair_replaces_warm=True,
            final_counts={a: expected_counts(HORIZON, 'qr' if a == 'qr20' else 'warm',
                10 if a == 'warm10' else 20, 0 if a == 'qr20' else 200) for a in ARMS}),
        gate=dict(exact_preflight_all_arms=True, actual_first_matches_preflight=True,
            same_batch_rng_recipe=True, no_qr_repeat_arm=True,
            performance='Exploratory paired deltas only; no seed uncertainty or timing qualification',
            old_failed_labels_unchanged=True, retries=0),
        guards=dict(nonfinite_abort=True, refresh_and_observation_gram_limit=.05,
            per_parameter_cadence_counters=True, resource_interval_calls=50),
        bounds=dict(internal_wall_seconds=3600, external_watchdog_seconds=3660,
            maximum_peak_allocated_bytes=24 * 2**30,
            maximum_initial_snapshot_purpose_cpu_bytes=256 * 2**20,
            maximum_evidence_bytes=100 * 2**20, queued=False, retries=0),
        artifact_policy=dict(checkpoints=False, final_weights=False, tensor_files=False,
            profiler_traces=False, optimizer_snapshots=False,
            initial_model_snapshot=dict(purpose='Exact initial reset across smoke/preflight/cadence arms',
                expected_bytes=114641296, location='CPU RAM only', retention='Release at completion/abort'),
            purpose_budget_expected_bytes=237631888, expected_compact_evidence_bytes=10 * 2**20),
        timing=dict(steady_excludes_first_calls=50, qualified_speedup=False,
            instrumented_wall=True, cadence_dependent_guard_work_included_in_wall=True,
            diagnostic_costs_not_subtracted_to_infer_uninstrumented_wall=True,
            no_matched_wall_endpoint=True, global_qr10_smoke_only=True,
            no_claim_of_policy_specific_burn_in=True, randomized_order_not_counterbalanced=True),
        implementation=dict(entry='source/benchmarks/cifar_cadence_training.py',
            private_loaded_orchestrator='matched_training_diagnostic_runtime.py',
            unchanged_shared_trainer=True, cpu116_after_smoke=True,
            original_masks_restored=True, streaming_hash_chunk_bytes=2**20,
            no_allocator_zero_gate=True),
        remaining_before_launch=['Bind free GPU index/UUID and actual private-supervisor environment',
            'Verify all frozen hashes, live resources and inherited launch environment'])
