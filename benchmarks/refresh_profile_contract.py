"""Prospective width1536 actual-trainer cost diagnostic, not a learning study."""
import copy
from pathlib import Path

BASELINE = Path('/workspace/optimizer_replay_results/cifar_soap_longer_seed140_frozen_20261010')
RUNS = ('qr20_plain', 'qr20_profile', 'warm20_plain', 'warm20_profile')
PROFILE_CALLS = (281, 282, 401)
HORIZON = 420


def recipe(base):
    result = copy.deepcopy(base)
    result['max_wall_seconds'] = 600
    result['matched_comparison'] = dict(arms=list(RUNS), steps_per_arm=HORIZON,
        execution_order=list(RUNS), scope='Paired plain/profile observer invariance and cost attribution')
    return result


def protocol():
    return dict(status='PREPARED_NOT_LAUNCHED', scope='Actual width1536 optimizer profile diagnostic',
        arms=list(RUNS), execution_order=list(RUNS), steps_per_arm=HORIZON,
        profile_calls=list(PROFILE_CALLS), disposable_smoke_calls=25,
        maximum_optimizer_updates=1705, empty_preflight_backward_calls=2,
        schedule=dict(total=35000, lr_warmup=875), seed=140,
        precision=dict(basis='FP32 with high/TF32 allowed', momentum='highest FP32',
            changed=False, optimizer_compiled=False, model_compiled=True),
        gates=dict(full_plain_profile_endpoint_and_loss_sequence_exact=True,
            prefix_hashes=[280, 282, 400, 401, 420], exact_first_preupdate=True,
            Gram_limit=.05, all_state_finite=True, coverage_factors=8,
            counter_check=True, registered_ownership_release=True),
        bounds=dict(internal_wall_seconds=600, external_watchdog_seconds=660,
            maximum_peak_allocated_bytes=24*2**30,
            maximum_initial_snapshot_purpose_cpu_bytes=256*2**20,
            maximum_evidence_bytes=250*2**20, queued=False, retries=0),
        artifacts=dict(checkpoints=False, final_weights=False, tensor_files=False,
            initial_snapshot=dict(purpose='Reset plain/profile trajectories and smoke/preflight',
                location='CPU RAM only', expected_bytes=114641296,
                purpose_expected_bytes=237631888, retention='Release completion/abort'),
            traces=dict(purpose='Attribute optimizer stages, dtypes/casts and kernels by factor shape',
                count=6, expected_compressed_bytes=30*2**20,
                retention='Keep compressed diagnostic evidence; delete only own compression temporary'),
            expected_compact_bytes=50*2**20),
        timing=dict(profile_windows_excluded_from_plain_means=True,
            startup_and_all_in_wall_retained=True, warmed_refresh_calls_before_profile=14,
            warm_before_profile=12, repair_before_profile=1,
            profiler_device_work_not_wall_speedup=True, no_repeat_speed_qualification=True,
            no_learning_or_precision_ranking=True),
        instrumentation=dict(private_instance_scopes_only=True,
            expression_and_control_flow_AST_erasure_exact=True,
            op_dispatch_metadata_only=True, record_shapes=False,
            profile_memory=False, with_stack=False))


def validate(config, declared, base):
    if config != recipe(base) or declared != protocol():
        raise ValueError('Undeclared profiling recipe or protocol delta')
