"""Prospective fresh native-trainer integration feasibility; no quality claim."""
import copy
from pathlib import Path

BASELINE = Path('/workspace/optimizer_replay_results/cifar_soap_refresh_profile_frozen_20261010')
PRECISION_BASELINE = Path('/workspace/optimizer_replay_results/soap_common_y_fixture_frozen_20261011')
ARMS = ('qr20', 'warm20', 'hybrid20')
HORIZON = 420
WARMUP = 245
PROBE_CALLS = (21, 281)


def expected_counts(calls, method, frequency, reset):
    age=calls-1;events=age//frequency
    repairs=events if method=='qr' else age//reset if reset else 0
    return dict(step=age,research_qr_refreshes=repairs,
        research_warm_refreshes=events-repairs,research_strict_transports=events)


def recipe(base):
    c = copy.deepcopy(base)
    c.update(seed=141, max_wall_seconds=900)
    c['matched_comparison'] = dict(arms=list(ARMS), steps_per_arm=HORIZON,
        execution_order=list(ARMS), scope='Compiled hybrid actual-trainer feasibility only')
    return c


def protocol():
    return dict(status='PREPARED_NOT_LAUNCHED', seed=141, arms=list(ARMS), execution_order=list(ARMS),
        steps_per_arm=HORIZON, disposable_smoke_calls=25, disposable_calls_per_arm=WARMUP,
        maximum_optimizer_updates=25+len(ARMS)*(WARMUP+HORIZON), empty_preflight_backward_calls=3,
        schedule=dict(total=35000, lr_warmup=875), probe_calls=list(PROBE_CALLS),
        sampled_batch_calls=[21,281,401],
        prefix_hashes=[1,20,21,200,201,280,281,400,401,420],
        coverage=dict(matrix_parameters=4, two_sided_factors=8,
            dimensions_per_refresh={'1536':5,'4608':1,'6144':2}, matrix_max_axis=10000),
        precision=dict(basis_state='FP32', covariance_basis='FP32 high', candidate='BF16 Q@A only',
            retraction='highest FP32 cubic2', momentum='highest FP32 unchanged',
            default_compile_fullgraph=True, dynamic=False, variance='carry warm; QR permutation unchanged',
            common_backend_delta={'allow_bf16_reduced_precision_reduction':False},
            reference='Eager FP32 high-prefix/highest-retraction hybrid on exact same actual refresh C,Q'),
        warmup=dict(purpose='Exercise actual model and all ordinary/warm/QR repair paths and input layouts',
            optimizer='Disposable independent optimizer per arm; fresh empty optimizer after release',
            reset='Exact model/module flags/global+crop+augmentation RNG restored before each phase',
            covers_repair_at_call=201, covers_post_repair_warm_calls=[221,241],
            all_in_charged=True, steady_only_excludes_disposable=True),
        expected=dict(primary={a:expected_counts(HORIZON,'qr' if a=='qr20' else 'warm',20,0 if a=='qr20' else 200) for a in ARMS},
            disposable={a:expected_counts(WARMUP,'qr' if a=='qr20' else 'warm',20,0 if a=='qr20' else 200) for a in ARMS},
            hybrid_kernel_calls=232, hybrid_calls_by_dimension={'1536':145,'4608':29,'6144':58},
            diagnostic_reference_calls=16),
        gates=dict(exact_initial_preflight=True, actual_first_matches_preflight=True,
            all_state_finite=True, state_gram_limit=.05, mixed_basis_relative=1e-3,
            movement_relative=.1, movement_cosine=.995, diagnostic_gram=.005,
            unchanged_original_moment_and_qr_paths=True, exact_pair_data_rng_plan=True,
            no_graph_breaks=True, no_steady_recompilation=True, full_registered_release=True,
            no_speed_or_learning_qualification=True),
        bounds=dict(internal_wall_seconds=900,external_watchdog_seconds=960,
            maximum_peak_allocated_bytes=24*2**30,maximum_initial_snapshot_purpose_cpu_bytes=256*2**20,
            maximum_evidence_bytes=50*2**20,maximum_compiler_cache_bytes=512*2**20,
            minimum_free_disk_bytes=5*2**30,queued=False,retries=0),
        artifacts=dict(checkpoints=False,final_weights=False,tensor_files=False,profiler_traces=False,
            initial_snapshot=dict(purpose='Reset fresh smoke/disposable warmup/preflight/primary branches',
                location='CPU RAM only',expected_bytes=114641296,purpose_expected_bytes=237631888,
                retention='Release completion/abort'), optimizer_snapshots=False,
            probe=dict(purpose='Same actual refresh covariance/Q against matched eager FP32 hybrid',
                location='GPU RAM only, one factor at a time',expected_max_extra_bytes=7*6144**2*4,
                retention='Release after each factor before next ordinary step'),
            compiler_cache=dict(purpose='Generated arithmetic/graph provenance, no model state',
                expected_bytes=150*2**20,retention='Through next integration review'),expected_compact_bytes=10*2**20),
        timing=dict(serialized_optimizer_host_and_events=True,diagnostic_calls_excluded=list(PROBE_CALLS),
            includes_precision_context_casts_boundary=True,all_in_setup_warmup_retained=True,
            no_repeat_qualification=True,primary_within_lr_warmup=True),
        implementation=dict(entry='source/benchmarks/cifar_hybrid_training.py',private_supervisor_only=True,
            cpu116_after_smoke=True,original_masks_restored=True),
        remaining_before_launch=['Bind a currently free GPU index/UUID once under a nonqueued lifetime claim',
            'Check actual private-supervisor environment and launch hashes; never start historical programs'])


def validate(config, declared, base):
    if config != recipe(base) or declared != protocol():
        raise ValueError('Undeclared hybrid recipe/protocol delta')
