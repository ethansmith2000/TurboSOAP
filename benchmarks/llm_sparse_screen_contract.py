"""Prospective two-seed cadence screen. No numerical-framework imports."""
import copy
import json
from pathlib import Path
from llm_math_learning_contract import evaluation_ids, ids_sha
from llm_feasibility_contract import CACHE, PARAMETERS, SNAPSHOT_CAP, SNAPSHOT_BYTES, SNAPSHOT_PURPOSE, DIMENSIONS
from hybrid_training_contract import expected_counts

PARENT = Path('/workspace/optimizer_replay_results/llm_soap_cubic3_learning_seed179_frozen_20261014')
SEEDS = (180, 181)
ARMS = ('qr20', 'highest_residual20', 'hold200', 'qr40')
ORDERS = {180: ARMS, 181: tuple(reversed(ARMS))}
HORIZON = 3500
WARMUP = 245
SPECS = {'qr20': ('qr', 20, 0), 'highest_residual20': ('warm', 20, 200),
         'hold200': ('qr', 200, 0), 'qr40': ('qr', 40, 0)}
EVALUATIONS = (500, 1000, 1500, 2000, 2500, 3000, 3500)
GUARDS = (1, 21, 41, 201, 501, 1001, 1501, 2001, 2501, 3001, 3500)
OWN_REFERENCE_CALLS = (21, 41, 221)  # Disposable J1 only, both incoming layouts.


def checked_seed(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Only the two prospectively declared fresh seeds are supported')
    return seed


def counts(calls, arm):
    if arm not in SPECS:
        raise ValueError('Undeclared cadence arm')
    return expected_counts(calls, *SPECS[arm])


def event_kind(call, arm):
    if type(call) is not int or call < 1 or arm not in SPECS:
        raise ValueError('Invalid training call or arm')
    if call == 1:
        return 'cold'
    age = call - 1
    method, frequency, repair = SPECS[arm]
    if age % frequency:
        return 'ordinary'
    return 'qr' if method == 'qr' or (repair and age % repair == 0) else 'warm'


def recipe(seed):
    checked_seed(seed)
    c = json.loads((PARENT / 'config.json').read_text())
    c['seed'] = seed
    c['matched_comparison'] = dict(arms=list(ARMS), steps_per_arm=HORIZON,
        execution_order=list(ORDERS[seed]), scope='Paired sparse-QR screen, not equivalence or qualified production timing')
    return c


def protocol(seed):
    checked_seed(seed)
    old = json.loads((PARENT / 'protocol.json').read_text())
    keys = ('coverage', 'bounds', 'artifacts', 'model_parity', 'llm_data', 'llm_evaluation',
            'compiler_isolation', 'evaluation_policy', 'evaluation_transaction', 'schedule')
    p = {k: copy.deepcopy(old[k]) for k in keys}
    p.update(status='PREPARED_NOT_LAUNCHED', seed=seed, arms=list(ARMS), execution_order=list(ORDERS[seed]),
        steps_per_arm=HORIZON, disposable_calls_per_arm=WARMUP, disposable_smoke_calls=25,
        maximum_optimizer_updates=15005, empty_preflight_backward_calls=4,
        evaluation_calls=list(EVALUATIONS), guard_calls=list(GUARDS), probe_calls=[],
        sampled_batch_calls=[21, 41, 201, 501, 1001, 2001, 3001, 3500], prefix_hashes=[245, 3500])
    p['expected'] = {phase: {a: counts(n, a) for a in ARMS}
        for phase, n in [('disposable', WARMUP), ('primary', HORIZON)]}
    warm = sum(counts(n, 'highest_residual20')['research_warm_refreshes'] for n in (WARMUP, HORIZON))
    p['expected'].update(hybrid_calls_by_dimension={d: k * warm for d, k in DIMENSIONS.items()},
        own_eager_factor_references=24, disposable_actual_m_transactions=20,
        primary_counterfactual_basis_calls=0, evaluations_per_arm=7)
    p['gates'] = dict(state_gram_limit=.05, absolute_gram_fro=.05, normalized_gram=.005,
        physical_m_relative=.01, reference_basis_relative=1e-4, movement_relative=.1, movement_cosine=.995,
        exact_initial_preflight=True, exact_disposable_primary_scalar_and_state_prefix=True,
        no_primary_compilation=True, compiler_limits=dict(recompile_limit=8, accumulated_recompile_limit=256),
        finite=True, exact_evaluation_restoration=True, registered_release=True)
    p['cadence'] = {a: dict(method=m, frequency=f, repair=r) for a, (m, f, r) in SPECS.items()}
    p['cadence_semantics'] = 'Call1 initializes matrices without a matrix parameter update; call k is matrix age k-1. Refresh follows update/covariance. Coincident QR replaces warm. Hold still accumulates covariance every step.'
    p['qualification'] = dict(phase='disposable only', own_eager_calls=list(OWN_REFERENCE_CALLS),
        actual_m='Highest physical worldM before/after the actual native first refresh per matrix, plus J1 first repair200; 20 transactions. No counterfactual basis banks or history oracle.',
        exact_prefix='At call245 compare entire model/optimizer/RNG endpoint and every loss/gradient norm with disposable run',
        no_primary_diagnostics=True)
    p['llm_evaluation']['panels'] = {n: dict(blocks=len(v), valid_tokens=len(v)*512, ids_sha256=ids_sha(v))
        for n, v in evaluation_ids(seed).items()}
    p['timing'] = dict(serialized_optimizer_host_and_events=True, steady_excludes_first_calls=50,
        primary_diagnostic_calls=[], complete_training_iteration_intervals=True,
        common_guard_and_evaluation_calls=True, evaluation_guard_and_provenance_overhead_separately_measured=True,
        all_in_setup_warmup_retained=True, no_repeat_qualification=True,
        interpretation='Measured iteration intervals are synchronized and exclude separately observed checks/evaluations; phase/all-in include them. Equal-wall interpolation only within common observed intervals; no uninstrumented/deployed speed claim.')
    p['artifacts']['probe'] = dict(purpose='Disposable actual-M qualification, streamed two rectangular worldM buffers and original own-eager J1 references; no persistent statistics.',
        location='GPU RAM only', expected_max_extra_bytes=12*2**30, retention='Release every native refresh on success/failure')
    p['artifacts']['compiler_cache']['retention'] = 'Arithmetic provenance through this sparse-screen review; separate inventory, not model state'
    p['artifacts']['expected_compact_bytes'] = 25*2**20
    p['memory'] = dict(maximum_tensor_purpose_bytes=12*2**30, largest_factor_workspace_bytes=32*8192**2*4,
        rectangular_m_workspace_bytes=4*8192*2048*4, borrowed_gradients_bytes=PARAMETERS*4,
        fingerprint_workspace_bytes=12*2**20, initial_snapshot_purpose_bytes=SNAPSHOT_PURPOSE,
        retention='No checkpoints/final weights/tensor files/traces. Initial snapshot RAM only, released completion/abort.')
    p['precision'] = dict(state='FP32 covariance/basis/M/V', ordinary='Original global high/TF32 context',
        warm='Unchanged J1 highest prefix/Gram, BF16 candidate/corrections, two corrections',
        momentum='Original highest FP32 world transport', variance='Warm carry, QR permutation',
        compile='Default/fullgraph=True/dynamic=False; original limits8/256')
    p['implementation'] = dict(entry='source/benchmarks/llm_sparse_screen.py',
        source_delta='Additive screen only; native step/QR/covariance/M/V and evaluator unchanged; pure QR interval200 for hold',
        model='Original common eager SDPA-MATH model; original compiled J1', cpu116_after_smoke=True)
    p['decision'] = dict(primary_question='Does intervening warm motion earn useful learning per complete cost versus pure hold200 or QR40?',
        practical_screen='Prospective utility margin0.01 CE: sparse candidate is provisional only if every post-warmup panel1000..3500 on BOTH seeds is <= J1+0.01 and QR20+0.01, with lower measured optimizer and iteration cost. Report all paired differences. Otherwise report tradeoff/unresolved; no automatic extra seed or policy.',
        limits='Two seeds are not an equivalence proof; no claim from nonsignificance, diagnostic sign counts or projected timing. No deployed/default promotion; efficient QR still unqualified.',
        further_work='Efficient QR + ordinary profiling; longer/deeper/backend confirmation only for useful survivor.')
    p['bound_rationale'] = 'Per-seed15005calls=25smoke+4*(245disposable+3500primary),28evaluations; retain3600internal/3660external,24GiBallocated,3GiBinitialCPU,100MiBcompact,512MiBcache,min5GiBfree. Previous17911260calls1924s included616extra endpoints; new screen removes those and uses lean common guards. Two one-attempt bundles, no retry or auto expansion.'
    return p


def validate(config, declared):
    seed = checked_seed(config.get('seed'))
    if config != recipe(seed) or declared != protocol(seed):
        raise ValueError('Undeclared sparse-screen policy or criterion')
