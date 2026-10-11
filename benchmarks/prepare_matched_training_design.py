"""Freeze a CPU-tested comparison DESIGN; never launch or import a GPU trainer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'soap.py', 'soap_reference.py', 'benchmarks/cifar_optimizers.py',
    'benchmarks/cifar_optimizer_benchmark.py', 'benchmarks/movement_retraction.py',
    'benchmarks/warm_retraction.py', 'benchmarks/legacy_soap.py',
    'benchmarks/reference/soap_may_930a5a3.py',
    'benchmarks/matched_training_contract.py', 'benchmarks/prepare_matched_training_design.py',
    'benchmarks/execution_fingerprints.py', 'benchmarks/launch_environment_contract.py',
    'benchmarks/launch_environment_wire_contract.py',
    'benchmarks/cifar_support/cifar100_qualification.py',
    'benchmarks/cifar_support/cifar100_strong_augmentation.py',
    'benchmarks/cifar_support/vit.py', 'tests/test_matched_training_contract.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def protocol():
    return dict(status='FROZEN_DESIGN_NOT_LAUNCHABLE',
        scope='Common CIFAR recipe, policy bundles; neither blog reproduction nor isolated refresh ablation',
        arms=['qr20', 'may_literal', 'fro_cubic2_cap05', 'qr20_repeat'],
        steps_per_arm=3500, total_measured_updates=14000, disposable_qr10_smoke_updates=25,
        all_arm_preflight_backward_calls=4, maximum_optimizer_updates=14025,
        observation_steps=[0, 100, 400, 875, 1000, 1750, 2625, 3500],
        schedule=dict(steps_per_epoch=175, original_total_updates=35000,
                      warmup_updates=875, short_horizon_does_not_compress_schedule=True),
        process=dict(shared_model=True, shared_compiled_forward=True, shared_batch_plan=True,
                     independent_optimizer_states=True, only_one_populated_optimizer_at_a_time=True,
                     selected_device=None, seeded_batch_order=2140,
                     global_seed=139, crop_seed=1140, mix_seed=3140),
        gate=dict(exact=True, all_arms_before_any_comparison_step=True,
                  observed=['full_indices', 'full_input', 'full_target', 'global_crop_augmentation_RNG',
                            'full_output_loss', 'all_gradients_before_and_after_clip', 'clip_norm'],
                  actual_training_first_preupdate_must_match_preflight=True,
                  matched_hash_family_selection=False, retries=0,
                  qr_repeat=['sampled_update_loss_and_model_hashes', 'full_validation_metrics',
                             'final_model_full_optimizer_and_RNG_hashes'],
                  quality_qualification='Requires all four complete, exact preflight and QR-repeat gates',
                  comparison='Single-seed exploratory validation deltas; no statistical winner/default promotion'),
        guards=dict(nonfinite_abort=True, refresh_boundary_checks=True,
                    maximum_gram_error=.05, full_finite_state_check_at_observations=True,
                    live_resource_checks_every_updates=50),
        bounds=dict(internal_wall_seconds=3600, external_watchdog_seconds=3660,
                    maximum_peak_allocated_bytes=24*2**30,
                    maximum_initial_snapshot_purpose_cpu_bytes=256*2**20,
                    maximum_evidence_bytes=100*2**20, queued=False, retries=0),
        artifact_policy=dict(checkpoints=False, final_weights=False, tensor_files=False,
            profiler_traces=False,
            initial_model_snapshot=dict(purpose='Exact paired initial-weight resets across arms and preflight',
                location='CPU RAM only', expected_bytes=114641296,
                retention='Only until final comparison fingerprints or abort; release on all exits'),
            optimizer_snapshots=False,
            retained=['source/config/provenance/seeds', 'compact hashes', 'metrics/logs', 'inventories']),
        timing=dict(first_updates_excluded_from_steady_optimizer_summary=50,
            exclusions=['compile', 'disposable smoke', 'preflight', 'evaluation', 'hashing', 'online guards'],
            excluded_costs_reported=True, all_in_wall_includes_all_excluded_costs=True,
            stop_clock_not_allowed_for_wall_cap=True,
            interpretation='Instrumented ordered-arm cost diagnostic; no qualified timing speedup claim'),
        historical_recipe=dict(recalled_shampoo_beta=.999, recalled_adam_beta=.999,
            adam_beta_interpretation='Provisionally beta2', tentative_lr=.0006,
            beta1='Unknown for the blog 50k-step run; May .95 default is not evidence',
            trainer='Unavailable; CIFAR data/model/recipe differ from the blog LLM'),
        remaining_before_launch=['Common GPU runner wiring and CPU source-scope/abort tests',
            'May-compatible all-state finite/geometry and registered storage ownership checks',
            'Final runnable source/environment/supervisor freeze and free-device binding',
            'Actual supervisor inheritance precheck, nonqueued claim, one bounded managed execution'])


def validate_config(config, baseline):
    removed = {'prefix_steps', 'branch_steps', 'execution_diagnostic'}
    expected = {k: v for k, v in baseline.items() if k not in removed}
    expected['max_wall_seconds'] = 3600
    expected['matched_comparison'] = dict(arms=protocol()['arms'], steps_per_arm=3500)
    if json.dumps(config, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('Matched design changes more than declared study/horizon fields')
    if config['checkpoint'] or config['official_test']:
        raise ValueError('No weights or official test evaluation')
    return True


def verify(bundle):
    bundle = Path(bundle)
    m = read(bundle / 'manifest.json')
    for name, expected in m['sources'].items():
        if sha(bundle / 'source' / name) != expected:
            raise ValueError('Frozen source drift: ' + name)
    for name, expected in m['artifacts'].items():
        if sha(bundle / name) != expected:
            raise ValueError('Frozen artifact drift: ' + name)
    validate_config(read(bundle / 'config.json'), read(bundle / 'reference/config.json'))
    if read(bundle / 'protocol.json') != protocol():
        raise ValueError('Protocol drift')
    return m


def prepare(out, baseline):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU preparation requires CUDA_VISIBLE_DEVICES empty')
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preparation requires uninitialized CUDA')
    def forbidden(*args, **kwargs):
        raise RuntimeError('CPU preparation attempted CUDA initialization')
    torch.cuda._lazy_init = forbidden
    sys.path[:0] = [str(ROOT), str(ROOT/'benchmarks'), str(ROOT/'benchmarks/cifar_support')]
    from matched_training_contract import (ARMS, RandomStreams, SharedInitialState,
        build_arm, optimizer_metadata, assert_fresh_independent)
    import cifar100_qualification as support
    from types import SimpleNamespace
    import numpy as np

    baseline = Path(baseline).resolve()
    bm = read(baseline / 'manifest.json')
    for name, expected in bm['source_sha256'].items():
        if sha(baseline / 'source' / name) != expected:
            raise ValueError('Baseline source drift: ' + name)
    for name, field in [('config.json', 'config_sha256'), ('split.json', 'split_sha256'),
                        ('data_prepared.json', 'base_prepared_sha256')]:
        if sha(baseline/name) != bm[field]:
            raise ValueError('Baseline artifact drift: ' + name)
    for path, expected in read(baseline/'data_prepared.json')['data'].items():
        if sha(path) != expected:
            raise ValueError('Pinned dataset changed')
    out = Path(out).resolve()
    out.mkdir(exist_ok=False)
    (out/'reference').mkdir(); (out/'source').mkdir()
    for name in ('config.json', 'split.json', 'data_prepared.json', 'manifest.json', 'environment_contract.json'):
        shutil.copy2(baseline/name, out/'reference'/name)
    original = read(baseline/'config.json')
    config = {k: v for k, v in original.items() if k not in {'prefix_steps', 'branch_steps', 'execution_diagnostic'}}
    config.update(max_wall_seconds=3600, matched_comparison=dict(arms=list(ARMS), steps_per_arm=3500))
    validate_config(config, original)
    write(out/'config.json', config); write(out/'protocol.json', protocol())
    for name in SOURCES:
        destination = out/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name in bm['source_sha256'] and sha(ROOT/name) != bm['source_sha256'][name]:
            raise ValueError('Previously frozen shared source changed: ' + name)
        shutil.copy2(ROOT/name, destination)

    # Shape/policy audit only, no optimizer update, compilation, dataset decode or CUDA.
    torch.set_num_threads(4); torch.manual_seed(config['seed'])
    model = support.build_model(config)
    streams = RandomStreams(torch.Generator().manual_seed(1140),
        SimpleNamespace(rng=np.random.default_rng(3140), counts={'mixup': 0, 'cutmix': 0}))
    # Three logical input copies bound hashing and batch RAM beyond largest-gradient workspace.
    extra = 3*config['batch_size']*3*32*32*4 + 3*config['batch_size']*config['classes']*4
    snapshot = SharedInitialState(model, streams, extra_workspace_bytes=extra)
    opts = [build_arm(model, config, arm) for arm in ARMS]
    assert_fresh_independent(opts)
    policies = [optimizer_metadata(opt, arm) for arm, opt in zip(ARMS, opts)]
    audit = dict(scope='CPU construction and declared shape/state budget only; no GPU compatibility claim',
        parameter_count=sum(p.numel() for p in model.parameters()),
        snapshot_bytes=snapshot.snapshot_bytes, purpose_budget_bytes=snapshot.budget_bytes,
        initial_model_sha256=snapshot.initial_digest,
        all_groups=[dict(arm=arm, groups=[dict(role=g['role'], names=g['param_names'],
            shapes=[list(p.shape) for p in g['params']]) for g in opt.param_groups])
            for arm, opt in zip(ARMS, opts)],
        cuda_initialized=torch.cuda.is_initialized(), optimizer_updates=0,
        snapshot_released=False)
    snapshot.reset(); snapshot.close()
    audit['snapshot_released'] = snapshot.weights is None and snapshot.rng is None
    write(out/'effective_policies.json', policies); write(out/'cpu_shape_audit.json', audit)
    write(out/'supervisor_contract.json', dict(status='DESIGN_ONLY_NO_PROGRAM_INSTALLED',
        private_config='/workspace/manifold_opt/supervisord.conf',
        intended_program='manifold_matched_learning', autostart=False, autorestart=False,
        claim_launcher='/workspace/bin/gpu-claim run', nonqueued=True,
        selected_device=None, runnable_command=None,
        required_env=dict(TORCHINDUCTOR_COMPILE_THREADS='4', OMP_NUM_THREADS='4',
            MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1'),
        inherited_environment_gate='Tracked environment exact after prospective UUID binding; no cache/library exemptions',
        backend_gate='JSON-wire normalized historical phases with UUID-only delta, before smoke and after measured',
        system_services_modified=False))
    artifacts = {str(p.relative_to(out)): sha(p) for p in out.rglob('*')
                 if p.is_file() and out/'source' not in p.parents}
    write(out/'manifest.json', dict(status='FROZEN_DESIGN_NOT_LAUNCHABLE',
        baseline=str(baseline), baseline_manifest_sha256=sha(baseline/'manifest.json'),
        sources={name: sha(out/'source'/name) for name in SOURCES}, artifacts=artifacts,
        immutable_scope='All listed files; report and next-task notes separate'))
    verify(out)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.baseline), indent=2))
