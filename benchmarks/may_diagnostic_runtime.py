"""Single May arm; frozen shared preflight/smoke/math and release primitives."""
try:
    from .matched_training_diagnostic_runtime import (build_arm, TensorOwnership,
        disposable_smoke, preflight, train_arm, failure_record, collect_abort_audit)
except ImportError:
    from matched_training_diagnostic_runtime import (build_arm, TensorOwnership,
        disposable_smoke, preflight, train_arm, failure_record, collect_abort_audit)

def run_may(runtime, protocol, *, factory=build_arm):
    """One code path, tested with a tiny CPU runtime; production horizon is frozen."""
    results = []; smoke_completed = None; status = 'failed'; error = None; abort_audit = None
    snapshots = TensorOwnership(); snapshots.watch_tree(runtime.snapshot.weights, 'initial_weights')
    snapshots.watch_tree(runtime.snapshot.rng, 'initial_rng')
    try:
        runtime.emit('started', dict(initial_model_sha256=runtime.snapshot.initial_digest,
            snapshot_bytes=runtime.snapshot.snapshot_bytes, purpose_budget_bytes=runtime.snapshot.budget_bytes))
        runtime.before_smoke()
        disposable_smoke(runtime, protocol['disposable_qr10_smoke_updates'])
        smoke_completed = protocol['disposable_qr10_smoke_updates']
        runtime.after_smoke()
        expected_first = preflight(runtime, factory)
        runtime.current_arm = 'may_literal'
        result = train_arm(runtime, 'may_literal', expected_first, steps=protocol['may_calls'],
            observation_steps=protocol['observation_steps'], factory=factory)
        results.append(result)
        runtime.after_short_diagnostic()
        runtime.budget(resources=True)
        status = 'completed'
    except BaseException as exc:
        error = failure_record(exc)
        abort_audit = collect_abort_audit(runtime)
    finally:
        cleanup_errors = []
        for label, action in [('gradients', lambda: runtime.model.zero_grad(set_to_none=True)),
                ('reset', lambda: runtime.snapshot.reset() if status != 'completed' else None),
                ('snapshot_close', runtime.snapshot.close)]:
            try:
                action()
            except BaseException as exc:
                cleanup_errors.append(dict(stage=label, **failure_record(exc)))
        try:
            snapshot_release = snapshots.report()
        except BaseException as exc:
            snapshot_release = dict(all_registered_released=False, failure=failure_record(exc))
        finally:
            snapshots.close()
        try:
            placement = runtime.close()
        except BaseException as exc:
            placement = dict(passed=False, failure=failure_record(exc))
    if cleanup_errors or not snapshot_release['all_registered_released'] or not placement['passed']:
        status = 'failed'
    if status == 'failed' and abort_audit is None:
        abort_audit = collect_abort_audit(runtime)
    report = dict(status=status, runs=results, failure=error, abort_audit=abort_audit, cleanup_errors=cleanup_errors, snapshot_release=snapshot_release,
                  placement_release=placement, diagnostic_completed=status == 'completed',
                  quality_qualified=False, timing_qualified=False, full_comparison_completed=False, qr_repeat_completed=False,
                  smoke_calls_completed=smoke_completed, may_calls_completed=getattr(runtime, 'last_arm_result', {}).get('steps', 0),
                  maximum_optimizer_calls=protocol['maximum_optimizer_updates'])
    last = getattr(runtime, 'last_arm_result', None)
    if last is not None and last.get('status') == 'failed':
        report['failed_arm'] = last
        report['orchestration_failure'] = report['failure']
        report['failure'] = last.get('failure') or report['failure']
    runtime.emit('finished', report)
    return report
