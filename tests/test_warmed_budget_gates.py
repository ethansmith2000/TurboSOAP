from benchmarks.summarize_warmed_budget import repeat_gate, endpoint_gate


def run(wall=66., opt=9.7, endpoint=90.4, previous=89.9):
    return dict(training_seconds=endpoint, optimizer_ms=100., wall_target_reached=True,
                observations=[dict(step=3500, training_seconds=wall,
                                   timing=dict(optimizer_ms=opt), reasons=['fixed_updates']),
                              dict(step=4775, training_seconds=endpoint, reasons=['equal_wall'])],
                history=[dict(step=4750, training_seconds=previous),
                         dict(step=4775, training_seconds=endpoint)])


def test_gate_uses_identical_fixed_work_despite_variable_final_lengths():
    runs=[run(endpoint=t) for t in [90.1,90.9,90.4]]
    runs[1]['training_seconds']=1000.
    runs[2]['optimizer_ms']=.01
    assert repeat_gate(runs)['repeat_gate']
    assert repeat_gate(runs)['mean_fixed_training_seconds']==66.


def test_repeat_gate_withholds_quality_when_either_original_limit_fails():
    assert not repeat_gate([run(wall=w) for w in [66.,66.,69.]])['repeat_gate']
    assert not repeat_gate([run(opt=o) for o in [9.7,9.7,10.]])['repeat_gate']


def test_wall_endpoint_must_be_first_crossing_within_declared_overshoot():
    assert endpoint_gate(run())
    assert not endpoint_gate(run(endpoint=89.9))
    assert not endpoint_gate(run(endpoint=91.01))
    assert not endpoint_gate(run(previous=90.1))
    missing=run();missing['observations']=missing['observations'][:1]
    assert not endpoint_gate(missing)
    capped=run();capped['wall_target_reached']=False
    assert not endpoint_gate(capped)
