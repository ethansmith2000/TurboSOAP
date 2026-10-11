import pytest
from benchmarks.wide_learning_checks import chunk_boundaries,timing_repeat


def runs(walls=(100.,100.,100.),opts=(20.,20.,20.)):
    return [dict(steps=3500,training_seconds=w,optimizer_ms=o) for w,o in zip(walls,opts)]


def test_partial420_observation_has_no_missing_or_double_counted_updates():
    boundaries=chunk_boundaries(3500,[100,200,400,420,875,1750,3500])
    lengths=[b-a for a,b in zip([0,*boundaries[:-1]],boundaries)]
    assert len(boundaries)==141 and sum(lengths)==3500
    i=boundaries.index(420)
    assert boundaries[i-1:i+2]==[400,420,425] and lengths[i:i+2]==[20,5]
    assert all(x in boundaries for x in [100,200,400,420,875,1750,3500])


def test_all_three_controls_are_used_so_middle_outlier_cannot_be_trimmed():
    assert not timing_repeat(runs(walls=(100.,104.,100.)))['repeat_gate']
    assert not timing_repeat(runs(opts=(20.,20.5,20.)))['repeat_gate']
    assert timing_repeat(runs(walls=(100.,102.,100.),opts=(20.,20.2,20.)))['repeat_gate']


@pytest.mark.parametrize('case',[runs()[:2],runs()+runs()[:1],
    [dict(steps=420,training_seconds=10.,optimizer_ms=20.),*runs()[1:]],
    runs(walls=(100.,float('nan'),100.)),runs(opts=(20.,0.,20.))])
def test_missing_extra_unequal_work_and_invalid_timing_fail_closed(case):
    with pytest.raises(ValueError):timing_repeat(case)
