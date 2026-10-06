import gc
import pytest
from benchmarks.timing_diagnostics import HostTiming, interval_union_us, range_over_mean


def test_interval_union_counts_overlapping_device_work_once():
    assert interval_union_us([(10,20),(0,5),(2,4),(4,12),(22,25)]) == 23
    assert interval_union_us([]) == 0
    assert interval_union_us([(0,0),(0,1),(1,2)]) == 2
    with pytest.raises(ValueError):interval_union_us([(2,1)])


def test_repeat_range_includes_all_observations():
    assert range_over_mean([100,100,100]) == 0
    assert range_over_mean([98,100,102]) == .04
    with pytest.raises(ValueError):range_over_mean([])
    with pytest.raises(ValueError):range_over_mean([0,1])


def test_host_observer_reset_does_not_accumulate_prior_chunks():
    before=len(gc.callbacks)
    h=HostTiming()
    try:
        h.reset();h.begin();h.mark();h.mark();h.finish()
        first=h.flush();second=h.flush()
        assert first['steps']==1 and second['steps']==0
        assert all(v>=0 for phase in first['phases'].values() for v in phase.values())
        assert all(v==0 for phase in second['phases'].values() for v in phase.values())
    finally:h.close()
    assert len(gc.callbacks)==before
