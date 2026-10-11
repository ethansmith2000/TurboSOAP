import os
import time
from benchmarks.dispatch_diagnostics import parse_stat,thread_deltas,thread_snapshot,DispatchHostTiming
from benchmarks.prepare_dispatch_screen import arms
from benchmarks.summarize_dispatch_screen import material_effect

def test_stat_parser_handles_parentheses_and_spaces_in_comm():
    fields=['0']*50;fields[11]='14';fields[12]='15';fields[36]='39'
    assert parse_stat('123 (strange (worker) name) '+' '.join(fields))==dict(user_ticks=14,system_ticks=15,last_cpu=39)

def test_thread_runtime_and_wait_are_separate_and_new_threads_not_fabricated():
    a=dict(name='worker',affinity=[116],last_cpu=116,runtime_ns=100,runqueue_wait_ns=30,timeslices=2,user_ticks=1,system_ticks=1)
    z=dict(a,runtime_ns=170,runqueue_wait_ns=90,timeslices=3,user_ticks=3,system_ticks=2)
    rows=thread_deltas({1:a},{1:z,2:z})
    assert len(rows)==1 and rows[0]['runtime_ns']==70 and rows[0]['runqueue_wait_ns']==60

def test_live_own_thread_snapshot_contains_main_and_monotonic_scheduler_counters():
    a=thread_snapshot();sum(i*i for i in range(10000));z=thread_snapshot()
    row=next(x for x in thread_deltas(a,z) if x['tid']==os.getpid())
    assert row['runtime_ns']>0 and row['runqueue_wait_ns']>=0 and row['affinity']==sorted(os.sched_getaffinity(0))

def test_phase_user_system_cpu_survives_base_flush_reset():
    h=DispatchHostTiming()
    try:
        h.reset();h.begin()
        for i in range(3):
            sum(x*x for x in range(25000))
            if i<2:h.mark()
        h.finish();r=h.flush()
        assert r['steps']==1 and len(r['phases'])==3 and r['thread_interval']
        for p in r['phases'].values():
            assert p['main_user_cpu_ms']>=0 and p['main_system_cpu_ms']>=0
            assert abs(p['main_user_cpu_ms']+p['main_system_cpu_ms']-p['thread_cpu_ms'])<2
        assert not h.rows
    finally:h.close()

def test_arm_plan_changes_only_autograd_core_with_balanced_order():
    aa=arms();assert [x['kind'] for x in aa]==['near','same','same','near','near','same']
    assert all(x['timing_affinity']['main']==116 for x in aa)
    assert all(x['timing_affinity']['autograd']==(116 if x['kind']=='same' else 117) for x in aa)
    assert len([x for x in aa if x['capture_replay']])==2

def test_material_effect_requires_repeats_separation_and_consistent_pairs():
    a=dict(model_mean_ms=10,model_min_ms=9.9,model_max_ms=10.1)
    z=dict(model_mean_ms=9,model_min_ms=8.9,model_max_ms=9.1)
    assert material_effect([a,z],[-.1]*3,True,True)==(-.09999999999999998,True)
    assert not material_effect([a,z],[-.1]*3,True,False)[1]
    assert not material_effect([a,z],[-.1,.1,-.1],True,True)[1]
    assert not material_effect([a,dict(z,model_max_ms=10)],[-.1]*3,True,True)[1]
