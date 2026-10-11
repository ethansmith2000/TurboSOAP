from copy import deepcopy
from benchmarks.compact_dispatch_screen import CompactDispatchHostTiming
from benchmarks.dispatch_diagnostics import DispatchHostTiming

def test_affinity_catalog_losslessly_reconstructs_repeated_and_new_masks(monkeypatch):
    original=dict(thread_interval=[dict(tid=1,affinity=[116],runtime_ns=10),
                                  dict(tid=2,affinity=list(range(256)),runtime_ns=20)],observation_seconds=0.)
    monkeypatch.setattr(DispatchHostTiming,'flush',lambda self:deepcopy(original))
    h=CompactDispatchHostTiming()
    try:
        first=h.flush();second=h.flush();catalog=first['affinity_definitions']
        assert 'affinity_definitions' not in second
        for result in [first,second]:
            restored=[{**{k:v for k,v in row.items() if k!='affinity_ref'},'affinity':catalog[row['affinity_ref']]}
                      for row in result['thread_interval']]
            assert restored==original['thread_interval']
        original['thread_interval'].append(dict(tid=3,affinity=[117],runtime_ns=30))
        third=h.flush();assert list(third['affinity_definitions'].values())==[[117]]
    finally:h.close()
