"""Same dispatch observer, with lossless deduplication of repeated affinity masks."""
import sys
from pathlib import Path
import time
try:
    from .dispatch_diagnostics import DispatchHostTiming
except ImportError:
    from dispatch_diagnostics import DispatchHostTiming

class CompactDispatchHostTiming(DispatchHostTiming):
    def __init__(self):
        super().__init__();self.affinity_catalog={}

    def flush(self):
        start=time.monotonic();result=super().flush();definitions={}
        for row in result['thread_interval']:
            mask=tuple(row.pop('affinity'))
            if mask not in self.affinity_catalog:
                key=f'mask_{len(self.affinity_catalog)}';self.affinity_catalog[mask]=key
                definitions[key]=list(mask)
            row['affinity_ref']=self.affinity_catalog[mask]
        if definitions:result['affinity_definitions']=definitions
        result['observation_seconds']=time.monotonic()-start
        return result

def main():
    here=Path(__file__).resolve().parent;sys.path[:0]=[str(here.parent),str(here)]
    import cifar_radius_learning_runner as trainer
    import cifar_affinity_screen as affinity
    trainer.HostTiming=CompactDispatchHostTiming
    sys.modules['cifar_timing_screen']=trainer
    affinity.main()

if __name__=='__main__':main()
