"""Actual shallow wide ViT, isolated runner and unchanged compact observer."""
from pathlib import Path
import sys

def main():
    here=Path(__file__).resolve().parent
    sys.path[:0]=[str(here.parent),str(here)]
    import cifar_wide_runner as trainer
    import cifar_affinity_screen as affinity
    from compact_dispatch_screen import CompactDispatchHostTiming
    trainer.HostTiming=CompactDispatchHostTiming
    sys.modules['cifar_timing_screen']=trainer
    affinity.main()

if __name__=='__main__':main()
