"""One-factor autograd co-location diagnostic using the unchanged isolated runner."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]

def main():
    import cifar_radius_learning_runner as trainer
    from dispatch_diagnostics import DispatchHostTiming
    import cifar_affinity_screen as affinity
    trainer.HostTiming=DispatchHostTiming
    sys.modules['cifar_timing_screen']=trainer
    affinity.main()

if __name__=='__main__':main()
