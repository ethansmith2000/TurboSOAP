"""Use the existing natural-cycle harness with isolated skew-Frobenius policies."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
import retraction_diagnostic as runner
from movement_retraction import POLICIES, warm


def main():
    original=runner.warm
    runner.VARIANTS=(*runner.VARIANTS,*POLICIES.values())
    def dispatch(cov,q,group,variant):
        return warm(cov,q,group,variant) if variant.name in POLICIES else original(cov,q,group,variant)
    runner.warm=dispatch
    runner.main()


if __name__=='__main__':main()
