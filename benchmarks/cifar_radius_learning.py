"""Wire qualified fixed-radius policies into the isolated placed CIFAR runner."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
from radius_natural import POLICIES
from movement_retraction import warm

def configure_matrix(matrix,name):
    policy=POLICIES[name]
    for g in matrix.param_groups:
        g.update(research_movement_policy=name,basis_ns_iterations=policy.iterations,
                 basis_rotation_cap=policy.cap,basis_rotation_cap_mode='skew_frobenius')
    matrix._gauge_step_one=lambda cov,q,g:warm(cov,q,g,POLICIES[g['research_movement_policy']])

def main():
    import cifar_radius_learning_runner as trainer
    trainer.configure_matrix=configure_matrix
    sys.modules['cifar_timing_screen']=trainer
    import cifar_affinity_screen
    cifar_affinity_screen.main()

if __name__=='__main__':main()
