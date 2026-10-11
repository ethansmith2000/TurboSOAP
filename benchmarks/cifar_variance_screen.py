"""Opt-in variance policy adapter; shared trainer and optimizer core unchanged."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
import torch
from variance_probe import VarianceProbe

def configure_variance(matrix,policy):
    if policy not in ('permutation','overlap'):
        raise ValueError('Unknown variance policy')
    for g in matrix.param_groups:g['variance_policy']=policy

@torch.no_grad()
def nonnegative_variance(opt):
    minima=[v['exp_avg_sq'].min() for v in opt.matrix.state.values()]
    minimum=float(torch.stack(minima).min())
    if minimum<0:raise FloatingPointError('Negative stored variance')
    return minimum

def main():
    import cifar_radius_learning_runner as trainer
    original_config,original_build,original_guard=trainer.screen_arm_config,trainer.build_optimizer,trainer.check_basis_state
    def config(c,arm):
        result=original_config(c,arm);result['variance_policy']=arm['variance_policy'];return result
    def build(model,c,policy):
        opt=original_build(model,c,policy);configure_variance(opt.matrix,c.get('variance_policy','permutation'));return opt
    def guard(opt,limit):
        result=original_guard(opt,limit);result['minimum_variance']=nonnegative_variance(opt);return result
    trainer.screen_arm_config,trainer.build_optimizer,trainer.check_basis_state=config,build,guard
    import cifar_strict_transport
    cifar_strict_transport.StrictTransportProbe=VarianceProbe
    sys.modules['cifar_timing_screen']=trainer
    import cifar_affinity_screen
    cifar_affinity_screen.main()

if __name__=='__main__':main()
