"""Freeze the first real-data NS6 cadence/covariance screen; no model files."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_cifar_benchmark import prepare,sha,REPO


def study():
    pairs=[('qr20',.999),('warm20',.99),('qr40',.999),('warm40',.99),
           ('warm20',.999),('qr20',.99),('warm40',.999),('qr40',.99)]
    arms=[dict(label=f'{policy}_cov{str(beta).replace(".","p")}',policy=policy,
        matrix_lr=.0005,covariance_beta=beta,matrix_beta2=.999,
        basis_ns_iterations=6 if policy.startswith('warm') else 2) for policy,beta in pairs]
    arms.append({**arms[0],'label':arms[0]['label']+'_repeat_end','repeat_of':arms[0]['label']})
    return dict(arms=arms,equal_updates=3500,maximum_updates=3500,evaluation_updates=[875,1750,3500],
        wall_seconds=None,timing_chunk=25,maximum_gram_error=.05,timing_version=2,guard_every_refresh=True,
        control_pairs=[[arms[0]['label'],arms[-1]['label']]],
        auxiliary_beta2_fixed=.999,matrix_beta1_fixed=.9,
        selection='Validation CE at 3500 updates primary; 875/1750 curves, accuracy and train panel secondary. No best-checkpoint selection.',
        timing='Descriptive full optimizer intervals only; all refresh diagnostics excluded. No equal-wall quality claim.',
        scope='One seed139,20 of the original200 scheduled epochs, LR.0005 fixed, no per-policy LR retuning or convergence claim.',
        checkpoint_policy='No weights/checkpoints/moment snapshots. Initial model state in RAM, discarded.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle.resolve()
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),matrix_lr=.0005,branch_steps=3500)
    for name in ['cifar_lr_screen.py','prepare_cifar_cadence.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    c=json.loads((b/'config.json').read_text());c['max_wall_seconds']=1800;c['memory_screen']=study()
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    shutil.copy2(REPO/'benchmarks/CIFAR_CADENCE.md',b/'PROTOCOL.md')
    m=json.loads((b/'manifest.json').read_text());m.update(config_sha256=sha(b/'config.json'),
        source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        purpose='Real CIFAR learning screen of NS6 warm20/40 vs QR20/40 with covariance .99/.999',
        protocol_sha256=sha(b/'PROTOCOL.md'))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(b)


if __name__=='__main__':main()
