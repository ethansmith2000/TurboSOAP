"""Freeze the predeclared 3-policy x 3-LR CIFAR screen and unchanged repeat."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_cifar_benchmark import prepare,sha,REPO


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--bracket',choices=['original','lower'],default='original')
    a=p.parse_args();b=a.bundle.resolve()
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),branch_steps=3500)
    for name in ['cifar_lr_screen.py','prepare_cifar_lr_screen.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    c=json.loads((b/'config.json').read_text())
    # Interleave policies and rates; repeat the central QR control last.
    pairs=[('qr10',.001),('warm10',.0005),('qr20',.002),
           ('warm10',.001),('qr20',.0005),('qr10',.002),
           ('qr20',.001),('qr10',.0005),('warm10',.002)]
    arms=[dict(label=f'{policy}_lr{lr:g}',policy=policy,matrix_lr=lr) for policy,lr in pairs]
    arms.append(dict(label='qr10_lr0.001_repeat',policy='qr10',matrix_lr=.001))
    if a.bracket=='lower':
        pairs=[('qr10',.0005),('warm10',.000125),('qr20',.00025),
               ('warm10',.0005),('qr20',.000125),('qr10',.00025),
               ('qr20',.0005),('qr10',.000125),('warm10',.00025)]
        arms=[dict(label=f'{policy}_lr{lr:g}',policy=policy,matrix_lr=lr) for policy,lr in pairs]
        for index,suffix in [(5,'mid'),(10,'end')]:
            arms.insert(index,dict(label=f'qr10_lr0.0005_repeat_{suffix}',policy='qr10',
                                  matrix_lr=.0005,repeat_of='qr10_lr0.0005'))
    c['max_wall_seconds']=1800 if a.bracket=='lower' else 1500
    c['lr_screen']=dict(arms=arms,equal_updates=3500,maximum_updates=4200,
        evaluation_updates=[875,1750,3500],wall_seconds=75.,timing_chunk=25,
        maximum_gram_error=.05,selection='Minimum validation CE at 3500 updates within each policy; repeat excluded. Boundary winners require further calibration.',
        timing='Training wall includes cold basis initialization and synchronized training chunks; excludes discarded compile smoke, evaluation, hash probes and logging. Equal-wall observation is first 25-step boundary at/after 75 seconds; report overshoot.',
        checkpoint_policy='No saved weights; initial states in RAM, discarded at exit.',
        scope='One seed, 20-epoch early-learning screen on original 200-epoch LR horizon; auxiliary LR fixed .0005; covariance beta and beta2 .999. No convergence claim or automatic expansion.')
    c['lr_screen']['bracket']=a.bracket
    c['lr_screen']['timing_version']=2
    if a.bracket=='lower':
        c['lr_screen']['control_pairs']=[['qr10_lr0.0005',f'qr10_lr0.0005_repeat_{suffix}'] for suffix in ['mid','end']]
    c['lr_screen']['timing']+=' Four CUDA events partition augmentation, model/backward and optimizer; these elapsed intervals include host dispatch gaps, not pure kernel time. CPU process time and cumulative CUDA time are recorded independently; diagnostic accounting is outside training wall.'
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    m=json.loads((b/'manifest.json').read_text());m.update(config_sha256=sha(b/'config.json'),
        source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        purpose='Bounded CIFAR LR screen, equal-update/equal-training-wall observations, unchanged-control repeat')
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(b)


if __name__=='__main__':main()
