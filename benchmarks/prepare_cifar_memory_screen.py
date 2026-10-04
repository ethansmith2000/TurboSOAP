"""Freeze the matrix-only one-factor covariance/Adam memory screen."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_cifar_benchmark import prepare,sha,REPO


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args();b=a.bundle.resolve()
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),
            matrix_lr=.0005,branch_steps=3500)
    for name in ['cifar_lr_screen.py','prepare_cifar_memory_screen.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    c=json.loads((b/'config.json').read_text())
    settings={'base':(.999,.999),'cov99':(.99,.999),'v99':(.999,.99)}
    pairs=[('qr10','base'),('warm10','cov99'),('qr20','v99'),
           ('warm10','base'),('qr20','cov99'),('qr10','v99'),
           ('qr20','base'),('qr10','cov99'),('warm10','v99')]
    arms=[dict(label=f'{policy}_{variant}',policy=policy,variant=variant,matrix_lr=.0005,
               covariance_beta=settings[variant][0],matrix_beta2=settings[variant][1])
          for policy,variant in pairs]
    for index,suffix in [(5,'mid'),(10,'end')]:
        arms.insert(index,{**arms[0],'label':f'qr10_base_repeat_{suffix}','repeat_of':'qr10_base'})
    c['max_wall_seconds']=1800
    c['memory_screen']=dict(arms=arms,equal_updates=3500,maximum_updates=3500,
        evaluation_updates=[875,1750,3500],wall_seconds=None,timing_chunk=25,
        maximum_gram_error=.05,timing_version=2,
        control_pairs=[['qr10_base',f'qr10_base_repeat_{s}'] for s in ['mid','end']],
        auxiliary_beta2_fixed=.999,matrix_beta1_fixed=.9,
        selection='Within-policy CE deltas at 3500 updates for one-factor changes vs baseline; fixed LR attribution, not tuned-memory or convergence ranking. No automatic combined-beta run.',
        timing='Fixed updates only. Four CUDA events partition augmentation/model-backward/optimizer. Intervals include host dispatch gaps. Training wall excludes discarded compile smoke, evaluation, probes, logging and timing accounting. Report repeated-control timing descriptively; no wall-quality claim.',
        scope='One seed, 20 of 200 scheduled epochs, independent covariance and matrix Adam beta2 tests; auxiliary betas unchanged.',
        checkpoint_policy='No saved weights; initial states held in RAM and discarded.')
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    m=json.loads((b/'manifest.json').read_text());m.update(config_sha256=sha(b/'config.json'),
        source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        purpose='One-factor covariance and matrix Adam beta2 attribution with fixed auxiliary optimizer')
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(b)


if __name__=='__main__':main()
