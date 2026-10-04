"""Freeze a 100-epoch paired covariance-memory extension; no saved weights."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_cifar_benchmark import prepare,sha,REPO


def long_study():
    # Interleave settings across methods; repeat the opening control at the end.
    pairs=[('qr10','base'),('warm10','cov99'),('qr20','base'),
           ('qr10','cov99'),('warm10','base'),('qr20','cov99')]
    arms=[dict(label=f'{policy}_{variant}',policy=policy,variant=variant,
               matrix_lr=.0005,covariance_beta=.999 if variant=='base' else .99,
               matrix_beta2=.999) for policy,variant in pairs]
    arms.append({**arms[0],'label':'qr10_base_repeat_end','repeat_of':'qr10_base'})
    return dict(arms=arms,equal_updates=17500,maximum_updates=17500,
        evaluation_updates=[875,1750,3500,7000,10500,14000,17500],
        wall_seconds=None,timing_chunk=25,maximum_gram_error=.05,timing_version=2,
        control_pairs=[['qr10_base','qr10_base_repeat_end']],
        auxiliary_beta2_fixed=.999,matrix_beta1_fixed=.9,
        selection='Primary: within-policy validation CE delta at 17500 updates (100 epochs); secondary: direction at 40/60/80 epochs, accuracy, train panel and Gram error. No endpoint or best-checkpoint selection.',
        timing='Descriptive elapsed component and control timings only; no equal-wall quality ranking.',
        scope='Seed 139, 100 of 200 scheduled epochs. Covariance-only .99 vs .999; matrix/auxiliary Adam beta2 .999 and both LRs .0005. No seed replication claim or automatic grid expansion.',
        checkpoint_policy='No saved weights/checkpoints/moment snapshots; initial states held in RAM and discarded.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args();b=a.bundle.resolve()
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),
            matrix_lr=.0005,branch_steps=17500)
    for name in ['cifar_lr_screen.py','prepare_cifar_covariance_long.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    c=json.loads((b/'config.json').read_text())
    c['max_wall_seconds']=4200;c['memory_screen']=long_study()
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    m=json.loads((b/'manifest.json').read_text());m.update(config_sha256=sha(b/'config.json'),
        source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        purpose='Longer covariance-only paired learning at fixed matrix/auxiliary memory and LR')
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    print(b)


if __name__=='__main__':main()
