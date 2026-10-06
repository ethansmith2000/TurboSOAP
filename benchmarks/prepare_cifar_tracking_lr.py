"""Freeze paired LR attribution with sparse offline covariance tracking probes."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_cifar_benchmark import prepare,sha,REPO


def study():
    pairs=[('qr20',.0005),('warm20',.00025),('qr20',.001),('warm20',.0005),('qr20',.00025),('warm20',.001)]
    arms=[dict(label=f'{policy}_lr{rate}',policy=policy,matrix_lr=rate,covariance_beta=.99,matrix_beta2=.999,
        basis_ns_iterations=6 if policy=='warm20' else 2,tracking_probe=rate==.0005) for policy,rate in pairs]
    arms.append({**arms[0],'label':'qr20_lr0.0005_repeat_end','repeat_of':arms[0]['label']})
    return dict(arms=arms,equal_updates=3500,maximum_updates=3500,evaluation_updates=[875,1750,3500],
        wall_seconds=None,timing_chunk=25,maximum_gram_error=.05,timing_version=2,guard_every_refresh=True,
        tracking_ages=[20,180,200,220,860,1740,3480],control_pairs=[[arms[0]['label'],arms[-1]['label']]],
        auxiliary_beta2_fixed=.999,matrix_beta1_fixed=.9,
        selection='Primary validation CE at3500 for each family/rate. Compare best of three tested rates, not best observed step. No automatic bracket extension.',
        scope='Seed139,20 of200 scheduled epochs. Covariance.99 fixed, auxiliary LR.0005 fixed; only matrix LR varies. Local tracking shadows are not training interventions.',
        checkpoint_policy='No saved weights/checkpoints/tensor snapshots; in-RAM states discarded.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle.resolve()
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),matrix_lr=.0005,branch_steps=3500,covariance_beta=.99)
    for name in ['cifar_lr_screen.py','cifar_tracking_probe.py','prepare_cifar_tracking_lr.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    c=json.loads((b/'config.json').read_text());c['max_wall_seconds']=1800;c['memory_screen']=study()
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    shutil.copy2(REPO/'benchmarks/CIFAR_TRACKING_LR.md',b/'PROTOCOL.md')
    m=json.loads((b/'manifest.json').read_text());m.update(config_sha256=sha(b/'config.json'),
        source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        purpose='Bounded QR20/NS6-warm20 matrix-LR sensitivity and read-only tracking alternatives',protocol_sha256=sha(b/'PROTOCOL.md'))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(b)


if __name__=='__main__':main()
