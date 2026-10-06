"""Freeze a precision/cap/beta factorial, plus two reproducibility controls."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_cifar_benchmark import prepare,sha,REPO


def arm(policy,precision,cap,beta):
    return dict(label=f'{policy}_{precision}_cap{cap}_beta{beta}',policy=policy,
                matrix_lr=.0005,matrix_beta2=.999,covariance_beta=beta,
                transport_precision=precision,basis_rotation_cap=cap,
                basis_ns_iterations=6 if policy=='warm20' else 2,tracking_probe=True)


def study():
    settings=[('qr20','inherit',.1),('warm20','highest',.2),('warm20','inherit',.1),
              ('qr20','highest',.1),('warm20','highest',.1),('warm20','inherit',.2)]
    arms=[arm(policy,precision,cap,beta) for beta in [.99,.999] for policy,precision,cap in settings]
    for original in [arms[0],arms[1]]:
        arms.append({**original,'label':original['label']+'_repeat_end','repeat_of':original['label']})
    return dict(arms=arms,equal_updates=3500,maximum_updates=3500,evaluation_updates=[875,1750,3500],
                wall_seconds=None,timing_chunk=25,maximum_gram_error=.05,timing_version=2,guard_every_refresh=True,
                tracking_probe_kind='strict_transport',tracking_ages=[20,180,200,220,860,1740,3480],
                control_pairs=[[arms[0]['label'],arms[-2]['label']],[arms[1]['label'],arms[-1]['label']]],
                selection='Primary validation CE at3500. Report paired transport, cap and covariance contrasts; no automatic LR retuning or horizon extension.',
                scope='Seed139,20 of200 scheduled epochs. Matched 2x2 warm precision/cap at each .99/.999 covariance plus QR precision controls.',
                checkpoint_policy='No saved weights/checkpoints/tensor snapshots; temporary states discarded.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle.resolve()
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),
            matrix_lr=.0005,branch_steps=3500,covariance_beta=.99)
    for name in ['cifar_lr_screen.py','cifar_tracking_probe.py','cifar_strict_transport.py',
                 'strict_transport_cost_gate.py','run_strict_transport_round.py','prepare_cifar_strict_transport.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    c=json.loads((b/'config.json').read_text());c['memory_screen']=study();c['max_wall_seconds']=2100
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    shutil.copy2(REPO/'benchmarks/CIFAR_STRICT_TRANSPORT.md',b/'PROTOCOL.md')
    m=json.loads((b/'manifest.json').read_text());m.update(config_sha256=sha(b/'config.json'),
        source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        purpose='Opt-in strict m transport implementation, dimension-specific refresh cost and paired precision/cap/beta learning',
        protocol_sha256=sha(b/'PROTOCOL.md'))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(b)


if __name__=='__main__':main()
