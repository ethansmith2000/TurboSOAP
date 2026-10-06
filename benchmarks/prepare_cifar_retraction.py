"""Freeze the bounded one-step CIFAR learning intervention and exact controls."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_cifar_benchmark import prepare,sha,REPO


def study():
    arms=[]
    for beta in (.99,.999):
        for kind in ('qr20','ns6','cubic2','quintic1'):
            arm=dict(label=f'{kind}_beta{beta}',policy='qr20' if kind=='qr20' else 'warm20',
                     matrix_lr=.0005,matrix_beta2=.999,covariance_beta=beta,
                     transport_precision='highest',basis_ns_iterations=6 if kind=='ns6' else 2,
                     basis_rotation_cap=.1,tracking_probe=True)
            if kind in ('cubic2','quintic1'):
                arm.update(retraction_variant=f'spectral_{kind}_cap05',basis_ns_iterations=2 if kind=='cubic2' else 1,basis_rotation_cap=.5)
            arms.append(arm)
    for original in (arms[0],arms[3]):
        arms.append({**original,'label':original['label']+'_repeat','repeat_of':original['label']})
    return dict(arms=arms,equal_updates=3500,maximum_updates=3500,evaluation_updates=[875,1750,3500],
                wall_seconds=None,timing_chunk=25,maximum_gram_error=.05,timing_version=2,
                guard_every_refresh=True,tracking_probe_kind='strict_transport',
                tracking_ages=[20,180,200,220,860,1740,3480],
                control_pairs=[[arms[0]['label'],arms[-2]['label']],[arms[3]['label'],arms[-1]['label']]],
                selection='Primary CE at3500; fixed-setting comparison, no LR tuning, no default promotion.',
                checkpoint_policy='No weights/checkpoints/tensor snapshots; common state in RAM only.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle.resolve()
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),
            matrix_lr=.0005,branch_steps=3500,covariance_beta=.99)
    for name in ['cifar_lr_screen.py','cifar_tracking_probe.py','cifar_strict_transport.py',
                 'warm_retraction.py','cifar_retraction_screen.py','prepare_cifar_retraction.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    c=json.loads((b/'config.json').read_text());c['memory_screen']=study();c['max_wall_seconds']=1500
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (b/'PROTOCOL.md').write_text('''# Bounded low-step retraction: revised real CIFAR intervention

Predeclared after the cubic1 failure and diagnostic replay. Cubic1/cap.5 failed
the unchanged world-M<.01 condition at age220: observed.0149173 with exact
selected-basis computation. Preserve that failed screen; this is a new policy.
Ten3500-update arms:
QR20, original NS6/average-cap.1, cubic2/row-sum-cap.5 and quintic1/row-sum-cap.5,
each at covariance .99/.999; final QR20 and quintic1 repeats at .99. This is a
joint cap/retraction intervention, not attribution to iteration count alone.
Raw-covariance normalization removal is not included in this learning stage.

Seed139, both factors, Adam(.9,.999), matrix/auxLR.0005, original200-epoch schedule,
20-epoch endpoint and same frozen CIFAR100 split/augmentation as prior screens.
Warm20/QR200, V carry/reindex, high basis computation and strict FP32 m transport.
No optimizer compilation; common initial weights/RNG reset for every arm.
Observe875/1750/3500; no new LR tuning or best-checkpoint selection.

Every refresh gets the same offline finite/Gram guard (<.05). Selected ages
20/180/200/220/860/1740/3480 retain selected-basis transport implementation
checks (update error<1e-6, world-m error<.01, Gram<.05), separately from the
unchanged 1% full strict-refresh diagnostic. No whole-policy full-reference pass
is assumed; QR/reset failures remain explicit. Offline probes are outside timing.
Report effective experimental groups, all timing components and exact historical
QR/NS6 anchors plus repeated controls. Primary CE at3500; report accuracy too.
No time-to-quality claim from synthetic timing paired with CIFAR loss.

One claimed GPU3, no queue, internal1500s/external1560s. Before launch require
all bounded synthetic cases completed, exact repeats, and prototype tests pass.
Retain compact evidence under20MB; capacity326GiB at round start. No weights,
checkpoints or tensor snapshots. No production/default change. Any failure ends
this screen for diagnosis; no retrospective threshold relaxation.
''')
    m=json.loads((b/'manifest.json').read_text());m.update(config_sha256=sha(b/'config.json'),
        source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        purpose='Paired fixed-work bounded retraction learning intervention',protocol_sha256=sha(b/'PROTOCOL.md'))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(b)


if __name__=='__main__':main()
