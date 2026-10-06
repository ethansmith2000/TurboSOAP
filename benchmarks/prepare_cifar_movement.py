"""Freeze a compact movement attribution screen before GPU execution."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_cifar_benchmark import prepare, sha, REPO


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    b=parser.parse_args().bundle.resolve()
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),
            matrix_lr=.0005,branch_steps=1000,covariance_beta=.99)
    arms=[]
    for beta in (.99,.999):
        for kind in ('ns6','row_cubic2'):
            a=dict(label=f'{kind}_beta{beta}',policy='warm20',matrix_lr=.0005,
                matrix_beta2=.999,covariance_beta=beta,transport_precision='highest',
                basis_ns_iterations=6 if kind=='ns6' else 2,basis_rotation_cap=.1,
                tracking_probe=True)
            if kind=='row_cubic2':a['retraction_variant']='spectral_cubic2_cap05'
            arms.append(a)
    arms.append({**arms[0],'label':arms[0]['label']+'_repeat','repeat_of':arms[0]['label']})
    c=json.loads((b/'config.json').read_text())
    c['memory_screen']=dict(arms=arms,equal_updates=1000,maximum_updates=1000,
        evaluation_updates=[875,1000],wall_seconds=None,timing_chunk=25,
        maximum_gram_error=.05,timing_version=2,guard_every_refresh=True,
        tracking_probe_kind='strict_transport',movement_attribution=True,
        tracking_ages=[20,180,200,220,400,420,860],
        control_pairs=[[arms[0]['label'],arms[-1]['label']]],
        checkpoint_policy='No weights or tensor snapshots. Common states exist in RAM only.')
    c['max_wall_seconds']=900
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    for name in ['cifar_lr_screen.py','cifar_tracking_probe.py','cifar_strict_transport.py',
                 'warm_retraction.py','cifar_retraction_screen.py','movement_retraction.py',
                 'cifar_movement_probe.py','cifar_movement_screen.py','prepare_cifar_movement.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    (b/'PROTOCOL.md').write_text('''# Actual CIFAR movement attribution — predeclared 2026-10-06

Five 1000-step reconstructed trajectories: NS6/average cap .1 and cubic2/row cap
.5 at covariance .99/.999, plus NS6/.99 repeat. Same seed 139, paired initial
state/RNG/data, LR .0005, Adam (.9,.999), high basis arithmetic, strict M transport,
warm20/QR200, both factors, original 200-epoch LR schedule and cached CIFAR split.
At ages 20/180/200/220/400/420/860, compare six alternatives on identical incoming
covariances, bases and moments. They never feed back into training:
NS6 average .1; row .5 cubic2/quintic1; skew-Frobenius .5 cubic2 and .4 quintic1;
exact-spectral .5 cubic2 as an offline oracle, never an efficiency candidate.
No candidate rescaling for low-step polynomials. Generator and candidate/retraction
use high precision setting; diagnostic eigensolves and moment checks use highest.
For real skew A, paired singular values imply ||A||2 <= ||A||F/sqrt(2).
This is an alternative upper bound, not guaranteed tighter than row sum on all A.

Record raw spectral/Frobenius/row norms, cap scale, post-retraction Gram, movement,
same-state residual change and original-coordinate M preservation. Do not equate
instantaneous covariance residual improvement with better learning. Shadow
alternatives do not abort for exceeding the old 1% M guard: record their failures
and exclude them from the subsequent unchanged-gate qualification shortlist.
The actually trained controls retain all existing geometry and transport guards.
Exact historical step875 observations and repeat metrics/probes must match.
This is attribution, not a speed benchmark or convergence ranking.

One claimed GPU3 with no queue. Internal 900s / external 960s bound. Capacity at
preparation: 326 GiB free. Retain source/config/metrics/logs under 25 MiB; no saved
weights, checkpoints or tensor snapshots. Keep compact evidence for research
audit. No new compiler cache requested; inherited model compile cache may be reused.
No production/default changes. Only qualify policies whose sampled shadow
geometry and M errors pass .05/1%, then measure natural cycles independently.
''')
    m=json.loads((b/'manifest.json').read_text());m.update(config_sha256=sha(b/'config.json'),
        source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        protocol_sha256=sha(b/'PROTOCOL.md'),evidence_budget_bytes=25*2**20)
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(b)


if __name__=='__main__':main()
