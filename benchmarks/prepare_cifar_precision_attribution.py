"""Freeze operation-level precision probes on established CIFAR trajectories."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_cifar_benchmark import prepare,sha,REPO
from prepare_cifar_tracking_lr import study


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle.resolve()
    prepare(b,Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'),
            matrix_lr=.0005,branch_steps=3500,covariance_beta=.99)
    for name in ['cifar_lr_screen.py','cifar_tracking_probe.py','cifar_refresh_efficiency.py',
                 'cifar_precision_attribution.py','prepare_cifar_tracking_lr.py','prepare_cifar_precision_attribution.py']:
        shutil.copy2(REPO/'benchmarks'/name,b/'source/benchmarks'/name)
    c=json.loads((b/'config.json').read_text());s=study()
    s['arms']=[a for a in s['arms'] if a['matrix_lr']==.0005]
    s.update(tracking_probe_kind='precision_attribution',
             selection='No training intervention. Attribute full-refresh high-vs-highest differences with fixed-basis transport and QR ordering controls.',
             scope='Unchanged QR20 and NS6 warm20 trajectories plus QR repeat; seed139, covariance .99, both LRs .0005.')
    c['memory_screen']=s;c['max_wall_seconds']=900
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    shutil.copy2(REPO/'benchmarks/CIFAR_PRECISION_ATTRIBUTION.md',b/'PROTOCOL.md')
    m=json.loads((b/'manifest.json').read_text());m.update(config_sha256=sha(b/'config.json'),
        source_sha256={str(p.relative_to(b/'source')):sha(p) for p in (b/'source').rglob('*') if p.is_file()},
        purpose='Separate fixed-basis m transport, basis arithmetic, QR ordering and v permutation precision effects',
        protocol_sha256=sha(b/'PROTOCOL.md'))
    (b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(b)


if __name__=='__main__':main()
