"""Freeze a new, bounded mock-trainer runtime map without launching a GPU job."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--widths',type=int,nargs='+',default=[192,768,1536])
    p.add_argument('--tokens',type=int,nargs='+',default=[64,512])
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--maximum-seconds',type=int,default=1200)
    a=p.parse_args();b=a.bundle.resolve();root=Path(__file__).resolve().parents[1]
    if b.exists():raise ValueError('Use a new bundle directory')
    if any(d<1 or 4*d>10000 for d in a.widths):raise ValueError('Widths must fit all requested factors under the 10000 dimension cap')
    if len(set(a.widths))!=len(a.widths) or len(set(a.tokens))!=len(a.tokens):raise ValueError('Duplicate grid values')
    if min(a.tokens)<1 or a.rounds<2 or a.maximum_seconds<1:raise ValueError('Invalid budget or grid')
    if shutil.disk_usage(b.parent).free<5*2**30:raise RuntimeError('Require 5 GiB free')
    c=dict(widths=a.widths,tokens=a.tokens,modes=['all','smaller_side'],rounds=a.rounds,
           samples_per_event=5,discarded_event_warmups=2,seed=20261002,maximum_seconds=a.maximum_seconds,
           covariance_beta=.999,adam_betas=[.9,.999],lr=.0005,weight_decay=.05,eps=1e-8,
           variance_policy='permutation',checkpoint=False,artifact_budget_bytes=20*2**20)
    (b/'source/benchmarks').mkdir(parents=True)
    for name in ['soap.py','soap_reference.py','benchmarks/benchmark_synthetic_scaling.py',
                 'benchmarks/prepare_synthetic_scaling.py']:
        shutil.copy2(root/name,b/'source'/name)
    (b/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (b/'PROTOCOL.md').write_text((root/'benchmarks/SYNTHETIC_SCALING.md').read_text()+
        '\n## Frozen grid and budget\n\n```json\n'+json.dumps(c,indent=2)+'\n```\n')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=dict(sha256={str(f.relative_to(b)):sha(f) for f in b.rglob('*') if f.is_file()},
                  purpose='Separate synthetic runtime characterization',checkpoint_policy='No weights or moment snapshots')
    (b/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(b)


if __name__=='__main__':main()
