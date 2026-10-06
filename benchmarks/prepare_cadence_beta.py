"""Freeze the bounded all-factor covariance-beta/cadence screen; no GPU launch."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def arms():
    return [dict(name=f'{method}{frequency}_cov{str(beta).replace(".", "p")}',
        method=method,frequency=frequency,covariance_beta=beta,iterations=6,cap='average',
        precision='high',covariance='float32',bf16_gauge=False,compiled=False)
        for beta in [.999,.99] for frequency in [10,20,40] for method in ['qr','warm']]


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle.resolve()
    if b.exists():raise ValueError('Use a new bundle directory')
    if shutil.disk_usage(b.parent).free<5*2**30:raise RuntimeError('Require 5 GiB free')
    root=Path(__file__).resolve().parents[1]
    config=dict(width=768,tokens=64,seed=20261002,updates=200,repeats=2,modes=['all'],arms=arms(),
        maximum_seconds=600,checkpoint=False,artifact_budget_bytes=20*2**20,
        diagnostic_budgets=dict(momentum_relative=.01,update_relative=.01,update_cosine=.999),
        budget_scope='Provisional numerical/cost screen, not learning-quality selection')
    (b/'source/benchmarks').mkdir(parents=True)
    for name in ['soap.py','soap_reference.py','benchmarks/benchmark_synthetic_scaling.py',
        'benchmarks/benchmark_large_precision.py','benchmarks/rectangular_precision_gate.py',
        'benchmarks/summarize_rectangular_precision.py','benchmarks/prepare_cadence_beta.py']:
        shutil.copy2(root/name,b/'source'/name)
    shutil.copy2(root/'benchmarks/CADENCE_BETA.md',b/'PROTOCOL.md')
    (b/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (b/'manifest.json').write_text(json.dumps({str(f.relative_to(b)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in b.rglob('*') if f.is_file()},indent=2)+'\n')
    print(b)


if __name__=='__main__':main()
