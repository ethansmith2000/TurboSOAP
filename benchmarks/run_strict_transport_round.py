"""One claimed-GPU lifetime: shape gate, then conditional paired training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle.resolve()
    assert HERE.parent==b/'source'
    manifest=json.loads((b/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((b/'source'/name).read_bytes()).hexdigest()==digest,name
    subprocess.run([sys.executable,'-u',str(HERE/'strict_transport_cost_gate.py'),'--bundle',str(b)],check=True)
    gate=json.loads((b/'cost_gate.json').read_text());assert gate['cifar_gate_pass']
    subprocess.run([sys.executable,'-u',str(HERE/'cifar_lr_screen.py'),'--bundle',str(b)],check=True)


if __name__=='__main__':main()
