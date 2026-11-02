"""Fixed full-BF16 seed204 replication, reused quality control and timing replay."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
from bf16_full_horizon import make_protocol
from bf16_hybrid_confirm import campaign,worker_protocol
from bf16_storage_screen import run
from owt_restart import sha

CASES=('qr10','bf16')


def normalize_reference(original):
    assert original['seed']==204 and original['calls']==10000
    assert original['profile']=='stock' and original['arm']=='qr'
    reference=copy.deepcopy(original)
    reference['initial_gradient_sha256']=reference['initial_gradient']['hash']
    return reference


def validate_manifest(m):
    assert tuple(m['cases'])==CASES
    assert m['calls_per_case']==dict(qr10=201,bf16=10000)
    assert m['seed']==204 and m['schedule_horizon']==10000
    assert tuple(m['stock_betas'])==(.95,.95,.95)
    assert m['protocol']==make_protocol(204)
    assert m['numerical_error_role']=='diagnostic'


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--worker',choices=CASES);a=p.parse_args();b=a.bundle
    assert os.environ.get('GPU_CLAIM_UUIDS'),'Lifetime GPU claim required'
    assert Path(__file__).resolve()==b/'source/benchmarks/bf16_seed204_replicate.py'
    m=json.loads((b/'manifest.json').read_text());validate_manifest(m)
    for name,digest in m['source_files'].items():assert sha(b/'source'/name)==digest,name
    for name,digest in {**m['inputs'],**m['support_files']}.items():assert sha(b/name)==digest,name
    if a.worker:run(b,a.worker,m['calls_per_case'][a.worker],protocol=worker_protocol(a.worker,m))
    else:campaign(b,m,runner_path=Path(__file__).resolve())


if __name__=='__main__':main()
