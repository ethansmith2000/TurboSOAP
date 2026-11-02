"""Approved two-control comparison; frozen inputs, no warm retraining or retries."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
from bf16_storage_screen import run,event_kind,CONTROL_CADENCES
from owt_restart import sha,write

ROOT=HERE.parent
PARENT=Path('/workspace/optimizer_replay_results/soap_bf16_replicate_20261027_seed204_v1')
CASES=('bf16_qr20','bf16_qr50')
CALLS=2000


def make_protocol(case,expected):
    cadence=CONTROL_CADENCES[case]
    ages=([20,100,200,500,1000,1500,1980] if cadence==20 else
          [50,100,200,500,1000,1500,1950])
    return dict(model_seed=204,validation_rows=1024,
        validation_calls=[50,100,200,500,1000,1500,2000],
        evaluation_hash_calls=[0,1000,2000],diagnostic_calls=[a+1 for a in ages],
        residual_calls=[501,1001,1501,ages[-1]+1],
        capture_calls=[101,201,501,1001,1501,2000],
        repeat_initial_backward=True,expected_startup=expected,
        startup_reference='verified full BF16 seed204 replication',
        check_final_state_finite=True,arm_timeout_seconds=1800)


def validate_manifest(m):
    assert tuple(m['cases'])==CASES and m['calls']==CALLS
    assert m['seed']==204 and m['schedule_horizon']==10000
    assert m['gpu_execution_seconds_cap']==1800
    assert m['numerical_error_role']=='diagnostic'
    assert tuple(m['stock_betas'])==(.95,.95,.95)
    assert m['protocols']=={c:make_protocol(c,m['reference_startup']) for c in CASES}


def verify(bundle):
    m=json.loads((bundle/'manifest.json').read_text());validate_manifest(m)
    for name,digest in m['source_files'].items():assert sha(bundle/'source'/name)==digest,name
    for name,digest in {**m['inputs'],**m['support_files']}.items():assert sha(bundle/name)==digest,name
    for name,digest in m['production_source_sha256'].items():
        assert sha(bundle/'source'/name)==digest,name
    return m


def audit_arm(bundle,case,m):
    folder=bundle/'runs'/case;p=m['protocols'][case]
    r=json.loads((folder/'result.json').read_text())
    events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
    updates=[e for e in events if e['event']=='update']
    assert [e['call'] for e in updates]==list(range(1,CALLS+1))
    assert all(e['kind']==event_kind(case,e['call']) for e in updates)
    assert all(math.isfinite(e[k]) for e in updates for k in ('loss','fb_ms','optimizer_ms','step_ms'))
    counts=Counter(e['kind'] for e in updates)
    assert counts==Counter(event_kind(case,c) for c in range(1,CALLS+1))
    assert not counts['warm'] and r['seed']==204 and r['clean_cycle_calls']==1900
    assert [v['call'] for v in r['validation']]==[0,*p['validation_calls']]
    assert all(math.isfinite(v['ce']) and v['targets']==1024*1023 for v in r['validation'])
    assert sum(v['listed_tensor_hash_preservation_checked'] for v in r['validation'])==3
    for name in ('initial_parameters_sha256','initial_gradient_sha256'):
        assert r[name]==m['reference_startup'][name]
    assert r['validation'][0]['ce']==m['reference_startup']['initial_validation_ce']
    assert r['initial_backward_repeat_passed']
    for name in ('startup_match','initial_validation_match','final_state_finite'):
        assert any(e['event']==name and e['passed'] for e in events),name
    captures=[e for e in events if e['event']=='capture_counters']
    assert [e['call'] for e in captures]==p['capture_calls']
    assert all(not e['graph_breaks'] and e['stats']['unique_graphs']==8 for e in captures)
    assert not r['dynamo_counters'].get('graph_break',{})
    geometry=json.loads((folder/'geometry.json').read_text())
    assert [g['call'] for g in geometry]==[1,*p['diagnostic_calls']]
    assert all(event_kind(case,c)=='qr' for c in p['diagnostic_calls'])
    assert all(len(g['factors'])==96 for g in geometry)
    assert all(len(g['momentum'])==48 for g in geometry[1:])
    assert all(len(g.get('coordinate_residuals',[]))==96 for g in geometry if g['call'] in p['residual_calls'])
    assert r['matrix_state_bytes']==m['reference_matrix_state_bytes']
    assert r['peak_gpu_bytes']<24*2**30
    return dict(status='passed',case=case,event_counts=dict(counts),unique_graphs=8,
        clean_cycle_calls=1900,final_state_finite=True,numerical_error_role='diagnostic')


def cached_prefix(bundle):
    r=json.loads((bundle/'warm_result.json').read_text())
    events=[json.loads(s) for s in (bundle/'warm_events.jsonl').read_text().splitlines()]
    clean=[e for e in events if e['event']=='update' and 101<=e['call']<=2000]
    assert len(clean)==1900
    assert Counter(e['kind'] for e in clean)==dict(ordinary=1710,warm=152,qr=38)
    return dict(case='cached_bf16_warm10_qr50',validation=[v for v in r['validation'] if v['call']<=2000],
        mean_step_ms=sum(e['step_ms'] for e in clean)/len(clean),
        mean_optimizer_ms=sum(e['optimizer_ms'] for e in clean)/len(clean),
        clean_cycle_calls=1900,event_counts=dict(Counter(e['kind'] for e in clean)),
        matrix_state_bytes=r['matrix_state_bytes'],timing_scope='Cached prefix on original claimed device, contextual across launches')


def campaign(bundle,m):
    start=time.monotonic();progress=dict(status='running',cases=list(CASES),completed=[],active=None)
    write(bundle/'progress.json',progress);audits=[];results=[]
    try:
        for case in CASES:
            progress['active']=case;write(bundle/'progress.json',progress)
            remaining=m['gpu_execution_seconds_cap']-(time.monotonic()-start)
            assert remaining>0,'Combined 1800-second GPU execution cap'
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--bundle',str(bundle),'--worker',case],
                check=True,timeout=remaining)
            audits.append(audit_arm(bundle,case,m))
            results.append(json.loads((bundle/'runs'/case/'result.json').read_text()))
            progress['completed'].append(case);write(bundle/'progress.json',progress)
        comparison=dict(status='completed',seed=204,calls=CALLS,results=results,cached_warm_prefix=cached_prefix(bundle),
            timing_scope='New controls sequentially on one GPU; cached warm prefix is not newly paired timing.',
            numerical_error_role='diagnostic',elapsed_seconds=time.monotonic()-start)
        write(bundle/'comparison.json',comparison)
        write(bundle/'completion_audit.json',dict(status='passed',arms=audits,new_training_calls=4000,
            model_or_optimizer_tensor_checkpoints_written=False))
        d=Path(m['compact_evidence_directory']);d.mkdir(parents=True,exist_ok=True)
        for name in ('comparison.json','completion_audit.json','manifest.json','PROTOCOL.md','cpu_checks.xml'):
            shutil.copy2(bundle/name,d/name)
        shutil.copytree(bundle/'runs',d/'runs',dirs_exist_ok=True)
        progress.update(status='completed',active=None,elapsed_seconds=time.monotonic()-start)
        write(bundle/'progress.json',progress)
    except BaseException as error:
        progress.update(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-start)
        write(bundle/'progress.json',progress);raise


def prepare(bundle,checks):
    assert not bundle.exists(),'Fresh bundle required'
    ledger=json.loads((ROOT/'automation/remaining_experiments_20261027.json').read_text())
    parent=json.loads((PARENT/'manifest.json').read_text())
    assert json.loads((PARENT/'progress.json').read_text())['status']=='completed'
    assert json.loads((PARENT/'completion_audit.json').read_text())['status']=='passed'
    for name,digest in ledger['production_source_sha256'].items():
        assert sha(ROOT/name)==sha(PARENT/'source'/name)==digest,name
    for name,digest in parent['source_files'].items():
        assert sha(PARENT/'source'/name)==digest,name
        if name!='benchmarks/bf16_storage_screen.py':assert sha(ROOT/name)==digest,name
    assert 'failures="0"' in checks.read_text() and 'errors="0"' in checks.read_text()
    bundle.mkdir();(bundle/'source').mkdir()
    names=set(parent['source_files'])|{'benchmarks/bf16_refresh_controls.py','tests/test_bf16_refresh_controls.py'}
    for name in sorted(names):
        target=bundle/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    for name in ('train.npy','validation.npy'):(bundle/name).symlink_to((PARENT/name).resolve())
    shutil.copy2(ROOT/'BF16_REFRESH_CONTROLS_20261027.md',bundle/'PROTOCOL.md')
    shutil.copy2(checks,bundle/'cpu_checks.xml')
    for source,target in [('runs/bf16/result.json','warm_result.json'),('runs/bf16/events.jsonl','warm_events.jsonl'),
                          ('manifest.json','parent_manifest.json'),('completion_audit.json','parent_audit.json')]:
        shutil.copy2(PARENT/source,bundle/target)
    warm=json.loads((bundle/'warm_result.json').read_text())
    m=dict(cases=list(CASES),calls=CALLS,seed=204,schedule_horizon=10000,stock_betas=[.95,.95,.95],
        numerical_error_role='diagnostic',gpu_execution_seconds_cap=1800,
        reference_startup=parent['reference_startup'],reference_matrix_state_bytes=warm['matrix_state_bytes'],
        protocols={c:make_protocol(c,parent['reference_startup']) for c in CASES},
        source_files={name:sha(bundle/'source'/name) for name in sorted(names)},
        inputs={name:sha(bundle/name) for name in ('train.npy','validation.npy')},
        support_files={name:sha(bundle/name) for name in ('PROTOCOL.md','cpu_checks.xml','warm_result.json','warm_events.jsonl','parent_manifest.json','parent_audit.json')},
        production_source_sha256=ledger['production_source_sha256'],parent_bundle=str(PARENT),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        compact_evidence_directory=str(ROOT/'reports/bf16_refresh_controls_20261027'))
    assert m['inputs']==parent['inputs']
    write(bundle/'manifest.json',m);verify(bundle);cached_prefix(bundle)
    print(json.dumps(dict(status='prepared',bundle=str(bundle),sources=len(names))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--checks',type=Path)
    p.add_argument('--worker',choices=CASES);a=p.parse_args()
    if a.prepare:prepare(a.bundle,a.checks);return
    assert os.environ.get('GPU_CLAIM_UUIDS'),'Lifetime GPU claim required'
    assert Path(__file__).resolve()==a.bundle/'source/benchmarks/bf16_refresh_controls.py'
    m=verify(a.bundle)
    if a.worker:run(a.bundle,a.worker,CALLS,protocol=m['protocols'][a.worker])
    else:campaign(a.bundle,m)


if __name__=='__main__':main()
