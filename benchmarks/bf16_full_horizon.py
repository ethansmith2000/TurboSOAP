"""Fresh model-seed paired full-horizon confirmation; no adaptive successor."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
from bf16_storage_screen import run,event_kind
from owt_restart import sha,write

CASES=('qr10','bf16')
CALLS=10000
SEED=205


def make_protocol(seed=SEED):
    ages=(10,20,50,100,150,200,500,510,1000,1010,1500,1510,1990,
          2500,2510,5000,5010,7500,7510,9950,9990)
    return dict(model_seed=seed,validation_rows=1024,
        validation_calls=[50,100,200,500,1000,1500,2000,2500,3000,4000,5000,
                          6000,7000,7500,8000,9000,10000],
        evaluation_hash_calls=[0,1000,5000,10000],diagnostic_calls=[a+1 for a in ages],
        residual_calls=[501,1001,1501,1991,2501,5001,7501,9991],
        capture_calls=[51,101,201,501,1001,2501,5001,7501,10000],
        repeat_initial_backward=True,arm_timeout_seconds=5400)


def validate_manifest(manifest):
    assert tuple(manifest['cases'])==CASES
    assert manifest['calls']==CALLS and manifest['seed']==SEED
    assert manifest['schedule_horizon']==CALLS
    assert tuple(manifest['stock_betas'])==(.95,.95,.95)
    assert manifest['protocol']==make_protocol()


def worker_protocol(bundle,case,manifest):
    protocol=copy.deepcopy(manifest['protocol'])
    if case=='bf16':
        baseline=json.loads((bundle/'runs/qr10/result.json').read_text())
        first=next(json.loads(line) for line in (bundle/'runs/qr10/events.jsonl').read_text().splitlines()
                   if json.loads(line)['event']=='initial_gradient')
        protocol.update(expected_startup=dict(initial_parameters_sha256=baseline['initial_parameters_sha256'],
            initial_gradient_sha256=baseline['initial_gradient_sha256'],initial_loss=first['loss'],
            initial_norm=first['norm'],initial_validation_ce=baseline['validation'][0]['ce']),
            startup_reference='this paired seed205 stock QR10 control')
    return protocol


def audit_arm(bundle,case,manifest):
    seed=manifest.get('seed',SEED)
    folder=bundle/'runs'/case
    result=json.loads((folder/'result.json').read_text())
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    updates=[e for e in events if e['event']=='update']
    assert [e['call'] for e in updates]==list(range(1,CALLS+1))
    assert all(e['kind']==event_kind(case,e['call']) for e in updates)
    assert result['seed']==seed and result['clean_cycle_calls']==9900
    expected_counts=Counter(event_kind(case,c) for c in range(1,CALLS+1))
    assert Counter(e['kind'] for e in updates)==expected_counts
    protocol=manifest['protocol']
    assert [v['call'] for v in result['validation']]==[0,*protocol['validation_calls']]
    assert all(v['targets']==1024*1023 for v in result['validation'])
    assert sum(v['listed_tensor_hash_preservation_checked'] for v in result['validation'])==4
    geometry=json.loads((folder/'geometry.json').read_text())
    assert [g['call'] for g in geometry]==[1,*protocol['diagnostic_calls']]
    assert all(len(g['factors'])==96 for g in geometry)
    assert all(len(g['momentum'])==48 for g in geometry[1:])
    assert all(len(g.get('coordinate_residuals',[]))==96 for g in geometry if g['call'] in protocol['residual_calls'])
    captures=[e for e in events if e['event']=='capture_counters']
    assert [e['call'] for e in captures]==protocol['capture_calls']
    assert all(not e['graph_breaks'] for e in captures)
    if case in ('bf16','bf16_fp32_basis'):
        assert all(e['stats']['unique_graphs']==12 for e in captures)
        assert any(e['event']=='startup_match' and e['passed'] for e in events)
        assert any(e['event']=='initial_validation_match' and e['passed'] for e in events)
    assert result['initial_backward_repeat_passed']
    return dict(case=case,status='passed',calls=CALLS,seed=seed,event_counts=dict(expected_counts),
        factor_observations=sum(len(g['factors']) for g in geometry),
        actual_m_observations=sum(len(g['momentum']) for g in geometry),
        coordinate_residual_observations=sum(len(g.get('coordinate_residuals',[])) for g in geometry),
        graph_breaks=0,original_geometry_guard_passed=result['original_geometry_guard_passed'],
        precision_specific_guard_passed=result['precision_specific_guard_passed'])


def campaign(bundle,manifest):
    start=time.monotonic();results=[];audits=[]
    progress=dict(status='running',seed=SEED,calls_per_arm=CALLS,cases=list(CASES),completed=[],active=None)
    write(bundle/'progress.json',progress)
    try:
        for case in CASES:
            progress['active']=case;write(bundle/'progress.json',progress)
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--bundle',str(bundle),'--worker',case],
                check=True,timeout=manifest['protocol']['arm_timeout_seconds'])
            audits.append(audit_arm(bundle,case,manifest))
            results.append(json.loads((bundle/'runs'/case/'result.json').read_text()))
            progress['completed'].append(case);write(bundle/'progress.json',progress)
            assert time.monotonic()-start<10800,'Three-hour paired campaign budget'
        assert len({r['initial_parameters_sha256'] for r in results})==1
        assert len({r['initial_gradient_sha256'] for r in results})==1
        assert len({r['validation'][0]['ce'] for r in results})==1
        reference,candidate=results
        comparison=dict(status='completed',calls_per_arm=CALLS,seed=SEED,
            elapsed_seconds=time.monotonic()-start,results=results,
            final_ce_delta=candidate['validation'][-1]['ce']-reference['validation'][-1]['ce'],
            measured_step_time_saving_percent=100*(1-candidate['mean_step_ms']/reference['mean_step_ms']),
            matrix_state_saving_percent=100*(1-candidate['matrix_state_bytes']/reference['matrix_state_bytes']))
        write(bundle/'comparison.json',comparison)
        write(bundle/'completion_audit.json',dict(status='passed',arms=audits,
            same_initial_parameters=True,same_initial_gradients=True,same_initial_ce=True,
            training_targets=2*CALLS*32*1023,model_or_optimizer_tensor_checkpoints_written=False))
        destination=Path(manifest['compact_evidence_directory']);destination.mkdir(parents=True,exist_ok=True)
        for name in ('comparison.json','completion_audit.json','manifest.json'):
            shutil.copy2(bundle/name,destination/name)
        shutil.copytree(bundle/'runs',destination/'runs',dirs_exist_ok=True)
        progress.update(status='completed',active=None,elapsed_seconds=time.monotonic()-start)
        write(bundle/'progress.json',progress);print(json.dumps(progress),flush=True)
    except BaseException as error:
        progress.update(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-start)
        write(bundle/'progress.json',progress)
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',required=True,type=Path)
    parser.add_argument('--worker',choices=CASES);args=parser.parse_args();bundle=args.bundle
    assert os.environ.get('GPU_CLAIM_UUIDS'),'Lifetime GPU claim required'
    assert Path(__file__).resolve()==bundle/'source/benchmarks/bf16_full_horizon.py'
    manifest=json.loads((bundle/'manifest.json').read_text());validate_manifest(manifest)
    for name,digest in manifest['source_files'].items():assert sha(bundle/'source'/name)==digest,name
    for name,digest in manifest['inputs'].items():assert sha(bundle/name)==digest,name
    for name,digest in manifest['support_files'].items():assert sha(bundle/name)==digest,name
    if args.worker:run(bundle,args.worker,CALLS,protocol=worker_protocol(bundle,args.worker,manifest))
    else:campaign(bundle,manifest)


if __name__=='__main__':main()
