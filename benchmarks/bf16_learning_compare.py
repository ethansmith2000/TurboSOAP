"""Frozen five-arm stock-beta OWT learning comparison, sequential lifetime claim."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
from bf16_storage_screen import run,observation_calls,event_kind
from owt_restart import sha,write

CASES=('qr10','original','compiled_fp32','bf16','bf16_fp32_basis')


def make_protocol(expected,startup_reference):
    return dict(validation_rows=1024,validation_calls=[50,100,200,500,1000,1500,2000],
        evaluation_hash_calls=[0,1000,2000],diagnostic_calls=sorted(observation_calls(2000)),
        residual_calls=[501,1001,1501,1991],capture_calls=[51,101,201,501,1001,1501,2000],
        repeat_initial_backward=True,expected_startup=expected,startup_reference=startup_reference,
        arm_timeout_seconds=1800)


def validate_manifest(manifest):
    assert manifest['calls']==2000
    assert tuple(manifest['cases'])==CASES
    assert manifest['schedule_horizon']==10000
    assert tuple(manifest['stock_betas'])==(.95,.95,.95)
    assert manifest['protocol']['validation_rows']==1024
    assert manifest['protocol']['diagnostic_calls']==sorted(observation_calls(2000))


def audit_arm(bundle,case):
    folder=bundle/'runs'/case
    result=json.loads((folder/'result.json').read_text())
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    updates=[e for e in events if e['event']=='update']
    assert [e['call'] for e in updates]==list(range(1,2001))
    assert all(e['kind']==event_kind(case,e['call']) for e in updates)
    assert result['clean_cycle_calls']==1900
    expected_counts=Counter(event_kind(case,c) for c in range(1,2001))
    assert Counter(e['kind'] for e in updates)==expected_counts
    assert [v['call'] for v in result['validation']]==[0,50,100,200,500,1000,1500,2000]
    assert all(v['targets']==1024*1023 for v in result['validation'])
    assert sum(v['listed_tensor_hash_preservation_checked'] for v in result['validation'])==3
    geometry=json.loads((folder/'geometry.json').read_text())
    assert [g['call'] for g in geometry]==[1,*sorted(observation_calls(2000))]
    assert all(len(g['factors'])==96 for g in geometry)
    assert all(len(g['momentum'])==48 for g in geometry[1:])
    assert all(len(g.get('coordinate_residuals',[]))==96 for g in geometry if g['call'] in (501,1001,1501,1991))
    captures=[e for e in events if e['event']=='capture_counters']
    assert all(not e['graph_breaks'] for e in captures)
    if case not in ('qr10','original'):assert all(e['stats']['unique_graphs']==12 for e in captures)
    assert result['initial_backward_repeat_passed']
    assert any(e['event']=='startup_match' and e['passed'] for e in events)
    return dict(case=case,status='passed',training_calls=2000,event_counts=dict(expected_counts),
        clean_cycle_calls=1900,factor_observations=sum(len(g['factors']) for g in geometry),
        actual_m_observations=sum(len(g['momentum']) for g in geometry),
        coordinate_residual_observations=sum(len(g.get('coordinate_residuals',[])) for g in geometry),
        graph_breaks=0,original_geometry_guard_passed=result['original_geometry_guard_passed'],
        precision_specific_guard_passed=result['precision_specific_guard_passed'])


def campaign(bundle,manifest):
    start=time.monotonic();results=[];audits=[]
    progress=dict(status='running',cases=list(CASES),completed=[],active=None)
    write(bundle/'progress.json',progress)
    try:
        for case in CASES:
            progress['active']=case;write(bundle/'progress.json',progress)
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--bundle',str(bundle),'--worker',case],
                check=True,timeout=manifest['protocol']['arm_timeout_seconds'])
            audits.append(audit_arm(bundle,case))
            results.append(json.loads((bundle/'runs'/case/'result.json').read_text()))
            progress['completed'].append(case);write(bundle/'progress.json',progress)
            assert time.monotonic()-start<7200,'Two-hour campaign limit'
        assert len({r['initial_parameters_sha256'] for r in results})==1
        assert len({r['initial_gradient_sha256'] for r in results})==1
        assert len({r['validation'][0]['ce'] for r in results})==1
        baseline=results[0];warm=results[1];compiled=results[2]
        for r in results:
            r['comparisons']={}
            for name,reference in [('stock_qr10',baseline),('existing_warm',warm),('compiled_fp32',compiled)]:
                r['comparisons'][name]=dict(final_ce_delta=r['validation'][-1]['ce']-reference['validation'][-1]['ce'],
                    mean_step_time_saving_percent=100*(1-r['mean_step_ms']/reference['mean_step_ms']),
                    matrix_state_saving_percent=100*(1-r['matrix_state_bytes']/reference['matrix_state_bytes']))
        write(bundle/'comparison.json',dict(status='completed',calls_per_arm=2000,seed=204,
            elapsed_seconds=time.monotonic()-start,results=results))
        write(bundle/'completion_audit.json',dict(status='passed',arms=audits,
            same_initial_parameters=True,same_initial_gradients=True,same_initial_ce=True,
            training_targets=5*2000*32*1023,model_or_optimizer_tensor_checkpoints_written=False))
        destination=Path(manifest['compact_evidence_directory']);destination.mkdir(parents=True,exist_ok=True)
        for name in ('comparison.json','completion_audit.json','manifest.json'):
            shutil.copy2(bundle/name,destination/name)
        shutil.copytree(bundle/'runs',destination/'runs',dirs_exist_ok=True)
        progress.update(status='completed',active=None,elapsed_seconds=time.monotonic()-start)
        write(bundle/'progress.json',progress)
        print(json.dumps(progress),flush=True)
    except BaseException as error:
        progress.update(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-start)
        write(bundle/'progress.json',progress)
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',required=True,type=Path)
    parser.add_argument('--worker',choices=CASES);args=parser.parse_args();bundle=args.bundle
    assert os.environ.get('GPU_CLAIM_UUIDS'),'Lifetime GPU claim required'
    assert Path(__file__).resolve()==bundle/'source/benchmarks/bf16_learning_compare.py'
    manifest=json.loads((bundle/'manifest.json').read_text());validate_manifest(manifest)
    for name,digest in manifest['source_files'].items():assert sha(bundle/'source'/name)==digest,name
    for name,digest in manifest['inputs'].items():assert sha(bundle/name)==digest,name
    if args.worker:run(bundle,args.worker,manifest['calls'],protocol=manifest['protocol'])
    else:campaign(bundle,manifest)


if __name__=='__main__':main()
