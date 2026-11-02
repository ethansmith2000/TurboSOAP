"""One full FP32-basis candidate, cached quality control and short timing replay."""
import argparse
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
from bf16_full_horizon import make_protocol,audit_arm
from bf16_storage_screen import run,event_kind
from owt_restart import sha,write

CASES=('qr10','bf16_fp32_basis')


def worker_protocol(case,manifest):
    p=copy.deepcopy(manifest['protocol'])
    if case=='qr10':
        for key in ('validation_calls','evaluation_hash_calls','diagnostic_calls','residual_calls','capture_calls'):
            p[key]=[call for call in p[key] if call<=201]
        p['validation_calls'].append(201)
        p['capture_calls']=sorted(set([*p['capture_calls'],201]))
    p.update(expected_startup=copy.deepcopy(manifest['reference_startup']),
             startup_reference=f"hash-verified completed stock QR10, seed{manifest['seed']} full horizon")
    return p


def validate_manifest(m):
    assert tuple(m['cases'])==CASES
    assert m['calls_per_case']==dict(qr10=201,bf16_fp32_basis=10000)
    assert m['seed']==205 and m['schedule_horizon']==10000
    assert tuple(m['stock_betas'])==(.95,.95,.95)
    assert m['protocol']==make_protocol()


def check_reference_reproduction(prefix,reference):
    assert prefix['initial_parameters_sha256']==reference['initial_parameters_sha256']
    assert prefix['initial_gradient_sha256']==reference['initial_gradient_sha256']
    old={v['call']:v['ce'] for v in reference['validation']}
    matched=[]
    for row in prefix['validation']:
        if row['call'] in old:
            assert row['ce']==old[row['call']],('Control reproduction mismatch',row['call'],row['ce'],old[row['call']])
            matched.append(row['call'])
    assert matched==[0,50,100,200]
    return matched


def prefix_audit(bundle,reference,diagnostics_advisory=False):
    folder=bundle/'runs/qr10';r=json.loads((folder/'result.json').read_text())
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    updates=[e for e in events if e['event']=='update']
    assert [e['call'] for e in updates]==list(range(1,202))
    assert all(e['kind']==event_kind('qr10',e['call']) for e in updates)
    assert r['clean_cycle_calls']==100 and r['seed']==reference['seed']
    assert r['initial_backward_repeat_passed']
    if not diagnostics_advisory:
        assert r['original_geometry_guard_passed'] and r['precision_specific_guard_passed']
    assert any(e['event']=='startup_match' and e['passed'] for e in events)
    assert not r['dynamo_counters'].get('graph_break',{})
    return dict(status='passed',case='qr10',calls=201,
                diagnostics_advisory=diagnostics_advisory,
                original_geometry_guard_passed=r['original_geometry_guard_passed'],
                precision_specific_guard_passed=r['precision_specific_guard_passed'],
                exactly_reproduced_validation_calls=check_reference_reproduction(r,reference))


def campaign(bundle,m,runner_path=None):
    cases=tuple(m['cases']);seed=m['seed'];candidate_case=cases[1]
    runner=Path(runner_path) if runner_path is not None else Path(__file__).resolve()
    advisory=m.get('numerical_error_role')=='diagnostic'
    start=time.monotonic();progress=dict(status='running',seed=seed,cases=list(cases),
        calls_per_case=m['calls_per_case'],completed=[],active=None)
    write(bundle/'progress.json',progress)
    reference=json.loads((bundle/'reference_result.json').read_text());audits=[]
    try:
        for case in cases:
            progress['active']=case;write(bundle/'progress.json',progress)
            remaining=7200-(time.monotonic()-start)
            assert remaining>0,'Two-hour execution limit'
            subprocess.run([sys.executable,str(runner),'--bundle',str(bundle),'--worker',case],
                check=True,timeout=min(m['protocol']['arm_timeout_seconds'],remaining))
            audits.append(prefix_audit(bundle,reference,diagnostics_advisory=advisory) if case=='qr10' else audit_arm(bundle,case,m))
            progress['completed'].append(case);write(bundle/'progress.json',progress)
            assert time.monotonic()-start<7200,'Two-hour execution limit'
        candidate=json.loads((bundle/'runs'/candidate_case/'result.json').read_text())
        prefix=json.loads((bundle/'runs/qr10/result.json').read_text())
        window={}
        for case in cases:
            events=[json.loads(line) for line in (bundle/'runs'/case/'events.jsonl').read_text().splitlines()]
            clean=[e for e in events if e['event']=='update' and 101<=e['call']<=200]
            assert len(clean)==100
            window[case]=sum(e['step_ms'] for e in clean)/100
        comparison=dict(status='completed',seed=seed,candidate=candidate,
            cached_quality_reference=reference,quality_reference_source=m['reference_source_bundle'],
            final_ce_delta=candidate['validation'][-1]['ce']-reference['validation'][-1]['ce'],
            matrix_state_saving_percent=100*(1-candidate['matrix_state_bytes']/prefix['matrix_state_bytes']),
            memory_reference_source='new stock replay, same matrix coverage',
            numerical_error_role=m.get('numerical_error_role','legacy_engineering_screen'),
            paired_prefix_timing=dict(calls=[101,200],number_of_calls=100,mean_step_ms=window,
                measured_step_time_saving_percent=100*(1-window[candidate_case]/window[cases[0]])),
            timing_scope='Matched 100-call prefix on the newly claimed device; cached full-run timings are contextual, not a newly paired full-horizon wall-time measurement.',
            elapsed_seconds=time.monotonic()-start)
        write(bundle/'comparison.json',comparison)
        write(bundle/'completion_audit.json',dict(status='passed',arms=audits,
            cached_reference_verified=True,cached_reference_call50_100_200_reproduced=True,
            new_training_calls=10201,new_training_targets=10201*32*1023,
            model_or_optimizer_tensor_checkpoints_written=False))
        destination=Path(m['compact_evidence_directory']);destination.mkdir(parents=True,exist_ok=True)
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
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--worker',choices=CASES);a=p.parse_args();b=a.bundle
    assert os.environ.get('GPU_CLAIM_UUIDS'),'Lifetime GPU claim required'
    assert Path(__file__).resolve()==b/'source/benchmarks/bf16_hybrid_confirm.py'
    m=json.loads((b/'manifest.json').read_text());validate_manifest(m)
    for name,digest in m['source_files'].items():assert sha(b/'source'/name)==digest,name
    for name,digest in {**m['inputs'],**m['support_files']}.items():assert sha(b/name)==digest,name
    if a.worker:run(b,a.worker,m['calls_per_case'][a.worker],protocol=worker_protocol(a.worker,m))
    else:campaign(b,m)


if __name__=='__main__':main()
