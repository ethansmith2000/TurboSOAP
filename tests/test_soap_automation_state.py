from datetime import datetime,timedelta,timezone
import json
import multiprocessing
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from soap_automation_state import ORDER,reserve,transition,attach_schedule


@pytest.fixture
def ledger(tmp_path):
    path=tmp_path/'state.json'
    path.write_text(json.dumps(dict(status='active',scheduled_task_id=None,
        expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        additional_gpu_seconds_cap=2700,gpu_seconds_reserved=0,
        stages={name:dict(status='pending') for name in ORDER})))
    return path


def complete_review(path):
    reserve(path,'replication_review','review',0)
    transition(path,'replication_review','review','completed')


def test_stage_order_and_no_duplicate_reservations(ledger):
    with pytest.raises(ValueError,match='Predecessor'):reserve(ledger,'refresh_controls','a',1800)
    reserve(ledger,'replication_review','a',0)
    with pytest.raises(ValueError,match='already reserved'):reserve(ledger,'replication_review','b',0)
    with pytest.raises(ValueError,match='owner'):transition(ledger,'replication_review','b','completed')
    transition(ledger,'replication_review','a','completed')
    reserve(ledger,'refresh_controls','controls',1800)
    transition(ledger,'refresh_controls','controls','running',dict(program='controlled-job'))
    with pytest.raises(ValueError):reserve(ledger,'refresh_controls','other',1800)


def test_gpu_caps_and_reserved_total(ledger):
    complete_review(ledger)
    with pytest.raises(ValueError,match='Stage GPU'):reserve(ledger,'refresh_controls','a',1801)
    reserve(ledger,'refresh_controls','a',1800)
    transition(ledger,'refresh_controls','a','completed')
    state=json.loads(ledger.read_text());state['additional_gpu_seconds_cap']=2000
    ledger.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='Total GPU'):reserve(ledger,'qr_profile','b',600)


def test_failure_freezes_future_work(ledger):
    reserve(ledger,'replication_review','a',0)
    transition(ledger,'replication_review','a','needs_review',dict(reason='Run failure'))
    assert json.loads(ledger.read_text())['status']=='needs_review'
    with pytest.raises(ValueError,match='not active'):reserve(ledger,'refresh_controls','b',1800)


def test_completion_and_schedule_attachment(ledger):
    attach_schedule(ledger,'task1','later')
    with pytest.raises(ValueError,match='Different schedule'):attach_schedule(ledger,'task2','later')
    for stage,budget in zip(ORDER,(0,1800,600,0)):
        reserve(ledger,stage,stage,budget);transition(ledger,stage,stage,'completed')
    state=json.loads(ledger.read_text())
    assert state['status']=='completed' and state['gpu_seconds_reserved']==2400


def test_expiry_does_not_launch_work(ledger):
    state=json.loads(ledger.read_text());state['expires_at_utc']='2000-01-01T00:00:00+00:00'
    ledger.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='expired'):reserve(ledger,'replication_review','a',0)


def reserve_once(path,queue):
    try:reserve(path,'replication_review',str(os.getpid()),0);queue.put('reserved')
    except ValueError:queue.put('rejected')


def test_competing_processes_cannot_reserve_the_same_stage(ledger):
    context=multiprocessing.get_context('fork');queue=context.Queue()
    workers=[context.Process(target=reserve_once,args=(ledger,queue)) for _ in range(2)]
    for worker in workers:worker.start()
    for worker in workers:worker.join(timeout=5);assert worker.exitcode==0
    assert sorted(queue.get(timeout=1) for _ in workers)==['rejected','reserved']
