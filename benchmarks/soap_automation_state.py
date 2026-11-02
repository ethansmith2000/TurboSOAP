"""Atomic reservations for the bounded app-owned SOAP experiment monitor.

This CLI records ownership/budget only; it never launches or stops GPU jobs.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile

ORDER=('replication_review','refresh_controls','qr_profile','final_report')
GPU_CAPS={'replication_review':0,'refresh_controls':1800,'qr_profile':600,'final_report':0}


def now():return datetime.now(timezone.utc)


def save(path,state):
    with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix=path.name+'.',delete=False) as stream:
        json.dump(state,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        temporary=stream.name
    os.replace(temporary,path)


@contextmanager
def locked(path):
    path=Path(path)
    with path.with_suffix(path.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(path.read_text())
        yield path,state


def reserve(path,stage,owner,gpu_seconds):
    with locked(path) as (path,state):
        if state['status']!='active':raise ValueError('Automation is not active')
        if now()>=datetime.fromisoformat(state['expires_at_utc']):raise ValueError('Automation expired')
        if stage not in ORDER:raise ValueError('Unknown stage')
        row=state['stages'][stage]
        if row['status']!='pending':raise ValueError('Stage already reserved or finished; reconcile it, never duplicate it')
        if any(state['stages'][name]['status']!='completed' for name in ORDER[:ORDER.index(stage)]):
            raise ValueError('Predecessor stage incomplete')
        if any(r['status'] in ('preparing','running') for r in state['stages'].values()):
            raise ValueError('Another stage is active')
        if type(gpu_seconds) is not int or not 0<=gpu_seconds<=GPU_CAPS[stage]:raise ValueError('Stage GPU budget exceeded')
        if state['gpu_seconds_reserved']+gpu_seconds>state['additional_gpu_seconds_cap']:
            raise ValueError('Total GPU budget exceeded')
        row.update(status='preparing',owner=owner,reserved_at_utc=now().isoformat(),gpu_seconds_reserved=gpu_seconds)
        state['gpu_seconds_reserved']+=gpu_seconds
        save(path,state)
        return row


def transition(path,stage,owner,status,details=None):
    with locked(path) as (path,state):
        row=state['stages'][stage]
        if row.get('owner')!=owner:raise ValueError('Reservation owner mismatch')
        allowed={'preparing':('running','completed','needs_review'),
                 'running':('completed','needs_review')}
        if status not in allowed.get(row['status'],()):raise ValueError('Invalid stage transition')
        row.update(status=status,updated_at_utc=now().isoformat())
        if details is not None:row['details']=details
        if status=='needs_review':state['status']='needs_review'
        elif all(r['status']=='completed' for r in state['stages'].values()):state['status']='completed'
        save(path,state)
        return row


def attach_schedule(path,task_id,next_run):
    with locked(path) as (path,state):
        if state.get('scheduled_task_id') not in (None,task_id):raise ValueError('Different schedule already attached')
        state.update(scheduled_task_id=task_id,initial_next_run_at_utc=next_run)
        save(path,state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--state',required=True,type=Path)
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('status')
    r=sub.add_parser('reserve');r.add_argument('--stage',choices=ORDER,required=True)
    r.add_argument('--owner',required=True);r.add_argument('--gpu-seconds',type=int,required=True)
    t=sub.add_parser('transition');t.add_argument('--stage',choices=ORDER,required=True)
    t.add_argument('--owner',required=True);t.add_argument('--status',choices=('running','completed','needs_review'),required=True)
    t.add_argument('--details-file',type=Path)
    a=sub.add_parser('attach');a.add_argument('--task-id',required=True);a.add_argument('--next-run',required=True)
    args=p.parse_args()
    if args.action=='status':print(args.state.read_text());return
    if args.action=='reserve':result=reserve(args.state,args.stage,args.owner,args.gpu_seconds)
    elif args.action=='transition':
        details=json.loads(args.details_file.read_text()) if args.details_file else None
        result=transition(args.state,args.stage,args.owner,args.status,details)
    else:attach_schedule(args.state,args.task_id,args.next_run);result={'attached':args.task_id}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
