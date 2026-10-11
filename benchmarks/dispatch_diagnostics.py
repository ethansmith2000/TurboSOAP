"""Own-process CPU accounting outside update timing; no policy feedback."""
import os
from pathlib import Path
import resource
import subprocess
import time
try:
    from .timing_diagnostics import HostTiming, PHASES
except ImportError:
    from timing_diagnostics import HostTiming, PHASES

def parse_stat(text):
    # comm may contain whitespace and parentheses; fields after the last ')' start at field3.
    parts=text[text.rindex(')')+2:].split()
    return dict(user_ticks=int(parts[11]),system_ticks=int(parts[12]),last_cpu=int(parts[36]))

def thread_snapshot():
    rows={}
    for p in Path('/proc/self/task').iterdir():
        try:
            tid=int(p.name);stats=parse_stat((p/'stat').read_text());sched=(p/'schedstat').read_text().split()
            rows[tid]=dict(name=(p/'comm').read_text().strip(),affinity=sorted(os.sched_getaffinity(tid)),
                          runtime_ns=int(sched[0]),runqueue_wait_ns=int(sched[1]),timeslices=int(sched[2]),**stats)
        except (FileNotFoundError,ProcessLookupError):pass
    return rows

def thread_deltas(before,after):
    rows=[]
    for tid,end in after.items():
        if tid not in before:continue
        start=before[tid];keys=['runtime_ns','runqueue_wait_ns','timeslices','user_ticks','system_ticks']
        delta={k:end[k]-start[k] for k in keys};assert all(x>=0 for x in delta.values())
        rows.append(dict(tid=tid,name=end['name'],affinity=end['affinity'],last_cpu=end['last_cpu'],**delta))
    return rows

class DispatchHostTiming(HostTiming):
    def __init__(self):
        super().__init__();self.flush_count=0

    def reset(self):
        super().reset();self.thread_start=thread_snapshot()

    def begin(self):
        super().begin();u=resource.getrusage(resource.RUSAGE_THREAD)
        self.points[0]=(*self.points[0],u.ru_utime,u.ru_stime)

    def mark(self):
        super().mark();u=resource.getrusage(resource.RUSAGE_THREAD)
        self.points[-1]=(*self.points[-1],u.ru_utime,u.ru_stime)

    def finish(self):
        super().finish()
        for phase,a,b in zip(PHASES,self.points,self.points[1:]):
            self.rows[-1][phase].update(main_user_cpu_ms=1000*(b[3]-a[3]),main_system_cpu_ms=1000*(b[4]-a[4]))

    def flush(self):
        started=time.monotonic();before=self.thread_start;after=thread_snapshot()
        main_cpu={phase:{key:sum(row[phase][key] for row in self.rows)
                        for key in ['main_user_cpu_ms','main_system_cpu_ms']} for phase in PHASES}
        result=super().flush()
        for phase in PHASES:
            result['phases'][phase].update(main_cpu[phase])
        result['thread_interval']=thread_deltas(before,after)
        result['thread_interval_scope']='Reset-to-flush interval includes offline guards/probes/hashes; not phase-specific. Runtime and runqueue wait are distinct; user/system ticks have kernel clock resolution.'
        self.flush_count+=1
        if self.flush_count%10==0:
            result['gpu_sample']=subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
                '--query-gpu=uuid,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu',
                '--format=csv,noheader'],text=True,timeout=5).strip()
        result['observation_seconds']=time.monotonic()-started
        return result
