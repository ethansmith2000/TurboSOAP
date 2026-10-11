"""Host observations and short, separately scoped CUDA profiler evidence."""
from contextlib import nullcontext
import gc
import gzip
import json
import resource
import subprocess
import time

import torch


PHASES = ('augmentation', 'model_backward', 'optimizer')


def interval_union_us(intervals):
    total = 0.0
    end = None
    for left, right in sorted(intervals):
        if right < left:
            raise ValueError('Reversed interval')
        total += right - max(left, end if end is not None else left) if end is None or right > end else 0
        end = max(end, right) if end is not None else right
    return total


def range_over_mean(values):
    if not values or min(values) <= 0:
        raise ValueError('Positive samples required')
    return (max(values) - min(values)) / (sum(values) / len(values))


class HostTiming:
    def __init__(self):
        self.profile = False
        self.rows = []
        self.collections = []
        self.pending_gc = {}
        gc.callbacks.append(self.gc_callback)

    def gc_callback(self, phase, info):
        generation = info['generation']
        if phase == 'start':
            self.pending_gc[generation] = time.perf_counter()
        else:
            start = self.pending_gc.pop(generation, None)
            if start is not None:
                self.collections.append(dict(generation=generation, wall_ms=1000*(time.perf_counter()-start),
                                             collected=info['collected'], uncollectable=info['uncollectable']))

    def scope(self, phase):
        return torch.profiler.record_function('timing/'+phase) if self.profile else nullcontext()

    def begin(self):
        self.points = [(time.perf_counter(), time.thread_time(), time.process_time())]

    def mark(self):
        self.points.append((time.perf_counter(), time.thread_time(), time.process_time()))

    def finish(self):
        self.mark()
        self.rows.append({phase: {name: 1000*(b[i]-a[i]) for i,name in enumerate(('host_wall_ms','thread_cpu_ms','process_cpu_ms'))}
                          for phase,a,b in zip(PHASES,self.points,self.points[1:])})

    def reset(self):
        self.rows.clear()
        self.collections.clear()
        self.usage = resource.getrusage(resource.RUSAGE_SELF)

    def flush(self):
        after = resource.getrusage(resource.RUSAGE_SELF)
        result = dict(steps=len(self.rows), phases={phase: {name: sum(row[phase][name] for row in self.rows)
                      for name in ('host_wall_ms','thread_cpu_ms','process_cpu_ms')} for phase in PHASES},
                      gc=list(self.collections), voluntary_context_switches=after.ru_nvcsw-self.usage.ru_nvcsw,
                      involuntary_context_switches=after.ru_nivcsw-self.usage.ru_nivcsw)
        self.reset()
        return result

    def close(self):
        gc.callbacks.remove(self.gc_callback)


def telemetry():
    try:
        value = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu',
                                         '--format=csv,noheader'], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        value = str(exc)
    from torch._dynamo.utils import counters
    return dict(gpu=value, compiler_counters={k:dict(v) for k,v in counters.items()}, gc_counts=gc.get_count())


def profile_summary(prof, bundle, name):
    # Kineto emits CUDA user annotations as device events too. Their spans are
    # not kernel work; filter exported trace categories and retain CPU scope rows.
    try:
        from .analyze_timing_traces import analyze_trace
    except ImportError:
        from analyze_timing_traces import analyze_trace
    raw = bundle/(name+'.json')
    prof.export_chrome_trace(str(raw))
    with raw.open() as source:
        corrected = analyze_trace(json.load(source))
    target = bundle/(name+'.json.gz')
    with raw.open('rb') as source, gzip.open(target,'wb') as out:
        import shutil
        shutil.copyfileobj(source,out)
    raw.unlink()
    device = corrected['gpu_work']
    return dict(phases=corrected['phases'], device_events=device['count'],
                device_sum_us=device['sum_us'], device_union_us=device['union_us'],
                device_span_us=device.get('span_us',0), ignored_gpu_annotations=corrected['ignored_gpu_annotations'],
                unattributed_gpu_events=corrected['unattributed_gpu_events'],
                trace=target.name, trace_bytes=target.stat().st_size,
                caveat='Profiler perturbation: diagnostic replay only; excluded from primary repeat timing.')
