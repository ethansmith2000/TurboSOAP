"""Prospective fixed-work timing checks, independent of GPU execution."""
import math
import statistics


def chunk_boundaries(updates,evaluations,chunk=25):
    if updates<=50 or chunk<1 or any(type(x) is not int or not 0<x<=updates for x in evaluations):
        raise ValueError('Invalid prospective observation schedule')
    return sorted(set(range(chunk,updates+1,chunk))|set(evaluations)|{updates})


def timing_repeat(runs):
    if len(runs)!=3 or any(r['steps']!=3500 for r in runs):
        raise ValueError('Need exactly three complete equal3500 controls')
    values=[[r[k] for r in runs] for k in ['training_seconds','optimizer_ms']]
    if any(not math.isfinite(x) or x<=0 for xs in values for x in xs):
        raise ValueError('Missing or nonfinite timing')
    wall,opt=[(max(xs)-min(xs))/statistics.mean(xs) for xs in values]
    return dict(wall_range_over_mean=wall,optimizer_range_over_mean=opt,
        mean_training_seconds=statistics.mean(values[0]),mean_optimizer_ms=statistics.mean(values[1]),
        repeat_gate=wall<=.03 and opt<=.02)
