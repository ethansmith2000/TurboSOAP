"""Separate real GPU work from profiler GPU annotation spans using Chrome categories."""
import gzip
import json
from pathlib import Path
import sys
try:
    from .timing_diagnostics import interval_union_us
except ImportError:
    from timing_diagnostics import interval_union_us


def analyze_trace(trace):
    events=trace['traceEvents']
    phases=[e for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('timing/') and e.get('ph')=='X']
    launches={e.get('args',{}).get('correlation'):e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{})}
    gpu=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and e.get('ph')=='X']
    def spans(rows):return [(e['ts'],e['ts']+e['dur']) for e in rows]
    def metrics(rows):return dict(count=len(rows),sum_us=sum(e['dur'] for e in rows),union_us=interval_union_us(spans(rows)))
    result=dict(gpu_work=metrics(gpu),phases={},unattributed_gpu_events=0,
        ignored_gpu_annotations=sum(e.get('cat')=='gpu_user_annotation' for e in events),
        method='Only kernel/gpu_memcpy/gpu_memset are device work. Attribute by CUDA launch correlation and enclosing CPU phase time. Host/device overlap; profiler overhead excluded from primary timing.')
    if gpu:result['gpu_work']['span_us']=max(e['ts']+e['dur'] for e in gpu)-min(e['ts'] for e in gpu)
    classified={e['name']:[] for e in phases}
    for e in gpu:
        launch=launches.get(e.get('args',{}).get('correlation'))
        owner=next((p for p in phases if launch is not None and p['ts']<=launch['ts']<p['ts']+p['dur']),None)
        if owner is None:result['unattributed_gpu_events']+=1
        else:classified[owner['name']].append(e)
    for name,rows in classified.items():
        host=[e for e in phases if e['name']==name]
        result['phases'][name]=dict(host_wall_us=sum(e['dur'] for e in host),host_calls=len(host),gpu_work=metrics(rows))
    return result


def main():
    b=Path(sys.argv[1]);rows=[]
    for p in sorted(b.glob('*_profile*.json.gz')):
        with gzip.open(p,'rt') as f:trace=json.load(f)
        rows.append(dict(trace=p.name,**analyze_trace(trace)))
    (b/'trace_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
