"""Read-only attribution of retained timing records; no experiment replay."""
import csv
import hashlib
import json
from pathlib import Path
import sys

STUDIES=['cifar_soap_pinned_timing_20261006','cifar_soap_radius_learning_20261006','cifar_soap_variance_20261006']
PHASES=['augmentation','model_backward','optimizer']

def audit(root, output):
    rows=[];inputs={}
    for name in STUDIES:
        bundle=root/name;p=bundle/'screen.json';inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        screen=json.loads(p.read_text());uid=screen['gpu_uuid']
        for run in screen['runs']:
            if run['covariance_beta']!=.99:continue
            history=[x for x in run['history'] if x['step']<=3500]
            assert len(history)==140 and all(x['host']['steps']==25 for x in history)
            fixed=next(x for x in run['observations'] if x['step']==3500)
            phases={phase:{key:sum(x['host']['phases'][phase][key] for x in history)/1000
                          for key in ['host_wall_ms','thread_cpu_ms','process_cpu_ms']} for phase in PHASES}
            for x in phases.values():x['non_main_cpu_seconds']=x['process_cpu_ms']-x['thread_cpu_ms']
            snapshots=[]
            for event in ['telemetry_start','telemetry_end']:
                fields=next(x for x in csv.reader(run[event]['gpu'].splitlines(),skipinitialspace=True) if x[0]==uid)
                snapshots.append(dict(event=event,temperature_c=float(fields[1]),sm_mhz=float(fields[2].split()[0]),
                                      memory_mhz=float(fields[3].split()[0]),power_w=float(fields[4].split()[0])))
            rows.append(dict(study=name,label=run['label'],kind=run['kind'],training_seconds=fixed['training_seconds'],
                optimizer_ms=fixed['timing']['optimizer_ms'],model_event_ms=fixed['timing']['model_backward_cuda_ms'],
                host_seconds=phases,gc_seconds=sum(g['wall_ms'] for x in history for g in x['host']['gc'])/1000,
                voluntary_context_switches=sum(x['host']['voluntary_context_switches'] for x in history),
                involuntary_context_switches=sum(x['host']['involuntary_context_switches'] for x in history),
                compiler_stable=run['telemetry_start']['compiler_counters']==run['telemetry_end']['compiler_counters'],
                gpu_endpoint_snapshots=snapshots))
    # Field names in the input retain the original ms suffix; values here are seconds.
    for row in rows:
        row['host_seconds']={p:{k.removesuffix('_ms'):v for k,v in x.items()} for p,x in row['host_seconds'].items()}
    result=dict(input_sha256=inputs,rows=rows,
        scope='Full first3500 updates; host wall/current-thread/process CPU overlap with CUDA events. Non-main CPU is a process-minus-main residual, not an identified thread. Only two GPU endpoint snapshots; no in-arm clock causal claim.')
    output.mkdir(parents=True,exist_ok=True);(output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Retained timing audit','',result['scope'],'',
           '| Study | Arm | Wall s | Model host s | Model main CPU s | Model non-main CPU s | Optimizer host s | Optimizer main CPU s | GC s |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        m=x['host_seconds']['model_backward'];o=x['host_seconds']['optimizer']
        lines.append(f"| {x['study']} | {x['label']} | {x['training_seconds']:.3f} | {m['host_wall']:.3f} | {m['thread_cpu']:.3f} | {m['non_main_cpu_seconds']:.3f} | {o['host_wall']:.3f} | {o['thread_cpu']:.3f} | {x['gc_seconds']:.4f} |")
    lines+=['','## Selected question','',
        'The radius QR slowdown is concentrated in non-main model/backward CPU time; '
        'optimizer wall is almost entirely main-thread CPU work. This supports a host '
        'dispatch/locality diagnostic, not proof of scheduler wait, GPU slowdown or '
        'unplaced OpenMP workers. Prior same-L3 versus different-L3 placement changed '
        'model time by8.52% with exact work. Test worker on main core116 versus '
        'current adjacent core117, holding main116 and everything else fixed. '
        'Co-location may help cache locality or hurt scheduling; outcome is unknown.','',
        'Add own-process per-thread runtime/runqueue-wait/affinity snapshots and main '
        'user/system phase CPU accounting. Collect four sparse own-GPU readings per '
        'arm outside training timing. Preserve corrected short profiler traces to '
        'separate actual GPU work from event spans. No other process changes. '
        'Retained clock endpoints alone cannot identify frequency dynamics.','',
        'A short1000-update QR result does not qualify3500-update QR/Fro timing. '
        'No historical replay was run for this audit. See the separately frozen '
        'prospective protocol for repeat/effect/numerical gates.']
    (output/'README.md').write_text('\n'.join(lines)+'\n');return result

if __name__=='__main__':audit(Path(sys.argv[1]),Path(sys.argv[2]))
