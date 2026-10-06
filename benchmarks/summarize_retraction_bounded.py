"""Summarize bounded-rotation natural cycles and unchanged repeats."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(b):
    d=json.loads((b/'results.json').read_text());assert d['status']=='completed' and len(d['cases'])==26
    rows=[];controls=[]
    for c in d['cases']:
        obs=c['observations'];warm=[m for o in obs if o['event']=='warm' for m in o['refreshes']]
        row=dict(width=c['width'],beta=c['beta'],name=c['variant']['name'],repeat=c['repeat'],status=c['status'],
                 updates=c['steps'][-1]['step'],max_gram=max(float(o['maximum_gram']) for o in obs),
                 max_global_m=max(float(m['global_world_m_error']) for o in obs for m in o['matrices']),
                 max_transport_m=max(float(m['world_m_transport_error']) for o in obs for m in o['refreshes']),
                 optimizer_ms=c['timing']['optimizer_ms'],events=c['timing']['events'],
                 mean_warm_movement=statistics.mean(v for m in warm for v in m['movement_rms']) if warm else None,
                 mean_warm_residual_change=statistics.mean(a-z for m in warm for a,z in zip(m['residual_after'],m['residual_before'])) if warm else None,
                 final_synthetic_loss=c['steps'][-1]['loss'])
        rows.append(row)
        if c['status']=='completed':assert row['updates']==420 and len(obs)==22
        if c['repeat']:
            base=next(x for x in d['cases'] if not x['repeat'] and (x['width'],x['beta'],x['variant']['name'])==(c['width'],c['beta'],c['variant']['name']))
            assert base['model_sha256']==c['model_sha256']
            assert [s['loss'] for s in base['steps']]==[s['loss'] for s in c['steps']]
            assert base['observations']==c['observations']
            a,z=base['timing']['optimizer_ms'],c['timing']['optimizer_ms']
            controls.append(dict(name=row['name'],exact_model_losses_diagnostics=True,optimizer_range_over_mean=abs(a-z)/((a+z)/2)))
    result=dict(status='completed',rows=rows,controls=controls,wall_seconds=d['wall_seconds'],updates=sum(r['updates'] for r in rows))
    (b/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Bounded rotation and one-step retraction','',
           'Closed-loop synthetic screen at covariance .99/.999, widths768/1536 (largest factors3072/6144), 420updates each, warm20 and QR200/400. Every candidate uses fixed work. The row-sum cap replaces the existing average norm reduction; there is no extra error-controller decision. Cap numbers under the two norms are not equivalent movement budgets.','',
           '| Width | Cov beta | Policy | Repeat | Status | Max Gram | Max global M error | Optimizer ms/update |',
           '|---:|---:|---|---:|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['width']} | {r['beta']} | {r['name']} | {r['repeat']} | {r['status']} | {r['max_gram']:.6g} | {r['max_global_m']:.6g} | {r['optimizer_ms']:.3f} |")
    lines+=['','## Movement and tracking','',
            'Means below span actual warm refreshes and all active factors on each policy’s own trajectory. They quantify behavior but are not same-state causal comparisons. Negative residual change means improved diagonalization. Small movement need not mean good tracking.','',
            '| Width | Cov beta | Policy | Mean movement RMS | Mean residual change | Warm-event optimizer ms |',
            '|---:|---:|---|---:|---:|---:|']
    for r in rows:
        if r['mean_warm_movement'] is not None and not r['repeat']:
            lines.append(f"| {r['width']} | {r['beta']} | {r['name']} | {r['mean_warm_movement']:.6g} | {r['mean_warm_residual_change']:+.6g} | {r['events']['warm']:.3f} |")
    lines+=['','## Verification and limits','']
    for c in controls:lines.append(f"- {c['name']}: weights, every loss and every numerical diagnostic exactly repeat; optimizer range/mean {100*c['optimizer_range_over_mean']:.2f}%.")
    lines+=['',f"Executed {result['updates']:,} updates in {d['wall_seconds']:.1f}s. The .05 normalized-Gram guard remains a coarse stability stop, not a practical learning criterion. Strict m transport is fixed throughout. V remains carry/reindex. No strict full-refresh-equivalence pass is claimed.",'',
            'Timing excludes offline guards and momentum oracles; they are not deployed adaptive checks. Initialization is separate. Repeated uninstrumented natural cycles and compiler profiling remain necessary before production-throughput claims. Synthetic loss is not generalization evidence. There is no matched H200 measurement.','',
            'No weights or tensor snapshots retained. See PROTOCOL.md, config.json, manifest.json and frozen source/.']
    (b/'README.md').write_text('\n'.join(lines)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    s=summarize(p.parse_args().bundle);print(json.dumps({k:v for k,v in s.items() if k!='rows'},indent=2))
