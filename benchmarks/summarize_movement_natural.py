"""Natural-cycle qualification and timing; no synthetic generalization claims."""
import argparse
import json
from pathlib import Path
import statistics as st


def summarize(b):
    r=json.loads((b/'results.json').read_text());assert r['status']=='completed'
    assert len(r['cases'])==18
    rows=[];controls=[]
    for c in r['cases']:
        obs=c['observations'];refresh=[m for o in obs for m in o['refreshes']]
        gram=max(o['maximum_gram'] for o in obs)
        global_m=max(m['global_world_m_error'] for o in obs for m in o['matrices'])
        transport=max(m['world_m_transport_error'] for m in refresh)
        qualified=c['status']=='completed' and c['steps'][-1]['step']==420 and gram<.05 and global_m<.01 and transport<.01
        rows.append(dict(width=c['width'],beta=c['beta'],policy=c['variant']['name'],repeat=c['repeat'],
            qualified=qualified,max_gram=gram,max_global_m=global_m,max_transport_m=transport,
            optimizer_ms=c['timing']['optimizer_ms'],events=c['timing']['events']))
        if c['repeat']:
            original=next(x for x in r['cases'] if not x['repeat'] and
                (x['width'],x['beta'],x['variant']['name'])==(c['width'],c['beta'],c['variant']['name']))
            assert c['model_sha256']==original['model_sha256']
            assert [x['loss'] for x in c['steps']]==[x['loss'] for x in original['steps']]
            assert c['observations']==original['observations']
            a,z=c['timing']['optimizer_ms'],original['timing']['optimizer_ms']
            controls.append(dict(policy=c['variant']['name'],exact_model_losses_diagnostics=True,
                optimizer_relative_range=abs(a-z)/st.mean([a,z])))
    result=dict(status='completed',rows=rows,exact_repeats=True,controls=controls,
        updates=sum(x['steps'][-1]['step'] for x in r['cases']),wall_seconds=r['wall_seconds'])
    (b/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Skew-Frobenius cubic2: natural-cycle qualification','',
        'Same synthetic workload and seeds as the preceding retraction study; 420 updates through QR resets at 200/400, both factors. Offline probes are excluded from step timing.','',
        '| Width | Cov beta | Policy | Repeat | Qualified | Max Gram | Max global M | Optimizer ms |',
        '|---:|---:|---|---:|---|---:|---:|---:|']
    for x in rows:lines.append(f"| {x['width']} | {x['beta']} | {x['policy']} | {x['repeat']} | {x['qualified']} | {x['max_gram']:.5g} | {x['max_global_m']:.5g} | {x['optimizer_ms']:.3f} |")
    lines+=['',f"All {len(controls)} repeated model/loss/diagnostic sequences are exact. Executed {result['updates']} updates in {r['wall_seconds']:.1f}s.",'',
        'Qualification requires completion, Gram <.05, per-refresh and whole-history original-coordinate M errors <1%. This is a numerical screen, not a universal learning criterion. Timings are instrumented synthetic cycles; do not combine them with CIFAR losses to infer an end-to-end advantage. No weights saved; production defaults unchanged.']
    (b/'README.md').write_text('\n'.join(lines)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    r=summarize(p.parse_args().bundle)
    print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
