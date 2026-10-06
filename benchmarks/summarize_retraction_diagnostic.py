"""Validate and summarize the fixed-policy diagnostic without changing its gates."""
import argparse
import json
import math
from pathlib import Path


def summarize(bundle):
    d=json.loads((bundle/'results.json').read_text())
    assert d['status']=='completed' and len(d['cases'])==18
    def maximum(values):
        vals=[float(v) for v in values]
        return max(vals) if all(math.isfinite(v) for v in vals) else float('inf')
    rows=[];attribution=[];controls=[]
    for c in d['cases']:
        observations=c['observations'];name=c['variant']['name'];width=c['width']
        assert len(c['steps'])==c['steps'][-1]['step']+1
        if c['status']=='completed':
            assert len(c['steps'])==421 and len(observations)==22
            if name!='qr20':assert sum(s['event']=='qr' for s in c['steps'])==2
        row=dict(width=width,name=name,repeat=c['repeat'],status=c['status'],
                 final_step=c['steps'][-1]['step'],model_sha256=c['model_sha256'],
                 max_gram=maximum(o['maximum_gram'] for o in observations),
                 max_global_m=maximum(m['global_world_m_error'] for o in observations for m in o['matrices']),
                 max_transport_m=maximum([m['world_m_transport_error'] for o in observations for m in o['refreshes']] or [0.]),
                 optimizer_ms=c['timing']['optimizer_ms'],events=c['timing']['events'],
                 final_synthetic_loss=c['steps'][-1]['loss'],diagnostic_seconds=c['diagnostic_seconds'])
        rows.append(row)
        for o in observations:
            for m in o['refreshes']:
                if 'factor_attribution' in m:
                    attribution.append(dict(width=width,step=o['step'],**m['factor_attribution']))
        if c['repeat']:
            base=next(x for x in d['cases'] if x['width']==width and x['variant']['name']==name and not x['repeat'])
            exact=(base['model_sha256']==c['model_sha256'] and
                   [s['loss'] for s in base['steps']]==[s['loss'] for s in c['steps']])
            assert exact,('Repeated state/loss differs',name)
            a,b=base['timing']['optimizer_ms'],c['timing']['optimizer_ms']
            controls.append(dict(name=name,exact_model_and_loss=exact,optimizer_range_over_mean=abs(a-b)/((a+b)/2)))
    result=dict(status='completed',rows=rows,attribution=attribution,controls=controls,
                wall_seconds=d['wall_seconds'],updates=sum(r['final_step'] for r in rows),
                caveat='Synthetic natural trajectories; no learning ranking, hardware attribution or production promotion.')
    def clean(v):
        if isinstance(v,float) and not math.isfinite(v):return str(v)
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        return v
    (bundle/'summary.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    lines=['# Local retraction: attribution and natural trajectories','',
           'Completed diagnostic. Fixed policies are evaluated over 420 updates, QR resets at 200/400 and warm updates every20. Both factors are active; maximum factor dimensions are768 and3072. Covariance beta .999, high matmul precision, strict first-moment transport. No model weights are saved.','',
           'The .05 normalized-Gram stop is a coarse failure screen. Passing it is not a learning-quality qualification. Whole-history momentum errors and per-refresh preservation errors are reported separately. Failed-case times are not eligible speed rankings.','',
           '| Width | Policy | Repeat | Status / last update | Max Gram | Max global M error | Max refresh M error | Optimizer ms |',
           '|---:|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['width']} | {r['name']} | {r['repeat']} | {r['status']} / {r['final_step']} | {r['max_gram']:.6g} | {r['max_global_m']:.6g} | {r['max_transport_m']:.6g} | {r['optimizer_ms']:.3f} |")
    lines+=['','## Same-state scaling attribution','',
            'Inputs come from the NS6 expansion-factor trajectory. Candidates and row-sum scales are computed at the actual high precision setting; geometry is measured at highest precision. Error curves apply alternate corrections to the same candidate, independently from the closed-loop policies above.','',
            '| Factor dim | Age | NS input scale | Candidate Gram | Candidate Gram eigenvalue min/max | Scaled eigenvalue min/max | Unscaled quintic1 Gram | Scaled quintic2 Gram | Scaled quintic6 Gram |',
            '|---:|---:|---:|---:|---|---|---:|---:|---:|']
    for a in attribution:
        c=a['candidate_spectrum'];s=a['scaled_spectrum'];v=a['curves']
        lines.append(f"| {a['dimension']} | {a['step']} | {a['scale']:.5f} | {a['candidate_gram']:.5g} | {c['minimum_gram_eigenvalue']:.5f} / {c['maximum_gram_eigenvalue']:.5f} | {s['minimum_gram_eigenvalue']:.5f} / {s['maximum_gram_eigenvalue']:.5f} | {v['order5_scaled0_n1']:.5g} | {v['order5_scaled1_n2']:.5g} | {v['order5_scaled1_n6']:.5g} |")
    lines+=['','## Reproducibility and scope','']
    for c in controls:lines.append(f"- {c['name']}: complete model/loss sequence exactly repeats; optimizer range/mean {100*c['optimizer_range_over_mean']:.2f}%.")
    lines+=['',f"Actual updates: {result['updates']:,}; total elapsed {d['wall_seconds']:.1f}s. Diagnostic seconds are recorded per case and excluded from optimizer timings. No timing claim that includes deployment guards is made: these are offline study checks, not part of the proposed fixed policy.",'',
            'Noisy timing differences, first-event effects and diagnostics can influence subsequent thermal/dispatch conditions. Use repeated isolated full-refresh timing and uninstrumented natural cycles before a sustained-throughput claim. Synthetic losses are sanity values only. This stage does not establish cap .2, covariance .99, BF16, compilation or factor6144 behavior.','',
            'See PROTOCOL.md for prospective stop conditions, PROVENANCE.md for unresolved H200 mapping, results.json for all observations, and source/ for frozen runtime code.']
    (bundle/'README.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    result=summarize(p.parse_args().bundle)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','attribution']},indent=2))
