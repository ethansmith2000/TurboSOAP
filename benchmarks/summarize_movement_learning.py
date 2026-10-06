"""Validate fixed-update anchors and qualify measured-wall interpretations."""
import argparse
import json
from pathlib import Path
import statistics as st
from summarize_cifar_movement import KEYS


def summarize(b, prior):
    r=json.loads((b/'screen.json').read_text());old=json.loads((prior/'screen.json').read_text())
    assert r['status']=='completed' and len(r['runs'])==10
    study=r['config']['memory_screen'];by={x['label']:x for x in r['runs']}
    assert set(by)=={a['label'] for a in study['arms']}
    groups={x['label']:x['groups'] for x in json.loads((b/'experimental_groups.json').read_text())}
    rows=[];anchors=[];diagnostics=[];guards=0;streams={}
    def fixed(run):return [o for o in run['observations'] if 'fixed_updates' in o['reasons']]
    for a in r['runs']:
        assert a['status']=='completed' and 3500<=a['steps']<=5000
        observations=fixed(a);assert [o['step'] for o in observations]==[875,1750,3500]
        kind=a['label'].split('_beta')[0];beta=a['covariance_beta']
        assert len(a['refresh_guards'])==1+(a['steps']-1)//20
        assert all(x['factor_count']==48 and x['maximum_gram_error']<.05 for x in a['refresh_guards'])
        guards+=len(a['refresh_guards'])
        for g in groups[a['label']]:
            assert g['transport_precision']=='highest' and g['shampoo_beta']==beta
            assert g['precondition_frequency']==20 and g['hard_reset_interval']==(0 if kind=='qr20' else 200)
            assert g['basis_rotation_cap_mode']==dict(qr20='average',ns6='average',row_cubic2='spectral_bound',fro_cubic2='skew_frobenius')[kind]
            assert g['basis_rotation_cap']==(.5 if kind.endswith('cubic2') else .1)
        for o in observations:
            stream={k:o[k] for k in KEYS[1:4]}
            assert o['step'] not in streams or stream==streams[o['step']]
            streams[o['step']]=stream
        if kind!='fro_cubic2' and not a.get('repeat_of'):
            oldkind='cubic2' if kind=='row_cubic2' else kind
            matched=next(x for x in old['runs'] if x['label']==f'{oldkind}_beta{beta}')
            for x,y in zip(observations,matched['observations']):
                assert all(x[k]==y[k] for k in KEYS),(a['label'],x['step'])
                anchors.append(dict(label=a['label'],step=x['step'],exact=True))
        assert [p['step'] for p in a['tracking_probes']]==[age+1 for age in study['tracking_ages']]
        probes=[m for p in a['tracking_probes'] for m in p['matrices']]
        assert len(probes)==168 and all(m['targeted_pass'] for m in probes)
        for event in ['warm','qr']:
            ms=[m for m in probes if m['executed_method']==event]
            if ms:diagnostics.append(dict(label=a['label'],event=event,count=len(ms),
                max_world_m=max(m['world_m_preservation_error'] for m in ms),
                max_selected_basis_error=max(m['selected_basis_update_error'] for m in ms),
                max_full_strict_difference=max(m['full_strict_update_difference'] for m in ms),
                full_strict_failures=sum(not m['full_strict_pass'] for m in ms)))
        end=observations[-1];wall=[o for o in a['observations'] if 'equal_wall' in o['reasons']]
        assert len(wall)<=1
        budget=None
        if wall:
            w=wall[0];assert w['training_seconds']>=90 and w['step']%25==0
            before=[x for x in a['history'] if x['step']<w['step']]
            assert not before or before[-1]['training_seconds']<90
            budget=dict(step=w['step'],ce=w['validation']['ce'],accuracy=w['validation']['accuracy'],
                        actual_seconds=w['training_seconds'],overshoot_seconds=w['training_seconds']-90)
        rows.append(dict(label=a['label'],kind=kind,beta=beta,repeat=bool(a.get('repeat_of')),
            ce=end['validation']['ce'],accuracy=end['validation']['accuracy'],
            optimizer_ms=end['timing']['optimizer_ms'],training_seconds=end['training_seconds'],
            total_steps=a['steps'],budget=budget))
    controls=[]
    for first,last in study['control_pairs']:
        x,y=by[first],by[last]
        assert all(all(a[k]==z[k] for k in KEYS) for a,z in zip(fixed(x),fixed(y)))
        assert [p['matrices'] for p in x['tracking_probes']]==[p['matrices'] for p in y['tracking_probes']]
        a,z=[next(row for row in rows if row['label']==k) for k in (first,last)]
        ranges={k:abs(a[k]-z[k])/st.mean([a[k],z[k]]) for k in ['optimizer_ms','training_seconds']}
        qualified=(ranges['optimizer_ms']<=study['timing_repeat_maximum_optimizer_range_over_mean'] and
                   ranges['training_seconds']<=study['timing_repeat_maximum_training_wall_range_over_mean'])
        controls.append(dict(label=first,observations_and_probes_exact=True,ranges=ranges,qualified=qualified))
    nm=json.loads((b/'manifest.json').read_text());om=json.loads((prior/'manifest.json').read_text())
    for key in ['split_sha256','base_prepared_sha256']:assert nm[key]==om[key]
    for key in ['soap.py','soap_reference.py','benchmarks/cifar_lr_screen.py','benchmarks/cifar_optimizers.py']:
        assert nm['source_sha256'][key]==om['source_sha256'][key]
    for key in ['torch','cuda','initial_model_sha256']:assert r[key]==old[key]
    result=dict(status='completed',rows=rows,diagnostics=diagnostics,anchors=anchors,controls=controls,
        repeat_qualified_wall_interpretation=all(c['qualified'] for c in controls) and all(x['budget'] for x in rows),
        refresh_guards=guards,matrix_probes=sum(x['count'] for x in diagnostics),
        trainer_steps=sum(a['steps'] for a in r['runs']),wall_seconds=r['all_in_seconds'])
    (b/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Movement bound: paired learning at fixed updates and measured wall budget','',
        'Only the clipping bound differs between row cubic2 and Frobenius cubic2. Seed 139, original 200-epoch LR schedule, both factors, strict M, high basis arithmetic, warm every 20 / QR every 200. Fixed3500 endpoint is primary; the secondary budget is 90 measured training seconds, excluding initialization/evaluation/offline probes.','',
        '| Cov beta | Policy | Repeat | CE at 3500 | Optimizer ms | Wall-budget steps | Wall-budget CE | Actual seconds |',
        '|---:|---|---|---:|---:|---:|---:|---:|']
    for x in rows:
        w=x['budget'];suffix=f"{w['step']} | {w['ce']:.6f} | {w['actual_seconds']:.3f}" if w else 'capped | unavailable | unavailable'
        lines.append(f"| {x['beta']} | {x['kind']} | {x['repeat']} | {x['ce']:.6f} | {x['optimizer_ms']:.3f} | {suffix} |")
    lines+=['','## Repeat qualification','',
        '| Control | Optimizer range/mean | Fixed3500 training wall range/mean | Predeclared repeat criterion |',
        '|---|---:|---:|---|']
    for x in controls:lines.append(f"| {x['label']} | {100*x['ranges']['optimizer_ms']:.2f}% | {100*x['ranges']['training_seconds']:.2f}% | {x['qualified']} |")
    lines+=['',f"Repeat-qualified wall interpretation: **{result['repeat_qualified_wall_interpretation']}**. Criteria were <=2% optimizer and <=3% training-wall range/mean for both .99 repeated controls. If false, budget endpoints remain exploratory; no time-to-quality winner is promoted.",'',
        f"All {len(anchors)} historical observations and both repeated fixed-update controls/probes match exactly. {guards} refresh guards and {result['matrix_probes']} targeted probes pass. Full strict-refresh disagreements, including QR/reset events, remain recorded in summary.json.",'',
        'A measured-wall budget includes Python/launch overhead and model work; this is not equal cumulative optimizer time. Original LR schedule is indexed by updates, with no per-arm horizon retuning. One seed and an early endpoint do not establish a tuned-method or convergence ranking. Large-factor synthetic timing is separate. No weights saved; defaults unchanged.']
    (b/'README.md').write_text('\n'.join(lines)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    a=p.parse_args();r=summarize(a.bundle,a.prior)
    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','anchors','diagnostics')},indent=2))
