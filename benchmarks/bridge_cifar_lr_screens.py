"""Combine fixed-update LR evidence only after exact shared-anchor verification."""
import argparse
import hashlib
import json
from pathlib import Path


def bridge(current,prior):
    def read(b,name):return json.loads((b/name).read_text())
    new,old=read(current,'screen.json'),read(prior,'screen.json')
    assert new['status']==old['status']=='completed','Require complete reproducible studies'
    nm,om=read(current,'manifest.json'),read(prior,'manifest.json')
    for key in ['split_sha256','base_prepared_sha256']:assert nm[key]==om[key],key
    for key in ['torch','cuda','initial_model_sha256']:assert new[key]==old[key],key
    keys=(set(new['config'])|set(old['config']))-{'lr_screen','max_wall_seconds'}
    for key in keys:assert new['config'][key]==old['config'][key],key
    source_keys=['soap.py','soap_reference.py','benchmarks/cifar_optimizers.py']
    source_keys+=sorted(k for k in nm['source_sha256'] if k.startswith('benchmarks/cifar_support/'))
    for key in source_keys:assert nm['source_sha256'][key]==om['source_sha256'][key],key
    def selected(runs,policy):return next(x for x in runs if x['label']==f'{policy}_lr0.0005')
    anchors=[]
    for policy in ['qr10','warm10','qr20']:
        a,b=selected(new['runs'],policy),selected(old['runs'],policy)
        for step in [875,1750,3500]:
            x=next(o for o in a['observations'] if o['step']==step)
            y=next(o for o in b['observations'] if o['step']==step)
            for key in ['model_sha256','rng_sha256','consumed_ids_sha256','augmented_batches','validation','train_panel']:
                assert x[key]==y[key],(policy,step,key)
        anchors.append(dict(policy=policy,steps=[875,1750,3500],status='exact_match'))
    rows=[]
    for bundle,report in [(current,new),(prior,old)]:
        for run in report['runs']:
            if run.get('repeat_of') or '_repeat' in run['label']:continue
            if bundle==prior and run['matrix_lr']==.0005:continue
            fixed=next(x for x in run['observations'] if x['step']==3500)
            rows.append(dict(policy=run['policy'],matrix_lr=run['matrix_lr'],
                validation_ce=fixed['validation']['ce'],accuracy=fixed['validation']['accuracy'],
                model_sha256=fixed['model_sha256'],source_bundle=str(bundle)))
    assert len(rows)==15 and len({(x['policy'],x['matrix_lr']) for x in rows})==15
    selections=[]
    for policy in ['qr10','warm10','qr20']:
        candidates=[x for x in rows if x['policy']==policy]
        rates=sorted(x['matrix_lr'] for x in candidates);best=min(candidates,key=lambda x:x['validation_ce'])
        selections.append({**best,'boundary_winner':best['matrix_lr'] in (rates[0],rates[-1])})
    result=dict(status='verified',scope='Fixed-update quality only; no pooled timing or additional seeds',
        prior_screen_sha256=hashlib.sha256((prior/'screen.json').read_bytes()).hexdigest(),
        current_screen_sha256=hashlib.sha256((current/'screen.json').read_bytes()).hexdigest(),
        anchors=anchors,rows=rows,selected_by_equal_update_ce=selections)
    (current/'combined_lr_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True);a=p.parse_args()
    print(json.dumps(bridge(a.bundle.resolve(),a.prior.resolve()),indent=2))
