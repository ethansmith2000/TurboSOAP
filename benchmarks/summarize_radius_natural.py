"""Fail-closed qualification and exact historical/repeat evidence."""
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

def numerical_observations(case):
    return [{k:v for k,v in o.items() if k!='optimizer_sha256'} for o in case['observations']]

def summarize(b):
    r=json.loads((b/'results.json').read_text()); m=json.loads((b/'manifest.json').read_text())
    assert r['status']=='completed' and len(r['cases'])==18
    historical=Path(m['historical_results'])
    assert hashlib.sha256(historical.read_bytes()).hexdigest()==m['historical_results_sha256']
    old=json.loads(historical.read_text())['cases']
    expected={(w,beta,p,0) for w in (768,1536) for beta in (.99,.999)
              for p in ('qr20','fro_cubic2_cap05','fro_cubic2_cap06','fro_cubic3_cap06')}
    expected|={(1536,.999,p,1) for p in ('fro_cubic2_cap06','fro_cubic3_cap06')}
    actual={(c['width'],c['beta'],c['variant']['name'],c['repeat']) for c in r['cases']}
    assert actual==expected
    rows=[]; controls=[]; anchors=[]
    for c in r['cases']:
        obs=c['observations']; refresh=[x for o in obs for x in o['refreshes']]
        gram=max(o['maximum_gram'] for o in obs)
        global_m=max(x['global_world_m_error'] for o in obs for x in o['matrices'])
        transport=max(x['world_m_transport_error'] for x in refresh)
        complete=(c['status']=='completed' and [x['step'] for x in c['steps']]==list(range(421))
                  and [o['step'] for o in obs]==list(range(0,421,20))
                  and len(refresh)==63 and all(len(o['matrices'])==3 and o['finite'] for o in obs)
                  and len(obs[-1].get('optimizer_sha256',''))==64)
        p=c['variant']['name']
        events={e:sum(x['event']==e for x in c['steps'][1:]) for e in ('ordinary','qr','warm')}
        assert events==dict(ordinary=399,qr=21 if p=='qr20' else 2,warm=0 if p=='qr20' else 19)
        rows.append(dict(width=c['width'],beta=c['beta'],policy=p,repeat=c['repeat'],
            qualified=complete and gram<.05 and global_m<.01 and transport<.01,
            max_gram=gram,max_global_m=global_m,max_transport_m=transport,
            optimizer_ms=c['timing']['optimizer_ms'],step_wall_ms=c['timing']['step_wall_ms'],
            events=c['timing']['events'],event_counts=events,
            cold_optimizer_ms=c['steps'][0]['optimizer_ms'],diagnostic_seconds=c['diagnostic_seconds']))
        if c['repeat']:
            original=next(x for x in r['cases'] if not x['repeat'] and (x['width'],x['beta'],x['variant']['name'])==(c['width'],c['beta'],p))
            exact=(c['model_sha256']==original['model_sha256'] and c['observations']==original['observations']
                   and [x['loss'] for x in c['steps']]==[x['loss'] for x in original['steps']])
            a,z=c['timing']['optimizer_ms'],original['timing']['optimizer_ms']
            controls.append(dict(policy=p,exact_model_optimizer_losses_probes=exact,optimizer_relative_range=abs(a-z)/st.mean([a,z])))
        elif p in ('qr20','fro_cubic2_cap05'):
            previous=next(x for x in old if not x['repeat'] and (x['width'],x['beta'],x['variant']['name'])==(c['width'],c['beta'],p))
            exact=(c['model_sha256']==previous['model_sha256'] and numerical_observations(c)==numerical_observations(previous)
                   and [x['loss'] for x in c['steps']]==[x['loss'] for x in previous['steps']])
            anchors.append(dict(width=c['width'],beta=c['beta'],policy=p,exact=exact))
    result=dict(status='completed',qualified=all(x['qualified'] for x in rows) and len(controls)==2
                and all(x['exact_model_optimizer_losses_probes'] for x in controls)
                and len(anchors)==8 and all(x['exact'] for x in anchors),
                rows=rows,controls=controls,historical_anchors=anchors,
                updates=7560,refresh_matrix_checks=1134,history_matrix_checks=1188,wall_seconds=r['wall_seconds'])
    (b/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    r=summarize(Path(sys.argv[1])); print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
