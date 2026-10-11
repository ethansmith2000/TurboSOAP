"""Prospective scope and exact record stream for a fresh 420-update pilot."""
import ast
import copy
import math

KINDS = {'initial', 'fingerprint', 'smoke', 'code', 'backend', 'guard',
         'probe', 'chunk', 'observation', 'final'}

def validate_config(c, baseline):
    expected = copy.deepcopy(baseline)
    expected['max_wall_seconds'] = 1200
    expected['fresh420'] = dict(per_process_seconds=390, smoke_steps=25,
        measured_steps=420, fingerprint_steps=[1,2,20,21,25])
    if c != expected:
        raise ValueError('Only declared wall bounds and fresh-process protocol may change')
    s=c['memory_screen']
    if [a['label'] for a in s['arms']] != ['qr_0','fro_1','qr_2']:
        raise ValueError('Require QR/Fro/QR in fixed order')
    if s['evaluation_updates'] != [100,200,400,420] or s['tracking_ages'] != [20,180,200,220,380,400]:
        raise ValueError('Fixed evaluation/probe schedule required')
    if c['checkpoint'] or c['official_test'] or any(a['capture_replay'] for a in s['arms']):
        raise ValueError('No checkpoint or optimizer snapshot')
    return True

def canonical(kind,row):
    if kind not in KINDS: raise ValueError('Unknown comparison kind')
    value=copy.deepcopy(row)
    if kind in ['guard','probe']: value.pop('diagnostic_seconds',None)
    if kind=='chunk':
        if set(value) != {'step','samples','loss','gradient_norm','host'}:
            raise ValueError('Unexpected chunk schema')
        value={k:value[k] for k in ['step','loss','gradient_norm']}
    if kind=='observation': value.pop('training_seconds')
    return dict(kind=kind,value=value)

def check_record(reference, index, record):
    if index >= len(reference): raise ValueError('QR repeat produced an unexpected record')
    if reference[index] != record:
        # Field path is compact evidence, with no tensor data or tolerant equality.
        try:
            from execution_fingerprints import first_difference
        except ImportError:
            from .execution_fingerprints import first_difference
        raise ValueError('Exact QR repeat mismatch at record '+str(index)+': '+str(first_difference(reference[index],record)))

def reconstruct_stream(r):
    """Recompute from raw semantic records so a changed summary cannot hide state."""
    rows=[canonical('initial',dict(model=r['initial_model_sha256'],rng=r['initial_rng_sha256'],batch_plan=r['batch_plan'])),
          canonical('backend',dict(phase='before_smoke',record=r['backend_before_smoke']))]
    rows.extend(canonical('fingerprint',dict(phase='smoke',record=x)) for x in r['smoke_fingerprints'])
    rows.extend([canonical('smoke',r['smoke']),canonical('code',dict(phase='after_smoke',record=r['code_after_smoke'])),
                 canonical('initial',dict(effective_groups=r['effective_groups']))])
    for step in range(1,421):
        for kind,key in [('fingerprint','measured'),('guard','refresh_guards'),('probe','tracking_probes'),('chunk','timing_chunks'),('observation','observations')]:
            for x in r[key]:
                if x['step']==step:rows.append(canonical(kind,dict(phase='measured',record=x) if kind=='fingerprint' else x))
    rows.extend([canonical('final',r['final_state']),canonical('code',dict(phase='after_measured',record=r['code_after_measured'])),
                 canonical('backend',dict(phase='after_measured',record=r['backend_after_measured'])),
                 canonical('code',dict(phase='after_saved_operand_replay',record=r['code_after_saved_operand_replay']))])
    return rows

def audit_primary_math(before,after):
    """Preserve ordered primary recipe calls; diagnostics remain caller-side."""
    def calls(source):
        tree=ast.parse(source)
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='step')
        selected=[]
        names={'support.lr_at','apply_lr_multiplier','optimizer.zero_grad','torch.autocast',
               'active','F.cross_entropy','loss.backward','torch.nn.utils.clip_grad_norm_',
               'optimizer.step','strong','torch.randint','torch.rand'}
        for n in ast.walk(fn):
            if isinstance(n,ast.Call) and ast.unparse(n.func) in names:
                selected.append((n.lineno,ast.dump(n,include_attributes=False)))
        return [x[1] for x in sorted(selected)]
    if calls(before)!=calls(after): raise ValueError('Primary recipe math changed')
    tree=ast.parse(after)
    for flag,value in [('torch.backends.cudnn.benchmark',False),('torch.backends.cudnn.deterministic',True)]:
        assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(ast.unparse(t)==flag for t in n.targets)]
        if len(assignments)!=1 or ast.literal_eval(assignments[0].value)!=value: raise ValueError(flag)
    return True

def numerical_complete(r):
    if r.get('status')!='completed':return False
    try:
        assert r['parameter_count']==28660324
        assert sorted(tuple(x['shape']) for x in r['matrix_inventory'])==sorted([(4608,1536),(1536,1536),(6144,1536),(1536,6144)])
        assert [x['step'] for x in r['smoke_fingerprints']]==[1,25]
        assert [x['step'] for x in r['measured']]==[1,2,20,21,25]
        assert [o['step'] for o in r['observations']]==[100,200,400,420]
        assert [g['step'] for g in r['refresh_guards']]==[1]+list(range(21,421,20))
        assert [p['step'] for p in r['tracking_probes']]==[21,181,201,221,381,401]
        assert [h['step'] for h in r['timing_chunks']]==list(range(10,421,10))
        assert all(g['factor_count']==8 and g['maximum_gram_error']<.05 for g in r['refresh_guards'])
        rows=[x for p in r['tracking_probes'] for x in p['matrices']]
        assert len(rows)==24 and sum(len(x['factors']) for x in rows)==48
        assert all(x['targeted_pass'] and x['selected_basis_update_error']<1e-6 and x['world_m_preservation_error']<.01 and x['maximum_gram']<.05 and not x['full_reference_collected'] for x in rows)
        assert r['owned_optimizer_released'] and r['own_affinity_restored']
        assert r['peak_allocated_bytes']<24*2**30
        assert all(math.isfinite(h['loss']) and math.isfinite(h['gradient_norm']) for h in r['timing_chunks'])
        assert all(r['initial_state_replay'][k] is None for k in ['actual_first_vs_plain','actual_first_vs_profiled','plain_vs_profiled'])
        assert r['saved_operand_replay']['interpretation']['supported']
        assert reconstruct_stream(r)==r['comparison_records']
        final=r['observations'][-1]
        assert r['final_state']==dict(model=final['model_sha256'],optimizer=final['optimizer_sha256'],rng=final['rng_sha256'])
    except (AssertionError,KeyError,TypeError):return False
    return True
