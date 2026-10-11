"""Validate scope of the prospective one-flag execution control."""
import ast
import copy
import math


def validate_child_change(baseline,control):
    before=ast.parse(baseline);after=ast.parse(control)
    def assignments(tree):
        return [n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                len(n.targets)==1 and ast.unparse(n.targets[0])=='torch.backends.cudnn.benchmark']
    left=assignments(before);right=assignments(after)
    if len(left)!=1 or len(right)!=1 or not isinstance(left[0].value,ast.Constant) or left[0].value.value is not True:
        raise ValueError('Baseline must set benchmark=True exactly once')
    if not isinstance(right[0].value,ast.Constant) or right[0].value.value is not False:
        raise ValueError('Control must set benchmark=False exactly once')
    right[0].value=ast.Constant(value=True)
    if ast.dump(before)!=ast.dump(after):raise ValueError('Control changes more than the single backend flag')
    return True


def validate_config(c):
    d=c['execution_diagnostic'];contexts=d['contexts']
    if len(contexts)!=4 or len({x['label'] for x in contexts})!=4:
        raise ValueError('Need four uniquely labeled fresh processes')
    if [x['allocation_horizon'] for x in contexts]!=[420,3500,3500,420]:
        raise ValueError('Use declared balanced allocation order')
    if any(x['timing_chunk']!=25 for x in contexts):raise ValueError('Flush must stay25')
    for horizon in [420,3500]:
        if sorted(x['repeat'] for x in contexts if x['allocation_horizon']==horizon)!=[0,1]:
            raise ValueError('Each allocation needs two distinct repeats')
    if d['smoke_steps']!=25 or d['measured_steps']!=25 or d['fingerprint_steps']!=[1,2,20,21,25]:
        raise ValueError('No automatic work/observation extension')
    if d['per_process_seconds']!=150 or c['max_wall_seconds']!=600 or d['maximum_peak_allocated_bytes']!=24*2**30:
        raise ValueError('Keep prospectively declared resource limits')
    expected=dict(width=1536,depth=1,heads=24,seed=139,batch_size=256,epochs=200,
                  warmup_epochs=5,shampoo_beta=.99,matrix_lr=.0005,checkpoint=False,official_test=False)
    if any(c[k]!=v for k,v in expected.items()) or c['betas']!=[.9,.999]:
        raise ValueError('Keep model/optimizer/recipe fixed')
    return True


def backend_delta(left,right,path=''):
    if type(left)!=type(right):return [path]
    if isinstance(left,dict):
        if left.keys()!=right.keys():return [path+'.keys']
        return [p for k in left for p in backend_delta(left[k],right[k],(path+'.'+k).lstrip('.'))]
    return [] if left==right else [path]


def finite_records(value):
    if isinstance(value,dict):return value.get('finite',True) and all(finite_records(v) for v in value.values())
    if isinstance(value,list):return all(finite_records(v) for v in value)
    if isinstance(value,float):return math.isfinite(value)
    return True
