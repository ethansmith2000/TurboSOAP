"""Independent leaf/block oracle for the base-width PyTorch adaptation."""
import copy
import math
import pytest
import torch
from reference_blocked_soap import ReferenceBlockedSOAP,kind,metadata,tiles,merge
from reference_llama import ReferenceLlama,rope
from fast_soap import fast_warm_refresh


def oracle_init(p,vector,block,maximum):
    m,n=p.shape; rs=[(0,m)] if m>maximum else [(i,min(i+block,m)) for i in range(0,m,block)]
    cs=[(0,n)] if n>maximum else [(i,min(i+block,n)) for i in range(0,n,block)]
    r=block if len(rs)>1 else m;c=block if len(cs)>1 else n
    blocks=[]
    for a,b in rs:
        for x,y in cs:
            blocks.append(dict(slice=(a,b,x,y),v=torch.zeros(r,c),l=torch.zeros(r,r) if m<=maximum else None,
                               r=torch.zeros(c,c) if n<=maximum else None,ql=torch.eye(r) if m<=maximum else None,
                               qr=torch.eye(c) if n<=maximum else None))
    return dict(momentum=torch.zeros_like(p),blocks=blocks,vector=vector.clone(),p=p.clone())


def oracle_step(s,g,call,arm,embedding=False):
    # Explicit independent block loops; refresh order follows the pinned source.
    s['momentum']=.9*s['momentum']+.1*g; full=torch.zeros_like(g)
    event='eigh' if call<=50 else 'ordinary' if call%10 else 'qr' if arm=='qr' or call%50==0 else 'warm'
    for block in s['blocks']:
        a,b,x,y=block['slice'];v=block['v']; gg=torch.zeros_like(v);mm=torch.zeros_like(v)
        gg[:b-a,:y-x]=g[a:b,x:y];mm[:b-a,:y-x]=s['momentum'][a:b,x:y]
        for side,axis in (('l',0),('r',1)):
            cov=block[side]
            if cov is None:continue
            cov=.999*cov+.001*(gg@gg.T if side=='l' else gg.T@gg);block[side]=cov
            q=block['q'+side]
            if event!='ordinary':
                regular=cov+1e-10*torch.eye(len(cov))
                if event=='eigh':q=torch.linalg.eigh(regular).eigenvectors[:,torch.arange(len(cov)-1,-1,-1)]
                elif event=='qr':
                    estimates=torch.diag(q.T@regular@q);order=estimates.argsort(descending=True)
                    q=torch.linalg.qr(regular@q[:,order]).Q
                    v=v[order,:] if side=='l' else v[:,order]
                else:q=fast_warm_refresh(regular,q)
                block['q'+side]=q
        gp,mp=gg,mm
        if block['ql'] is not None:gp=block['ql'].T@gp;mp=block['ql'].T@mp
        if block['qr'] is not None:gp=gp@block['qr'];mp=mp@block['qr']
        v=.999*v+.001*gp**2;block['v']=v
        u=mp/(v.sqrt()+1e-8)
        if block['qr'] is not None:u=u@block['qr'].T
        if block['ql'] is not None:u=block['ql']@u
        full[a:b,x:y]=u[:b-a,:y-x]
    step=torch.tensor(float(call));full*=torch.sqrt(1-.999**step)/(1-.9**step)
    vector=s['vector']
    for _ in range(5 if call<=50 else 1):
        yy=full@vector;sigma=yy.norm();z=full.T@yy;candidate=z/(z.norm()+1e-10)
        if candidate.norm()>.5:vector=candidate
    s['vector']=vector
    if embedding:full=full/(full.square().mean().sqrt()+1e-10)
    else:full=full*math.sqrt(g.shape[1]/g.shape[0])/(sigma+1e-10)
    t=call-1;schedule=t/177 if t<177 else max(0,1-(t-177)/(28332-177))
    s['p']=s['p']-schedule*(.032*full+.0002*s['p'])


@pytest.mark.parametrize('shape,name', [((5,7),'weight'),((11,4),'embed'),((4,11),'weight')])
@pytest.mark.parametrize('arm',['qr','warm'])
def test_complete_transition_matches_independent_oracle_through_initialization_and_repairs(shape,name,arm):
    torch.set_float32_matmul_precision('highest')
    generator=torch.Generator().manual_seed(991)
    p=torch.nn.Parameter(torch.randn(*shape,generator=generator)*.01)
    opt=ReferenceBlockedSOAP([(name,p)],arm=arm,block=4,maximum=8,compile_kernels=False)
    state=oracle_init(p.detach(),opt.groups[0]['state']['vector'][0],4,8)
    for call in range(1,112):
        g=.03*torch.randn(*shape,generator=generator);p.grad=g.clone()
        oracle_step(state,g,call,arm,name=='embed');event=opt.step()
        assert event==kind(call,arm)
        torch.testing.assert_close(p,state['p'],atol=3e-5,rtol=3e-4)
        actual=opt.groups[0]['state']
        torch.testing.assert_close(actual['momentum'][0],state['momentum'],atol=1e-8,rtol=1e-6)
        for i,block in enumerate(state['blocks']):
            torch.testing.assert_close(actual['nu'][i],block['v'],atol=2e-8,rtol=2e-3)
            for side in ('l','r'):
                if block[side] is not None:torch.testing.assert_close(actual[side+'c'][i],block[side],atol=1e-9,rtol=1e-5)


def test_grouping_multiple_leaves_preserves_independent_results():
    torch.manual_seed(9);ps=[torch.nn.Parameter(torch.randn(4,4)) for _ in range(2)]
    opt=ReferenceBlockedSOAP([(str(i),p) for i,p in enumerate(ps)],block=4,compile_kernels=False)
    assert len(opt.groups)==1
    initial=[p.detach().clone() for p in ps]
    for p in ps:p.grad=torch.ones_like(p)
    opt.step()
    # Source schedule starts at zero but moment/covariance/variance still update.
    for p,q in zip(ps,initial):torch.testing.assert_close(p,q,atol=0,rtol=0)
    torch.testing.assert_close(opt.groups[0]['state']['momentum'],torch.full((2,4,4),.1),atol=0,rtol=0)


def test_model_count_and_zero_initialized_outputs():
    meta=ReferenceLlama(device='meta')
    assert sum(p.numel() for p in meta.parameters())==185680896
    assert len(list(meta.parameters()))==226
    model=ReferenceLlama(width=8,depth=2,expansion=4,head_dim=4,vocab=19,sequence=7,recompute=False)
    assert not any('bias' in name for name,_ in model.named_parameters())
    assert torch.count_nonzero(model.readout)==0
    tokens=torch.arange(14).reshape(2,7)%19
    torch.testing.assert_close(model(tokens),torch.zeros(2,7,19),atol=0,rtol=0)


def test_activation_recomputation_and_rope_math():
    torch.manual_seed(44)
    model=ReferenceLlama(width=8,depth=2,expansion=4,head_dim=4,vocab=19,sequence=7,recompute=False)
    with torch.no_grad():model.readout.normal_()
    other=copy.deepcopy(model);other.recompute=True
    tokens=torch.arange(14).reshape(2,7)%19
    a,b=model(tokens),other(tokens)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    a.square().mean().backward();b.square().mean().backward()
    for p,q in zip(model.parameters(),other.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
    x=torch.arange(4.).reshape(1,1,1,4)
    cos=torch.ones(1,1,1,2);sin=torch.zeros_like(cos)
    torch.testing.assert_close(rope(x,cos,sin),x,atol=0,rtol=0)


def test_preflight_logging_preserves_event_type_and_optimizer_kind(tmp_path):
    import json
    from reference_soap_cost import Run
    (tmp_path/'config.json').write_text('{}')
    run=Run(tmp_path)
    run.emit('optimizer_preflight',kind='eigh',call=1,seconds_including_compile=1.)
    run.emit('hot_optimizer_preflight',kind='qr',repeat=0,host_ms=2.)
    rows=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [(row['event'],row['kind']) for row in rows]==[('optimizer_preflight','eigh'),('hot_optimizer_preflight','qr')]
