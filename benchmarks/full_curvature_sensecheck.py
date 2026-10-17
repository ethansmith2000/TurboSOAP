"""Bounded CPU-only exact-curvature/preconditioning mathematical study."""
from pathlib import Path
import argparse,json,math,sys,time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
import torch
from full_directional_curvature import offdiag,explicit_commutator_terms,single_product_terms,quadratic_alpha,ideal_cubic_two
from tracking_curvature_kernel import generator

def forbidden(*a,**k):raise AssertionError('CPU sensecheck attempted CUDA initialization')
torch.cuda._lazy_init=forbidden

def matrices():
    for n in (8,16,32):
        for seed in (161,162,163):
            rng=torch.Generator().manual_seed(seed+n)
            u=torch.linalg.qr(torch.randn(n,n,generator=rng,dtype=torch.float64)).Q
            for family in ('spread','clustered','spiked'):
                if family=='spread':d=torch.linspace(.1,2.,n,dtype=torch.float64)
                elif family=='clustered':d=torch.cat((torch.linspace(.1,.101,n//2,dtype=torch.float64),torch.linspace(3.,3.002,n-n//2,dtype=torch.float64)))
                else:d=torch.cat((torch.linspace(.1,.2,n-1,dtype=torch.float64),torch.tensor([4.*n],dtype=torch.float64)))
                b=(u*d)@u.T;b=.5*(b+b.T);b=b/(b.trace()/n)
                yield f'{family}_n{n}_s{seed}',b,rng
    for name,b in (('zero8',torch.zeros(8,8,dtype=torch.float64)),('identity8',torch.eye(8,dtype=torch.float64)),('diagonal8',torch.diag(torch.linspace(.1,1.9,8,dtype=torch.float64))),('equal_diagonal_rankone8',torch.ones(8,8,dtype=torch.float64))):
        yield name,b,torch.Generator().manual_seed(164)

def energy(b,rot):return offdiag(rot.T@b@rot).square().sum()
def geometry(q):
    err=(q.T@q-torch.eye(len(q),dtype=q.dtype)).norm()
    return dict(absolute_gram=float(err),normalized_gram=float(err/math.sqrt(len(q))))

def run(out):
    started=time.monotonic();out=Path(out);out.mkdir(exist_ok=True)
    if (out/'cpu_result.json').exists() or (out/'checks.jsonl').exists():raise ValueError('Fresh CPU attempt required')
    checked=[];finite=[];jacobi_identity=[]
    def append(path,row):
        with (out/path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    for name,b,rng in matrices():
        # A fresh independent skew controls derivatives even when raw-gradient
        # directions vanish at equal diagonals or exactly diagonal covariance.
        z=torch.randn(len(b),len(b),generator=rng,dtype=torch.float64)
        random_a=.5*(z-z.T);random_a=random_a*(.5/(random_a.norm()/math.sqrt(2)).clamp_min(1e-12)).clamp(max=1)
        jac=generator(b,'jacobi')[2];eng=generator(b,'energy')[2]
        for direction,a in (('jacobi',jac),('energy',eng),('independent_skew',random_a)):
            s,h=explicit_commutator_terms(b,a);parts=single_product_terms(b,a)
            t=torch.zeros((),dtype=torch.float64,requires_grad=True)
            f=energy(b,torch.matrix_exp(t*a));ds,=torch.autograd.grad(f,t,create_graph=True);dh,=torch.autograd.grad(ds,t)
            x=b@a-a@b;y=x@a-a@x
            invariant_s=-2*(b.diagonal()*x.diagonal()).sum()
            invariant_h=-2*(x.diagonal().square()+b.diagonal()*y.diagonal()).sum()
            eps=2e-4;f0=float(f.detach());fp=float(energy(b,torch.matrix_exp(eps*a)));fm=float(energy(b,torch.matrix_exp(-eps*a)))
            fd_s=(fp-fm)/(2*eps);fd_h=(fp-2*f0+fm)/eps**2
            scale_s=max(1.,abs(float(s)));scale_h=max(1.,abs(float(h)))
            row=dict(matrix=name,direction=direction,n=len(b),slope=float(s),exact_curvature=float(h),autograd_slope=float(ds.detach()),autograd_curvature=float(dh.detach()),
                single_product_slope=float(parts['slope']),single_product_curvature=float(parts['curvature']),invariant_slope=float(invariant_s),invariant_curvature=float(invariant_h),
                diagonal_model=float(parts['diagonal_model']),offdiagonal_commutator=float(parts['offdiagonal_commutator']),diagonal_drift=float(parts['diagonal_drift']),
                fd_slope=fd_s,fd_curvature=fd_h,fd_slope_normalized_error=abs(fd_s-float(s))/scale_s,fd_curvature_normalized_error=abs(fd_h-float(h))/scale_h)
            row['passed']=all(torch.allclose(v,s,rtol=1e-9,atol=1e-10) for v in (ds.detach(),parts['slope'],invariant_s)) and all(torch.allclose(v,h,rtol=1e-9,atol=1e-10) for v in (dh.detach(),parts['curvature'],invariant_h)) and row['fd_slope_normalized_error']<=2e-5 and row['fd_curvature_normalized_error']<=2e-5
            append('checks.jsonl',row);checked.append(row)
            if not row['passed']:raise AssertionError('Exact derivative check failed: '+name+'/'+direction)
        raw,bound,a,cap,chosen=generator(b,'jacobi');terms=single_product_terms(b,raw)
        # With J=-.5 B*gap/(gap^2+tau^2), -s >= h_diag;
        # unconstrained diagonal-model optimum >=1, so adding this scalar
        # damping to capped Jacobi should be inert (except stationary zeros).
        model=quadratic_alpha(terms['slope'],terms['diagonal_model'])
        j_diag=a*quadratic_alpha(*[single_product_terms(b,a)[k] for k in ('slope','diagonal_model')])
        identity=dict(matrix=name,raw_slope=float(terms['slope']),raw_diagonal_curvature=float(terms['diagonal_model']),raw_model_optimal_scale=model,capped_norm=float(a.norm()),jacobi_diag_change=float((a-j_diag).norm()),passed=torch.allclose(a,j_diag,rtol=1e-10,atol=1e-12))
        append('jacobi_model_identity.jsonl',identity);jacobi_identity.append(identity)
        if not identity['passed']:raise AssertionError('Jacobi diagonal model unexpectedly changed step')
        def damped_full(a):
            terms=single_product_terms(b,a);return a*quadratic_alpha(terms['slope'],terms['curvature'])
        policies=dict(jacobi_cap=jac,energy_cap=eng,energy_diag_damped=generator(b,'damped')[2],energy_full_damped=damped_full(eng),jacobi_diag_damped=j_diag,jacobi_full_damped=damped_full(jac))
        before=float(offdiag(b).square().sum())
        for policy,a in policies.items():
            for curve in ('exp','ideal_cubic2'):
                rot=torch.matrix_exp(a) if curve=='exp' else ideal_cubic_two(a)
                after=float(energy(b,rot));g=geometry(rot);terms=single_product_terms(b,a)
                row=dict(matrix=name,n=len(b),policy=policy,curve=curve,energy_before=before,energy_after=after,energy_ratio=None if before==0 else after/before,
                    skew_fro_bound=float(a.norm()/math.sqrt(2)),slope=float(terms['slope']),exact_curvature=float(terms['curvature']),diagonal_curvature=float(terms['diagonal_model']),
                    quadratic_predicted_energy_change=float(terms['slope']+.5*terms['curvature']),geometry=g,
                    geometry_admitted=g['absolute_gram']<=.05 and g['normalized_gram']<=.005,input_overshoot=after>before+1e-12*max(1.,before),
                    finite=math.isfinite(after))
                append('finite_steps.jsonl',row);finite.append(row)
                if not row['finite'] or row['skew_fro_bound']>.5*(1+1e-12):raise AssertionError('Nonfinite/cap failure')
    result=dict(status='PASS_CPU_FULL_DIRECTIONAL_CURVATURE',matrices=31,derivative_checks=len(checked),jacobi_identity_checks=len(jacobi_identity),finite_step_cases=len(finite),
        max_autograd_curvature_normalized_error=max(abs(r['exact_curvature']-r['autograd_curvature'])/max(1.,abs(r['exact_curvature'])) for r in checked),
        max_fd_curvature_normalized_error=max(r['fd_curvature_normalized_error'] for r in checked),negative_exact_curvature_checks=sum(r['exact_curvature']< -1e-12 for r in checked),
        geometry_admitted=sum(r['geometry_admitted'] for r in finite),policy_outcomes={p:{curve:dict(count=sum(r['policy']==p and r['curve']==curve for r in finite),overshoots=sum(r['policy']==p and r['curve']==curve and r['input_overshoot'] for r in finite)) for curve in ('exp','ideal_cubic2')} for p in policies},
        cuda_initialized=torch.cuda.is_initialized(),seconds=time.monotonic()-started,
        scope='Fresh31small artificial FP64 symmetric-covariance CPU states; exact exp and ideal same-dtype cubic2, not native QKV/compiled/BF16/cost/learning qualification. Full quadratic optimum is not a finite-descent certificate.',
        new_persistent_optimizer_state=False,live_defaults_changed=False)
    assert result['derivative_checks']==93 and result['finite_step_cases']==372 and not result['cuda_initialized']
    (out/'cpu_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
