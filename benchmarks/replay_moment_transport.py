"""Full raw-second-moment oracle for diagonal transport under prescribed bases.

CPU float64, deterministic gradients; this isolates coordinate history rather
than comparing eigentrackers or asserting a training-quality result.
"""
import argparse
import json
import math
from pathlib import Path

import torch


def rotation(n, i, j, angle):
    q = torch.eye(n, dtype=torch.float64)
    q[i, i] = q[j, j] = math.cos(angle)
    q[i, j], q[j, i] = -math.sin(angle), math.sin(angle)
    return q


def basis(step, scenario):
    if scenario == 'stationary':
        angle = 0.
    elif scenario == 'smooth':
        angle = .8*math.sin(step/180.)
    else:
        angle = 0. if step < 500 else (.8 if step < 750 else -.4)
    left = rotation(4, 0, 1, angle) @ rotation(4, 2, 3, .4*angle)
    right = rotation(4, 1, 2, -.7*angle)
    # Row-major vectorization of QL.T @ G @ QR.
    return torch.kron(left.T.contiguous(), right.T.contiguous())


def replay(gradients, beta, interval, policy, scenario):
    transform = basis(0, scenario)
    m = torch.zeros(16, dtype=torch.float64)
    v = torch.zeros_like(m)
    world_m = torch.zeros_like(m)
    world_s = torch.zeros(16, 16, dtype=torch.float64)
    rows = []
    for step, grad in enumerate(gradients, 1):
        projected = transform @ grad
        m = .95*m + .05*projected
        v = beta*v + (1-beta)*projected.square()
        world_m = .95*world_m + .05*grad
        world_s = beta*world_s + (1-beta)*torch.outer(grad, grad)
        if step % interval == 0:
            new = basis(step, scenario)
            transition = new @ transform.T
            m = transition @ m
            if policy == 'squared_overlap':
                v = transition.square() @ v
            transform = new
        exact_m = transform @ world_m
        exact_v = (transform @ world_s @ transform.T).diagonal()
        m_error = float((m-exact_m).norm()/exact_m.norm().clamp_min(1e-15))
        assert m_error < 1e-10, (step, m_error)
        exact_update = transform.T @ (exact_m/exact_v.clamp_min(1e-15).sqrt())
        approximate_update = transform.T @ (m/v.clamp_min(1e-15).sqrt())
        rows.append(dict(m_relative_error=m_error,
            v_relative_error=float((v-exact_v).norm()/exact_v.norm().clamp_min(1e-15)),
            update_relative_error=float((approximate_update-exact_update).norm()/exact_update.norm().clamp_min(1e-15)),
            update_cosine=float(torch.nn.functional.cosine_similarity(exact_update,approximate_update,dim=0))))
    if scenario == 'stationary':
        assert max(r['v_relative_error'] for r in rows) < 1e-12
    return dict(beta2=beta, refresh_interval=interval, policy=policy, scenario=scenario,
        final=rows[-1], mean_after_step200={k:sum(r[k] for r in rows[200:])/len(rows[200:]) for k in rows[0]},
        maximum_m_relative_error=max(r['m_relative_error'] for r in rows))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    gen=torch.Generator().manual_seed(123)
    mixing=torch.randn(16,16,generator=gen,dtype=torch.float64)
    mixing=mixing @ torch.diag(torch.linspace(.2,2.,16,dtype=torch.float64)) / 4
    gradients=torch.randn(1000,16,generator=gen,dtype=torch.float64) @ mixing.T
    gradients+=torch.linspace(-.5,.5,16,dtype=torch.float64)
    # Analytic controls: permutation and sign changes are exact for diagonal S.
    s=torch.diag(torch.tensor([100.,1.],dtype=torch.float64))
    controls={}
    for label,t in [('sign',torch.diag(torch.tensor([-1.,1.],dtype=torch.float64))),
                    ('permutation',torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64))]:
        error=float(((t@s@t.T).diagonal()-t.square()@s.diagonal()).abs().max())
        assert error==0
        controls[label+'_error']=error
    paths=[]
    for pieces in [1,10,100]:
        t=rotation(2,0,1,math.pi/4/pieces).T
        v=s.diagonal().clone()
        for _ in range(pieces):v=t.square()@v
        paths.append(dict(pieces=pieces,stored_diagonal=v.tolist(),exact_final_diagonal=[50.5,50.5]))
    results=[replay(gradients,beta,interval,policy,scenario)
             for scenario in ['stationary','smooth','shock']
             for beta in [.99,.999] for interval in [1,20]
             for policy in ['carry','squared_overlap']]
    report=dict(seed=123,dtype='float64',steps=1000,matrix_shape=[4,4],beta1=.95,
                controls=controls,rotation_only_paths=paths,results=results,
                scope='Prescribed orthogonal bases and common gradients; exact raw-second-moment oracle. No eigentracker, covariance-beta study, or training-quality claim.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(controls=controls,rotation_only_paths=paths,
                         cases=len(results),maximum_m_relative_error=max(r['maximum_m_relative_error'] for r in results)),indent=2))


if __name__=='__main__':main()
