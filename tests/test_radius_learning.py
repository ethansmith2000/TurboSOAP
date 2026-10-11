import torch
from benchmarks.cifar_radius_learning import configure_matrix
from benchmarks.movement_retraction import configure_matrix as old_configure
from benchmarks.prepare_radius_learning import arms
from soap_reference import ResearchSOAP

def test_baseline_adapter_exact_and_new_policies_preserve_inputs():
    p=torch.nn.Parameter(torch.zeros(8,12)); old=ResearchSOAP([p],basis_method='warm')
    new=ResearchSOAP([p],basis_method='warm');old_configure(old,'fro_cubic2_cap05');configure_matrix(new,'fro_cubic2_cap05')
    gen=torch.Generator().manual_seed(419);g=torch.randn(8,12,generator=gen);cov=g@g.T
    q=torch.linalg.qr(torch.randn(8,8,generator=gen)).Q;saved=(cov.clone(),q.clone())
    assert old.param_groups[0]==new.param_groups[0]
    assert torch.equal(old._gauge_step_one(cov,q,old.param_groups[0]),new._gauge_step_one(cov,q,new.param_groups[0]))
    for name in ['fro_cubic2_cap06','fro_cubic3_cap06']:
        configure_matrix(new,name);new._gauge_step_one(cov,q,new.param_groups[0])
    assert torch.equal(cov,saved[0]) and torch.equal(q,saved[1])

def test_order_includes_three_interleaved_controls_and_paired_betas():
    aa=arms();assert len(aa)==16 and len({x['label'] for x in aa})==16
    for kind in ['qr','fro05','fro06','fro06strong']:
        group=[x for x in aa if x['kind']==kind and x['covariance_beta']==.99]
        assert len(group)==3 and all(x['repeat_of']==group[0]['label'] for x in group[1:])
        assert len([x for x in aa if x['kind']==kind and x['covariance_beta']==.999])==1
