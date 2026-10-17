"""Only fresh3500/evaluation/probe-contract deltas; CUDA forbidden by CPU controller."""
import ast,copy,json
from pathlib import Path
import pytest,torch
from damped_confirmation_contract import recipe,protocol,validate,ARMS,POLICIES,PROBES
from hybrid_training_contract import BASELINE

def test_original_native_functions_unchanged_and_recipe_delta_bounded():
    root=Path(__file__).resolve().parents[1]/'benchmarks'
    def functions(name):
        tree=ast.parse((root/name).read_text())
        return [ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    assert functions('cifar_damped_learning.py')==functions('cifar_damped_confirmation.py')
    from damped_learning_contract import recipe as old_recipe,protocol as old_protocol
    base=json.loads((BASELINE/'config.json').read_text());c=recipe(base);p=protocol();validate(c,p,base)
    old=old_recipe(base)
    assert c['seed']==160 and c['max_wall_seconds']==old['max_wall_seconds']
    assert c['matched_comparison']['execution_order']==['highest_residual20','qr20','damped20']
    changed=copy.deepcopy(old);changed['seed']=160
    changed['matched_comparison']['execution_order']=c['matched_comparison']['execution_order']
    assert c==changed
    prior=old_protocol()
    ignored={'seed','execution_order','implementation','decision','bound_rationale'}
    assert {k:v for k,v in p.items() if k not in ignored}=={k:v for k,v in prior.items() if k not in ignored}
    assert {k:v for k,v in p['implementation'].items() if k!='entry'}=={k:v for k,v in prior['implementation'].items() if k!='entry'}
    assert {k:v for k,v in c.items() if k not in ('seed','max_wall_seconds','matched_comparison')}=={k:v for k,v in old.items() if k not in ('seed','max_wall_seconds','matched_comparison')}
    assert p['gates']['own_eager_basis_relative']==old_protocol()['gates']['own_eager_basis_relative']
    assert p['diagnostic_admission']==old_protocol()['diagnostic_admission']
    assert p['bounds']['maximum_peak_allocated_bytes']==24*2**30
    assert p['bounds']['maximum_initial_snapshot_purpose_cpu_bytes']==512*2**20
    assert p['full_moment_diagnostic']['calls']==p['probe_calls']==list(PROBES)
    assert min(p['evaluation_calls'])>p['schedule']['lr_warmup']
    assert p['evaluation_calls']==[1000,2000,3000,3500]
    for field,value in [('seed',159),('max_wall_seconds',1800)]:
        wrong=copy.deepcopy(c);wrong[field]=value
        with pytest.raises(ValueError,match='Undeclared'):validate(wrong,p,base)

def test_long_counters_probe_masks_and_resource_contract():
    p=protocol();e=p['expected']
    assert p['maximum_optimizer_updates']==11260 and p['steps_per_arm']==3500
    assert e['hybrid_kernel_calls']==2688 and e['hybrid_calls_by_dimension']=={'2048':1680,'6144':336,'8192':672}
    assert all(e['policy_calls_by_dimension'][a]=={'2048':840,'6144':168,'8192':336} for a in POLICIES)
    assert e['diagnostic_reference_calls']==32 and e['full_moment_transactions']==24 and e['counterfactual_basis_calls']==96
    assert all((v-1)%20==0 and (v-1)%200 for v in PROBES)
    assert all(e['primary'][a]['research_warm_refreshes']==157 and e['primary'][a]['research_qr_refreshes']==17 for a in POLICIES)
    assert e['primary']['qr20']['research_qr_refreshes']==174
    assert p['bounds']['internal_wall_seconds']==3600 and p['bounds']['external_watchdog_seconds']==3660
    assert not any(p['artifacts'][k] for k in ('checkpoints','final_weights','tensor_files','profiler_traces'))

def test_late_populated_all_arm_momentum_geometry_nonmutation_and_release(monkeypatch):
    from soap_reference import ResearchSOAP
    from movement_retraction import configure_matrix
    from cifar_optimizers import SplitOptimizer
    from native_damped_kernel import PolicyRouter
    from native_damped_diagnostic import configure_diagnostic
    from matched_training_runtime import OptimizerLease
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda:0);monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    for arm in ARMS:
        torch.manual_seed(160)
        model=torch.nn.Module();model.register_parameter('matrix',torch.nn.Parameter(torch.randn(12,8)))
        model.register_parameter('aux',torch.nn.Parameter(torch.randn(3)))
        router=PolicyRouter(lambda *a:None,lambda:'rng',lambda fn,**kw:fn)
        matrix=ResearchSOAP([model.matrix],basis_method='qr' if arm=='qr20' else 'warm',shampoo_beta=.99,
            precondition_frequency=20,hard_reset_interval=0 if arm=='qr20' else 200,transport_precision='highest')
        matrix.param_groups[0]['param_names']=['matrix']
        if arm in POLICIES:
            configure_matrix(matrix,'fro_cubic2_cap05');matrix.param_groups[0].update(basis_lr=.5,basis_jacobi_damping=.01)
            matrix._gauge_step_one=router.kernels[arm]
        configure_diagnostic(matrix,router,arm)
        lease=OptimizerLease(model,SplitOptimizer(matrix,torch.optim.AdamW([model.aux])));del matrix
        for call in range(1,3501):
            model.matrix.grad=torch.randn_like(model.matrix);router.probe=call in PROBES;lease.opt.step()
        router.probe=False;state=lease.opt.matrix.state[model.matrix]
        assert state['research_qr_refreshes']==(174 if arm=='qr20' else 17)
        assert state['research_warm_refreshes']==(0 if arm=='qr20' else 157)
        del state
        assert len(router.moment_probes)==2 and router.cf_calls==8
        assert len(router.probes)==(4 if arm in POLICIES else 0)
        assert all(row['source_arm']==arm and row['source_step'] in (1020,3480) and row['nonmutation_pass'] and row['registered_release_pass'] and row['physical_m_pass'] for row in router.moment_probes)
        assert lease.release()['passed']

def test_actual_entry_eval_restoration_failure_and_mutation(tmp_path):
    from test_hybrid_learning import LearningCPU
    from hybrid_training_kernel import build_arm
    from cifar_damped_confirmation import evaluate_transaction,endpoint
    r=LearningCPU(tmp_path);r.model.blocks[1].eval();opt=build_arm(r.model,r.config,'hybrid20',r.kernel)
    for param in r.model.parameters():param.grad=torch.ones_like(param)
    flags=[m.training for m in r.model.modules()];before=endpoint(r,opt)
    row=evaluate_transaction(r,opt)
    assert row['restoration_passed'] and row['before']==row['after']==before
    assert [m.training for m in r.model.modules()]==flags
    r.evaluation_fault='raise'
    with pytest.raises(RuntimeError,match='injected evaluation'):evaluate_transaction(r,opt)
    assert endpoint(r,opt)==before and [m.training for m in r.model.modules()]==flags
    r.evaluation_fault='weights'
    with pytest.raises(RuntimeError,match='Evaluation mutated'):evaluate_transaction(r,opt)
    r.snapshot.close()

def test_fresh_preparation_seal_rejects_semantic_eval_tamper(tmp_path,monkeypatch):
    import prepare_damped_confirmation as prep
    parent=tmp_path/'parent';parent.mkdir();(parent/'source').mkdir()
    prep.write(parent/'manifest.json',dict(source_sha256={}));prep.write(parent/'execution_status.json',dict(status='completed'))
    prep.write(parent/'summary.json',{});monkeypatch.setattr(prep,'PARENTS',(parent,))
    ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=160,parameters=50796644,snapshot_purpose_bytes=414257552,cuda_initialized=False))
    b=tmp_path/'fresh';m=prep.prepare(b,ref);prep.verify(b)
    assert len(m['source_sha256'])==4 and '3660 /venv/main/bin/python' in (b/'supervisor_template.conf').read_text()
    p=prep.read(b/'protocol.json');p['evaluation_calls']=[3500];prep.write(b/'protocol.json',p)
    wrong=copy.deepcopy(m);wrong['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',wrong)
    with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
