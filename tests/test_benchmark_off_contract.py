import copy
import json
from pathlib import Path
import pytest
from benchmarks.benchmark_off_contract import validate_child_change,validate_config,backend_delta,finite_records

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/workspace/optimizer_replay_results/cifar_soap_execution_diagnostic_20261007')


def config():
    c=json.loads((BASE/'config.json').read_text())
    c['execution_diagnostic']['contexts']=[dict(label=f'case{i}',allocation_horizon=h,
        timing_chunk=25,repeat=r) for i,(h,r) in enumerate([(420,0),(3500,0),(3500,1),(420,1)])]
    return c


def test_only_backend_flag_changes_in_actual_control_source():
    before=(BASE/'source/benchmarks/cifar_execution_child.py').read_text()
    after=(ROOT/'benchmarks/cifar_benchmark_off_child.py').read_text()
    assert validate_child_change(before,after)
    with pytest.raises(ValueError):validate_child_change(before,after.replace("torch.set_float32_matmul_precision('high')","torch.set_float32_matmul_precision('highest')"))
    with pytest.raises(ValueError):validate_child_change(before,before)


@pytest.mark.parametrize('mutation',[
    lambda c:c['execution_diagnostic']['contexts'].pop(),
    lambda c:c['execution_diagnostic']['contexts'][1].update(label='case0'),
    lambda c:c['execution_diagnostic']['contexts'][0].update(timing_chunk=10),
    lambda c:c['execution_diagnostic']['contexts'][2].update(repeat=0),
    lambda c:c['execution_diagnostic'].update(measured_steps=100),
    lambda c:c.update(max_wall_seconds=700),
    lambda c:c.update(matrix_lr=.001),
])
def test_reject_extra_flags_work_and_scope_drift(mutation):
    c=config();assert validate_config(c);mutation(c)
    with pytest.raises(ValueError):validate_config(c)


def test_metadata_delta_and_nonfinite_guard():
    a={'cudnn_flags':{'benchmark':True,'deterministic':False},'cuda':'12.8'}
    b=copy.deepcopy(a);b['cudnn_flags']['benchmark']=False
    assert backend_delta(a,b)==['cudnn_flags.benchmark']
    b['cudnn_flags']['deterministic']=True
    assert backend_delta(a,b)==['cudnn_flags.benchmark','cudnn_flags.deterministic']
    assert finite_records({'finite':True,'norm':1.})
    assert not finite_records({'finite':False})
    assert not finite_records({'loss':float('nan')})


def test_qualifier_rejects_incomplete_and_replay_or_trajectory_failures():
    from benchmarks.summarize_benchmark_off import qualify
    labels=['plan420_flush10_r0','plan420_flush25_r0','plan420_flush10_r1','plan3500_flush25_r1']
    # Retained real exact-group records; reading them does not rerun GPU work.
    runs=[json.loads((BASE/'contexts'/label/'result.json').read_text()) for label in labels]
    for r in runs:
        for phase in ['backend_before_smoke','backend_after_measured']:
            r[phase]['cudnn_flags']['benchmark']=False
    assert qualify(runs)['qualified']
    assert not qualify(runs[:3])['qualified']
    bad=copy.deepcopy(runs);bad[1]['measured'][1]['model_sha256']='changed'
    assert not qualify(bad)['qualified']
    bad=copy.deepcopy(runs);bad[1]['runtime_profile']['invariants_after']['model']='changed'
    assert not qualify(bad)['qualified']
    bad=copy.deepcopy(runs);bad[1]['measured'][1]['loss']=float('nan')
    assert not qualify(bad)['qualified']
