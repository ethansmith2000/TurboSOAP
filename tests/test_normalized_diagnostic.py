import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks.normalized_diagnostic_contract import (
    BASE, NORMALIZED_POLICY, validate_config, validate_child_change,
    validate_outer_change, validate_reference_region, verify_launch, localize,
)
from benchmarks.normalized_input_replay import (
    purpose_bytes, normalized_interpretation, RAM_CAP, run_normalized_replay,
)
from benchmarks.prepare_normalized_input import prepare
from benchmarks.normalized_input_abi import _select_saved, EXPECTED_PROOF
from benchmarks.saved_operand_abi import select_operand, OperandUnavailable
from benchmarks.execution_fingerprints import tensor_record, gradient_records
from benchmarks.summarize_normalized_input import qualify


ROOT = Path(__file__).resolve().parents[1]


def config():
    value = json.loads((BASE/'config.json').read_text())
    value['max_wall_seconds'] = 900
    d = value['execution_diagnostic']; d['per_process_seconds'] = 210
    d['normalized_input'] = copy.deepcopy(NORMALIZED_POLICY)
    for arm in d['contexts']:
        arm['label'] = arm['label'].replace('saved_operand', 'normalized_input')
    return value


def test_primary_outer_and_preceding_replays_are_ast_unchanged():
    before = (BASE/'source/benchmarks/cifar_saved_operand_child.py').read_text()
    after = (ROOT/'benchmarks/cifar_normalized_child.py').read_text()
    assert validate_child_change(before, after)
    for changed in (after.replace("build_optimizer(model,cfg,'qr20')", "build_optimizer(model,cfg,'qr10')"),
                    after.replace('cudnn.deterministic=True', 'cudnn.deterministic=False'),
                    after.replace('support.lr_at(index+1', 'support.lr_at(index+2'),
                    after.replace("ram_cap_bytes=d['normalized_input']['ram_cap_bytes']", "ram_cap_bytes=0")
                    .replace('# NORMALIZED_REPLAY_END', '# BROKEN_END')):
        with pytest.raises(ValueError):
            validate_child_change(before, changed)
    assert validate_outer_change((BASE/'source/benchmarks/cifar_saved_operand.py').read_text(),
                                (ROOT/'benchmarks/cifar_normalized.py').read_text())
    changed = (ROOT/'benchmarks/cifar_normalized.py').read_text().replace('if code:', 'if False:')
    with pytest.raises(ValueError):
        validate_outer_change((BASE/'source/benchmarks/cifar_saved_operand.py').read_text(), changed)


@pytest.mark.parametrize('field', ['lr', 'recipe', 'betas', 'measured', 'allocation', 'ram', 'save'])
def test_complete_config_recipe_and_work_bounds_reject_changes(field):
    value = config(); assert validate_config(value)
    if field in ('lr', 'recipe', 'betas'):
        value[field] = 'changed'
    elif field == 'measured':
        value['execution_diagnostic']['measured_steps'] = 420
    elif field == 'allocation':
        value['execution_diagnostic']['contexts'][0]['allocation_horizon'] = 3500
    elif field == 'ram':
        value['execution_diagnostic']['normalized_input']['ram_cap_bytes'] *= 2
    else:
        value['checkpoint'] = True
    with pytest.raises(ValueError):
        validate_config(value)


def test_pre_backward_capture_calls_are_metadata_only():
    text = (ROOT/'benchmarks/normalized_input_replay.py').read_text()
    assert validate_reference_region(text)
    for added in ('normalized.clone()', 'tensor_record(normalized)', 'normalized.cpu()'):
        changed = text.replace('# NORMALIZED_REFERENCE_END', added+'\n                        # NORMALIZED_REFERENCE_END')
        with pytest.raises(ValueError):
            validate_reference_region(changed)


def test_purpose_memory_counts_unique_native_storage_and_cap():
    x = torch.empty(4); alias = x.detach(); target = torch.empty(2)
    assert purpose_bytes(100, x, target, normalized=alias) == 100+24
    with pytest.raises(MemoryError):
        purpose_bytes(RAM_CAP, x, target)


def test_prepared_bundle_has_no_device_or_launch_and_preserves_frozen_dependencies(tmp_path):
    b = tmp_path/'prepared'; manifest = prepare(b)
    assert manifest['device'] is None and not manifest['gpu_selected']
    assert not (b/'launch_manifest.json').exists()
    validate_config(json.loads((b/'config.json').read_text()))
    old = json.loads((BASE/'manifest.json').read_text())
    added = {'benchmarks/'+name for name in [
        'normalized_input_replay.py', 'normalized_diagnostic_contract.py', 'normalized_input_abi.py',
        'tensor_ownership.py', 'cifar_normalized_child.py', 'cifar_normalized.py',
        'summarize_normalized_input.py', 'prepare_normalized_input.py']}
    for name, digest in old['source_sha256'].items():
        if name not in added:
            assert hashlib.sha256((b/'source'/name).read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError):
        prepare(b)
    assert not torch.cuda.is_initialized()


def test_launch_requires_frozen_manifest_config_and_exact_claim(tmp_path, monkeypatch):
    b = tmp_path/'prepared'; m = prepare(b)
    manifest_sha = hashlib.sha256((b/'manifest.json').read_bytes()).hexdigest()
    launch = dict(manifest_sha256=manifest_sha, config_sha256=m['config_sha256'], gpu=dict(index=6, uuid='GPU-fixture'))
    (b/'launch_manifest.json').write_text(json.dumps(launch))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-wrong'); monkeypatch.setenv('GPU_CLAIM_INDICES', '6')
    with pytest.raises(ValueError, match='device'):
        verify_launch(b)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-fixture'); assert verify_launch(b) == launch
    (b/'config.json').write_text('{}')
    with pytest.raises(ValueError, match='configuration'):
        verify_launch(b)


@pytest.mark.parametrize('valid,grad,norm,operand,expected', [
    (False, 'b', 'b', 'b', 'INCONCLUSIVE_PARITY_ABI_OR_RESOURCE_GATE'),
    (True, 'a', 'b', 'b', 'INCONCLUSIVE_NO_FRESH_HEAD_GRADIENT_VARIATION'),
    (True, 'b', 'b', 'b', 'VARIATION_UPSTREAM_OF_FINAL_AFFINE_CAST'),
    (True, 'b', 'a', 'b', 'VARIATION_IN_FINAL_AFFINE_CAST_OR_LIFETIME_PATH'),
    (True, 'b', 'a', 'a', 'INCONCLUSIVE_INPUTS_EQUAL_WITH_HEAD_GRADIENT_VARIATION'),
])
def test_localization_requires_fresh_primary_variation_and_parity(valid, grad, norm, operand, expected):
    left = dict(primary_head_gradient='a', normalized_cls='a', operand='a')
    right = dict(primary_head_gradient=grad, normalized_cls=norm, operand=operand)
    assert localize(left, right, valid) == expected


def fake_runs():
    c = json.loads((BASE/'config.json').read_text())
    runs = [json.loads((BASE/'contexts'/ctx['label']/'result.json').read_text()) for ctx in c['execution_diagnostic']['contexts']]
    for run in runs:
        old = run['saved_operand_replay']; caps = copy.deepcopy(old['captures'])
        release = dict(all_registered_released=True, tracked_tensors=3, tracked_unique_storages=3,
                       live_tensors=0, live_unique_storages=0, live_unique_storage_bytes=0)
        for cap in caps:
            cap.update(normalized_cls=dict(dtype='torch.float32', shape=[256, 1536], sha256='fixture', finite=True),
                       normalized_selection=copy.deepcopy(cap['selection']),
                       capture_registered_release=copy.deepcopy(release),
                       purpose_before_backward_bytes=220912016, purpose_after_backward_bytes=223000000)
            cap['normalized_selection'].update(selected_index=21, shape_matches=[6, 17, 21])
            for row in cap['normalized_selection']['saved_metadata']:
                row['device'] = 'cuda:0'
        abi = dict(forward_sha256='b606a746c3bba1b875a186ba582ec3ae463640eb097f6cd1671e4c551f1e66cc',
                   backward_sha256='e8105f7122c5762a23f1143e4c12da4c9621d28c553391bb14a5dea09e5d2bbe', proof=EXPECTED_PROOF)
        run.update(own_affinity_restored=True, code_after_normalized_replay=copy.deepcopy(run['code_after_saved_operand_replay']))
        run['normalized_input_replay'] = dict(status='completed', captures=caps, heads=copy.deepcopy(old['heads']),
            final_invariants_before=old['final_invariants_before'], final_invariants_after=old['final_invariants_after'],
            held_invariants_before=old['held_invariants_before'], held_invariants_after=old['held_invariants_after'],
            rebuilt_input=old['rebuilt_input'], rebuilt_target=old['rebuilt_target'],
            ram_cap_bytes=RAM_CAP, maximum_purpose_bytes=223000000,
            registered_outer_release=copy.deepcopy(release), held_operand=old['held_operand'],
            capture_kernel_names_counts=old['capture_kernel_names_counts'], head_kernel_names_counts=old['head_kernel_names_counts'], abi=abi)
        a = run['measured'][0]; actual = dict(output=a['output'], loss=a['loss'], grad=a['grad_before_clip'])
        run['normalized_input_replay']['interpretation'] = normalized_interpretation(actual, run['initial_state_replay']['plain'], caps, old['heads'], old)
    return runs


@pytest.mark.parametrize('change', ['slot', 'storage', 'live_storage', 'device', 'cap', 'gradient', 'code', 'mask', 'count', 'input', 'restore'])
def test_result_qualifier_rejects_missing_wrong_or_nonparity_evidence(change):
    runs = fake_runs(); value = qualify(runs)
    assert value['observer_valid'] and value['localization_supported'] and not value['fresh_variation_observed']
    assert all('INCONCLUSIVE_NO_FRESH' in p['classification'] for p in value['pairs'])
    if change == 'count':
        runs.pop()
    elif change == 'code':
        runs[0]['code_after_normalized_replay'] = []
    elif change == 'mask':
        runs[0]['own_affinity_restored'] = False
    else:
        n = runs[0]['normalized_input_replay']
        if change == 'slot':
            n['captures'][0]['normalized_selection']['selected_index'] = 6
        elif change == 'storage':
            n['captures'][0]['capture_registered_release']['all_registered_released'] = False
        elif change == 'live_storage':
            n['captures'][0]['capture_registered_release']['live_unique_storages'] = 1
        elif change == 'device':
            n['captures'][0]['normalized_selection']['saved_metadata'][21]['device'] = 'cpu'
        elif change == 'cap':
            n['maximum_purpose_bytes'] = RAM_CAP
        elif change == 'gradient':
            n['captures'][0]['execution']['grad']['head.weight']['sha256'] = 'changed'
        elif change == 'input':
            n['rebuilt_input'] = {}
        else:
            n['final_invariants_after'] = {}
    assert not qualify(runs)['localization_supported']


def test_complete_new_observer_on_native_cpu_aot_with_explicit_cuda_boundary_stubs(monkeypatch):
    # Execute the NEW observer end-to-end on CPU. Native graph/backward/reference
    # selection are real; CUDA sync/profile/code/production ABI boundaries are
    # explicit stubs, so this test claims no CUDA qualification/provenance.
    import benchmarks.normalized_input_replay as replay
    from torch._dynamo.backends.common import aot_autograd
    from functorch.compile import make_boxed_func
    class Tiny(torch.nn.Module):
        embed_dim = 12; num_classes = 4
        def __init__(self):
            super().__init__(); self.scale=torch.nn.Parameter(torch.ones(12)); self.bias=torch.nn.Parameter(torch.zeros(12)); self.head=torch.nn.Linear(12,4)
        def forward(self, x):
            u=x*1.1+self.bias; var, mean=torch.var_mean(u,dim=-1,correction=0,keepdim=True)
            normalized=(u-mean)*torch.rsqrt(var+1e-5)
            return self.head((normalized*self.scale+self.bias)[:,0,:])
    graphs = {}
    def fw(gm, inputs):
        graphs['fw']=gm; return make_boxed_func(gm.forward)
    def bw(gm, inputs):
        graphs['bw']=gm; return make_boxed_func(gm.forward)
    torch.manual_seed(175); model=Tiny(); initial=copy.deepcopy(model.state_dict())
    x=torch.randn(3,4,12); target=torch.tensor([0,1,2]); initial_rng=torch.random.get_rng_state()
    native_compile=torch.compile; native_autocast=torch.autocast
    active=native_compile(model,backend=aot_autograd(fw_compiler=fw,bw_compiler=bw),fullgraph=True,dynamic=False)
    with native_autocast('cpu',dtype=torch.bfloat16):
        output=active(x); loss=torch.nn.functional.cross_entropy(output.float(),target)
    loss.backward(); execution=dict(output=tensor_record(output),loss=float(loss.detach()),grad=gradient_records(model))
    roles=list(list(graphs['fw'].graph.nodes)[-1].args[0])[1:]
    normalized=next(n for n in roles if str(n.target)=='aten.mul.Tensor' and all(isinstance(a,torch.fx.Node) for a in n.args)
                    and str(n.args[0].target)=='aten.sub.Tensor' and str(n.args[1].target)=='aten.rsqrt.default')
    index=roles.index(normalized)
    matches=[i for i,n in enumerate(roles) if tuple(n.meta['val'].shape)==(3,4,12) and n.meta['val'].dtype==torch.float32]
    opindex=next(i for i,n in enumerate(roles) if tuple(n.meta['val'].shape)==(3,12) and n.meta['val'].dtype==torch.bfloat16)
    opabi=dict(saved_count=len(roles),saved_index=opindex,shape=[3,12],stride=[12,1])
    monkeypatch.setattr(replay,'loaded_code',lambda:[])
    monkeypatch.setattr(replay,'normalized_abi',lambda rows:SimpleNamespace(record=lambda:{'scope':'CPU fixture only'}))
    monkeypatch.setattr(replay,'classifier_abi',lambda *args:opabi)
    monkeypatch.setattr(replay,'select_normalized',lambda node,abi:_select_saved(node,saved_count=len(roles),saved_index=index,
        shape=(3,4,12),stride=(48,12,1),same_shape_indices=matches,device='cpu'))
    monkeypatch.setattr(torch,'autocast',lambda device,**kwargs:native_autocast('cpu',**kwargs))
    monkeypatch.setattr(torch,'compile',lambda model,**kwargs:native_compile(model,backend='aot_eager',fullgraph=True,dynamic=False))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda:0)
    class Profile:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def events(self):return [SimpleNamespace(device_type='DeviceType.CUDA',name='CPU_BOUNDARY_STUB_ONLY')]
    monkeypatch.setattr(torch.profiler,'profile',lambda **kwargs:Profile())
    first=dict(input=tensor_record(x),target=tensor_record(target),output=execution['output'],loss=execution['loss'],grad_before_clip=execution['grad'])
    previous=dict(status='completed',captures=[dict(execution=copy.deepcopy(execution),operand=None) for _ in range(3)])
    def invariants():
        return dict(model={n:tensor_record(t) for n,t in model.state_dict().items()},optimizer={},rng=tensor_record(torch.random.get_rng_state()))
    with torch.no_grad():model.head.weight.add_(.01)
    before=invariants(); progress=[]
    value=run_normalized_replay(model=model,active=active,initial=initial,initial_rng=initial_rng,
        get_rng=torch.random.get_rng_state,restore_rng=torch.random.set_rng_state,
        make_batch=lambda:(x.clone(),target.clone()),first_record=first,warm_record=execution,invariants=invariants,
        progress=lambda r:progress.append(r['status']),previous=previous,ram_cap_bytes=RAM_CAP)
    assert value['status']=='completed' and value['registered_outer_release']['all_registered_released']
    assert invariants()==before and len(value['captures'])==len(value['heads'])==3
    assert all(c['capture_registered_release']['all_registered_released'] for c in value['captures'])
    assert all(c['execution']==execution for c in value['captures'])
    assert value['interpretation']['checks']['primary_gradients_exact']
    assert value['interpretation']['checks']['held_head_matches']
    assert not value['interpretation']['checks']['earlier_original_replay_exact'] # fixture previous operand intentionally absent
    assert all(c['normalized_cls']==value['captures'][0]['normalized_cls'] for c in value['captures'])
    assert not torch.cuda.is_initialized()
