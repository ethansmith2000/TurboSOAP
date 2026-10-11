import copy
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from benchmarks.normalized_input_abi import (
    FORWARD_SHA, BACKWARD_SHA, PRODUCER, CONSUMER, BACKWARD_USE,
    normalized_abi, unique_loaded_rows, inspect_sources, select_normalized, _select_saved, OperandUnavailable,
)
from benchmarks.tensor_ownership import TensorOwnership
from benchmarks.audit_normalized_input import audit


BASE = Path('/workspace/optimizer_replay_results/soap_fresh420_json_round_20261008')


def source_rows():
    return [dict(path=str(BASE / 'compiled_code' / (h + '.py')), source_sha256=h)
            for h in (FORWARD_SHA, BACKWARD_SHA)]


def test_frozen_original_source_and_all_retained_metadata_without_gpu():
    abi = normalized_abi(source_rows())
    assert abi.record()['proof']['saved_index'] == 21
    assert abi.record()['proof']['producer_output_argument'] == 4
    assert abi.record()['same_shape_indices'] == [6, 17, 21]
    result = audit(BASE.parent)
    assert len(result['runtime_metadata']) == 9 and not result['gpu_executed']


def test_missing_duplicate_changed_and_wrong_sources_rejected(tmp_path):
    rows = source_rows()
    for bad in (rows[:1], rows + [dict(rows[0], module='distinct')],
                list(reversed(rows)) + [dict(rows[1], module='distinct')]):
        with pytest.raises(OperandUnavailable):
            normalized_abi(bad)
    path = tmp_path / 'forward.py'
    path.write_bytes(Path(rows[0]['path']).read_bytes() + b'\n# changed\n')
    bad = [dict(rows[0], path=str(path)), rows[1]]
    with pytest.raises(OperandUnavailable, match='hash'):
        normalized_abi(bad)
    bad[0]['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(OperandUnavailable, match='Missing pinned'):
        normalized_abi(bad)


def test_identical_inventory_entries_are_verified_without_hiding_conflicting_records():
    rows = source_rows()
    assert len(unique_loaded_rows(rows + copy.deepcopy(rows))) == 2
    assert normalized_abi(rows + copy.deepcopy(rows)) == normalized_abi(rows)
    conflicting = rows + [dict(rows[0], kernels=[{'name': 'changed'}])]
    with pytest.raises(OperandUnavailable, match='Duplicate'):
        normalized_abi(conflicting)


@pytest.mark.parametrize('side,old,new', [
    ('fw', 'buf35, buf36, buf37, buf40', 'buf36, buf35, buf37, buf40'),
    ('fw', PRODUCER + '.run(buf3, primals_5, buf19, buf31, buf35,',
     PRODUCER + '.run(buf3, primals_5, buf19, buf31, buf36,'),
    ('fw', CONSUMER + '.run(buf35,', CONSUMER + '.run(buf23,'),
    ('fw', 'tmp31 = tmp25 * tmp30', 'tmp31 = tmp25 + tmp30'),
    ('fw', 'tl.store(out_ptr2 + (r0_2 + 1536*x3), tmp31',
     'tl.store(out_ptr2 + (r0_2 + 1536*x3), tmp24'),
    ('bw', 'view_9, mul_7, convert_element_type_27,', 'view_9, convert_element_type_27, mul_7,'),
    ('bw', BACKWARD_USE + '.run(buf0, primals_17, mul_7,', BACKWARD_USE + '.run(buf0, primals_17, mul_2,'),
    ('bw', 'assert_size_stride(mul_7, (256, 65, 1536)', 'assert_size_stride(mul_7, (256, 64, 1536)'),
])
def test_structural_roles_rejected_even_independently_of_hash(side, old, new):
    fw, bw = [Path(r['path']).read_text() for r in source_rows()]
    original = fw if side == 'fw' else bw
    assert old in original
    changed = original.replace(old, new)
    with pytest.raises(OperandUnavailable):
        inspect_sources(changed if side == 'fw' else fw, changed if side == 'bw' else bw)


def synthetic():
    node = type('CompiledFunctionBackward', (), {})()
    tensors = [torch.zeros(12), torch.randn(3, 4, 12), torch.randn(3, 4, 12),
               torch.randn(3, 4, 12), torch.randn(3, 12, dtype=torch.bfloat16)]
    node.saved_tensors = tuple(tensors[:2])
    node._tensors_no_vc_check = tensors[2:]
    args = dict(saved_count=5, saved_index=3, shape=(3, 4, 12), stride=(48, 12, 1),
                same_shape_indices=(1, 2, 3), device='cpu', classifier_index=4, classifier_shape=(3, 12))
    return node, tensors, args


def test_both_saved_channels_use_role_with_three_shape_matches_and_no_tensor_math(monkeypatch):
    node, tensors, args = synthetic()
    def forbidden(*a, **k):
        raise AssertionError('Data read/copy before backward')
    for method in ('clone', 'cpu', 'numpy', 'item', 'sum'):
        monkeypatch.setattr(torch.Tensor, method, forbidden)
    selected, metadata = _select_saved(node, **args)
    assert selected is tensors[3]
    assert metadata['shape_matches'] == [1, 2, 3] and metadata['no_version_check_count'] == 3
    assert metadata['selected_index'] == 3


@pytest.mark.parametrize('change', ['count', 'slot', 'dtype', 'stride', 'offset', 'extent', 'alias',
                                  'duplicate', 'device', 'classifier', 'node'])
def test_runtime_missing_wrong_layout_device_or_alias_fails_closed(change):
    node, tensors, args = synthetic()
    if change == 'count':
        args['saved_count'] = 4
    elif change == 'slot':
        args['saved_index'] = 0
    elif change == 'dtype':
        tensors[3] = tensors[3].to(torch.bfloat16)
    elif change == 'stride':
        tensors[3] = tensors[3].transpose(1, 2)
    elif change == 'offset':
        tensors[3] = torch.empty(145)[1:].view(3, 4, 12)
    elif change == 'extent':
        tensors[3] = torch.empty(288)[:144].view(3, 4, 12)
    elif change == 'alias':
        tensors[2] = tensors[3].detach()
    elif change == 'duplicate':
        tensors[0] = torch.randn(3, 4, 12)
    elif change == 'device':
        args['device'] = 'meta'
    elif change == 'classifier':
        tensors[4] = torch.randn(3, 12)
    else:
        node = None
    if node is not None:
        node.saved_tensors = tuple(tensors[:2]); node._tensors_no_vc_check = tensors[2:]
    with pytest.raises(OperandUnavailable):
        _select_saved(node, **args)


def test_public_selector_requires_native_cuda_proof():
    abi = normalized_abi(source_rows())
    with pytest.raises(OperandUnavailable, match='CUDA device'):
        select_normalized(None, abi, device='cpu')
    with pytest.raises(OperandUnavailable, match='pinned'):
        select_normalized(None, replace(abi, forward_sha256='bad'))
    proof = json.loads(abi.proof_json); proof['saved_index'] = 6
    with pytest.raises(OperandUnavailable, match='role changed'):
        select_normalized(None, replace(abi, proof_json=json.dumps(proof)))
    proof = json.loads(abi.proof_json); proof['consumer'] = 'wrong'
    with pytest.raises(OperandUnavailable, match='role changed'):
        select_normalized(None, replace(abi, proof_json=json.dumps(proof)))


class TinyNormalized(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(12))
        self.bias = torch.nn.Parameter(torch.zeros(12))
        self.head = torch.nn.Linear(12, 4)

    def forward(self, x):
        u = x * 1.1 + self.bias
        variance, mean = torch.var_mean(u, dim=-1, correction=0, keepdim=True)
        normalized = (u - mean) * torch.rsqrt(variance + 1e-5)
        return self.head((normalized * self.weight + self.bias)[:, 0, :])


@pytest.mark.parametrize('bf16', [False, True])
def test_actual_cpu_aot_reference_preserves_outputs_all_gradients_and_releases_aliases(bf16):
    from torch._dynamo.backends.common import aot_autograd
    from functorch.compile import make_boxed_func
    graphs = {}
    def compiler(role):
        def compile_graph(gm, examples):
            graphs[role] = gm
            return make_boxed_func(gm.forward)
        return compile_graph
    torch.manual_seed(174)
    model = TinyNormalized(); reference = copy.deepcopy(model)
    x = torch.randn(3, 4, 12, requires_grad=True); xr = x.detach().clone().requires_grad_()
    active = torch.compile(model, backend=aot_autograd(fw_compiler=compiler('fw'),
                           bw_compiler=compiler('bw')), fullgraph=True, dynamic=False)
    with torch.autocast('cpu', dtype=torch.bfloat16) if bf16 else nullcontext():
        out = active(x); plain = reference(xr)
    assert torch.equal(out, plain)
    # Derive the small fixture's ordered saved role from its ACTUAL native FX graph,
    # independently of shape. This is a CPU fixture, not the pinned CUDA proof.
    nodes = list(graphs['fw'].graph.nodes)
    normalized = [n for n in nodes if str(n.target) == 'aten.mul.Tensor'
                  and all(isinstance(a, torch.fx.Node) for a in n.args)
                  and str(n.args[0].target) == 'aten.sub.Tensor'
                  and str(n.args[1].target) == 'aten.rsqrt.default']
    assert len(normalized) == 1
    role = normalized[0]
    affine = [n for n in role.users if str(n.target) == 'aten.mul.Tensor']
    assert len(affine) == 1 and str(affine[0].args[1].target).startswith('primals_')
    add = next(n for n in affine[0].users if str(n.target) == 'aten.add.Tensor')
    cls = next(n for n in add.users if str(n.target) == 'aten.select.int')
    assert cls.args[1:] == (1, 0)
    head_inputs = [n for n in cls.users]
    if bf16:
        assert len(head_inputs) == 1 and str(head_inputs[0].target) == 'aten._to_copy.default'
        assert head_inputs[0].kwargs['dtype'] == torch.bfloat16
        head_inputs = list(head_inputs[0].users)
    assert any(str(n.target) == 'aten.addmm.default' for n in head_inputs)
    saved_roles = list(nodes[-1].args[0])[1:]
    index = saved_roles.index(role)
    matching = [i for i, n in enumerate(saved_roles) if tuple(n.meta['val'].shape) == (3, 4, 12)
                and n.meta['val'].dtype == torch.float32]
    assert len(matching) == 3
    selected, metadata = _select_saved(out.grad_fn, saved_count=len(saved_roles), saved_index=index,
                                      shape=(3, 4, 12), stride=(48, 12, 1),
                                      same_shape_indices=matching, device='cpu')
    observer = TensorOwnership(); observer.watch(selected, 'normalized.original_reference')
    # No inspection/copy of selected data until full backward.
    out.float().square().sum().backward(); plain.float().square().sum().backward()
    expected = (x.detach() * 1.1 + model.bias.detach())
    var, mean = torch.var_mean(expected, dim=-1, correction=0, keepdim=True)
    expected = (expected - mean) * torch.rsqrt(var + 1e-5)
    assert all(torch.equal(p.grad, q.grad) for p, q in zip(model.parameters(), reference.parameters()))
    assert torch.equal(x.grad, xr.grad) and torch.equal(selected, expected)
    backward_inputs = [n for n in graphs['bw'].graph.nodes if n.op == 'placeholder']
    assert backward_inputs[index].name == role.name
    assert any(str(n.target) == 'aten.mul.Tensor' for n in backward_inputs[index].users)
    cls_copy = selected[:, 0, :].detach().clone()
    assert tuple(cls_copy.shape) == (3, 12) and cls_copy.untyped_storage()._cdata != selected.untyped_storage()._cdata
    alias = selected.detach().view(-1)
    del selected, out
    assert observer.report()['live_unique_storages'] == 1
    del alias
    assert observer.report()['all_registered_released']
    observer.close()
    # Fixture compile callbacks deliberately retain graph objects, never runtime tensors.
    assert metadata['selected_index'] == index
    assert not torch.cuda.is_initialized()
