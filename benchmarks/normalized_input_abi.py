"""Select the existing pre-affine saved tensor from a pinned native graph.

Only source parsing and tensor metadata occur before backward. This module
does not change a graph, run kernels, copy tensor data, or own replay state.
"""
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

try:
    from .saved_operand_abi import OperandUnavailable, classifier_abi
except ImportError:
    from saved_operand_abi import OperandUnavailable, classifier_abi


FORWARD_SHA = 'b606a746c3bba1b875a186ba582ec3ae463640eb097f6cd1671e4c551f1e66cc'
BACKWARD_SHA = 'e8105f7122c5762a23f1143e4c12da4c9621d28c553391bb14a5dea09e5d2bbe'
PRODUCER = 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_10'
CONSUMER = 'triton_poi_fused__to_copy_native_layer_norm_select_11'
BACKWARD_USE = 'triton_per_fused__to_copy_native_layer_norm_backward_select_backward_5'
EXPECTED_PROOF = dict(
    producer=PRODUCER + '.run(buf3, primals_5, buf19, buf31, buf35, buf40, 16640, 1536, stream=stream0)',
    consumer=CONSUMER + '.run(buf35, primals_17, primals_18, buf36, 393216, stream=stream0)',
    backward_use=BACKWARD_USE + '.run(buf0, primals_17, mul_7, div, buf5, buf7, workspace_0, 16640, 1536, stream=stream0)',
    producer_output_argument=4, producer_store_value='tmp31', normalized_name='buf35',
    backward_name='mul_7', saved_index=21, saved_count=30, classifier_index=22)


def _require(condition, message):
    if not condition:
        raise OperandUnavailable(message)


def _one(items, role):
    items = list(items)
    _require(len(items) == 1, f'Expected one {role}, found {len(items)}')
    return items[0]


def _named(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _call_function(data):
    tree = ast.parse(data)
    fn = _one((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == 'call'), 'compiled call')
    return tree, fn


def _run(fn, name):
    return _one((n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == name + '.run'), name + ' launch')


def _allocation(fn, name, shape, stride, dtype):
    alloc = _one((n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                  and len(n.targets) == 1 and _named(n.targets[0], name)), name + ' allocation')
    value = alloc.value
    _require(isinstance(value, ast.Call) and ast.unparse(value.func) == 'empty_strided_cuda'
             and len(value.args) == 3, 'Expected native allocation for ' + name)
    _require(ast.literal_eval(value.args[0]) == shape
             and ast.literal_eval(value.args[1]) == stride
             and ast.unparse(value.args[2]) == dtype, 'Allocation layout changed: ' + name)
    return alloc


def _kernel(tree, name):
    assignment = _one((n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and len(n.targets) == 1 and _named(n.targets[0], name)), name + ' definition')
    call = assignment.value
    _require(isinstance(call, ast.Call) and ast.unparse(call.func) == 'async_compile.triton'
             and len(call.args) >= 2 and isinstance(call.args[1], ast.Constant)
             and isinstance(call.args[1].value, str), 'Missing embedded kernel source')
    kt = ast.parse(call.args[1].value)
    return _one((n for n in ast.walk(kt) if isinstance(n, ast.FunctionDef)
                 and n.name == name), 'embedded kernel function')


def inspect_sources(forward, backward):
    """Structural proof; the public factory additionally pins both full hashes."""
    ft, ff = _call_function(forward)
    _, bf = _call_function(backward)
    ret = _one((n for n in ast.walk(ff) if isinstance(n, ast.Return)), 'forward return')
    _require(isinstance(ret.value, ast.Tuple) and len(ret.value.elts) == 31,
             'Forward saved count changed')
    items = ret.value.elts
    _require(_named(items[0], 'buf39'), 'Classifier output changed')
    indexes = [i - 1 for i, item in enumerate(items) if _named(item, 'buf35')]
    _require(indexes == [21] and _named(items[23], 'buf36'), 'Normalized/classifier saved slot changed')
    alloc = _allocation(ff, 'buf35', (256, 65, 1536), (99840, 1536, 1), 'torch.float32')
    _allocation(ff, 'buf36', (256, 1536), (1536, 1), 'torch.bfloat16')
    producer, consumer = _run(ff, PRODUCER), _run(ff, CONSUMER)
    _require(ast.unparse(producer) == PRODUCER + '.run(buf3, primals_5, buf19, buf31, buf35, buf40, 16640, 1536, stream=stream0)',
             'Normalization producer/output role changed')
    _require(ast.unparse(consumer) == CONSUMER + '.run(buf35, primals_17, primals_18, buf36, 393216, stream=stream0)',
             'Normalized affine/CLS consumer role changed')
    _require(alloc.lineno < producer.lineno < consumer.lineno < ret.lineno,
             'Normalization producer/consumer order changed')
    kernel = _kernel(ft, PRODUCER)
    _require(kernel.args.args[4].arg == 'out_ptr2', 'Producer pointer binding changed')
    stores = [n for n in ast.walk(kernel) if isinstance(n, ast.Call)
              and ast.unparse(n.func) == 'tl.store' and len(n.args) >= 2
              and any(_named(x, 'out_ptr2') for x in ast.walk(n.args[0]))]
    store = _one(stores, 'normalized output store')
    _require(_named(store.args[1], 'tmp31'), 'Producer normalized value changed')
    definitions = {n.targets[0].id: ast.unparse(n.value) for n in ast.walk(kernel)
                   if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name)}
    _require(all(definitions.get(k) == v for k, v in {
        'tmp25': 'tmp24 - tmp10', 'tmp31': 'tmp25 * tmp30',
        'tmp30': 'libdevice.rsqrt(tmp29)', 'tmp29': 'tmp27 + tmp28',
        'tmp27': 'tmp14 / tmp26'}.items()), 'Producer normalization arithmetic changed')
    _kernel(ft, CONSUMER)
    unpack = _one((n for n in ast.walk(bf) if isinstance(n, ast.Assign)
                   and _named(n.value, 'args') and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Tuple)), 'backward unpack')
    _require(len(unpack.targets[0].elts) == 31 and _named(unpack.targets[0].elts[21], 'mul_7')
             and _named(unpack.targets[0].elts[22], 'convert_element_type_27'),
             'Backward normalized/classifier slot changed')
    stride = _one((n for n in ast.walk(bf) if isinstance(n, ast.Call)
                   and ast.unparse(n.func) == 'assert_size_stride'
                   and len(n.args) == 3 and _named(n.args[0], 'mul_7')), 'backward normalized layout')
    _require(ast.literal_eval(stride.args[1]) == (256, 65, 1536)
             and ast.literal_eval(stride.args[2]) == (99840, 1536, 1), 'Backward normalized layout changed')
    use = _run(bf, BACKWARD_USE)
    _require(ast.unparse(use) == BACKWARD_USE + '.run(buf0, primals_17, mul_7, div, buf5, buf7, workspace_0, 16640, 1536, stream=stream0)',
             'Backward normalized input use changed')
    return dict(producer=ast.unparse(producer), consumer=ast.unparse(consumer),
                backward_use=ast.unparse(use), producer_output_argument=4,
                producer_store_value='tmp31', normalized_name='buf35', backward_name='mul_7',
                saved_index=21, saved_count=30, classifier_index=22)


@dataclass(frozen=True)
class NormalizedABI:
    forward_sha256: str
    backward_sha256: str
    proof_json: str

    def record(self):
        return dict(**asdict(self), proof=json.loads(self.proof_json),
                    shape=[256, 65, 1536], stride=[99840, 1536, 1], dtype='torch.float32',
                    storage_offset=0, same_shape_indices=[6, 17, 21],
                    scope='Pinned native CUDA graph; runtime ordered saved-tensor ABI')


def unique_loaded_rows(rows):
    """Collapse only completely identical inventory entries, never distinct code."""
    unique = []
    for row in rows:
        data = Path(row['path']).read_bytes()
        _require(hashlib.sha256(data).hexdigest() == row['source_sha256'], 'Loaded source hash changed')
        if row in unique:
            # Loaded-code inventories can enumerate the same module twice.
            # Byte/hash AND the entire retained row have already been checked.
            continue
        unique.append(row)
    return unique


def normalized_abi(rows):
    """Reject source ambiguity and verify the original graph without importing it."""
    rows = unique_loaded_rows(rows)
    sources = {}
    for row in rows:
        data = Path(row['path']).read_bytes()
        _require(hashlib.sha256(data).hexdigest() == row['source_sha256'], 'Loaded source hash changed')
        if row['source_sha256'] in (FORWARD_SHA, BACKWARD_SHA):
            _require(row['source_sha256'] not in sources, 'Duplicate required compiled source')
            sources[row['source_sha256']] = data.decode()
    _require(set(sources) == {FORWARD_SHA, BACKWARD_SHA}, 'Missing pinned forward/backward source')
    proof = inspect_sources(sources[FORWARD_SHA], sources[BACKWARD_SHA])
    _require(proof == EXPECTED_PROOF, 'Normalized source proof changed')
    classifier = classifier_abi(rows, 256, 1536, 100)
    _require(classifier['forward_sha256'] == FORWARD_SHA and classifier['backward_sha256'] == BACKWARD_SHA
             and classifier['saved_index'] == 22 and classifier['saved_count'] == 30,
             'Classifier ABI disagrees with normalized path')
    return NormalizedABI(FORWARD_SHA, BACKWARD_SHA, json.dumps(proof, sort_keys=True))


def _select_saved(node, *, saved_count, saved_index, shape, stride, same_shape_indices, device,
                  classifier_index=None, classifier_shape=None, classifier_dtype=torch.bfloat16):
    """Common metadata-only runtime checks, also exercised on a native CPU AOT fixture."""
    _require(node is not None and type(node).__name__ == 'CompiledFunctionBackward',
             'Output lacks direct CompiledFunctionBackward')
    try:
        checked = list(node.saved_tensors)
        unchecked = list(getattr(node, '_tensors_no_vc_check', []))
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise OperandUnavailable('Saved tensor access unavailable') from exc
    tensors = checked + unchecked
    _require(len(tensors) == saved_count, 'Saved tensor count disagrees with verified ABI')
    _require(all(torch.is_tensor(t) and t.layout == torch.strided for t in tensors),
             'Saved ABI contains a non-strided tensor')
    metadata = [dict(index=i, channel='version_checked' if i < len(checked) else 'no_version_check',
                     dtype=str(t.dtype), shape=list(t.shape), stride=list(t.stride()),
                     storage_offset=t.storage_offset(), device=str(t.device)) for i, t in enumerate(tensors)]
    matches = [i for i, t in enumerate(tensors) if t.dtype == torch.float32
               and tuple(t.shape) == tuple(shape) and tuple(t.stride()) == tuple(stride)
               and t.storage_offset() == 0]
    _require(matches == list(same_shape_indices), 'Equal-shaped saved roles disagree with verified ABI')
    _require(saved_index in matches, 'Normalized slot absent from verified saved roles')
    selected = tensors[saved_index]
    _require(selected.device == torch.device(device), 'Normalized device mismatch')
    storage = selected.untyped_storage()
    _require(storage.nbytes() == selected.numel() * selected.element_size(),
             'Normalized storage extent changed')
    _require(all(t.untyped_storage()._cdata != storage._cdata
                 for i, t in enumerate(tensors) if i != saved_index), 'Normalized storage aliases another saved role')
    if classifier_index is not None:
        operand = tensors[classifier_index]
        _require(operand.dtype == classifier_dtype and tuple(operand.shape) == tuple(classifier_shape)
                 and tuple(operand.stride()) == (classifier_shape[-1], 1)
                 and operand.storage_offset() == 0 and operand.device == selected.device,
                 'Classifier neighbor layout/device changed')
    return selected, dict(node_type=type(node).__name__, selected_index=saved_index,
                          selection_basis='verified source/return/unpack role; not shape uniqueness',
                          shape_matches=matches, version_checked_count=len(checked),
                          no_version_check_count=len(unchecked), saved_metadata=metadata)


def select_normalized(node, abi, device='cuda:0'):
    _require(isinstance(abi, NormalizedABI) and abi.forward_sha256 == FORWARD_SHA
             and abi.backward_sha256 == BACKWARD_SHA, 'Require pinned normalized ABI proof')
    proof = json.loads(abi.proof_json)
    _require(proof == EXPECTED_PROOF,
             'Normalized ABI proof role changed')
    _require(torch.device(device).type == 'cuda', 'Native normalized ABI requires CUDA device')
    return _select_saved(node, saved_count=30, saved_index=21, shape=(256, 65, 1536),
                         stride=(99840, 1536, 1), same_shape_indices=(6, 17, 21), device=device,
                         classifier_index=22, classifier_shape=(256, 1536))
