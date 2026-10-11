"""Post-trajectory native replay with registered normalized-input lifetimes."""
from contextlib import nullcontext
import time
import torch
from torch.nn import functional as F

try:
    from .initial_state_replay import initial_state_scope
    from .saved_operand_replay import HeldHead, head_execution, interpretation
    from .saved_operand_abi import classifier_abi, select_operand, OperandUnavailable
    from .normalized_input_abi import normalized_abi, select_normalized
    from .tensor_ownership import TensorOwnership
    from .execution_fingerprints import tensor_record, gradient_records, first_difference, loaded_code
    from .benchmark_off_contract import finite_records
except ImportError:
    from initial_state_replay import initial_state_scope
    from saved_operand_replay import HeldHead, head_execution, interpretation
    from saved_operand_abi import classifier_abi, select_operand, OperandUnavailable
    from normalized_input_abi import normalized_abi, select_normalized
    from tensor_ownership import TensorOwnership
    from execution_fingerprints import tensor_record, gradient_records, first_difference, loaded_code
    from benchmark_off_contract import finite_records


RAM_CAP = 256 * 2**20


def purpose_bytes(snapshot_bytes, x, target, held=None, normalized=None, operand=None, cls_copy=None):
    # Native saved references have no copy cost but extend full storage lifetime.
    # Snapshot state is owned by the unchanged initial_state_scope dependency.
    storages = {}
    for tensor in (x, target, held, normalized, operand, cls_copy):
        if tensor is not None:
            storage = tensor.untyped_storage()
            storages[storage._cdata] = storage.nbytes()
    value = snapshot_bytes + sum(storages.values())
    if value >= RAM_CAP:
        raise MemoryError('New normalized replay purpose RAM cap exceeded')
    return value


def capture_view(capture):
    return {key: capture[key] for key in ('execution', 'operand', 'selection', 'selection_failure')}


def normalized_interpretation(actual, warm, captures, heads, previous):
    legacy = interpretation(actual, warm, [capture_view(c) for c in captures], heads)
    checks = dict(legacy['checks'])
    checks.update(normalized_available=len(captures) == 3 and all(c.get('normalized_cls') is not None for c in captures),
                  registered_capture_release=len(captures) == 3 and all(c.get('capture_registered_release', {}).get('all_registered_released') is True
                      and c['capture_registered_release'].get('tracked_tensors') == 3 for c in captures),
                  normalized_repeated_exact=len(captures) == 3 and all(captures[0].get('normalized_cls') == c.get('normalized_cls') for c in captures[1:]),
                  earlier_original_replay_exact=previous.get('status') == 'completed'
                      and len(previous.get('captures', [])) == 3
                      and len(captures) == 3
                      and all(first_difference(c['execution'], p['execution']) is None
                              and c['operand'] == p['operand'] for c, p in zip(captures, previous.get('captures', []))))
    return dict(supported=all(checks.values()), checks=checks, original=legacy)


def run_normalized_replay(*, model, active, initial, initial_rng, get_rng, restore_rng, make_batch,
                          first_record, warm_record, invariants, progress, previous, ram_cap_bytes):
    if ram_cap_bytes != RAM_CAP:
        raise ValueError('Require the separately declared new256MiB purpose cap')
    started = time.monotonic()
    result = dict(status='running', captures=[], heads=[], maximum_purpose_bytes=0,
                  ram_cap_bytes=ram_cap_bytes, optimizer_steps=0,
                  ownership_scope='Registered normalized/operand/CLS references plus replay input/target/held operand; '
                                  'snapshot byte accounting/restoration uses unchanged initial_state_scope; not whole allocator ownership')
    outer_owner = TensorOwnership()
    try:
        with initial_state_scope(model, initial, initial_rng, get_rng, restore_rng, invariants) as scope:
            x, target = make_batch()
            outer_owner.watch_tree(dict(input=x, target=target), 'normalized_replay')
            assert tensor_record(x) == first_record['input'] and tensor_record(target) == first_record['target']
            restore_rng(scope['current_rng'])
            held_before = invariants()
            result.update(rebuilt_input=tensor_record(x), rebuilt_target=tensor_record(target),
                          held_invariants_before=held_before, final_invariants_before=scope['before'],
                          snapshot_bytes=scope['snapshot_bytes'])
            held = None
            try:
                rows = loaded_code()
                norm_abi = normalized_abi(rows)
                operand_abi = classifier_abi(rows, x.shape[0], model.embed_dim, model.num_classes)
                result.update(abi=norm_abi.record(), operand_abi=operand_abi)
            except (OperandUnavailable, OSError) as error:
                norm_abi = operand_abi = None
                result['abi_failure'] = str(error)

            def capture_call(profile=False, retain=False):
                model.zero_grad(set_to_none=True)
                owner = TensorOwnership()
                normalized = operand = cls_copy = None
                normal_selection = selection = None
                failure = result.get('abi_failure')
                window = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                         torch.profiler.ProfilerActivity.CUDA]) if profile else nullcontext()
                try:
                    with window as prof:
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            logits = active(x)
                            loss = F.cross_entropy(logits.float(), target)
                        # NORMALIZED_REFERENCE_START
                        # Metadata/references/native weak handles ONLY before backward.
                        if norm_abi is not None:
                            try:
                                normalized, normal_selection = select_normalized(logits.grad_fn, norm_abi)
                                owner.watch(normalized, 'native.normalized')
                                operand, selection = select_operand(logits.grad_fn, operand_abi)
                                owner.watch(operand, 'native.classifier_operand')
                            except OperandUnavailable as error:
                                failure = str(error)
                        retained = purpose_bytes(scope['snapshot_bytes'], x, target, held, normalized, operand)
                        result['maximum_purpose_bytes'] = max(result['maximum_purpose_bytes'], retained)
                        # NORMALIZED_REFERENCE_END
                        loss.backward()
                        torch.cuda.synchronize()
                    # Data reads/copies are after complete backward and synchronization.
                    if normalized is not None:
                        cls_copy = normalized[:, 0, :].detach().clone()
                        owner.watch(cls_copy, 'native.normalized_cls_copy_after_backward')
                    retained_after = purpose_bytes(scope['snapshot_bytes'], x, target, held,
                                                   normalized, operand, cls_copy)
                    result['maximum_purpose_bytes'] = max(result['maximum_purpose_bytes'], retained_after)
                    record = dict(execution=dict(output=tensor_record(logits), loss=float(loss.detach()),
                                                grad=gradient_records(model)),
                                  operand=tensor_record(operand) if operand is not None else None,
                                  selection=selection, selection_failure=failure,
                                  normalized_cls=tensor_record(cls_copy) if cls_copy is not None else None,
                                  normalized_selection=normal_selection,
                                  purpose_before_backward_bytes=retained, purpose_after_backward_bytes=retained_after,
                                  allocator_after_backward_bytes=torch.cuda.memory_allocated())
                    kept = operand.detach().clone() if retain and operand is not None else None
                    purpose_bytes(scope['snapshot_bytes'], x, target, held, normalized, operand, cls_copy)
                    if kept is not None:
                        # Transient first held clone coexists with originals; account it too.
                        peak = retained_after + kept.untyped_storage().nbytes()
                        assert peak < ram_cap_bytes
                        result['maximum_purpose_bytes'] = max(result['maximum_purpose_bytes'], peak)
                    kernels = {}
                    if profile:
                        for event in prof.events():
                            if str(event.device_type) == 'DeviceType.CUDA':
                                kernels[event.name] = kernels.get(event.name, 0) + 1
                        assert kernels, 'Missing normalized original replay CUDA provenance'
                    del logits, loss, normalized, operand, cls_copy
                    record['capture_registered_release'] = owner.report() if normal_selection is not None else dict(
                        all_registered_released=False, reason='No verified normalized registration')
                    if normal_selection is not None:
                        assert record['capture_registered_release']['all_registered_released'], 'Native capture reference/storage retained'
                    assert finite_records(record), 'Nonfinite normalized replay'
                    return record, kept, kernels
                finally:
                    owner.close()

            first, held, _ = capture_call(retain=True)
            result['captures'].append(first)
            if held is not None:
                outer_owner.watch(held, 'normalized_replay.held_classifier_operand')
            result['held_operand'] = tensor_record(held) if held is not None else None
            progress(result)
            for profile in (False, True):
                record, _, kernels = capture_call(profile)
                result['captures'].append(record)
                if profile:
                    result['capture_kernel_names_counts'] = kernels
                progress(result)
            if held is not None:
                # Same existing HeldHead math; only first BF16 operand held afterward.
                head = torch.compile(HeldHead(model.head), mode='default', fullgraph=True, dynamic=False)
                for profile in (False, False, True):
                    model.zero_grad(set_to_none=True)
                    window = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                             torch.profiler.ProfilerActivity.CUDA]) if profile else nullcontext()
                    with window as prof:
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            logits = head(held)
                            loss = F.cross_entropy(logits.float(), target)
                        loss.backward()
                        torch.cuda.synchronize()
                    value = head_execution(dict(output=tensor_record(logits), loss=float(loss.detach()),
                                                grad=gradient_records(model)))
                    assert finite_records(value)
                    result['heads'].append(value)
                    if profile:
                        kernels = {}
                        for event in prof.events():
                            if str(event.device_type) == 'DeviceType.CUDA':
                                kernels[event.name] = kernels.get(event.name, 0) + 1
                        assert kernels
                        result['head_kernel_names_counts'] = kernels
                    progress(result)
                assert tensor_record(held) == result['held_operand'], 'Held operand mutated'
                del head, logits, loss
            actual = dict(output=first_record['output'], loss=first_record['loss'], grad=first_record['grad_before_clip'])
            result['interpretation'] = normalized_interpretation(actual, warm_record, result['captures'], result['heads'], previous)
            result['held_invariants_after'] = invariants()
            assert result['held_invariants_after'] == held_before
            del x, target, held
            result['registered_outer_release'] = outer_owner.report()
            assert result['registered_outer_release']['all_registered_released'], 'Replay input/held storage retained'
        result.update(status='completed', final_invariants_after=invariants(), seconds=time.monotonic()-started)
        assert result['final_invariants_after'] == result['final_invariants_before']
        progress(result)
        return result
    finally:
        outer_owner.close()
