"""Keep repetition, observer validity and localization as separate results."""
import itertools
import json
from pathlib import Path
import sys

try:
    from .summarize_saved_operand import qualify as old_qualify
    from .normalized_input_replay import normalized_interpretation, RAM_CAP
    from .normalized_diagnostic_contract import validate_config, localize
    from .normalized_input_abi import FORWARD_SHA, BACKWARD_SHA, EXPECTED_PROOF
    from .benchmark_off_contract import finite_records, backend_delta
except ImportError:
    from summarize_saved_operand import qualify as old_qualify
    from normalized_input_replay import normalized_interpretation, RAM_CAP
    from normalized_diagnostic_contract import validate_config, localize
    from normalized_input_abi import FORWARD_SHA, BACKWARD_SHA, EXPECTED_PROOF
    from benchmark_off_contract import finite_records, backend_delta


def _released(report, expected):
    return (report.get('all_registered_released') is True and report.get('tracked_tensors') == expected
            and report.get('tracked_unique_storages') == expected and report.get('live_tensors') == 0
            and report.get('live_unique_storages') == 0 and report.get('live_unique_storage_bytes') == 0)


def normalized_valid(run):
    value = run.get('normalized_input_replay', {})
    captures = value.get('captures', [])
    if value.get('status') != 'completed' or len(captures) != 3 or len(value.get('heads', [])) not in (0, 3):
        return False, None
    a = run['measured'][0]
    actual = dict(output=a['output'], loss=a['loss'], grad=a['grad_before_clip'])
    interp = normalized_interpretation(actual, run['initial_state_replay']['plain'], captures,
                                     value['heads'], run['saved_operand_replay'])
    valid = (value['final_invariants_before'] == value['final_invariants_after']
             and value['held_invariants_before'] == value['held_invariants_after']
             and value['rebuilt_input'] == a['input'] and value['rebuilt_target'] == a['target']
             and value['ram_cap_bytes'] == RAM_CAP and value['maximum_purpose_bytes'] < RAM_CAP
             and _released(value['registered_outer_release'], 3 if value.get('held_operand') is not None else 2)
             and value['interpretation'] == interp and finite_records(captures)
             and finite_records(value['heads']) and bool(value.get('capture_kernel_names_counts'))
             and run.get('own_affinity_restored') is True)
    available = all(c.get('normalized_cls') is not None for c in captures)
    if available:
        abi = value.get('abi', {})
        valid = valid and abi.get('forward_sha256') == FORWARD_SHA and abi.get('backward_sha256') == BACKWARD_SHA
        valid = valid and abi.get('proof') == EXPECTED_PROOF
        for capture in captures:
            select = capture.get('normalized_selection') or {}
            metadata = select.get('saved_metadata', [])
            valid = valid and select.get('node_type') == 'CompiledFunctionBackward' and select.get('selected_index') == 21
            valid = valid and select.get('shape_matches') == [6, 17, 21] and len(metadata) == 30
            valid = valid and select.get('version_checked_count', 0) + select.get('no_version_check_count', 0) == 30
            if len(metadata) == 30:
                item = metadata[21]
                valid = valid and item.get('dtype') == 'torch.float32' and item.get('shape') == [256, 65, 1536]
                valid = valid and item.get('stride') == [99840, 1536, 1] and item.get('storage_offset') == 0
                valid = valid and item.get('device') == 'cuda:0' and metadata[22].get('device') == 'cuda:0'
            record = capture['normalized_cls']
            valid = valid and record.get('shape') == [256, 1536] and record.get('dtype') == 'torch.float32'
            released = capture.get('capture_registered_release', {})
            valid = valid and _released(released, 3)
            valid = valid and capture['purpose_before_backward_bytes'] < RAM_CAP and capture['purpose_after_backward_bytes'] < RAM_CAP
            valid = valid and value['maximum_purpose_bytes'] >= max(capture['purpose_before_backward_bytes'], capture['purpose_after_backward_bytes'])
        valid = valid and value['registered_outer_release']['tracked_tensors'] == 3
        valid = valid and len(value['heads']) == 3 and bool(value.get('head_kernel_names_counts'))
        valid = valid and value.get('held_operand') == captures[0].get('operand')
    else:
        # Completed unavailable capture is retained, but cannot qualify localization.
        valid = valid and all(c.get('selection_failure') for c in captures)
    return bool(valid), interp


def qualify(runs):
    previous = old_qualify(runs)
    if not previous['completed']:
        return dict(completed=False, previous=previous, observer_valid=False, localization_supported=False)
    rows = []
    for run in runs:
        valid, interp = normalized_valid(run)
        rows.append(dict(label=run['context']['label'], valid=valid, interpretation=interp))
    pairs = []
    code_ok = bool(runs[0].get('code_after_normalized_replay')) and all(
        r.get('code_after_normalized_replay') == runs[0].get('code_after_normalized_replay') for r in runs[1:])
    for a, b in itertools.combinations(runs, 2):
        exact = a.get('code_after_normalized_replay') == b.get('code_after_normalized_replay')
        ca = a['normalized_input_replay'].get('captures', [])
        cb = b['normalized_input_replay'].get('captures', [])
        valid = previous['interpretation_supported'] and all(r['valid'] and r['interpretation']['supported'] for r in rows) and code_ok
        left = dict(primary_head_gradient=a['measured'][0]['grad_before_clip']['head.weight'],
                    normalized_cls=ca[0].get('normalized_cls') if ca else None,
                    operand=ca[0].get('operand') if ca else None)
        right = dict(primary_head_gradient=b['measured'][0]['grad_before_clip']['head.weight'],
                     normalized_cls=cb[0].get('normalized_cls') if cb else None,
                     operand=cb[0].get('operand') if cb else None)
        pairs.append(dict(left=a['context']['label'], right=b['context']['label'],
                          new_observer_code_exact=exact, classification=localize(left, right, valid),
                          normalized_cls_exact=left['normalized_cls'] == right['normalized_cls'],
                          classifier_operand_exact=left['operand'] == right['operand'],
                          primary_head_gradient_exact=left['primary_head_gradient'] == right['primary_head_gradient']))
    valid = previous['observer_valid'] and all(r['valid'] for r in rows)
    supported = valid and previous['interpretation_supported'] and code_ok and all(r['interpretation']['supported'] for r in rows)
    return dict(completed=True, previous=previous, primary_exact_repeat=previous['primary']['qualified'],
                observer_valid=valid, new_observer_code_exact=code_ok, localization_supported=supported,
                fresh_variation_observed=any(not p['primary_head_gradient_exact'] for p in pairs),
                contexts=rows, pairs=pairs, scope='Input-path localization only; no named operator, learning or timing conclusion')


def main(bundle):
    config = json.loads((bundle/'config.json').read_text()); validate_config(config)
    runs = [json.loads((bundle/'contexts'/c['label']/'result.json').read_text())
            for c in config['execution_diagnostic']['contexts'] if (bundle/'contexts'/c['label']/'result.json').exists()]
    result = qualify(runs)
    m = json.loads((bundle/'manifest.json').read_text())
    prior = next((Path(m['baseline_bundle'])/'contexts').glob('*/result.json'))
    baseline = json.loads(prior.read_text())
    delta = [dict(label=r['context']['label'], phase=phase, fields=backend_delta(baseline[phase], r[phase]))
             for r in runs for phase in ('backend_before_smoke', 'backend_after_measured') if phase in r]
    backend_ok = len(delta) == 8 and all(not d['fields'] for d in delta)
    result.update(baseline_backend_exact=backend_ok, baseline_backend_deltas=delta)
    result['localization_supported'] = result['localization_supported'] and backend_ok
    if not backend_ok:
        for pair in result.get('pairs', []):
            pair['classification'] = 'INCONCLUSIVE_PARITY_ABI_OR_RESOURCE_GATE'
    with (bundle/'summary.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    value = main(Path(sys.argv[1]))
    print(json.dumps({k: v for k, v in value.items() if k not in ('previous', 'contexts', 'pairs', 'baseline_backend_deltas')}, indent=2))
