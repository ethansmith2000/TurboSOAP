"""CPU-only source and retained-metadata audit; no old graph is executed."""
import argparse
import hashlib
import json
from pathlib import Path

try:
    from .normalized_input_abi import normalized_abi, unique_loaded_rows, FORWARD_SHA, BACKWARD_SHA
except ImportError:
    from normalized_input_abi import normalized_abi, unique_loaded_rows, FORWARD_SHA, BACKWARD_SHA


def audit(evidence):
    plan = json.loads((evidence / 'soap_fresh420_json_round_20261008/upstream_abi_plan.json').read_text())
    records = []; inventory_records = []
    for label in ('qr_0', 'fro_1', 'qr_2'):
        result = json.loads((evidence / 'cifar_soap_fresh420_json_20261008/contexts' / label / 'result.json').read_text())
        rows = []
        # Verify retained inventory identity against archived bytes, rather than
        # depending on a mutable /tmp compiler cache for a CPU historical audit.
        archived = []
        for row in result['code_after_measured']:
            copy = dict(row)
            copy['path'] = str(evidence / 'soap_fresh420_json_round_20261008/compiled_code'
                               / (row['source_sha256'] + '.py'))
            archived.append(copy)
        unique = unique_loaded_rows(archived)
        inventory_records.append(dict(label=label, inventory_rows=len(archived), unique_rows=len(unique),
                                      identical_duplicate_rows=len(archived)-len(unique)))
        for h in (FORWARD_SHA, BACKWARD_SHA):
            path = evidence / 'soap_fresh420_json_round_20261008/compiled_code' / (h + '.py')
            assert sum(r['source_sha256'] == h for r in unique) == 1
            rows.append(dict(path=str(path), source_sha256=h))
        abi = normalized_abi(rows)
        assert abi.record()['proof']['saved_index'] == plan['saved_index'] == 21
        assert abi.forward_sha256 == plan['forward_sha256'] and abi.backward_sha256 == plan['backward_sha256']
        captures = result['saved_operand_replay']['captures']
        assert len(captures) == 3
        for index, capture in enumerate(captures):
            selection = capture['selection']; metadata = selection['saved_metadata']
            assert len(metadata) == 30
            matches = [r['index'] for r in metadata if r['dtype'] == 'torch.float32'
                       and r['shape'] == [256, 65, 1536] and r['stride'] == [99840, 1536, 1]
                       and r['storage_offset'] == 0]
            assert matches == [6, 17, 21]
            assert metadata[21]['index'] == 21 and metadata[22]['dtype'] == 'torch.bfloat16'
            assert selection['selected_index'] == 22
            records.append(dict(label=label, capture=index, normalized_index=21,
                                shape_matches=matches, retained_metadata_only=True))
    return dict(status='PASS_STATIC_AND_RETAINED_METADATA', abi=abi.record(), runtime_metadata=records,
                loaded_inventory=inventory_records,
                optimizer_steps=0, training_steps=0, gpu_executed=False, newly_captured_tensors=0,
                scope='CPU source proof and nine prior metadata records; new CUDA runtime selection remains unqualified',
                plan_sha256=hashlib.sha256((evidence / 'soap_fresh420_json_round_20261008/upstream_abi_plan.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=Path('/workspace/optimizer_replay_results'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.evidence)
    with args.output.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(value['status'])
