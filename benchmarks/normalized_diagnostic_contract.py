"""Prospective source, recipe, launch and localization contracts."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path


BASE = Path('/workspace/optimizer_replay_results/cifar_soap_saved_operand_20261008')
NORMALIZED_POLICY = dict(ram_cap_bytes=256*2**20, transient_full_reference_bytes=102236160,
                         prior_replay_reference_bytes=118675856, cls_copy_bytes=1572864,
                         tensor_files=False, retries=0)


def validate_config(config, baseline=None):
    baseline = json.loads((BASE/'config.json').read_text()) if baseline is None else baseline
    value = copy.deepcopy(config)
    d = value['execution_diagnostic']
    if value['max_wall_seconds'] != 900 or d['per_process_seconds'] != 210:
        raise ValueError('New bounds must remain900/210 seconds')
    if d.pop('normalized_input', None) != NORMALIZED_POLICY:
        raise ValueError('Normalized purpose/retention policy changed')
    labels = [x['label'] for x in d['contexts']]
    if len(labels) != 4 or len(set(labels)) != 4:
        raise ValueError('Four distinct fresh arms required')
    for context, old in zip(d['contexts'], baseline['execution_diagnostic']['contexts']):
        context['label'] = old['label']
    value['max_wall_seconds'] = baseline['max_wall_seconds']
    d['per_process_seconds'] = baseline['execution_diagnostic']['per_process_seconds']
    if value != baseline:
        raise ValueError('Primary recipe/steps/observations/allocation/backend protocol changed')
    return True


def _strip_region(source, name):
    start = '# ' + name + '_START'; end = '# ' + name + '_END'
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('One declared source region required: ' + name)
    a, b = source.index(start), source.index(end)
    if a >= b:
        raise ValueError('Source region order changed')
    return source[:a] + source[b + len(end):]


def validate_child_change(baseline, control):
    start = control.index('# NORMALIZED_REPLAY_START')
    if not control.index('# SAVED_OPERAND_REPLAY_END') < start < control.index("report['peak_allocated_bytes']"):
        raise ValueError('Normalized observer must follow earlier complete replays')
    stripped = control
    for name in ('NORMALIZED_LAUNCH', 'NORMALIZED_REPLAY', 'NORMALIZED_FINALIZE'):
        stripped = _strip_region(stripped, name)
    for line in ('from normalized_input_replay import run_normalized_replay\n',
                 'from normalized_diagnostic_contract import verify_launch\n'):
        if stripped.count(line) != 1:
            raise ValueError('Expected additive import exactly once')
        stripped = stripped.replace(line, '')
    if ast.dump(ast.parse(stripped)) != ast.dump(ast.parse(baseline)):
        raise ValueError('Original primary/preceding replay source changed')
    return True


def validate_outer_change(baseline, control):
    expected = baseline.replace('cifar_saved_operand_child.py', 'cifar_normalized_child.py')
    stripped = control.replace('from normalized_diagnostic_contract import verify_launch\n', '')
    stripped = _strip_region(stripped, 'NORMALIZED_OUTER_LAUNCH')
    if ast.dump(ast.parse(stripped)) != ast.dump(ast.parse(expected)):
        raise ValueError('Sequential outer behavior changed beyond entry/launch verification')
    return True


def validate_reference_region(source):
    start = source.index('# NORMALIZED_REFERENCE_START'); end = source.index('# NORMALIZED_REFERENCE_END')
    before_backward = source[start:end]
    # Strict list of metadata-only calls allowed in the new pre-backward region.
    allowed = {'select_normalized', 'select_operand', 'owner.watch', 'purpose_bytes', 'max', 'str'}
    tree = ast.parse(source)
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    region_calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and start <= offsets[n.lineno-1] < end]
    if set(region_calls) - allowed:
        raise ValueError('Data/math call before backward: ' + str(set(region_calls)-allowed))
    if any(word in before_backward for word in ('.clone(', 'tensor_record(', '.backward(', '.cpu(', 'retain_graph', 'register_hook')):
        raise ValueError('Pre-backward observer may hold references/metadata only')
    return True


def verify_launch(bundle):
    bundle = Path(bundle)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((bundle/'manifest.json').read_text())
    launch = json.loads((bundle/'launch_manifest.json').read_text())
    if launch['manifest_sha256'] != sha(bundle/'manifest.json'):
        raise ValueError('Launch not bound to frozen manifest')
    if launch['config_sha256'] != manifest['config_sha256'] or manifest['config_sha256'] != sha(bundle/'config.json'):
        raise ValueError('Frozen configuration changed')
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != launch['gpu']['uuid'] or
            os.environ.get('GPU_CLAIM_INDICES') != str(launch['gpu']['index'])):
        raise ValueError('Exclusive claimed device does not match launch manifest')
    return launch


def localize(left, right, valid):
    if not valid:
        return 'INCONCLUSIVE_PARITY_ABI_OR_RESOURCE_GATE'
    if left['primary_head_gradient'] == right['primary_head_gradient']:
        return 'INCONCLUSIVE_NO_FRESH_HEAD_GRADIENT_VARIATION'
    if left['normalized_cls'] != right['normalized_cls']:
        return 'VARIATION_UPSTREAM_OF_FINAL_AFFINE_CAST'
    if left['operand'] != right['operand']:
        return 'VARIATION_IN_FINAL_AFFINE_CAST_OR_LIFETIME_PATH'
    return 'INCONCLUSIVE_INPUTS_EQUAL_WITH_HEAD_GRADIENT_VARIATION'
