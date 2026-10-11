"""Prospective CPU-tested JSON-wire backend comparison; no old gate rewrite.

Live torch.__version__ is TorchVersion, a str subclass. Compare runtime metadata
in the representation written by json_write, retaining all fields and values.
This module has not been used by a GPU campaign.
"""
import ast
import json

try:
    from . import launch_environment_contract as original
except ImportError:
    import launch_environment_contract as original


def canonical_backend(record):
    def validate_keys(value):
        if isinstance(value, dict):
            if any(not isinstance(k, str) for k in value):
                raise TypeError('Backend JSON object keys must be strings')
            for item in value.values():
                validate_keys(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                validate_keys(item)
    validate_keys(record)
    return json.loads(json.dumps(record, allow_nan=False))


def check_backend(contract, device, record, phase):
    return original.check_backend(contract, device, canonical_backend(record), phase)


def verify_backend_launch(bundle, record, phase):
    contract, launch = original.read_launch(bundle)
    report = check_backend(contract, launch['device_binding'], record, phase)
    if not report['passed']:
        raise ValueError('Historical backend mismatch: '+json.dumps(report, sort_keys=True))
    return report


def validate_child_change(before, after):
    old = 'from launch_environment_contract import verify_environment_launch, verify_backend_launch\n'
    new = ('from launch_environment_contract import verify_environment_launch\n'
           'from launch_environment_wire_contract import verify_backend_launch\n')
    if before.count(old) != 1 or after.count(new) != 1:
        raise ValueError('One backend checker import substitution required')
    if ast.dump(ast.parse(after.replace(new, old))) != ast.dump(ast.parse(before)):
        raise ValueError('Child changed beyond backend comparison import')
    return True


def validate_outer_change(before, after):
    expected = before.replace('cifar_normalized_env_child.py', 'cifar_normalized_env_wire_child.py')
    if ast.dump(ast.parse(expected)) != ast.dump(ast.parse(after)):
        raise ValueError('Outer changed beyond child entry')
    return True


def validate_summary_change(before, after):
    old = 'from .launch_environment_contract import PHASES, check_backend, read_launch'
    new = ('from .launch_environment_contract import PHASES, read_launch\n'
           '    from .launch_environment_wire_contract import check_backend')
    old_direct = 'from launch_environment_contract import PHASES, check_backend, read_launch'
    new_direct = ('from launch_environment_contract import PHASES, read_launch\n'
                  '    from launch_environment_wire_contract import check_backend')
    if any(after.count(line) != 1 for line in (new, new_direct)):
        raise ValueError('Only backend checker summary imports may change')
    restored = after.replace(new, old).replace(new_direct, old_direct)
    if ast.dump(ast.parse(restored)) != ast.dump(ast.parse(before)):
        raise ValueError('Summary changed beyond backend comparator import')
    return True
