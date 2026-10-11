"""Normalize prospective comparison records to their persisted JSON representation."""
import json
try:
    from . import fresh420_contract as original
except ImportError:
    import fresh420_contract as original

def canonical(kind,row):
    # Original semantic field selection stays fixed. JSON arrays represent both
    # tuple and list sequences; finite floats round-trip exactly in Python JSON.
    # Unknown object types and nonfinite numbers fail rather than being coerced.
    return json.loads(json.dumps(original.canonical(kind,row),allow_nan=False))
