"""Opt-in, literal May SOAP; never imported by the shared trainer.

The loader executes verified historical bytes in an isolated module. It supplies
no optimizer defaults and makes no mathematical or state-handling corrections.
Policy inspection is offline; keep it outside any future timing span.
"""
import hashlib
import inspect
import json
from pathlib import Path
from types import ModuleType


COMMIT = '930a5a3a31a4bf74196eed358190c8f0bc71c454'
SOURCE_SHA256 = 'fff531c00b321531d07121f60cd25d4c7b46b9033d8617c5d03e11b28ac93b5d'
SOURCE = Path(__file__).parent/'reference/soap_may_930a5a3.py'


def load_legacy_class(source=SOURCE):
    """Reject modified source, then execute exactly the bytes that were hashed."""
    path = Path(source)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError(f'Historical SOAP source hash mismatch: {actual}')
    module = ModuleType('_turbosoap_may_930a5a3')
    module.__file__ = str(path.resolve())
    exec(compile(payload, module.__file__, 'exec'), module.__dict__)
    return module.SOAP


def build_legacy_optimizer(params, **kwargs):
    """Pass arguments straight to the original constructor, including groups."""
    return load_legacy_class()(params, **kwargs)


def effective_policy(optimizer):
    """Expose actual group overrides and parameter coverage without tensor data."""
    implementation = Path(optimizer.__class__.__init__.__code__.co_filename)
    if hashlib.sha256(implementation.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Policy provenance requires the literal historical class')
    groups = []
    for group in optimizer.param_groups:
        settings = {key:value for key,value in group.items() if key != 'params'}
        # Match the persisted representation and reject nonfinite/non-JSON
        # metadata instead of silently omitting an effective override.
        settings = json.loads(json.dumps(settings, allow_nan=False))
        groups.append(dict(settings=settings, parameters=[
            dict(index=i, shape=list(p.shape), dtype=str(p.dtype),
                 device=str(p.device), requires_grad=p.requires_grad)
            for i,p in enumerate(group['params'])]))
    defaults = {name:p.default for name,p in
                inspect.signature(optimizer.__class__.__init__).parameters.items()
                if p.default is not inspect.Parameter.empty}
    return json.loads(json.dumps(dict(commit=COMMIT, source_sha256=SOURCE_SHA256,
                                     constructor_defaults=defaults, groups=groups),
                                allow_nan=False))
