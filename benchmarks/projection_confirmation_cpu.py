"""Fresh full-shape CPU initial-state references, without CUDA or training."""
import argparse
import gc
import json
from pathlib import Path
import random
import resource
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE/'cifar_support')]
import numpy as np
import torch
from llm_feasibility_runtime import build_model
from projection_confirmation_contract import recipe, SEEDS, SNAPSHOT_CAP, SNAPSHOT_BYTES, SNAPSHOT_PURPOSE, PARAMETERS, DIMENSIONS
from cifar_optimizers import grouped_parameters
from matched_training_contract import RandomStreams, SharedInitialState, digest


def reference(seed):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU reference attempted CUDA initialization')
    torch.cuda._lazy_init = forbidden
    torch.set_num_threads(4)
    c = recipe(seed)
    with torch.device('meta'):
        meta = build_model(c)
    groups = grouped_parameters(meta, c)
    shapes = [(n, list(p.shape)) for g in groups if g['role']=='matrix' for n,p in zip(g['param_names'],g['params'])]
    dims = {}
    for _, shape in shapes:
        for d in shape:
            dims[str(d)] = dims.get(str(d), 0)+1
    parameters = sum(p.numel() for p in meta.parameters())
    matrix_parameters = sum(p.numel() for g in groups if g['role']=='matrix' for p in g['params'])
    if parameters != PARAMETERS or dims != DIMENSIONS or len(shapes) != 4:
        raise AssertionError('Actual model coverage differs')
    del groups, meta
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = build_model(c)
    streams = RandomStreams(torch.Generator().manual_seed(seed+1001),
        SimpleNamespace(rng=np.random.default_rng(seed+3001), counts={}))
    snapshot = SharedInitialState(model, streams, extra_workspace_bytes=4*2**20, ram_cap=SNAPSHOT_CAP)
    try:
        if snapshot.snapshot_bytes != SNAPSHOT_BYTES or snapshot.budget_bytes != SNAPSHOT_PURPOSE:
            raise AssertionError('Snapshot purpose changed')
        before, rng = snapshot.initial_digest, digest(streams.capture())
        identities = {n:id(v) for n,v in model.named_parameters()}
        buffers = {n:id(v) for n,v in model.named_buffers()}
        with torch.no_grad():
            next(model.parameters()).add_(.01)
        streams.crop.manual_seed(0); snapshot.reset()
        if digest(model.state_dict()) != before or digest(streams.capture()) != rng:
            raise AssertionError('Exact state/RNG reset failed')
        if identities != {n:id(v) for n,v in model.named_parameters()} or buffers != {n:id(v) for n,v in model.named_buffers()}:
            raise AssertionError('Parameter or buffer identity changed')
        row = dict(status='PASS_CPU_INITIAL_REFERENCE', seed=seed, parameters=parameters,
            matrix_parameters=matrix_parameters, matrix_shapes=shapes, factor_dimensions=dims,
            model_sha256=before, snapshot_bytes=snapshot.snapshot_bytes,
            snapshot_purpose_bytes=snapshot.budget_bytes, restored_exactly=True,
            parameter_and_buffer_identity_preserved=True, cuda_initialized=torch.cuda.is_initialized(),
            optimizer_updates=0, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    finally:
        snapshot.close()
    del snapshot, model
    gc.collect()
    if torch.cuda.is_initialized():
        raise AssertionError('CUDA was initialized')
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh reference output required')
    row = reference(args.seed)
    args.output.write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row))
