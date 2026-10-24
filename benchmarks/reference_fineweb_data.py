"""Pinned uint16 FineWeb inputs and disjoint, full-target next-token windows."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F


class FineWebWindows:
    def __init__(self, paths, sequence=1024, vocab=50257):
        self.paths = [Path(p) for p in paths]
        self.sequence = sequence
        self.maps = []
        self.counts = []
        for path in self.paths:
            header = np.fromfile(path, dtype='<i4', count=256)
            if len(header) != 256 or tuple(header[:2]) != (20240520, 1):
                raise ValueError(f'Invalid FineWeb header: {path}')
            count = int(header[2])
            if count <= 0 or path.stat().st_size != 1024 + 2 * count:
                raise ValueError(f'Invalid FineWeb length: {path}')
            data = np.memmap(path, mode='r', dtype='<u2', offset=1024, shape=(count,))
            if int(data.max()) >= vocab:
                raise ValueError(f'Token outside vocabulary: {path}')
            self.maps.append(data)
            self.counts.append(count // (sequence + 1))
        self.ends = np.cumsum(self.counts)

    def __len__(self):
        return int(self.ends[-1])

    def windows(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(self)):
            raise ValueError('Window IDs must be a vector within shard bounds')
        result = np.empty((len(ids), self.sequence + 1), dtype=np.int64)
        for i, index in enumerate(ids):
            shard = int(np.searchsorted(self.ends, index, side='right'))
            local = int(index - (self.ends[shard-1] if shard else 0))
            offset = local * (self.sequence + 1)
            result[i] = self.maps[shard][offset:offset+self.sequence+1]
        return result


def content_hash(windows):
    # Canonical bytes independent of host endianness or torch dtype.
    return hashlib.sha256(np.asarray(windows, dtype='<u2').tobytes()).hexdigest()


def sequence_losses(logits, windows):
    """Every target is valid, including GPT-2 ID zero; no padding is inserted."""
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           windows[:, 1:].reshape(-1), reduction='none').reshape(
                               len(windows), -1).mean(-1)


@contextmanager
def evaluation_mode(model):
    """Restore module modes, recomputation, and all RNG streams even on error."""
    modes = [(module, module.training) for module in model.modules()]
    recompute = model.recompute
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None
    numpy = np.random.get_state()
    python = random.getstate()
    try:
        model.eval()
        model.recompute = False
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        model.recompute = recompute
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        np.random.set_state(numpy)
        random.setstate(python)
