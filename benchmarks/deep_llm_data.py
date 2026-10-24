"""Read-only shared OWT adapter; blocks never cross cached row boundaries."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

CACHE = Path('/workspace/data/tokenized/openwebtext_gpt2_bs1024')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plan(rows, seed, training_calls, batch_size, panel_size=64):
    required = training_calls * batch_size + panel_size
    if required > rows or min(training_calls, batch_size, panel_size) < 1:
        raise ValueError('A complete nonrepeating plan and disjoint train panel must fit')
    order = np.random.default_rng(seed + 2001).permutation(rows)
    return order[:training_calls * batch_size].reshape(training_calls, batch_size), order[-panel_size:]


def tokens_from_rows(dataset, indices, width=1024):
    array = np.asarray(dataset[[int(i) for i in indices]]['input_ids'], dtype=np.int64)
    if array.shape != (len(indices), width) or array.min() < 0 or array.max() >= 50257:
        raise ValueError('Cached row width or vocabulary differs')
    return torch.from_numpy(array)


def shifted_loss(logits, tokens):
    if logits.shape[:2] != tokens.shape or tokens.shape[1] < 2:
        raise ValueError('Logits and complete cached token rows must match')
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           tokens[:, 1:].reshape(-1))


def audit(cache=CACHE, seed=191):
    """Verify all row widths and pin every batch consumed by this bounded run."""
    import datasets
    data = datasets.load_from_disk(str(cache), keep_in_memory=False)
    manifest = json.loads((cache / 'MANIFEST.json').read_text())
    result = dict(path=str(cache.resolve()), datasets_version=datasets.__version__,
                  declared_manifest=manifest, files={}, splits={},
                  provenance_limit='Builder/manifest record recipe but do not cryptographically pin upstream dataset/tokenizer revisions. Every consumed batch has an exact token hash; bulk Arrow shards are stat-pinned, not fully byte-hashed. No document IDs or content-deduplication proof.')
    expected = {'train': 8372843, 'validation': 443501}
    for split, dataset in data.items():
        if split not in expected or len(dataset) != expected[split] or dataset.column_names != ['input_ids']:
            raise ValueError('Shared cache split/count/columns changed')
        checked = 0
        for chunk in dataset.data.table['input_ids'].chunks:
            offsets = chunk.offsets.to_numpy(zero_copy_only=False)
            if chunk.null_count or not np.all(np.diff(offsets) == 1024):
                raise ValueError('Not every cached row contains 1024 tokens')
            checked += len(chunk)
        rng = np.random.default_rng(seed)
        ids = sorted(set([0, 1, 12345, len(dataset)-1, *rng.choice(len(dataset), 128, replace=False).tolist()]))
        tokens = tokens_from_rows(dataset, ids)
        result['splits'][split] = dict(rows=len(dataset), fingerprint=dataset._fingerprint,
            width_checked_rows=checked, stored_tokens=len(dataset)*1024,
            usable_targets=len(dataset)*1023, sampled_rows=ids,
            sampled_tokens_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest())
    if set(result['splits']) != set(expected):
        raise ValueError('Missing shared cache split')
    for path in sorted(cache.rglob('*')):
        if path.is_file():
            stat = path.stat()
            result['files'][str(path.relative_to(cache))] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                inode=stat.st_ino, sha256=None if path.suffix=='.arrow' else sha(path))
    order, train_ids = plan(len(data['train']),seed,35000,8)
    validation_ids = np.random.default_rng(seed+4001).choice(len(data['validation']),64,replace=False)
    def hashes(split, batches):
        return [hashlib.sha256(tokens_from_rows(data[split],ids).numpy().tobytes()).hexdigest() for ids in batches]
    result['consumed_inputs'] = dict(seed=seed,batch_size=8,primary_calls=125,
        train_batches_sha256=hashes('train',order[:125]),
        train_probe_batches_sha256=hashes('train',train_ids.reshape(-1,8)),
        validation_batches_sha256=hashes('validation',validation_ids.reshape(-1,8)))
    result['status'] = 'PASS_READ_ONLY_CACHE_AND_SELECTED_INPUT_AUDIT'
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=191)
    args = parser.parse_args()
    result = audit(seed=args.seed)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], splits=result['splits'], files=len(result['files']))))
