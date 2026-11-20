"""Regenerate the paper's frozen OWT row selection for the untied-head study.

Mirrors benchmarks/prepare_owt_restart.py (selection seeds 2205/4205, sorted-order
reads, uint16 [rows, 1024]) and then compares every output's sha256 with the
paper's recorded provenance (reports/owt_restart_20261026/provenance.json).
Also writes the training-row unigram counts used for the token-frequency buckets.

CPU only; run thread-capped and niced so it does not disturb timing runs:
  OMP_NUM_THREADS=2 nice -n 19 python benchmarks/untied_head_prepare.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PAPER_PROVENANCE = REPO / 'reports/owt_restart_20261026/provenance.json'
CACHE = Path('/workspace/data/tokenized/openwebtext_gpt2_u16_1024_v1')
SELECTION = [('train', 320000, 2205), ('validation', 1024, 4205)]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--output', type=Path, default=Path('/workspace/data/untied_head_20261120'))
    args = parser.parse_args()
    from datasets import load_from_disk

    paper = json.loads(PAPER_PROVENANCE.read_text())
    local_manifest = json.loads((args.cache / 'MANIFEST.json').read_text())
    shard_match = {k: v['sha256'] == local_manifest['files'].get(k, {}).get('sha256')
                   for k, v in paper['source_manifest']['files'].items()}
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_from_disk(str(args.cache))
    receipt = dict(source_cache=str(args.cache), source_manifest_sha256=sha(args.cache / 'MANIFEST.json'),
                   paper_source_manifest_sha256=paper['source_manifest_sha256'],
                   manifest_files_matching_paper=sum(shard_match.values()),
                   manifest_files_differing_from_paper=[k for k, ok in shard_match.items() if not ok],
                   selection=[dict(split=s, rows=n, seed=seed) for s, n, seed in SELECTION], files={})
    for split, count, seed in SELECTION:
        assert len(data[split]) == paper['source_manifest']['splits'][split]['rows']
        ids = np.random.default_rng(seed).choice(len(data[split]), count, replace=False)
        assert len(np.unique(ids)) == count
        ids_path = args.output / f'{split}_row_ids.npy'
        np.save(ids_path, ids)
        rows_path = args.output / f'{split}.npy'
        if not rows_path.exists() or sha(rows_path) != paper['files'][f'{split}.npy']['sha256']:
            temporary = args.output / f'{split}.tmp.npy'
            target = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint16, shape=(count, 1024))
            order = np.argsort(ids)
            for begin in range(0, count, 512):
                positions = order[begin:begin + 512]
                values = np.asarray(data[split][ids[positions].tolist()]['input_ids'], dtype=np.uint16)
                assert values.shape == (len(positions), 1024) and values.max() <= 50256
                target[positions] = values
            target.flush()
            del target
            temporary.replace(rows_path)
        for name in (f'{split}.npy', f'{split}_row_ids.npy'):
            digest = sha(args.output / name)
            receipt['files'][name] = dict(bytes=(args.output / name).stat().st_size, sha256=digest,
                                          paper_sha256=paper['files'][name]['sha256'],
                                          matches_paper=digest == paper['files'][name]['sha256'])
        print(json.dumps({split: {k: v['matches_paper'] for k, v in receipt['files'].items()}}), flush=True)
    train = np.load(args.output / 'train.npy', mmap_mode='r')
    counts = np.zeros(50257, dtype=np.int64)
    for begin in range(0, train.shape[0], 20000):
        counts += np.bincount(np.asarray(train[begin:begin + 20000]).reshape(-1), minlength=50257)
    np.save(args.output / 'train_token_counts.npy', counts)
    receipt['train_token_counts'] = dict(total=int(counts.sum()), seen_tokens=int((counts > 0).sum()),
                                         sha256=sha(args.output / 'train_token_counts.npy'))
    receipt['all_match_paper'] = all(v['matches_paper'] for v in receipt['files'].values())
    (args.output / 'provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(all_match_paper=receipt['all_match_paper'], **receipt['train_token_counts'])))


if __name__ == '__main__':
    main()
