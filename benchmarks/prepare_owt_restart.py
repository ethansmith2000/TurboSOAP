"""Prepare an immutable input cohort; run with /venv/owt/bin/python."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    from datasets import load_from_disk
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=Path('/workspace/data/tokenized/openwebtext_gpt2_u16_1024_v1'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.cache/'MANIFEST.json').read_text())
    assert manifest['status'] == 'complete' and manifest['block_size'] == 1024
    assert manifest['dtype'] == 'uint16' and not manifest['historical_replay']
    for relative, item in manifest['files'].items():
        assert sha(a.cache/relative) == item['sha256'], relative
    a.output.mkdir(parents=True, exist_ok=False)
    data = load_from_disk(str(a.cache))
    receipt = dict(source_cache=str(a.cache), source_manifest_sha256=sha(a.cache/'MANIFEST.json'),
                   source_manifest=manifest, seed=204, historical_replay=False,
                   purpose='Shared inputs for a fresh, bounded SOAP OpenWebText beta-profile comparison', files={})
    for split, count, seed in [('train', 320000, 2205), ('validation', 1024, 4205)]:
        ids = np.random.default_rng(seed).choice(len(data[split]), count, replace=False)
        assert len(np.unique(ids)) == count
        np.save(a.output/f'{split}_row_ids.npy', ids)
        target = np.lib.format.open_memmap(a.output/f'{split}.npy', mode='w+', dtype=np.uint16, shape=(count, 1024))
        order = np.argsort(ids)
        for begin in range(0, count, 512):
            positions = order[begin:begin+512]
            values = np.asarray(data[split][ids[positions].tolist()]['input_ids'], dtype=np.uint16)
            assert values.shape == (len(positions), 1024) and values.max() <= 50256
            target[positions] = values
        target.flush()
        del target
        for name in (f'{split}.npy', f'{split}_row_ids.npy'):
            receipt['files'][name] = dict(bytes=(a.output/name).stat().st_size, sha256=sha(a.output/name))
    (a.output/'provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    repo = Path(__file__).resolve().parents[1]
    source = a.output/'source'
    source.mkdir()
    for name in ['soap.py','soap_reference.py','fast_soap.py','transformer.py','benchmarks/cifar_optimizers.py',
                 'benchmarks/owt_restart.py','benchmarks/prepare_owt_restart.py',
                 'tests/reference/soap_upstream.py','tests/reference/provenance.json']:
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, target)
    receipt['source_files'] = {str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    (a.output/'provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(status='prepared', output=str(a.output), source_files=len(receipt['source_files']))))


if __name__ == '__main__':
    main()
