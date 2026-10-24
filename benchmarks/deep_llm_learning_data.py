"""Full-horizon, no-wrap inputs for the deeper native baseline; no cache writes."""
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from deep_llm_data import CACHE, sha, tokens_from_rows

PARENT = Path('/workspace/optimizer_replay_results/llm_soap_deep_feasibility_seed192_frozen_20261019')
HORIZON = 35000
BATCH = 8


def input_sha(tokens):
    value = tokens.numpy() if hasattr(tokens, 'numpy') else tokens
    return hashlib.sha256(np.asarray(value, dtype='<i4').tobytes()).hexdigest()


def window_plan(train_rows,seed):
    required=HORIZON*BATCH
    if train_rows<required+64:raise ValueError('Unique training windows plus unseen panel must fit')
    rng=np.random.default_rng(seed+2001);edges=np.linspace(0,train_rows,9,dtype=np.int64)
    windows=[];selected=[]
    for low,high in zip(edges[:-1],edges[1:]):
        if high-low<HORIZON:raise ValueError('Corpus stratum too small')
        start=int(rng.integers(low,high-HORIZON+1));windows.append([start,start+HORIZON])
        selected.append(np.arange(start,start+HORIZON,dtype=np.int64))
    ids=np.concatenate(selected);order=rng.permutation(ids).reshape(HORIZON,BATCH)
    unused=np.ones(train_rows,dtype=bool);unused[ids]=False
    unseen=rng.choice(np.flatnonzero(unused),64,replace=False)
    return order,unseen,windows


def data_plan(train_rows, validation_rows, seed):
    order,unseen,_=window_plan(train_rows,seed)
    validation = np.random.default_rng(seed + 4001).choice(validation_rows, 256, replace=False)
    panels = dict(seen_train=('train',order.reshape(-1)[:64]),
                  unseen_train=('train',unseen),validation=('validation',validation))
    return order, panels


def verify_stats(records):
    for name, record in records.items():
        stat = (CACHE / name).stat()
        if [stat.st_size, stat.st_mtime_ns, stat.st_ino] != record:
            raise RuntimeError('Shared cached file changed: ' + name)


def audit(output, seed=193):
    import datasets
    import psutil
    started = time.monotonic()
    parent = json.loads((PARENT/'data_audit.json').read_text())
    stats = json.loads((PARENT/'data_stat.json').read_text())
    verify_stats(stats)
    for name, record in parent['files'].items():
        if record['sha256'] is not None and sha(CACHE/name) != record['sha256']:
            raise RuntimeError('Shared cache metadata changed')
    data = datasets.load_from_disk(str(CACHE),keep_in_memory=False)
    for name, dataset in data.items():
        expected = parent['splits'][name]
        if len(dataset)!=expected['rows'] or dataset._fingerprint!=expected['fingerprint']:
            raise RuntimeError('Shared cache structure differs')
    order,panels=data_plan(len(data['train']),len(data['validation']),seed)
    _,_,windows=window_plan(len(data['train']),seed)
    flat=order.reshape(-1); sorted_positions=np.argsort(flat)
    # Preparation-only CPU token buffer: 1,146,880,000 bytes. Read selected rows
    # in source order to reduce shared-disk seeks, then hash exact batch order.
    tokens=np.empty((len(flat),1024),dtype='<i4'); peak_rss=0
    for start in range(0,len(flat),2048):
        positions=sorted_positions[start:start+2048]
        selected=tokens_from_rows(data['train'],flat[positions])
        tokens[positions]=selected.numpy().astype('<i4',copy=False)
        del selected
        peak_rss=max(peak_rss,psutil.Process().memory_info().rss)
        if peak_rss>12*2**30:raise MemoryError('Preparation process RSS limit')
        if start%(2048*32)==0:print(json.dumps(dict(event='audit_progress',selected_rows=min(start+2048,len(flat)),total=len(flat))),flush=True)
    hashes=[input_sha(tokens[start:start+BATCH]) for start in range(0,len(tokens),BATCH)]
    del tokens;gc.collect()
    panel_records={}
    for name,(split,ids) in panels.items():
        panel_records[name]=dict(split=split,ids=ids.tolist(),blocks=len(ids),valid_tokens=len(ids)*1023,
            batch_sha256=[input_sha(tokens_from_rows(data[split],ids[start:start+BATCH])) for start in range(0,len(ids),BATCH)])
    verify_stats(stats)
    result=dict(status='PASS_FULL_HORIZON_SELECTED_INPUT_AUDIT',seed=seed,
        shared_cache=str(CACHE),parent_cache_audit=str(PARENT/'data_audit.json'),
        parent_cache_audit_sha256=sha(PARENT/'data_audit.json'),parent_structure=parent['splits'],
        cache_file_stat=stats,training_calls=HORIZON,batch_size=BATCH,
        sampling='Eight equal nonoverlapping corpus strata, one35000row contiguous window per stratum; jointly permute280000selected rows with seed+2001. No within-run repeats; a sampled subset, not a whole-corpus pass.',
        stratified_training_windows=windows,
        training_plan_shape=list(order.shape),training_plan_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
        train_batches_sha256=hashes,panels=panel_records,
        unique_training_blocks=int(np.unique(flat).size),primary_targets=HORIZON*BATCH*1023,
        primary_input_tokens=HORIZON*BATCH*1024,seconds=time.monotonic()-started,
        preparation_cpu_token_buffer_bytes=1146880000,peak_process_rss=peak_rss,
        preparation_retention='Token buffer released before return/process exit; no new binary data or tensor file.',
        provenance_limit=parent['provenance_limit'],no_wrap=True,no_cache_writes=True)
    if result['unique_training_blocks']!=HORIZON*BATCH:raise RuntimeError('Repeated training rows')
    with Path(output).open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],unique_training_blocks=result['unique_training_blocks'],peak_process_rss=peak_rss)),flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,default=193)
    args=parser.parse_args();audit(args.output,args.seed)
