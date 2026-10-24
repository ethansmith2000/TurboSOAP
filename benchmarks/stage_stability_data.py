"""Deterministic repeated-data full-batch plan, isolated from training RNG."""
import numpy as np

def repeated_plan(seed,blocks,batch,calls):
 if min(blocks,batch,calls)<=0 or blocks<batch:raise ValueError('Nonempty full batch required')
 rng=np.random.default_rng(seed+2001);full=blocks//batch;chunks=[];epochs=[];remaining=calls;start=0
 while remaining:
  permutation=rng.permutation(blocks);take=min(full,remaining)
  chunks.append(permutation[:take*batch].reshape(take,batch).copy())
  epochs.append(dict(epoch=len(epochs),first_call=start+1,last_call=start+take,batches=take,full_epoch_batches=full,dropped_incomplete_tail_blocks=blocks%batch,unused_blocks_in_final_epoch=blocks-take*batch))
  remaining-=take;start+=take
 return np.concatenate(chunks),epochs
