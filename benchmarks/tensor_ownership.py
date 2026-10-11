"""Offline lifetime observation for explicitly registered tensors and storages.

Weak tensor references and PyTorch's native storage weak handles do not own data.
Storage handles detect aliases even after the registered tensor has died. This
does not enumerate every process allocation or identify library workspace owners.
"""
import weakref

import torch
from torch.multiprocessing.reductions import StorageWeakRef


class TensorOwnership:
    def __init__(self):
        self._tensors={}
        self._tensor_ids={}
        self._storages={}
        self._closed=False

    def watch(self,tensor,label):
        if self._closed:
            raise RuntimeError('Ownership observer is closed')
        if not torch.is_tensor(tensor) or tensor.layout!=torch.strided:
            raise TypeError('Register a dense strided tensor')
        if not isinstance(label,str) or not label:
            raise ValueError('Require a nonempty ownership label')
        tensor_id=id(tensor)
        key=self._tensor_ids.get(tensor_id)
        previous=self._tensors.get(key)
        if previous is not None and previous['ref']() is tensor:
            if label not in previous['labels']:
                previous['labels'].append(label)
                self._storages[previous['storage_key']]['labels'].append(label)
            return
        # Use an ordinal because Python object IDs can be reused after death.
        key=len(self._tensors)
        self._tensor_ids[tensor_id]=key
        storage=tensor.untyped_storage()
        storage_ref=StorageWeakRef(storage)
        storage_key=storage_ref.cdata
        if storage_key not in self._storages:
            self._storages[storage_key]=dict(ref=storage_ref,nbytes=storage.nbytes(),
                                            labels=[],device=str(tensor.device))
        entry=self._storages[storage_key]
        if label not in entry['labels']:
            entry['labels'].append(label)
        self._tensors[key]=dict(ref=weakref.ref(tensor),labels=[label],
                               storage_key=storage_key,
                               shape=list(tensor.shape),stride=list(tensor.stride()),
                               dtype=str(tensor.dtype),device=str(tensor.device))

    def watch_tree(self,value,label='purpose'):
        seen=set()
        def visit(item,path):
            if torch.is_tensor(item):
                self.watch(item,path)
            elif isinstance(item,(dict,list,tuple)):
                if id(item) in seen:
                    return
                seen.add(id(item))
                if isinstance(item,dict):
                    for index,(key,child) in enumerate(item.items()):
                        if torch.is_tensor(key):
                            self.watch(key,f'{path}.key[{index}]')
                        name=str(key) if isinstance(key,(str,int)) else f'item{index}'
                        visit(child,f'{path}.{name}')
                else:
                    for index,child in enumerate(item):
                        visit(child,f'{path}[{index}]')
        visit(value,label)

    def report(self):
        if self._closed:
            raise RuntimeError('Ownership observer is closed')
        if not self._tensors:
            raise RuntimeError('No registered owners; cannot qualify release')
        live_tensors=[entry for entry in self._tensors.values() if entry['ref']() is not None]
        live_storages=[entry for entry in self._storages.values() if not entry['ref'].expired()]
        return dict(tracked_tensors=len(self._tensors),tracked_unique_storages=len(self._storages),
                    live_tensors=len(live_tensors),live_unique_storages=len(live_storages),
                    live_tensor_labels=[entry['labels'][:] for entry in live_tensors],
                    live_storage_labels=[entry['labels'][:] for entry in live_storages],
                    live_unique_storage_bytes=sum(entry['nbytes'] for entry in live_storages),
                    registered_unique_storage_bytes=sum(entry['nbytes'] for entry in self._storages.values()),
                    all_registered_released=not live_tensors and not live_storages,
                    scope='Registered tensors and their storage aliases only; not whole-process allocation ownership')

    def close(self):
        self._tensors.clear();self._tensor_ids.clear();self._storages.clear();self._closed=True
