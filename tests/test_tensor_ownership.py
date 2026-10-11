import json
import weakref

import pytest
import torch

from benchmarks.tensor_ownership import TensorOwnership
from benchmarks.cuda_library_ownership import alias_control,library_scope


def test_observer_has_no_strong_tensor_or_storage_ownership():
    observer=TensorOwnership();x=torch.zeros(4,6);weak=weakref.ref(x)
    observer.watch(x,'owner');del x
    assert weak() is None
    report=observer.report()
    assert report['all_registered_released'] and report['live_unique_storage_bytes']==0
    json.dumps(report,allow_nan=False)


def test_unregistered_alias_is_detected_after_registered_tensor_dies():
    result=alias_control(torch.device('cpu'))
    assert result['intentional_alias_retention_detected']['live_tensors']==0
    assert result['intentional_alias_retention_detected']['live_unique_storages']==1
    assert result['after_release']['all_registered_released']


def test_regular_view_retains_registered_base_object():
    observer=TensorOwnership();value=torch.zeros(4,6);observer.watch(value,'view_base')
    alias=value.view(-1);del value
    report=observer.report()
    assert report['live_tensors']==1 and report['live_unique_storages']==1
    del alias
    assert observer.report()['all_registered_released']


def test_bare_storage_reference_is_detected_after_tensor_dies():
    observer=TensorOwnership();value=torch.zeros(4,6)
    storage=value.untyped_storage();observer.watch(value,'storage_reference')
    del value
    report=observer.report()
    assert report['live_tensors']==0 and report['live_unique_storages']==1
    del storage
    assert observer.report()['all_registered_released']


def test_retained_tensor_and_shared_storage_are_counted_once():
    observer=TensorOwnership();base=torch.zeros(4,6);alias=base.view(2,12)
    observer.watch(base,'base');observer.watch(alias,'alias');observer.watch(base,'base_again')
    report=observer.report()
    assert report['tracked_tensors']==2 and report['tracked_unique_storages']==1
    assert report['live_unique_storage_bytes']==base.numel()*base.element_size()
    del alias;assert not observer.report()['all_registered_released']
    del base;assert observer.report()['all_registered_released']


def test_exception_traceback_retention_is_visible_until_exception_released():
    observer=TensorOwnership()
    def failing_scope():
        value=torch.zeros(4,6);observer.watch(value,'exception_frame')
        raise RuntimeError('intentional')
    held=None
    try:failing_scope()
    except RuntimeError as error:held=error
    assert not observer.report()['all_registered_released']
    held=None
    assert observer.report()['all_registered_released']


def test_call_observer_retained_closure_is_visible():
    observer=TensorOwnership();value=torch.zeros(4,6);observer.watch(value,'captured')
    def capture(x):return lambda:x
    closure=capture(value);del value
    assert not observer.report()['all_registered_released']
    del closure;assert observer.report()['all_registered_released']


def test_nested_cycles_do_not_hold_roots_and_tensor_dict_keys_are_observed():
    observer=TensorOwnership();value=torch.zeros(4,6);key=torch.nn.Parameter(torch.zeros(2,3))
    root={'buffers':[value,value],key:value};root['cycle']=root
    observer.watch_tree(root)
    assert observer.report()['tracked_tensors']==2
    root.pop('cycle');del root,value,key
    assert observer.report()['all_registered_released']


def test_invalid_empty_and_closed_observation_fails_closed():
    observer=TensorOwnership()
    with pytest.raises(RuntimeError,match='No registered'):observer.report()
    with pytest.raises(TypeError):observer.watch('not tensor','bad')
    with pytest.raises(ValueError):observer.watch(torch.zeros(2,3),'')
    observer.close()
    with pytest.raises(RuntimeError,match='closed'):observer.report()
    with pytest.raises(RuntimeError,match='closed'):observer.watch(torch.zeros(2,3),'later')


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_library_scope_returns_only_scalar_metadata_and_releases_registered_owners(dtype):
    observer=TensorOwnership();record=library_scope(observer,torch.device('cpu'),dtype)
    assert observer.report()['all_registered_released']
    assert record['inside_scope']['tracked_tensors']==3
    json.dumps(record,allow_nan=False)
