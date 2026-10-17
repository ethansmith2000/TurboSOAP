"""Per-instance reuse installation and own-history physical-M diagnostics."""
import math
import weakref
import torch
from llm_sparse_screen_kernel import build_arm as native_arm, common_geometry_guard
from qr_reuse_profile_kernel import install
from hybrid_training_kernel import precision
from tensor_ownership import TensorOwnership
from reuse_learning_contract import ARMS, PRIMARY_M_AGES

def build_arm(model,config,arm,router):
    if arm not in ARMS:raise ValueError('Undeclared learning arm')
    opt=native_arm(model,config,'qr20',router)
    if arm=='reuse20':install(opt.matrix,'reuse')
    return opt

def qualify_actual_m(matrix,router,arm,phase):
    original=matrix._research_refresh
    ref=weakref.ref(matrix)
    # Retain a native bound method only through a weak reference to avoid an owner cycle.
    if getattr(original,'__self__',None) is matrix:
        method=original.__func__
        def refresh_original(obj,state,group):return method(obj,state,group)
    else:
        def refresh_original(obj,state,group):return original(state,group)
    def refresh(state,group):
        obj=ref()
        if obj is None:raise RuntimeError('Released optimizer owner')
        due=(not state.get('research_strict_transports',0)) if phase=='disposable' else state['step'] in PRIMARY_M_AGES
        if not due:return refresh_original(obj,state,group)
        owner=TensorOwnership();before=after=None;rng=router.rng_digest()
        try:
            with precision('highest'):
                before=obj._project_back_with_basis(state['exp_avg'],state['Q'],group);owner.watch(before,'actualM.before')
            refresh_original(obj,state,group)
            with precision('highest'):
                after=obj._project_back_with_basis(state['exp_avg'],state['Q'],group);owner.watch(after,'actualM.after')
                error=float((after-before).norm()/before.norm().clamp_min(1e-12))
            row=dict(arm=arm,phase=phase,age=int(state['step']),physical_m_relative=error,passed=math.isfinite(error) and error<=.01,rng_unchanged=rng==router.rng_digest())
            router.moments.append(row);router.emit('actual_m_before_gate',row)
            if not row['passed'] or not row['rng_unchanged']:raise FloatingPointError('Actual own-history momentum preservation failed')
        finally:
            before=after=None;release=owner.report();owner.close();router.emit('actual_m_release',release)
            if not release['all_registered_released']:raise RuntimeError('Physical-M buffers leaked')
    matrix._research_refresh=refresh
