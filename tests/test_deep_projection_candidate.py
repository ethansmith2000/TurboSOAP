"""Candidate/native separation, dispatch, reset metadata and construction restoration."""
import copy
import gc
import json
from pathlib import Path
import sys
import weakref
from collections import Counter

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'benchmarks'),str(Path(__file__).resolve().parents[1]/'benchmarks/cifar_support')]
import pytest
import torch
import deep_llm_learning as native
from deep_projection_candidate import CandidateRun,quality_row,recipe
from deep_projection_candidate_kernel import install_counted,native_adaptive,projection_metrics
from hybrid_training_kernel import precision
from matched_training_runtime import streaming_digest
from soap_reference import ResearchSOAP
from tensor_ownership import TensorOwnership


def fixture():
    torch.manual_seed(194)
    p=torch.nn.Parameter(torch.randn(7,3))
    opt=ResearchSOAP([p],precondition_frequency=20,hard_reset_interval=0,transport_precision='highest')
    counts=Counter();scope={'value':'training'};install_counted(opt,counts,scope)
    return p,opt,counts,scope


def test_actual_bf16_dispatch_and_native_highest_transport_keep_fp32_state():
    p,opt,counts,scope=fixture()
    with precision('high'):
        for _ in range(42):p.grad=torch.randn_like(p);opt.step()
    assert counts['training.bf16_forward']==41 and counts['training.bf16_back']==41
    assert counts['training.highest_forward']==2 and counts['training.highest_back']==2
    state=opt.state[p]
    assert all(x.dtype==torch.float32 for x in [state['exp_avg'],state['exp_avg_sq'],*state['GG'],*state['Q']])
    assert state['research_qr_refreshes']==2 and state['research_strict_transports']==2
    owner=weakref.ref(opt);del opt;gc.collect();assert owner() is None


def test_unhooked_reference_stays_native_and_live_adapter_parity_is_checked():
    p,opt,counts,scope=fixture()
    with precision('high'),torch.no_grad():
        for _ in range(22):p.grad=torch.randn_like(p);opt.step()
        state=opt.state[p];before=streaming_digest(opt.state_dict());original=dict(counts)
        native_adaptive(opt,state,opt.param_groups[0],p.grad)
        assert dict(counts)==original
        owner=TensorOwnership();row=projection_metrics(opt,state,opt.param_groups[0],p.grad,owner,check_live=True)
        assert row['relative']>0 and row['live_adapter_parity']
        assert before==streaming_digest(opt.state_dict())
        assert owner.report()['all_registered_released'];owner.close()


def test_projection_hook_corruption_is_rejected_without_retained_workspace():
    p,opt,counts,scope=fixture()
    with precision('high'),torch.no_grad():
        p.grad=torch.randn_like(p);opt.step()
        original=opt._project_with_basis
        opt._project_with_basis=lambda *args:original(*args)+1.
        owner=TensorOwnership()
        with pytest.raises(RuntimeError,match='forward dispatch differs'):
            projection_metrics(opt,opt.state[p],opt.param_groups[0],p.grad,owner,check_live=True)
        gc.collect();assert owner.report()['all_registered_released'];owner.close()


def test_construction_overlay_restores_native_module_on_phase_failure(tmp_path,monkeypatch):
    runtime=CandidateRun(tmp_path);builder=native.build_optimizer;observer=native.LayerObserver
    def fail(*args):
        assert native.build_optimizer is not builder and native.LayerObserver is not observer
        raise RuntimeError('injected phase failure')
    monkeypatch.setattr(native.Run,'trajectory',fail)
    with pytest.raises(RuntimeError,match='injected phase failure'):runtime.trajectory('disposable',25)
    assert native.build_optimizer is builder and native.LayerObserver is observer


def test_quality_margin_is_per_checkpoint_and_recipe_keeps_numeric_config():
    assert quality_row(1000,4.01,4.)['passes_panel']
    assert not quality_row(35000,4.010001,4.)['passes_panel']
    assert not quality_row(0,4.,4.)['postwarmup']
    base=dict(seed=193,lr=.0005,primary_calls=35000,matched_comparison={'arms':['qr20']})
    changed=recipe(base)
    assert {k:v for k,v in changed.items() if k!='matched_comparison'}=={k:v for k,v in base.items() if k!='matched_comparison'}
    assert changed['matched_comparison']['trial']==194 and base['matched_comparison']['arms']==['qr20']
