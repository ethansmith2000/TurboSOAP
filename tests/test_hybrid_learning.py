"""New evaluation transaction, longer cadence and immutable learning protocol checks."""
import copy
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from test_hybrid_training import CPU, Tiny, CONFIG, no_cuda
from hybrid_learning_contract import protocol,recipe,validate,ORDERS,expected_counts
from hybrid_training_kernel import build_arm
from cifar_hybrid_learning import evaluate_transaction,run_pilot,endpoint
from matched_training_runtime import streaming_digest

class LearningCPU(CPU):
    def __init__(self,bundle,fault=None):
        super().__init__(bundle,fault)
        self.protocol.update(execution_order=list(ORDERS[142]),probe_calls=[21],evaluation_calls=[100,221])
        self.ids={'validation':list(range(5)),'train_panel':list(range(3))}
        self.evaluation_fault=None
    def evaluate(self):
        assert not any(m.training for m in self.model.modules())
        torch.rand(3);np.random.random();random.random()
        torch.rand(2,generator=self.streams.crop);self.streams.augmentation.rng.random()
        self.streams.augmentation.counts['mixup']+=1
        if self.evaluation_fault=='raise':raise RuntimeError('injected evaluation failure')
        if self.evaluation_fault=='weights':
            with torch.no_grad():next(self.model.parameters()).add_(1.)
        return {name:dict(ce=1.,accuracy=.5,examples=len(ids)) for name,ids in self.ids.items()}

def test_eval_restores_rng_heterogeneous_flags_and_all_training_state(tmp_path):
    r=LearningCPU(tmp_path);r.model.blocks[1].eval()
    opt=build_arm(r.model,r.config,'hybrid20',r.kernel)
    for p in r.model.parameters():p.grad=torch.ones_like(p)
    flags=[m.training for m in r.model.modules()];before=endpoint(r,opt)
    row=evaluate_transaction(r,opt)
    assert row['restoration_passed'] and row['before']==row['after']==before
    assert [m.training for m in r.model.modules()]==flags
    r.evaluation_fault='raise'
    with pytest.raises(RuntimeError,match='injected evaluation'):evaluate_transaction(r,opt)
    assert endpoint(r,opt)==before and [m.training for m in r.model.modules()]==flags
    r.snapshot.close()

def test_evaluation_rejects_mutation(tmp_path):
    r=LearningCPU(tmp_path);opt=build_arm(r.model,r.config,'qr20',r.kernel);r.evaluation_fault='weights'
    with pytest.raises(RuntimeError,match='Evaluation mutated'):evaluate_transaction(r,opt)
    r.snapshot.close()

def test_cpu_orchestration_reversed_order_evaluation_and_later_reset(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    r=LearningCPU(tmp_path);result=run_pilot(r)
    assert result['optimizer_calls']==799
    assert [row['arm'] for row in result['primary']]==list(ORDERS[142])
    assert all(row['exact_disposable_scalar_prefix'] for row in result['primary'])
    assert all([v['call'] for v in row['evaluations']]==[100,221] for row in result['primary'])
    assert len({row['endpoint']['rng_sha256'] for row in result['primary']})==1
    assert all(row['release']['passed'] for row in result['primary']+result['disposable'])
    assert result['snapshot_release']['all_registered_released']

def test_longer_recipe_counters_and_resealed_policy_rejection(tmp_path):
    import prepare_hybrid_learning as prep
    for seed in (142,143):
        c=recipe(CONFIG,seed);p=protocol(seed);validate(c,p,CONFIG)
        assert p['maximum_optimizer_updates']==11260
        assert p['expected']['hybrid_kernel_calls']==1344
        assert p['expected']['hybrid_calls_by_dimension']=={'1536':840,'4608':168,'6144':336}
        assert p['expected']['diagnostic_reference_calls']==32
        assert p['expected']['primary']['warm20']['research_warm_refreshes']==157
        assert p['expected']['primary']['qr20']['research_qr_refreshes']==174
        assert p['expected']['primary']['warm20']['research_qr_refreshes']==17
        assert all((v-1)%20==0 and (v-1)%200 for v in p['probe_calls'])
        assert min(p['evaluation_calls'])>p['schedule']['lr_warmup']
    ref=tmp_path/'reference.json'
    prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=142,parameters=28660324,
        cuda_initialized=False,snapshot_purpose_bytes=237631888,model_sha256='0'*64))
    b=tmp_path/'fresh';m=prep.prepare(b,ref)
    assert len(m['source_sha256'])==66
    cfg=prep.read(b/'config.json');cfg['seed']=143;prep.write(b/'config.json',cfg)
    resealed=copy.deepcopy(m);resealed['artifacts']['config.json']=prep.sha(b/'config.json');prep.write(b/'manifest.json',resealed)
    with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
