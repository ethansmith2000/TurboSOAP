"""Larger native shapes and before-allocation cap, without CUDA or training."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_hybrid_training import Tiny,no_cuda
from cifar_optimizer_benchmark import support
from cifar_optimizers import grouped_parameters
from hybrid_training_contract import BASELINE
from hybrid_width_contract import protocol,recipe,validate,SNAPSHOT_CAP
from matched_training_contract import RandomStreams,SharedInitialState
from cifar_hybrid_width import larger_snapshot_cap
import cifar_matched_training as trainer

def test_native_width_factor_shapes_and_old_cap_fails_before_snapshot_allocation():
 c=recipe(json.loads((BASELINE/'config.json').read_text()))
 with torch.device('meta'):model=support.build_model(c)
 groups=grouped_parameters(model,c);mat=[p for g in groups if g['role']=='matrix' for p in g['params']]
 assert [list(p.shape) for p in mat]==[[6144,2048],[2048,2048],[8192,2048],[2048,8192]]
 assert sum(p.numel() for p in model.parameters())==50796644
 assert max(max(p.shape) for p in mat)==8192<10000
 extra=3*c['batch_size']*3*32*32*4+3*c['batch_size']*c['classes']*4
 assert sum(v.numel()*v.element_size() for v in model.state_dict().values())+3*max(p.numel()*p.element_size() for p in model.parameters())+extra==414257552
 with pytest.raises(MemoryError,match='RAM cap'):SharedInitialState(model,None,extra_workspace_bytes=extra)
 p=protocol();validate(c,p,json.loads((BASELINE/'config.json').read_text()))
 assert p['maximum_optimizer_updates']==2020 and p['expected']['hybrid_calls_by_dimension']=={'2048':145,'6144':29,'8192':58}
 assert p['bounds']['maximum_initial_snapshot_purpose_cpu_bytes']==512*2**20

def test_snapshot_binding_restores_after_success_and_failure(monkeypatch):
 original=trainer.SharedInitialState;calls=[]
 def observe(*a,**kw):calls.append(kw['ram_cap']);return original(*a,**kw)
 monkeypatch.setattr(trainer,'SharedInitialState',observe)
 streams=RandomStreams(torch.Generator().manual_seed(17),SimpleNamespace(rng=np.random.default_rng(19),counts={}))
 with larger_snapshot_cap():snapshot=trainer.SharedInitialState(Tiny(),streams)
 assert calls==[SNAPSHOT_CAP] and trainer.SharedInitialState is observe
 snapshot.close()
 with pytest.raises(RuntimeError,match='injected'):
  with larger_snapshot_cap():raise RuntimeError('injected')
 assert trainer.SharedInitialState is observe
 with pytest.raises(ValueError,match='nested'):
  with larger_snapshot_cap():trainer.SharedInitialState(Tiny(),streams,ram_cap=1)
 assert trainer.SharedInitialState is observe
