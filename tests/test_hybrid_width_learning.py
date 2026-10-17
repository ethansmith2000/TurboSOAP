"""New width2048 learning contract; reused tested runner/state restoration."""
import copy,json
from pathlib import Path
import pytest
from test_hybrid_training import no_cuda
from hybrid_training_contract import BASELINE
from hybrid_width_contract import recipe as width_recipe
from hybrid_width_learning_contract import recipe,protocol,validate,SNAPSHOT_CAP,PARAMETERS
from cifar_hybrid_width import Runtime as WidthRuntime
from cifar_hybrid_width_learning import Runtime

def test_only_declared_width_learning_changes_and_counters():
 base=json.loads((BASELINE/'config.json').read_text());c=recipe(base);p=protocol();validate(c,p,base)
 old=width_recipe(base)
 assert {k:v for k,v in c.items() if k not in ('seed','max_wall_seconds','matched_comparison')}=={k:v for k,v in old.items() if k not in ('seed','max_wall_seconds','matched_comparison')}
 assert c['seed']==145 and c['width']==2048 and c['heads']==32
 assert p['maximum_optimizer_updates']==11260 and p['expected']['hybrid_kernel_calls']==1344
 assert p['expected']['hybrid_calls_by_dimension']=={'2048':840,'6144':168,'8192':336}
 assert p['expected']['diagnostic_reference_calls']==32 and p['evaluation_calls']==[1000,1750,2625,3500]
 assert p['bounds']['maximum_initial_snapshot_purpose_cpu_bytes']==512*2**20
 assert Runtime.initialize is WidthRuntime.initialize and Runtime.budget is WidthRuntime.budget
 for field,value in [('seed',144),('width',1536),('heads',24)]:
  wrong=copy.deepcopy(c);wrong[field]=value
  with pytest.raises(ValueError,match='Undeclared'):validate(wrong,p,base)

def test_preparation_stdlibrary_seal_and_evaluation_policy_tamper(tmp_path,monkeypatch):
 import prepare_hybrid_width_learning as prep
 parent=tmp_path/'parent';parent.mkdir();(parent/'source').mkdir()
 prep.write(parent/'manifest.json',dict(source_sha256={}))
 prep.write(parent/'execution_status.json',dict(status='completed'));prep.write(parent/'summary.json',{})
 monkeypatch.setattr(prep,'PARENT',parent)
 ref=tmp_path/'ref.json';prep.write(ref,dict(status='PASS_CPU_INITIAL_REFERENCE',seed=145,parameters=PARAMETERS,
  snapshot_purpose_bytes=414257552,cuda_initialized=False,model_sha256='0'*64))
 b=tmp_path/'fresh';m=prep.prepare(b,ref);prep.verify(b)
 assert len(m['source_sha256'])==4 and 'manifold_hybrid_width2048_learning_seed145' in (b/'supervisor_template.conf').read_text()
 p=prep.read(b/'protocol.json');p['evaluation_calls']=[3500];prep.write(b/'protocol.json',p)
 altered=copy.deepcopy(m);altered['artifacts']['protocol.json']=prep.sha(b/'protocol.json');prep.write(b/'manifest.json',altered)
 with pytest.raises(ValueError,match='Undeclared'):prep.verify(b)
