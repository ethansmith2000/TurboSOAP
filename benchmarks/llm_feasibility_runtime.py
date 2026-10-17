"""New causal-LM runtime; original model math and optimizer paths preserved."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,math,random
import numpy as np
import torch
from torch.nn import functional as F
from transformer import Transformer
from train_llm import PackedTokenDataset
from native_damped_kernel import PolicyRouter
from cifar_native_damped import Runtime as NativeRuntime
from matched_training_contract import RandomStreams,SharedInitialState,wire
from matched_training_runtime import streaming_digest,streaming_tensor_record
from llm_feasibility_contract import SNAPSHOT_CAP,SNAPSHOT_BYTES,SNAPSHOT_PURPOSE
from execution_fingerprints import backend_record

def build_model(c):
    model=Transformer(dim=c['dim'],depth=c['depth'],heads=c['heads'],ff_mult=2.,ff_hidden_dim=c['ff_hidden_dim'],
        vocab_size=c['vocab_size'],max_seq_len=c['sequence_length'],tie_embeddings=c['tie_embeddings'],
        use_rope=c['use_rope'],qk_norm=c['qk_norm'],norm_eps=c['norm_eps'],bias=c['bias'],dropout=c['dropout'],
        input_projection=c['input_projection'],fused_qkv=c['fused_qkv'],fused_swiglu=c['fused_swiglu'],gradient_checkpointing=c['gradient_checkpointing'])
    for module in model.modules():
        if module._non_persistent_buffers_set:
            if module._non_persistent_buffers_set!={'_rope_cos','_rope_sin'}:raise ValueError('Undeclared nonpersistent model buffer')
            module._non_persistent_buffers_set.clear()
    return model

class FlatOutput(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):
        logits=self.model(x)
        return logits.reshape(-1,logits.shape[-1])

def captured_backward(model,active,make_batch,streams,*,clip_norm,autocast):
    before=streaming_digest(streams.capture());indices,x,target=make_batch();model.zero_grad(set_to_none=True)
    result=dict(indices=streaming_tensor_record(indices),input=streaming_tensor_record(x),target=streaming_tensor_record(target),rng_before=before,rng_after_augmentation=streaming_digest(streams.capture()))
    with autocast():logits=active(x);loss=F.cross_entropy(logits.float(),target)
    loss.backward()
    result.update(output=streaming_tensor_record(logits),loss=streaming_tensor_record(loss),gradients_before_clip={n:streaming_tensor_record(p.grad) if p.grad is not None else None for n,p in model.named_parameters()})
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip_norm,foreach=True)
    result.update(clip_norm=streaming_tensor_record(norm),gradients_after_clip={n:streaming_tensor_record(p.grad) if p.grad is not None else None for n,p in model.named_parameters()},rng_after_backward=streaming_digest(streams.capture()))
    records=[result[k] for k in ('indices','input','target','output','loss','clip_norm')]+[r for k in ('gradients_before_clip','gradients_after_clip') for r in result[k].values() if r is not None]
    if not all(r['finite'] for r in records):raise FloatingPointError('Nonfinite LLM preflight input/output/gradient')
    return wire(result),loss.detach(),norm.detach()

class Runtime(NativeRuntime):
    def initialize(self):
        c=self.config
        torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);random.seed(c['seed']);np.random.seed(c['seed'])
        torch.set_float32_matmul_precision('high');torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        cache=torch.load(c['token_cache'],map_location='cpu',weights_only=True);audit=json.loads((self.bundle/'token_cache_audit.json').read_text())
        if cache['metadata']!=audit['metadata'] or int(cache['vocab_size'])!=c['vocab_size']:raise ValueError('Cached LLM data metadata changed')
        self.train_dataset=PackedTokenDataset(cache['train_tokens'],c['sequence_length']);self.validation_dataset=PackedTokenDataset(cache['validation_tokens'],c['sequence_length'])
        source_bytes=sum(v.numel()*v.element_size() for v in (cache['train_tokens'],cache['validation_tokens']))
        dataset_bytes=sum(v.numel()*v.element_size() for v in (self.train_dataset.tokens,self.validation_dataset.tokens))
        if source_bytes+dataset_bytes>self.protocol['llm_data']['data_cpu_bound_bytes']:raise MemoryError('Declared token-cache CPU bound')
        del cache
        self.model=build_model(c).cuda();self.active=torch.compile(FlatOutput(self.model),mode='default',fullgraph=True,dynamic=False)
        self.crop=torch.Generator(device='cuda').manual_seed(c['seed']+1001)
        self.strong=SimpleNamespace(rng=np.random.default_rng(c['seed']+3001),counts={})
        self.streams=RandomStreams(self.crop,self.strong,include_cuda=True)
        self.snapshot=SharedInitialState(self.model,self.streams,extra_workspace_bytes=4*2**20,ram_cap=SNAPSHOT_CAP)
        initial=json.loads((self.bundle/'initial_reference.json').read_text())
        if self.snapshot.snapshot_bytes!=SNAPSHOT_BYTES or self.snapshot.budget_bytes!=SNAPSHOT_PURPOSE or self.snapshot.initial_digest!=initial['model_sha256']:raise RuntimeError('LLM initial state/resource reference mismatch')
        self.schedule_steps=35000;self.warmup_steps=875
        count=max(self.protocol['steps_per_arm'],self.protocol['disposable_calls_per_arm']);needed=count*c['batch_size']
        if needed>len(self.train_dataset):raise ValueError('Declared no-wrap batch plan exceeds cached data')
        plan=np.random.default_rng(c['seed']+2001).permutation(len(self.train_dataset))[:needed].reshape(count,c['batch_size'])
        self.batches=[torch.from_numpy(v.copy()) for v in plan];self.last_labels=None
        self.emit('batch_plan',dict(count=len(plan),sha256=hashlib.sha256(plan.tobytes()).hexdigest(),tokens_per_update=c['batch_size']*c['sequence_length'],primary_tokens_per_arm=self.protocol['steps_per_arm']*c['batch_size']*c['sequence_length'],unique_block_indices=len(np.unique(plan)),sampler_wrap=False))
        self.kernel=PolicyRouter(self.emit,lambda:streaming_digest(self.streams.capture()));self.kernel.covariance_samples={}
        self.budget(resources=True)

    def make_batch(self,index):
        ids=self.batches[index];seq=self.config['sequence_length']
        offsets=ids[:,None]*seq+torch.arange(seq+1,dtype=torch.int64)[None,:]
        block=self.train_dataset.tokens[offsets].to('cuda',dtype=torch.int64)
        x=block[:,:-1].contiguous();target=block[:,1:].contiguous().reshape(-1)
        self.last_labels=target
        return ids.to('cuda'),x,target

    @torch.no_grad()
    def smoke_parity(self,x):
        with self.autocast():
            actual=self.active(x);expected=FlatOutput(self.model)(x)
            actual_loss=F.cross_entropy(actual.float(),self.last_labels);expected_loss=F.cross_entropy(expected.float(),self.last_labels)
        relative=float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-24))
        p=self.protocol['model_parity'];loss_pass=bool(torch.isclose(actual_loss,expected_loss,atol=p['loss_absolute'],rtol=p['loss_relative']))
        row=dict(actual_loss=float(actual_loss),eager_loss=float(expected_loss),logits_fro_relative=relative,loss_pass=loss_pass,passed=loss_pass and relative<=p['logits_fro_relative'] and math.isfinite(relative))
        if not row['passed']:raise FloatingPointError('LLM eager/compiled model parity failed: '+str(row))
        return row

    def before_smoke(self):
        from prepare_llm_feasibility import verify_backend
        record=backend_record();verify_backend(self.bundle,record);self.emit('backend_before_smoke',dict(record=record,passed=True))

    def after_measured(self):
        from prepare_llm_feasibility import verify_backend
        record=backend_record();verify_backend(self.bundle,record);self.emit('backend_after_measured',dict(record=record,passed=True))
        self.retain_code('after_measured');self.budget(resources=True)
