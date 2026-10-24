"""Declared full-schedule data adapter; common eager/MATH model and native math."""
from llm_feasibility_runtime import *
from llm_sparse_screen_runtime import Runtime as PreviousRuntime
from llm_sparse_screen_kernel import Router
from llm_math_learning_contract import evaluation_ids,ids_sha
from stage_stability_data import repeated_plan
from llm_preserved_casts_runtime import global_guard

class Runtime(PreviousRuntime):
    def initialize(self):
        global_guard()
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
        self.model=build_model(c).cuda();self.active=FlatOutput(self.model)
        self.crop=torch.Generator(device='cuda').manual_seed(c['seed']+1001)
        self.strong=SimpleNamespace(rng=np.random.default_rng(c['seed']+3001),counts={})
        self.streams=RandomStreams(self.crop,self.strong,include_cuda=True)
        self.snapshot=SharedInitialState(self.model,self.streams,extra_workspace_bytes=4*2**20,ram_cap=SNAPSHOT_CAP)
        initial=json.loads((self.bundle/'initial_reference.json').read_text())
        if self.snapshot.snapshot_bytes!=SNAPSHOT_BYTES or self.snapshot.budget_bytes!=SNAPSHOT_PURPOSE or self.snapshot.initial_digest!=initial['model_sha256']:raise RuntimeError('LLM initial state/resource reference mismatch')
        self.schedule_steps=35000;self.warmup_steps=875
        count=max(self.protocol['steps_per_arm'],self.protocol['disposable_calls_per_arm']);needed=count*c['batch_size']
        plan,epochs=repeated_plan(c['seed'],len(self.train_dataset),c['batch_size'],count)
        self.batches=[torch.from_numpy(v.copy()) for v in plan];self.last_labels=None
        self.emit('batch_plan',dict(count=len(plan),sha256=hashlib.sha256(plan.tobytes()).hexdigest(),tokens_per_update=c['batch_size']*c['sequence_length'],primary_tokens_per_arm=self.protocol['steps_per_arm']*c['batch_size']*c['sequence_length'],unique_block_indices=len(np.unique(plan)),sampler_wrap=True,repeated_data=True,epochs=epochs))
        self.kernel=Router(self.emit,lambda:streaming_digest(self.streams.capture()))
        p=self.protocol['llm_evaluation']
        if len(self.train_dataset)!=p['train_blocks'] or len(self.validation_dataset)!=p['validation_blocks']:raise ValueError('Pinned evaluation coverage changed')
        ids=evaluation_ids(c['seed'],len(self.train_dataset),len(self.validation_dataset))
        if any(ids_sha(v)!=p['panels'][n]['ids_sha256'] for n,v in ids.items()):raise ValueError('Fixed evaluation IDs differ')
        self.ids={n:torch.tensor(v,dtype=torch.int64) for n,v in ids.items()}
        self.emit('evaluation_plan',dict(protocol=p,ids_sha256={n:ids_sha(v) for n,v in ids.items()},rng_untouched=True))
        self.emit('model_execution_contract',self.execution_contract());self.budget(resources=True)
