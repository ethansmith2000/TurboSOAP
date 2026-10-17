"""Bounded fresh LLM pilot; token averaging/domain hypothesis, not attribution."""
from pathlib import Path
from native_damped_contract import ARMS,POLICIES,HORIZON,WARMUP
from native_damped_contract import protocol as previous_protocol
SEED=162
ORDER=('highest_residual20','damped20','qr20')
PARENTS=(Path('/workspace/optimizer_replay_results/cifar_soap_damped_confirmation_seed160_frozen_20261012'),)
CACHE=Path('/workspace/.optimizer_data_cache/openwebtext_gpt2_train40000000_val262144.pt')
SNAPSHOT_CAP=3*2**30
PARAMETERS=144875648
SNAPSHOT_BYTES=991339008
SNAPSHOT_PURPOSE=2230649344
DIMENSIONS={'2048':5,'4096':1,'6144':1,'8192':1}

def recipe(base=None):
    return dict(seed=SEED,model='Existing modern Transformer, opt-in persistent fixed RoPE tables for exact snapshot',
        dim=2048,depth=1,heads=32,ff_hidden_dim=4096,sequence_length=512,batch_size=16,vocab_size=50257,
        tie_embeddings=True,use_rope=True,qk_norm=True,norm_eps=1e-5,dropout=0.,gradient_checkpointing=False,
        bias=False,input_projection=False,fused_qkv=True,fused_swiglu=True,
        lr=.0005,matrix_lr=.0005,betas=[.9,.999],shampoo_beta=.99,weight_decay=.05,
        precondition_mode='all',covariance_compute_dtype='float32',transport_precision='highest',
        basis_ns_iterations=2,basis_rotation_cap=.5,clip_norm=1.,min_lr_ratio=.1,
        gradient_accumulation_steps=1,loss='Mean cross entropy over all8192 valid target positions',
        token_cache=str(CACHE),max_wall_seconds=3600,
        matched_comparison=dict(arms=list(ARMS),steps_per_arm=HORIZON,execution_order=list(ORDER),
            scope='Fresh causal-LM420 engineering/covariance/actual-MV feasibility; not converged quality or causal token-batch effect'))

def protocol():
    p=previous_protocol();p.update(seed=SEED,execution_order=list(ORDER))
    p['coverage']['dimensions_per_refresh']=DIMENSIONS.copy()
    p['expected']['hybrid_calls_by_dimension']={n:count*58 for n,count in DIMENSIONS.items()}
    p['expected']['policy_calls_by_dimension']={a:{n:count*29 for n,count in DIMENSIONS.items()} for a in POLICIES}
    p['bounds'].update(internal_wall_seconds=3600,external_watchdog_seconds=3660,maximum_initial_snapshot_purpose_cpu_bytes=SNAPSHOT_CAP)
    p['implementation'].update(entry='source/benchmarks/llm_feasibility.py',
        model='Unmodified Transformer source, single instance fixed RoPE buffers promoted to persistent without value changes; flattened-logit wrapper, no math change',
        snapshot_adapter='Explicit3GiB cap for tied embedding/head state duplicate plus original largest-tensor fingerprint workspace; defaults unchanged',
        backward_adapter='Same mean tokenCE/backward/clip; stream all numerical fingerprints in1MiB chunks',
        trainer_delta='New causal-LM runtime and cached batch plan; original native optimizer/hook/QR/M/V paths retained')
    p['artifacts']['initial_snapshot'].update(expected_bytes=SNAPSHOT_BYTES,purpose_expected_bytes=SNAPSHOT_PURPOSE,
        purpose='Exact fresh LLM smoke/disposable/preflight/primary resets, including tied state aliases and fixed RoPE buffers',location='CPU RAM only',retention='Release completion/abort')
    p['artifacts']['probe'].update(purpose='Read-only actual C/Q/M/V and same-coordinate64x64 covariance samples at21/281; all numerical/ownership gates unchanged')
    p['artifacts']['compiler_cache'].update(expected_bytes=300*2**20,retention='Through LLM learning/token-batch review; arithmetic provenance only')
    p['model_parity']=dict(loss_absolute=.001,loss_relative=1e-4,logits_fro_relative=.005,finite=True,
        reference='Same instance eager modern Transformer versus compiled flat output under identical BF16 autocast')
    p['llm_data']=dict(cache=str(CACHE),token_batch=8192,sequence_length=512,sequences=16,accumulation=1,
        packing='Original PackedTokenDataset: non-overlapping512-input blocks plus next target, original cachedEOS/split',
        batch_plan='Freshseed+2001 permutation of cached train blocks, fixed420batches; same plan perarm/reset; no train sampler wrap',
        training_tokens_per_primary_arm=HORIZON*8192,loss_reduction='mean, no padding/ignored targets in this cache',
        data_cpu_bound_bytes=512*2**20,validation='Original262144-token validation stream pinned; no pilot evaluations inside875LRwarmup')
    p['covariance_samples']=dict(calls=[21,281],indices=64,coordinate='Original fixed row/column coordinates, never eigenbasis',
        retained_cpu_bytes=3*8*64**2*4,retention='RAM-only sampledarrays across selected observations, release at completion; scalars/hashes retained',
        scope='Sample drift across260updates includingmodelchange/repair; not fullmatrix norm or isolated gradient noise; not an optimizer EMA')
    p['bound_rationale']='New2020-update causal-LM3arm420/245 pilot, samewidth2048 but fourdimension layouts2048/4096/6144/8192; tied50257-vocab embedding/head AdamW increasesuniqueparameter and resetRAM. Declared snapshot991339008/purpose2230649344B under3GiB; streamednumerical fingerprints1MiB avoidfull logits/gradientCPUcopies.3600/3660s,24GiBallocated,100MiBcompact/512MiBisolatedcache; transientfullM probesexpected7GiB/conservative16GiB. No checkpoints/retry or quality/crossdomainwall/token-noise-cause claim.'
    p['decision']='User token-averaging hypothesis steers next stage to matched LLM feasibility before ViT generalization. Holdcurrent betas/cadence/MV fixed; task/architecture/embeddingcoverage differ, so no causal attribution to tokencount alone. Largerpostwarmuplearning and same-model token-batch ablation remain separately bounded.'
    return p

def validate(config,declared,base=None):
    if config!=recipe() or declared!=protocol():raise ValueError('Undeclared LLM feasibility delta')
