"""Same full-MV diagnostic plus bounded fixed-coordinate covariance samples."""
import gc,weakref,hashlib
import numpy as np
import torch
from native_damped_diagnostic import diagnose as original_diagnose
from native_damped_contract import ARMS,POLICIES
from hybrid_training_kernel import configure_hybrid
from cadence_training_contract import build_arm as baseline
from matched_training_runtime import streaming_digest

def sampled_diagnose(matrix,state,group,router,parameter,owners,arm):
    row=original_diagnose(matrix,state,group,router,parameter,owners)
    samples=[]
    for axis,cov in enumerate(state['GG']):
        index=torch.linspace(0,len(cov)-1,min(64,len(cov)),device=cov.device).long()
        sample=cov.index_select(0,index).index_select(1,index)
        owners.append(weakref.ref(sample))
        array=sample.detach().cpu().numpy().copy();key=(arm,parameter,axis)
        previous=router.covariance_samples.get(key)
        current=dict(axis=axis,dimension=len(cov),count=len(index),sha256=hashlib.sha256(array.tobytes()).hexdigest(),norm=float(np.linalg.norm(array)),
            previous_source_step=None,relative_sample_drift=None,cosine_to_previous=None)
        if previous is not None:
            age,old=previous
            current.update(previous_source_step=age,relative_sample_drift=float(np.linalg.norm(array-old)/max(float(np.linalg.norm(old)),1e-24)),
                cosine_to_previous=float((array*old).sum()/max(float(np.linalg.norm(array))*float(np.linalg.norm(old)),1e-24)))
        router.covariance_samples[key]=(int(state['step']),array)
        samples.append(current);del index,sample,previous,array
    row['fixed_coordinate_covariance_samples']=samples
    return row

def configure_llm_diagnostic(matrix,router,arm):
    original=type(matrix)._research_refresh;matrix_ref=weakref.ref(matrix)
    def refresh(state,group):
        self=matrix_ref()
        if self is None:raise RuntimeError('Missing live optimizer owner')
        if router.probe:
            name,p=next((name,p) for name,p in zip(group['param_names'],group['params']) if self.state[p] is state)
            before=streaming_digest(dict(state=state,parameter=p,gradient=p.grad));rng=router.rng_digest();owners=[]
            allocated=torch.cuda.memory_allocated()
            row=sampled_diagnose(self,state,group,router,name,owners,arm);row['source_arm']=arm
            gc.collect();alive=sum(ref() is not None for ref in owners)
            after=streaming_digest(dict(state=state,parameter=p,gradient=p.grad));rng_after=router.rng_digest()
            row.update(before_sha256=before,after_sha256=after,rng_before=rng,rng_after=rng_after,
                nonmutation_pass=before==after and rng==rng_after,registered_owners=len(owners),alive_after_release=alive,
                registered_release_pass=alive==0,conservative_peak_extra_bytes=max(0,torch.cuda.max_memory_allocated()-allocated))
            router.moment_probes.append(row);router.emit('full_moment_probe_before_gate',row)
            if not(row['nonmutation_pass'] and row['registered_release_pass'] and row['all_finite'] and row['physical_m_pass'] and row['conservative_peak_extra_bytes']<=16*2**30):
                raise FloatingPointError('LLM tracking transaction safety/release gate failed')
        return original(self,state,group)
    matrix._research_refresh=refresh

def build_arm(model,config,arm,router):
    if arm not in ARMS:raise ValueError('Only declared LLM arm')
    opt=baseline(model,config,'qr20' if arm=='qr20' else 'warm20')
    if arm in POLICIES:configure_hybrid(opt.matrix,router.kernels[arm])
    configure_llm_diagnostic(opt.matrix,router,arm)
    return opt
