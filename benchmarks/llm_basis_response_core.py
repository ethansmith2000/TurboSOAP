"""RAM-only fixed-state gradient aggregation, covariance sketches and lifetime checks."""
from contextlib import contextmanager
import gc,hashlib,json,math,time,weakref
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from matched_training_runtime import streaming_digest,failure_record
from tensor_ownership import TensorOwnership
from llm_basis_response_atoms import inherited_responses,prospective_responses

@contextmanager
def fp32_math_control():
 precision=torch.get_float32_matmul_precision();tf32=torch.backends.cuda.matmul.allow_tf32
 try:
  torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
  with sdpa_kernel([SDPBackend.MATH]),torch.autocast('cuda',enabled=False):yield
 finally:
  torch.set_float32_matmul_precision(precision);torch.backends.cuda.matmul.allow_tf32=tf32

def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def indices(n,device):return torch.linspace(0,n-1,min(64,n),device=device).long()
def scalar_array(a):
 off=a.copy();i=np.arange(min(a.shape));off[i,i]=0
 return dict(sha256=array_sha(a),sample_fro=float(np.linalg.norm(a)),sample_trace=float(np.trace(a)),sample_offdiag_fro=float(np.linalg.norm(off)))

class CrossPoolStats:
 """Online CPU statistics of fixed physical-coordinate samples, not full matrices."""
 def __init__(self,cap):self.cap=cap;self.cache={};self.refs=[];self.peak_bytes=0
 def add(self,key,value):
  a=np.asarray(value,dtype=np.float64)
  if key not in self.cache:
   mean=np.zeros_like(a);m2=np.zeros_like(a);self.refs.extend((weakref.ref(mean),weakref.ref(m2)));self.cache[key]=[0,mean,m2]
  row=self.cache[key];row[0]+=1;delta=a-row[1];row[1]+=delta/row[0];row[2]+=delta*(a-row[1])
  size=sum(v.nbytes for _,mean,m2 in self.cache.values() for v in (mean,m2));self.peak_bytes=max(self.peak_bytes,size)
  if size>self.cap:raise MemoryError('Declared CPU sketch-statistics cap exceeded')
 def finish(self):
  rows=[]
  for key,(count,mean,m2) in sorted(self.cache.items()):
   if count!=4:raise RuntimeError('Four disjoint pool observations required for every sketch')
   signal=float(np.square(mean).sum());variance=float(m2.sum()/(count-1))
   rows.append(dict(key=list(key),pools=count,mean_sha256=array_sha(mean),mean_sample_energy=signal,
    sample_variation_trace=variance,variation_over_signal=variance/signal if signal>0 else None))
  return rows
 def close(self):
  self.cache.clear();gc.collect();released=all(ref() is None for ref in self.refs);self.refs.clear();return released

def global_norm(values,divisor=1):
 norms=torch.stack([torch.linalg.vector_norm(v.detach(),2) for v in values]);norm=float(torch.linalg.vector_norm(norms,2))/divisor
 if not math.isfinite(norm):raise FloatingPointError('Nonfinite all-parameter gradient norm')
 return norm

def compare_gradients(sums,reference,names,matrix_names,factor,gates,owner):
 total=[0.,0.,0.,0.];per_matrix={}
 for name,summed,ref in zip(names,sums,reference):
  if not summed.is_contiguous() or not ref.is_contiguous():raise RuntimeError('Declared contiguous gradient workspace contract changed')
  a=summed.view(-1);b=ref.view(-1);q=[0.,0.,0.,0.]
  for start in range(0,a.numel(),2**18):
   u=a[start:start+2**18].double()*factor;v=b[start:start+2**18].double();d=u-v
   owner.watch(u,'comparison.expected');owner.watch(v,'comparison.reference');owner.watch(d,'comparison.delta')
   numbers=(float(d.square().sum()),float(u.square().sum()),float(v.square().sum()),float((u*v).sum()))
   q=[x+y for x,y in zip(q,numbers)];del u,v,d
  total=[x+y for x,y in zip(total,q)]
  def result(q):
   relative=math.sqrt(q[0]/max(q[2],1e-30));cosine=q[3]/math.sqrt(q[1]*q[2]) if q[1]>0 and q[2]>0 else (1. if q[1]==q[2]==0 else 0.)
   passed=math.isfinite(relative) and math.isfinite(cosine) and relative<=gates['gradient_relative'] and cosine>=gates['gradient_cosine']
   return dict(relative=relative,cosine=cosine,passed=passed)
  if name in matrix_names:per_matrix[name]=result(q)
 overall=result(total);overall['matrix_parameters']=per_matrix
 if not overall['passed'] or not all(v['passed'] for v in per_matrix.values()):raise FloatingPointError('Declared gradient aggregation fidelity gate failed: '+str(overall))
 return overall

def batch(runtime,ids):
 device=next(runtime.model.parameters()).device;seq=runtime.config['sequence_length'];ids=torch.tensor(ids,dtype=torch.int64)
 offsets=ids[:,None]*seq+torch.arange(seq+1,dtype=torch.int64)[None,:]
 blocks=runtime.train_dataset.tokens[offsets].to(device=device,dtype=torch.int64)
 return blocks[:,:-1].contiguous(),blocks[:,1:].contiguous().reshape(-1)

def gradients(runtime,ids,params,owner):
 with torch.enable_grad():
  x,target=batch(runtime,ids);logits=runtime.active(x);loss=F.cross_entropy(logits.float(),target)
  grads=torch.autograd.grad(loss,params,create_graph=False,retain_graph=False,allow_unused=False)
  value=float(loss.detach());del x,target,logits,loss
 owner.watch_tree(grads,'autograd.gradients')
 if not math.isfinite(value):raise FloatingPointError('Nonfinite FP32 diagnostic loss')
 if any(p.grad is not None and g.untyped_storage().data_ptr()==p.grad.untyped_storage().data_ptr() for p,g in zip(params,grads)):raise RuntimeError('Diagnostic gradients alias borrowed training gradients')
 return grads,value

def capture_group(runtime,all_sums,variations,n,matrix_names,stats,pool,owner):
 clip=runtime.config['clip_norm'];norm=global_norm(all_sums.values(),n)
 coefficient=float(torch.tensor(min(1.,clip/(norm+1e-6)),dtype=torch.float32));rows=[]
 for name in matrix_names:
  summed=all_sums[name];m2=variations[name];raw_norm=float(torch.linalg.vector_norm(summed,2))/n
  variation=float(m2.sum())/(n-1) if n>1 else None
  ri=indices(summed.shape[0],summed.device);ci=indices(summed.shape[1],summed.device)
  tile=summed.index_select(0,ri).index_select(1,ci)/n;owner.watch(tile,'gradient.sample')
  sample=tile.cpu().double().numpy().copy();stats.add((n,name,'gradient'),sample)
  entry=dict(parameter=name,shape=list(summed.shape),raw_gradient_norm=raw_norm,raw_gradient_energy=raw_norm**2,
   within_pool_sequence_variation_trace=variation,iid_sequence_variance_over_n_trace=variation/n if variation is not None else None,
   clipped_gradient_energy=raw_norm**2*coefficient**2,gradient_sample=scalar_array(sample),covariance_samples=[])
  for axis in (0,1):
   block=summed.index_select(axis,ri if axis==0 else ci);block.div_(n);owner.watch(block,'covariance.contraction_input')
   covariance=block@block.T if axis==0 else block.T@block;owner.watch(covariance,'covariance.sample')
   raw=covariance.cpu().double().numpy().copy();clipped=raw*coefficient**2
   stats.add((n,name,'covariance_raw',axis),raw);stats.add((n,name,'covariance_clipped',axis),clipped)
   entry['covariance_samples'].append(dict(axis=axis,dimension=summed.shape[axis],sample_size=min(64,summed.shape[axis]),raw=scalar_array(raw),clipped=scalar_array(clipped)))
   del block,covariance,raw,clipped
  rows.append(entry);del ri,ci,tile,sample,summed,m2
 return dict(pool=pool,sequences=n,valid_tokens=n*runtime.config['sequence_length'],all_parameter_gradient_norm=norm,global_clip_coefficient=coefficient,matrices=rows)

def run_pools(runtime,opt,owner,stats):
 spec=runtime.protocol['sequence_diagnostic'];gates=spec['gates'];named=list(runtime.model.named_parameters());names=[n for n,p in named];params=tuple(p for n,p in named)
 matrix_names=tuple(name for group in opt.matrix.param_groups for name in group['param_names'])
 if len(matrix_names)!=4 or any(p.dtype!=torch.float32 for p in params):raise RuntimeError('Original four SOAP weights and FP32 parameters required')
 parameter_bytes=sum(p.numel()*p.element_size() for p in params);matrix_bytes=sum(p.numel()*p.element_size() for name,p in named if name in matrix_names);largest=max(p.numel()*p.element_size() for name,p in named if name in matrix_names)
 factor_bytes=max(v['dimension']**2*4 for v in runtime.protocol['basis_response']['factors'])
 purpose_bytes=3*parameter_bytes+matrix_bytes+2*largest+12*2**20+28*factor_bytes
 if purpose_bytes>spec['memory']['maximum_tensor_purpose_bytes']:raise MemoryError('Declared diagnostic gradient tensor-purpose bound exceeded')
 groups=[];references=[];backwards=0
 for pool_id,pool in enumerate(spec['pools']):
  sums={n:torch.zeros_like(p,memory_format=torch.contiguous_format) for n,p in named};variations={n:torch.zeros_like(sums[n]) for n in matrix_names}
  owner.watch_tree(sums,'all_parameter.sum');owner.watch_tree(variations,'matrix.variation');losses=[];grads=None
  try:
   for n,idx in enumerate(pool['ids'],1):
    runtime.budget();grads,loss=gradients(runtime,[idx],params,owner);backwards+=1;losses.append(loss)
    if n>1:
     for name,g in zip(names,grads):
      if name in variations:
       previous=sums[name]/(n-1);delta=g-previous;owner.watch(previous,'variation.previous_mean');owner.watch(delta,'variation.delta')
       variations[name].addcmul_(delta,delta,value=(n-1)/n);del previous,delta
    with torch.no_grad():
     for name,g in zip(names,grads):sums[name].add_(g)
    g=None;grads=None
    if n in spec['sizes']:
     row=capture_group(runtime,sums,variations,n,matrix_names,stats,pool_id,owner);row['mean_sequence_loss']=math.fsum(losses)/n
     row['basis_responses']=prospective_responses(runtime,opt,sums,n,row['global_clip_coefficient'],stats,pool_id,owner)
     groups.append(row);runtime.emit('sequence_group',dict(pool=pool_id,sequences=n,loss=row['mean_sequence_loss'],gradient_norm=row['all_parameter_gradient_norm'],clip=row['global_clip_coefficient']))
    runtime.budget(resources=True)
   grads,full_loss=gradients(runtime,pool['ids'],params,owner);backwards+=1
   norm=global_norm(grads);mean_norm=global_norm(sums.values(),16);norm_relative=abs(mean_norm-norm)/max(norm,1e-30)
   mean_loss=math.fsum(losses)/16;loss_error=abs(mean_loss-full_loss)
   raw=compare_gradients(tuple(sums.values()),grads,names,matrix_names,1/16,gates,owner)
   coefficient=float(torch.tensor(min(1.,runtime.config['clip_norm']/(mean_norm+1e-6)),dtype=torch.float32))
   with torch.no_grad():reference_clip_norm=float(torch.nn.utils.clip_grad_norm_([SimpleNamespace(grad=g) for g in grads],runtime.config['clip_norm'],error_if_nonfinite=True,foreach=True))
   clipped=compare_gradients(tuple(sums.values()),grads,names,matrix_names,coefficient/16,gates,owner)
   passed=norm_relative<=gates['norm_relative'] and loss_error<=gates['loss_absolute']+gates['loss_relative']*abs(full_loss)
   record=dict(pool=pool_id,ids_sha256=pool['ids_sha256'],full_batch_loss=full_loss,mean_sequence_loss=mean_loss,loss_absolute_error=loss_error,
    raw_gradient=raw,clipped_gradient=clipped,mean_gradient_norm=mean_norm,full_batch_gradient_norm=norm,reference_clip_norm=reference_clip_norm,norm_relative=norm_relative,passed=passed)
   references.append(record);runtime.emit('sequence_reference',record)
   if not passed:raise FloatingPointError('FP32 aggregation loss/global-norm gate failed')
  finally:
   grads=None;g=None;sums.clear();variations.clear();gc.collect()
 return dict(groups=groups,references=references,backward_calls=backwards,cross_pool_sketch_statistics=stats.finish(),cpu_statistics_peak_bytes=stats.peak_bytes,tensor_purpose_bytes=purpose_bytes)

def diagnose(runtime,opt):
 from llm_math_confirmation import endpoint,compiler_stats
 started=time.monotonic();spec=runtime.protocol['sequence_diagnostic'];before=endpoint(runtime,opt);grad_before=streaming_digest([p.grad for p in runtime.model.parameters()]);grad_ids=[id(p.grad) for p in runtime.model.parameters()]
 modules=list(runtime.model.modules());flags=[m.training for m in modules];rng=runtime.streams.capture();compiler=compiler_stats()
 owner=TensorOwnership();sentinel=torch.empty(0);owner.watch(sentinel,'diagnostic.lifetime_sentinel');del sentinel
 stats=CrossPoolStats(spec['memory']['cpu_sketch_statistics_cap_bytes']);error=None;result=None
 try:
  runtime.model.eval()
  with fp32_math_control():
   inherited=inherited_responses(runtime,opt,owner)
   result=run_pools(runtime,opt,owner,stats)
   result['inherited_basis_responses']=inherited
   result['basis_policy_calls']=sum(len(v['policies']) for v in inherited)+sum(len(v['policies']) for row in result['groups'] for v in row['basis_responses'])
 except BaseException as exc:
  error=failure_record(exc);exc.__traceback__=None
 finally:
  runtime.streams.restore(rng)
  for module,flag in zip(modules,flags):module.training=flag
  cpu_released=stats.close();gc.collect();release=owner.report();owner.close()
  after=endpoint(runtime,opt);grad_after=streaming_digest([p.grad for p in runtime.model.parameters()])
  nonmutation=before==after and grad_before==grad_after and grad_ids==[id(p.grad) for p in runtime.model.parameters()] and flags==[m.training for m in modules] and compiler==compiler_stats()
  cleanup=dict(registered_release=release,cpu_statistics_released=cpu_released,state_rng_flags_gradients_and_identity_pass=nonmutation,before=before,after=after,gradient_before_sha256=grad_before,gradient_after_sha256=grad_after,seconds=time.monotonic()-started)
  runtime.emit('sequence_diagnostic_cleanup',cleanup)
 if not release['all_registered_released'] or not cpu_released or not nonmutation:raise RuntimeError('Sequence diagnostic nonmutation/release failed')
 if error is not None:raise RuntimeError('Sequence diagnostic failed: '+str(error))
 if result['backward_calls']!=spec['maximum_backward_calls'] or len(result['groups'])!=runtime.protocol['basis_response']['groups'] or len(result['references'])!=4:raise RuntimeError('Sequence diagnostic coverage differs')
 if result['basis_policy_calls']!=runtime.protocol['basis_response']['maximum_policy_calls'] or len(result['cross_pool_sketch_statistics'])!=runtime.protocol['basis_response']['maximum_cross_pool_statistics']:raise RuntimeError('Basis response/statistics coverage differs')
 result.update(cleanup=cleanup,optimizer_updates=0,passed=True);runtime.budget(resources=True);return result
