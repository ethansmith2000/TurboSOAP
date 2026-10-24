"""Managed synthetic full-step feasibility for an isolated reference adaptation."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from reference_llama import ReferenceLlama
from reference_blocked_soap import ReferenceBlockedSOAP
from tensor_ownership import TensorOwnership


def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def digest(named_tensors):
    h=hashlib.sha256()
    for name,tensor in named_tensors:
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Run:
    def __init__(self,bundle):
        self.bundle=bundle;self.c=json.loads((bundle/'config.json').read_text());self.start=time.monotonic();self.rows=[]

    def verify(self):
        for name,expected in json.loads((self.bundle/'manifest.json').read_text())['files'].items():
            with (self.bundle/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected,name
        for name,expected in json.loads((self.bundle/'reference_pins.json').read_text()).items():
            with Path('/workspace',name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected,name

    def emit(self,event,**kw):
        row=dict(event=event,elapsed_seconds=time.monotonic()-self.start,**kw)
        with (self.bundle/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if event!='update':print(json.dumps(row,allow_nan=False),flush=True)
        return row

    def budget(self):
        import psutil
        assert time.monotonic()-self.start<self.c['wall_seconds'],'wall budget'
        assert torch.cuda.max_memory_allocated()<self.c['gpu_bytes'],'GPU budget'
        assert psutil.Process().memory_info().rss<self.c['rss_bytes'],'RSS budget'
        assert shutil.disk_usage(self.bundle).free>5*2**30,'shared disk budget'
        if not hasattr(self,'last_disk') or time.monotonic()-self.last_disk>20:
            self.last_disk=time.monotonic()
            cache=sum(p.stat().st_size for p in (self.bundle/'compile_cache').rglob('*') if p.is_file())
            compact=sum(p.stat().st_size for p in self.bundle.rglob('*') if p.is_file() and 'compile_cache' not in p.parts)
            assert cache<self.c['cache_bytes'] and compact<100*2**20,'artifact budget'

    def model(self):
        torch.manual_seed(203)
        model=ReferenceLlama(device='cuda')
        assert sum(p.numel() for p in model.parameters())==185680896
        return model

    def backward(self,model,call):
        model.zero_grad(set_to_none=True)
        generator=torch.Generator(device='cuda').manual_seed(203000+call)
        data=torch.randint(50257,(128,1025),device='cuda',generator=generator)
        total=torch.zeros((),device='cuda')
        for tokens in data.split(self.c['physical_batch']):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(tokens[:,:1024])
                loss=F.cross_entropy(logits.float().reshape(-1,50257),tokens[:,1:].reshape(-1)) / self.c['accumulation']
            loss.backward();total+=loss.detach()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
        return float(total),float(norm)

    @torch.no_grad()
    def guard(self,model,opt,arm,call):
        previous=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
        maximum=0.;factor_count=0
        for group in opt.groups:
            state=group['state']
            assert all(bool(torch.isfinite(p).all()) for p in group['params'])
            assert all(bool(torch.isfinite(t).all()) for t in state.values())
            for name in ('lq','rq'):
                q=state[name]
                if not q.numel():continue
                error=q.transpose(-1,-2)@q
                error.diagonal(dim1=-2,dim2=-1).sub_(1)
                maximum=max(maximum,float(torch.linalg.vector_norm(error,dim=(-2,-1)).max()));factor_count+=len(q)
        torch.set_float32_matmul_precision(previous)
        assert factor_count==1026 and maximum<=.05,('Gram guard',maximum)
        self.emit('guard',arm=arm,call=call,absolute_gram_max=maximum,all_states_finite=True,factors=factor_count,passed=True)

    def preflight(self,owner):
        model=self.model();initial=digest(model.named_parameters())
        observations=[]
        for _ in range(2):
            start=time.perf_counter();loss,norm=self.backward(model,1);torch.cuda.synchronize()
            observations.append(dict(loss=loss,norm=norm,seconds=time.perf_counter()-start,
                                     gradient_sha256=digest((name,p.grad) for name,p in model.named_parameters())))
            self.budget()
        assert observations[0]['gradient_sha256']==observations[1]['gradient_sha256'],'initial Flash gradient repeat'
        assert observations[0]['loss']==observations[1]['loss'] and observations[0]['norm']==observations[1]['norm']
        self.emit('backward_preflight',initial_parameters_sha256=initial,observations=observations,exact_initial_repeat=True)
        opt=ReferenceBlockedSOAP(model.named_parameters(),compile_kernels=True)
        timings=[]
        for event,call,arm in [('eigh',1,'qr'),('ordinary',51,'qr'),('qr',60,'qr'),('warm',60,'warm')]:
            # Counter changes deliberately isolate compilation/shape feasibility;
            # this preflight does NOT represent a trained 60-update trajectory.
            opt.call=call-1;opt.step_tensor.fill_(call-1);opt.arm=arm
            start=time.perf_counter();actual=opt.step();torch.cuda.synchronize();seconds=time.perf_counter()-start
            assert actual==event
            self.guard(model,opt,arm,call)
            timings.append(dict(kind=event,call=call,seconds_including_compile=seconds))
            self.emit('optimizer_preflight',**timings[-1]);self.budget()
        # Two hot repetitions per event estimate a bounded main-job budget.
        for event,call,arm in [('ordinary',51,'qr'),('qr',60,'qr'),('warm',60,'warm')]:
            for repeat in range(2):
                opt.call=call-1;opt.step_tensor.fill_(call-1);opt.arm=arm
                start=time.perf_counter();opt.step();torch.cuda.synchronize()
                row=dict(kind=event,repeat=repeat,host_ms=(time.perf_counter()-start)*1000)
                timings.append(row);self.emit('hot_optimizer_preflight',**row);self.budget()
        owner.watch_tree(model.state_dict(),'model');owner.watch_tree([p.grad for p in model.parameters()],'gradients')
        for i,group in enumerate(opt.groups):owner.watch_tree(group['state'],f'optimizer.{i}')
        owner.watch(opt.step_tensor,'optimizer.step')
        return dict(initial_parameters_sha256=initial,backward=observations,timings=timings,
                    physical_batch=self.c['physical_batch'],accumulation=self.c['accumulation'],
                    scope='Synthetic shape/backend/port preflight; artificial counters after first EIGH, not a training trajectory')

    def arm(self,arm,owner):
        model=self.model();initial=digest(model.named_parameters());self.emit('arm_initial',arm=arm,sha256=initial)
        opt=ReferenceBlockedSOAP(model.named_parameters(),arm=arm,compile_kernels=True)
        for call in range(1,self.c['calls_per_arm']+1):
            torch.cuda.synchronize();start=time.perf_counter()
            loss,norm=self.backward(model,call);torch.cuda.synchronize();backward=(time.perf_counter()-start)*1000
            graphs_before=torch._dynamo.utils.counters['stats']['unique_graphs']
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            opt_start=time.perf_counter();begin.record();event=opt.step();end.record();torch.cuda.synchronize()
            optimizer=(time.perf_counter()-opt_start)*1000;whole=(time.perf_counter()-start)*1000
            graphs=torch._dynamo.utils.counters['stats']['unique_graphs']-graphs_before
            diagnostic=call in self.c['guard_calls']
            row=self.emit('update',arm=arm,call=call,kind=event,loss=loss,unclipped_norm=norm,
                          backward_host_ms=backward,optimizer_host_ms=optimizer,optimizer_event_ms=begin.elapsed_time(end),
                          iteration_host_ms=whole,new_graphs=graphs,diagnostic=diagnostic)
            assert torch.isfinite(torch.tensor([loss,norm])).all(),'loss/norm guard'
            # Call110 qualifies the first warm refresh after a QR repair;
            # its column-layout specialization is a declared diagnostic.
            if call>100 and not diagnostic:assert graphs==0,'new measured graph'
            self.rows.append(row)
            if diagnostic:self.guard(model,opt,arm,call)
            write(self.bundle/'execution_status.json',dict(status='running',trial=self.c['trial'],pid=os.getpid(),arm=arm,call=call,elapsed_seconds=time.monotonic()-self.start))
            self.budget()
        owner.watch_tree(model.state_dict(),'model');owner.watch_tree([p.grad for p in model.parameters()],'gradients')
        for i,group in enumerate(opt.groups):owner.watch_tree(group['state'],f'optimizer.{i}')
        owner.watch(opt.step_tensor,'optimizer.step')
        selected=[r for r in self.rows if r['arm']==arm and r['call']>100 and not r['diagnostic']]
        means={kind:{key:statistics.mean(r[key] for r in selected if r['kind']==kind) for key in ('backward_host_ms','optimizer_host_ms','optimizer_event_ms','iteration_host_ms')} for kind in ('ordinary','qr','warm') if any(r['kind']==kind for r in selected)}
        counts=Counter(r['kind'] for r in selected)
        weights={'ordinary':.9,'qr':.1} if arm=='qr' else {'ordinary':.9,'warm':.08,'qr':.02}
        costs={key:sum(weight*means[kind][key] for kind,weight in weights.items()) for key in ('backward_host_ms','optimizer_host_ms','optimizer_event_ms','iteration_host_ms')}
        self.emit('arm_complete',arm=arm,means=means,counts=dict(counts),costs=costs)
        return dict(arm=arm,initial_parameters_sha256=initial,means=means,counts=dict(counts),costs=costs)

    def run(self):
        self.verify();assert self.c['physical_batch']*self.c['accumulation']==128
        assert os.environ.get('GPU_CLAIM_INDICES')=='7' and torch.cuda.device_count()==1
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-866cad81-dc9b-cc40-afb8-c387d3ffba5f'
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch._dynamo.config.recompile_limit=64
        torch.cuda.set_per_process_memory_fraction(self.c['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        self.emit('started',pid=os.getpid(),config=self.c,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),attention='forced PyTorch FLASH_ATTENTION; no math fallback')
        results=[]
        for arm in (['preflight'] if self.c['mode']=='preflight' else ['qr','warm']):
            owner=TensorOwnership()
            result=self.preflight(owner) if arm=='preflight' else self.arm(arm,owner)
            gc.collect();release=owner.report();owner.close()
            self.emit('release',arm=arm,**release);assert release['all_registered_released']
            results.append(result)
            self.budget()
        if self.c['mode']=='cost':assert results[0]['initial_parameters_sha256']==results[1]['initial_parameters_sha256']
        self.verify()
        summary=dict(status='completed',results=results,all_in_seconds=time.monotonic()-self.start,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),compile_graphs=torch._dynamo.utils.counters['stats']['unique_graphs'],scope=self.c['scope'])
        write(self.bundle/'summary.json',summary);self.emit('finished',status='completed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);a=parser.parse_args()
    assert HERE.parent==a.bundle.resolve()/'source','frozen entry only'
    assert not (a.bundle/'execution_status.json').exists(),'one attempt only; never relaunch'
    trial=json.loads((a.bundle/'config.json').read_text())['trial']
    write(a.bundle/'execution_status.json',dict(status='running',trial=trial,pid=os.getpid()))
    try:
        Run(a.bundle).run();write(a.bundle/'execution_status.json',dict(status='completed',trial=trial,pid=os.getpid()))
    except BaseException as exc:
        import traceback
        write(a.bundle/'execution_status.json',dict(status='failed',trial=trial,pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc()));raise
