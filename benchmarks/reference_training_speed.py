"""Bounded fixed-state execution profile; no learning or optimizer-policy change."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from reference_llama import ReferenceLlama
from reference_execution import ExecutionLlama,token_loss
from reference_blocked_soap import ReferenceBlockedSOAP
from reference_fineweb_data import FineWebWindows,content_hash
from reference_soap_cost import Run,digest,write
from tensor_ownership import TensorOwnership


def gradient_comparison(model,reference):
    delta=norm=actual=dot=0.;per_parameter=[]
    for name,p in model.named_parameters():
        a=p.grad.detach().cpu().double();b=reference[name].double()
        error=float((a-b).square().sum());rn=float(b.square().sum())
        an=float(a.square().sum());dp=float((a*b).sum())
        delta+=error;norm+=rn;actual+=an;dot+=dp
        per_parameter.append(dict(name=name,relative_error=math.sqrt(error/max(rn,1e-30)),reference_norm=math.sqrt(rn)))
    return dict(global_relative_error=math.sqrt(delta/norm),cosine=dot/math.sqrt(norm*actual),
                per_parameter=per_parameter)


class SpeedRun(Run):
    def verify(self):
        super().verify()
        p=json.loads((self.bundle/'data_plan.json').read_text())
        for name,h in p['files'].items():
            with Path(p['local_dir'],name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h

    def budget(self):
        super().budget()
        import psutil
        process=psutil.Process()
        rss=process.memory_info().rss
        for child in process.children(recursive=True):
            try:rss+=child.memory_info().rss
            except psutil.NoSuchProcess:pass
        assert rss<self.c['rss_bytes'],'aggregate process/compile-worker RSS budget'
        self.peak_rss=max(getattr(self,'peak_rss',0),rss)
        self.peak_gpu=max(getattr(self,'peak_gpu',0),torch.cuda.max_memory_allocated())

    def backward(self,model,windows,case,compiled_loss):
        model.zero_grad(set_to_none=True)
        data=torch.from_numpy(windows).to('cuda')
        total=torch.zeros((),device='cuda')
        count=128//case['physical_batch']
        loss_function=compiled_loss if case['compiled_loss'] else token_loss
        for tokens in data.split(case['physical_batch']):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(tokens[:,:-1]) if case['compiled_blocks'] else ReferenceLlama.forward(model,tokens[:,:-1])
                loss=loss_function(logits,tokens[:,1:])/count
            loss.backward();total+=loss.detach()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
        return float(total),float(norm)

    def profile(self,owner):
        plan=json.loads((self.bundle/'data_plan.json').read_text())
        ids=json.loads((self.bundle/'window_ids.json').read_text())['train'][:128]
        data=FineWebWindows([Path(plan['local_dir'],n) for n in plan['train_files']])
        windows=data.windows(ids)
        assert content_hash(windows)==plan['batch_content_sha256'][0]
        torch.manual_seed(203);model=ExecutionLlama(device='cuda')
        assert digest(model.named_parameters())==self.c['initial_parameters_sha256']
        generator=torch.Generator(device='cuda').manual_seed(203208)
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name=='readout' or name.endswith(('.output','.fc2')):p.normal_(std=.01,generator=generator)
        point=digest(model.named_parameters())
        opt=ReferenceBlockedSOAP(model.named_parameters(),compile_kernels=False)
        optimizer_before=digest((f'{i}.{n}',t) for i,g in enumerate(opt.groups) for n,t in g['state'].items())
        compiled_loss=torch.compile(token_loss,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True})
        reference=None;reference_loss=None;results=[]
        for case in self.c['cases']:
            model.configure_execution(case['compiled_blocks'],case['checkpoint_stride'])
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            case_start=time.perf_counter()
            self.emit('case_start',case=case)
            try:
                for _ in range(case['warmups']):
                    loss,norm=self.backward(model,windows,case,compiled_loss);self.budget()
                values=[]
                for repeat in range(case['repeats']):
                    graphs=torch._dynamo.utils.counters['stats']['unique_graphs']
                    torch.cuda.synchronize();start=time.perf_counter()
                    loss,norm=self.backward(model,windows,case,compiled_loss);torch.cuda.synchronize()
                    seconds=time.perf_counter()-start
                    new_graphs=torch._dynamo.utils.counters['stats']['unique_graphs']-graphs
                    assert new_graphs==0,'new timed graph'
                    assert math.isfinite(loss) and math.isfinite(norm)
                    values.append(seconds)
                    self.emit('timing',name=case['name'],repeat=repeat,backward_seconds=seconds,loss=loss,norm=norm,new_graphs=new_graphs)
                    self.budget()
                if reference is None:
                    reference={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
                    reference_loss=loss
                    assert all(bool(torch.isfinite(g).all()) and bool(g.count_nonzero()) for g in reference.values())
                    owner.watch_tree(reference,'CPU.full-gradient-reference')
                    self.emit('gradient_reference',parameters=len(reference),bytes=sum(t.numel()*t.element_size() for t in reference.values()),nonzero_all=True)
                comparison=gradient_comparison(model,reference)
                passed=(abs(loss-reference_loss)<=self.c['loss_margin'] and
                        comparison['global_relative_error']<=self.c['gradient_relative_margin'] and
                        comparison['cosine']>=self.c['gradient_cosine_min'])
                row=dict(name=case['name'],status='completed',physical_batch=case['physical_batch'],
                         checkpoint_stride=case['checkpoint_stride'],compiled_blocks=case['compiled_blocks'],
                         compiled_loss=case['compiled_loss'],median_backward_seconds=statistics.median(values),
                         measured_seconds=values,loss=loss,loss_difference=loss-reference_loss,
                         fidelity= comparison,numerical_screen_passed=passed,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         case_all_in_seconds=time.perf_counter()-case_start)
                results.append(row)
                self.emit('case_complete',**{k:v for k,v in row.items() if k!='fidelity'},
                          global_gradient_relative_error=comparison['global_relative_error'],gradient_cosine=comparison['cosine'])
            except torch.OutOfMemoryError as exc:
                results.append(dict(name=case['name'],status='out_of_memory',error=str(exc)))
                self.emit('case_out_of_memory',name=case['name'],error=str(exc))
                # A spent feasibility case, not retried or silently substituted.
                model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            assert digest(model.named_parameters())==point,'parameters changed during fixed-state profiling'
            assert opt.call==0 and float(opt.step_tensor)==0
            self.budget()
        assert optimizer_before==digest((f'{i}.{n}',t) for i,g in enumerate(opt.groups) for n,t in g['state'].items())
        owner.watch_tree(model.state_dict(),'model')
        owner.watch_tree([p.grad for p in model.parameters()],'last-gradients')
        for i,g in enumerate(opt.groups):owner.watch_tree(g['state'],f'optimizer.{i}')
        owner.watch(opt.step_tensor,'optimizer.step')
        return dict(results=results,diagnostic_parameter_sha256=point,
                    reference_scope='Deterministic nonzero diagnostic parameter point, not trained-state quality or a convergence test',
                    optimizer_state_resident_unchanged=True)

    def run(self):
        self.verify()
        assert os.environ.get('GPU_CLAIM_INDICES')=='7' and torch.cuda.device_count()==1
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-866cad81-dc9b-cc40-afb8-c387d3ffba5f'
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch._dynamo.config.recompile_limit=64
        torch.cuda.set_per_process_memory_fraction(self.c['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        self.emit('started',pid=os.getpid(),config=self.c,torch=torch.__version__,cuda=torch.version.cuda)
        owner=TensorOwnership();result=self.profile(owner)
        gc.collect();release=owner.report();owner.close()
        self.emit('release',**release);assert release['all_registered_released']
        self.verify();self.budget()
        result.update(status='completed',all_in_seconds=time.monotonic()-self.start,
                      peak_allocated_bytes=self.peak_gpu,peak_aggregate_rss_bytes=self.peak_rss,
                      compile_graphs=torch._dynamo.utils.counters['stats']['unique_graphs'],release=release)
        write(self.bundle/'summary.json',result);self.emit('finished',status='completed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);args=parser.parse_args()
    assert HERE.parent==args.bundle.resolve()/'source','frozen entry only'
    assert not (args.bundle/'execution_status.json').exists(),'one attempt only'
    write(args.bundle/'execution_status.json',dict(status='starting',trial=208,pid=os.getpid()))
    try:
        SpeedRun(args.bundle).run();write(args.bundle/'execution_status.json',dict(status='completed',trial=208,pid=os.getpid()))
    except BaseException as exc:
        import traceback
        write(args.bundle/'execution_status.json',dict(status='failed',trial=208,pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc()));raise
