"""Longer matched FineWeb comparison using qualified CE compilation and physical16."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from reference_blocked_soap import ReferenceBlockedSOAP
from reference_execution import token_loss
from reference_fineweb_data import FineWebWindows, content_hash, sequence_losses, evaluation_mode
from reference_soap_cost import Run, digest, write
from tensor_ownership import TensorOwnership


class FastLearningRun(Run):
    def __init__(self,bundle):
        super().__init__(bundle)
        self.compiled_ce=torch.compile(token_loss,fullgraph=True,dynamic=False,options={"emulate_precision_casts":True})

    def verify(self):
        super().verify()
        plan = json.loads((self.bundle/'data_plan.json').read_text())
        for name, expected in plan['files'].items():
            with Path(plan['local_dir'], name).open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == expected, name

    def setup_data(self):
        self.plan = json.loads((self.bundle/'data_plan.json').read_text())
        p = self.plan
        self.train = FineWebWindows([Path(p['local_dir'], n) for n in p['train_files']])
        self.validation = FineWebWindows([Path(p['local_dir'], p['val_file'])])
        self.ids = json.loads((self.bundle/'window_ids.json').read_text())
        train_ids = self.ids['train']
        assert len(train_ids) == 128*self.c['calls_per_arm'] == len(set(train_ids))
        assert min(train_ids) >= 0 and max(train_ids) < len(self.train)
        assert len(self.ids['validation']) == len(set(self.ids['validation'])) == 2048
        self.val_windows = self.validation.windows(self.ids['validation'])
        assert content_hash(self.val_windows) == p['validation_content_sha256']
        self.emit('data_verified', revision=p['revision'], train_unique_windows=len(train_ids),
                  train_targets_per_arm=len(train_ids)*1024, validation_targets=2048*1024,
                  train_shards=p['train_files'], validation_shard=p['val_file'])

    def backward(self, model, call):
        model.zero_grad(set_to_none=True)
        ids = self.ids['train'][(call-1)*128:call*128]
        windows = self.train.windows(ids)
        assert content_hash(windows) == self.plan['batch_content_sha256'][call-1]
        data = torch.from_numpy(windows).to('cuda')
        total = torch.zeros((), device='cuda')
        for tokens in data.split(self.c['physical_batch']):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(tokens[:, :-1])
                loss = self.compiled_ce(logits,tokens[:,1:]) / self.c['accumulation']
            loss.backward()
            total += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
        return float(total),float(norm)

    def state_digest(self, model, opt):
        def tensors():
            yield from model.named_parameters()
            for name, p in model.named_parameters():
                if p.grad is not None:yield 'gradient.'+name,p.grad
            for i, group in enumerate(opt.groups):
                for name, t in group['state'].items():yield f'state.{i}.{name}',t
            yield 'step',opt.step_tensor
        return digest(tensors())

    def evaluate(self, model, opt, arm, call, arm_start, training_seconds):
        start = time.perf_counter()
        checked = call in (50, self.c['calls_per_arm'])
        before = self.state_digest(model,opt) if checked else None
        grad_ids = [id(p.grad) for p in model.parameters()]
        counter = opt.call
        modes = [m.training for m in model.modules()]
        recompute = model.recompute
        losses = []
        with evaluation_mode(model):
            for windows in np.split(self.val_windows, 2048//self.c['eval_physical_batch']):
                tokens = torch.from_numpy(windows).to('cuda')
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast('cuda', dtype=torch.bfloat16):
                    value = sequence_losses(model(tokens[:,:-1]),tokens)
                losses.extend(value.double().cpu().tolist())
                self.budget()
        assert grad_ids == [id(p.grad) for p in model.parameters()] and opt.call == counter
        assert recompute == model.recompute and modes == [m.training for m in model.modules()]
        if checked:assert before == self.state_digest(model,opt), 'evaluation state mutation'
        assert len(losses) == 2048 and all(np.isfinite(losses))
        seconds = time.perf_counter()-start
        result = dict(arm=arm,call=call,mean_ce=statistics.mean(losses),
                      panel_ce=[statistics.mean(losses[:1024]),statistics.mean(losses[1024:])],
                      targets=2048*1024,evaluation_seconds=seconds,
                      arm_elapsed_seconds=time.perf_counter()-arm_start,training_seconds=training_seconds,
                      full_state_preservation_checked=checked, gradient_identity_preserved=True,
                      modes_counters_rng_restored=True, validation_content_sha256=self.plan['validation_content_sha256'])
        write(self.bundle/f'validation_{arm}_{call:04d}.json',dict(**result,per_sequence_ce=losses))
        self.emit('validation',**result)
        return result

    def arm(self, arm, owner):
        arm_start = time.perf_counter()
        model = self.model()
        initial = digest(model.named_parameters())
        assert initial == self.c['initial_parameters_sha256']
        self.emit('arm_initial',arm=arm,sha256=initial)
        if arm in ('qr','warm'):
            observations = []
            for _ in range(2):
                loss,norm = self.backward(model,1)
                observations.append(dict(loss=loss,norm=norm,gradient_sha256=digest(
                    (n,p.grad) for n,p in model.named_parameters())))
                self.budget()
            assert observations[0] == observations[1], 'initial real-data backward repeat'
            if hasattr(self,'initial_gradient_sha256'):
                assert observations[0]['gradient_sha256']==self.initial_gradient_sha256,'cross-arm initial gradient'
            self.initial_gradient_sha256=observations[0]['gradient_sha256']
            self.emit('real_backward_preflight',observations=observations,exact_initial_repeat=True)
            model.zero_grad(set_to_none=True)
        opt = ReferenceBlockedSOAP(model.named_parameters(),arm=arm,compile_kernels=True)
        evaluations = [self.evaluate(model,opt,arm,0,arm_start,0.)]
        training_seconds = 0.
        for call in range(1,self.c['calls_per_arm']+1):
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss,norm = self.backward(model,call)
            torch.cuda.synchronize()
            backward = (time.perf_counter()-start)*1000
            graphs_before = torch._dynamo.utils.counters['stats']['unique_graphs']
            begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            opt_start = time.perf_counter()
            begin.record();kind = opt.step();end.record();torch.cuda.synchronize()
            optimizer = (time.perf_counter()-opt_start)*1000
            whole = (time.perf_counter()-start)*1000
            training_seconds += whole/1000
            graphs = torch._dynamo.utils.counters['stats']['unique_graphs']-graphs_before
            diagnostic = call in self.c['guard_calls']
            assert np.isfinite(loss) and np.isfinite(norm), 'loss/norm guard'
            if call>100 and not diagnostic:assert graphs == 0,'new measured graph'
            self.rows.append(self.emit('update',arm=arm,call=call,kind=kind,loss=loss,unclipped_norm=norm,
                backward_host_ms=backward,optimizer_host_ms=optimizer,optimizer_event_ms=begin.elapsed_time(end),
                iteration_host_ms=whole,new_graphs=graphs,diagnostic=diagnostic,
                training_seconds=training_seconds,arm_elapsed_seconds=time.perf_counter()-arm_start,
                batch_content_sha256=self.plan['batch_content_sha256'][call-1]))
            if call==self.c['speed_gate_call']:
                observed=[r for r in self.rows if r['arm']==arm and 51<=r['call']<=call]
                median=statistics.median(r['backward_host_ms']/1000 for r in observed)
                assert median<=self.c['max_median_fb_seconds'],('actual training speed gate',median)
                ordinary=statistics.median(r['iteration_host_ms']/1000 for r in observed if r['kind']=='ordinary')
                self.emit('training_speed_gate',arm=arm,call=call,median_backward_seconds=median,median_ordinary_step_seconds=ordinary,passed=True)
            if diagnostic:self.guard(model,opt,arm,call)
            if call in self.c['eval_calls']:
                evaluations.append(self.evaluate(model,opt,arm,call,arm_start,training_seconds))
            write(self.bundle/'execution_status.json',dict(status='running',trial=self.c['trial'],
                  pid=os.getpid(),arm=arm,call=call,elapsed_seconds=time.monotonic()-self.start))
            self.budget()
        owner.watch_tree(model.state_dict(),'model')
        owner.watch_tree([p.grad for p in model.parameters()],'gradients')
        for i,group in enumerate(opt.groups):owner.watch_tree(group['state'],f'optimizer.{i}')
        owner.watch(opt.step_tensor,'optimizer.step')
        selected = [r for r in self.rows if r['arm']==arm and r['call']>100 and not r['diagnostic']]
        keys = ('backward_host_ms','optimizer_host_ms','optimizer_event_ms','iteration_host_ms')
        counts = Counter(r['kind'] for r in selected)
        means = {k:{key:statistics.mean(r[key] for r in selected if r['kind']==k) for key in keys} for k in counts}
        weights = {'ordinary':.9,'qr':.1} if arm=='qr' else {'ordinary':.9,'warm':.08,'qr':.02}
        costs = {key:sum(w*means[k][key] for k,w in weights.items()) for key in keys}
        result = dict(arm=arm,initial_parameters_sha256=initial,means=means,counts=dict(counts),costs=costs,
                      evaluations=evaluations,training_seconds=training_seconds,
                      arm_elapsed_seconds=time.perf_counter()-arm_start)
        self.emit('arm_complete',**result)
        return result

    def run(self):
        self.verify();self.setup_data()
        assert self.c['physical_batch']*self.c['accumulation'] == 128
        assert os.environ.get('GPU_CLAIM_INDICES') == '7' and torch.cuda.device_count() == 1
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == 'GPU-866cad81-dc9b-cc40-afb8-c387d3ffba5f'
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch._dynamo.config.recompile_limit = 64
        torch.cuda.set_per_process_memory_fraction(self.c['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        self.emit('started',pid=os.getpid(),config=self.c,torch=torch.__version__,cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(),attention='forced FLASH_ATTENTION')
        results = []
        for arm in self.c['arm_order']:
            owner = TensorOwnership()
            results.append(self.arm(arm,owner))
            gc.collect()
            release = owner.report();owner.close()
            self.emit('release',arm=arm,**release)
            assert release['all_registered_released']
            self.budget()
        assert results[0]['initial_parameters_sha256'] == results[1]['initial_parameters_sha256']
        self.verify()
        write(self.bundle/'summary.json',dict(status='completed',results=results,
              all_in_seconds=time.monotonic()-self.start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              compile_graphs=torch._dynamo.utils.counters['stats']['unique_graphs'],scope=self.c['scope']))
        self.emit('finished',status='completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    args = parser.parse_args()
    assert HERE.parent == args.bundle.resolve()/'source','frozen entry only'
    assert not (args.bundle/'execution_status.json').exists(),'one attempt only; never relaunch'
    write(args.bundle/'execution_status.json',dict(status='starting',trial=210,pid=os.getpid()))
    try:
        FastLearningRun(args.bundle).run()
        write(args.bundle/'execution_status.json',dict(status='completed',trial=210,pid=os.getpid()))
    except BaseException as exc:
        import traceback
        write(args.bundle/'execution_status.json',dict(status='failed',trial=210,pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc()))
        raise
