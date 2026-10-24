"""Bounded synthetic blocked-factor qualification, not a full optimizer port."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import torch
from batched_soap_refresh import batched_qr_refresh, batched_warm_refresh
from fast_soap import fast_warm_refresh
from tensor_ownership import TensorOwnership


def config():
    return dict(trial=202, seed=202, dimension=512,
                cases=[dict(batch=b, padded=p) for b, p in ((1, False), (8, False), (64, False), (8, True))],
                layouts=['row', 'column'], history_refreshes=12, repair_every=5,
                timing_blocks=3, timing_repeats=10, absolute_gram_max=.05,
                normalized_gram_max=.005, own_scalar_basis_relative_max=1e-4,
                wall_seconds=900, external_seconds=960, gpu_bytes=24*2**30,
                rss_bytes=12*2**30, cache_bytes=512*2**20, evidence_bytes=100*2**20,
                minimum_free_disk_bytes=5*2**30,
                precision=dict(qr='highest', warm='high', metrics='highest'),
                scope='Synthetic post-initialization512 factor histories and compiled batched refresh latency; no model/full-step/learning/reference-port claim',
                artifacts='Compact source/config/events/timings/guards/code cache only; all tensors transient; no checkpoints, weights or tensor snapshots')


def write(p, x):
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n'); tmp.replace(p)


def verify(bundle):
    m = json.loads((bundle/'manifest.json').read_text())
    for name, digest in m['files'].items():
        assert hashlib.sha256((bundle/name).read_bytes()).hexdigest() == digest, name
    assert json.loads((bundle/'config.json').read_text()) == config()


class Run:
    def __init__(self, bundle):
        self.bundle = bundle; self.start = time.monotonic()
        self.guards = []; self.timings = []; self.releases = []

    def emit(self, event, **kw):
        row = dict(event=event, elapsed_seconds=time.monotonic()-self.start, **kw)
        with (self.bundle/'events.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)
        return row

    def budget(self):
        import psutil
        c = config()
        assert time.monotonic()-self.start < c['wall_seconds'], 'wall budget'
        assert torch.cuda.max_memory_allocated() < c['gpu_bytes'], 'GPU budget'
        assert psutil.Process().memory_info().rss < c['rss_bytes'], 'RSS budget'
        assert shutil.disk_usage(self.bundle).free > c['minimum_free_disk_bytes'], 'shared disk budget'
        cache = sum(p.stat().st_size for p in (self.bundle/'compile_cache').rglob('*') if p.is_file())
        compact = sum(p.stat().st_size for p in self.bundle.rglob('*') if p.is_file() and 'compile_cache' not in p.parts)
        assert cache < c['cache_bytes'] and compact < c['evidence_bytes'], 'artifact budget'

    def case(self, batch, padded, layout, owner):
        c = config(); n = c['dimension']; g = torch.Generator(device='cuda').manual_seed(202+batch)
        x = torch.randn(batch, n, 32, generator=g, device='cuda')
        diagonal = torch.linspace(.5, 1.5, n, device='cuda').expand(batch, -1).clone()
        active = torch.full((batch,), n, device='cuda', dtype=torch.int64)
        if padded:
            active = torch.tensor([512, 384, 257, 512, 384, 257, 512, 384], device='cuda')
            mask = torch.arange(n, device='cuda')[None, :] < active[:, None]
            x = x * mask[:, :, None]; diagonal = diagonal * mask
        covariance = torch.diag_embed(diagonal) + .02*(x@x.transpose(-1, -2))/32
        q0 = torch.eye(n, device='cuda').expand(batch, -1, -1).clone()
        if layout == 'column': q0 = q0.transpose(-1, -2)
        nu0 = torch.rand(batch, n, n, generator=g, device='cuda')
        epsilon = torch.full((batch,), 1e-10, device='cuda')
        raw_m = torch.randn(batch, n, 5, generator=g, device='cuda')
        raw_before = raw_m.clone()
        for z, label in ((covariance,'covariance'), (q0,'initial_basis'), (nu0,'variance'), (epsilon,'epsilon'), (raw_m,'raw_momentum'), (raw_before,'raw_momentum_copy')):
            owner.watch(z, label)
        for arm in ('qr', 'warm'):
            torch.set_float32_matmul_precision(c['precision'][arm])
            kernel = self.kernels[arm]
            qout, vout = kernel(covariance, q0, nu0, epsilon)
            torch.cuda.synchronize()
            owner.watch_tree((qout,vout), 'compile_output.'+arm)
        q = q0; nu = nu0
        for step in range(1, 13):
            arm = 'qr' if step % 5 == 0 else 'warm'
            torch.set_float32_matmul_precision(c['precision'][arm])
            current = covariance * (1 + step*1e-4)
            versions = [z._version for z in (current,q,nu,epsilon,raw_m)]
            rng = torch.cuda.get_rng_state().clone()
            result, new_nu = self.kernels[arm](current,q,nu,epsilon)
            own_error = 0.
            if arm == 'warm':
                scalar = torch.stack([fast_warm_refresh(current[i]+epsilon[i]*torch.eye(n, device='cuda'),q[i]) for i in range(batch)])
                torch.set_float32_matmul_precision('highest')
                own_error = float((torch.linalg.vector_norm(result-scalar,dim=(-2,-1))/torch.linalg.vector_norm(scalar,dim=(-2,-1))).max())
                assert new_nu is nu, 'warm variance must be unchanged'
                owner.watch(scalar, 'scalar_reference')
            torch.set_float32_matmul_precision('highest')
            residual = result.transpose(-1,-2)@result
            residual.diagonal(dim1=-2, dim2=-1).sub_(1)
            absolute = torch.linalg.vector_norm(residual, dim=(-2,-1))
            maximum = float(absolute.max()); normalized = maximum/math.sqrt(n)
            finite = bool(torch.isfinite(result).all() and torch.isfinite(new_nu).all())
            source = versions == [z._version for z in (current,q,nu,epsilon,raw_m)]
            rng_ok = torch.equal(rng,torch.cuda.get_rng_state())
            raw_ok = torch.equal(raw_m,raw_before)
            passed = finite and source and rng_ok and raw_ok and maximum <= .05 and normalized <= .005 and own_error <= 1e-4
            row = self.emit('geometry',batch=batch,padded=padded,layout=layout,step=step,arm=arm,
                            factors=batch,absolute_gram_max=maximum,normalized_gram_max=normalized,
                            own_scalar_basis_relative=own_error,source_versions_unchanged=source,
                            cuda_rng_unchanged=rng_ok,raw_momentum_unchanged=raw_ok,finite=finite,passed=passed)
            self.guards.append(row)
            assert passed, 'qualification guard failed'
            owner.watch_tree((current,result,new_nu,residual,absolute), 'history')
            q,nu = result,new_nu
            self.budget()
        # Stable inputs and balanced order; switch global precision outside the
        # timed region. This is refresh latency, not a complete optimizer cost.
        for block in range(3):
            for arm in (('qr','warm') if block % 2 == 0 else ('warm','qr')):
                torch.set_float32_matmul_precision(c['precision'][arm])
                before = torch._dynamo.utils.counters['stats']['unique_graphs']
                torch.cuda.synchronize()
                begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start = time.perf_counter(); begin.record()
                for _ in range(10): qout,vout = self.kernels[arm](covariance,q0,nu0,epsilon)
                end.record(); torch.cuda.synchronize()
                host = (time.perf_counter()-start)*1000/10
                graphs = torch._dynamo.utils.counters['stats']['unique_graphs']-before
                assert graphs == 0, 'new graph in timing phase'
                row = self.emit('timing',batch=batch,padded=padded,layout=layout,block=block,arm=arm,
                                repeats=10,host_ms=host,event_ms=begin.elapsed_time(end)/10,new_graphs=graphs)
                self.timings.append(row)
                owner.watch_tree((qout,vout), 'timing_output')
                self.budget()
        self.emit('case_complete',batch=batch,padded=padded,layout=layout)

    @torch.no_grad()
    def run(self):
        verify(self.bundle)
        assert os.environ.get('GPU_CLAIM_INDICES') == '7' and torch.cuda.device_count()==1
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == 'GPU-866cad81-dc9b-cc40-afb8-c387d3ffba5f'
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.cuda.set_per_process_memory_fraction(config()['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        self.kernels={name:torch.compile(fn,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True}) for name,fn in [('qr',batched_qr_refresh),('warm',batched_warm_refresh)]}
        self.emit('started',pid=os.getpid(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),config=config())
        for case in config()['cases']:
            for layout in config()['layouts']:
                owner = TensorOwnership()
                self.case(**case,layout=layout,owner=owner)
                gc.collect(); release = owner.report(); owner.close()
                self.releases.append(self.emit('release',**case,layout=layout,**release))
                assert release['all_registered_released'], 'registered storage retained'
        verify(self.bundle)
        summary=dict(status='completed',trial=202,guards=self.guards,timings=self.timings,releases=self.releases,
                     all_in_seconds=time.monotonic()-self.start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     compile_graphs=torch._dynamo.utils.counters['stats']['unique_graphs'],scope=config()['scope'])
        write(self.bundle/'summary.json',summary); self.emit('finished',status='completed')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--bundle',type=Path,required=True); a=parser.parse_args()
    assert HERE.parent == a.bundle.resolve()/'source', 'frozen entry only'
    assert not (a.bundle/'execution_status.json').exists(), 'one attempt only; never relaunch'
    write(a.bundle/'execution_status.json',dict(status='running',trial=202,pid=os.getpid()))
    try:
        Run(a.bundle).run()
        write(a.bundle/'execution_status.json',dict(status='completed',trial=202,pid=os.getpid()))
    except BaseException as exc:
        import traceback
        write(a.bundle/'execution_status.json',dict(status='failed',trial=202,pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc()))
        raise
