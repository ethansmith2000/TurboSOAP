"""Single managed process, shared compiled ViT, independently reset optimizer arms."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE/'cifar_support')]
import numpy as np
import torch
from cifar_optimizer_benchmark import support, StrongAugmentation
from matched_training_contract import RandomStreams, SharedInitialState
from matched_training_runtime import run_shared
from execution_fingerprints import backend_record, loaded_code
from launch_environment_contract import verify_environment_launch
from launch_environment_wire_contract import verify_backend_launch


def write(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


class GPURuntime:
    def __init__(self, bundle, started=None):
        self.bundle = bundle
        self.config = json.loads((bundle/'config.json').read_text())
        self.protocol = json.loads((bundle/'protocol.json').read_text())
        self.manifest = json.loads((bundle/'manifest.json').read_text())
        self.started = time.monotonic() if started is None else started
        self.original_affinities = {}; self.current_arm = None
        self.snapshot = None; self.model = None

    def initialize(self):
        bundle = self.bundle
        c = self.config
        torch.set_num_threads(4); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
        random.seed(c['seed']); np.random.seed(c['seed'])
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        raw, labels, _ = support.load_training_data(c)
        self.images = torch.from_numpy(raw).permute(0,3,1,2).contiguous().cuda()
        self.labels = torch.tensor(labels, device='cuda'); del raw
        split = json.loads((bundle/'split.json').read_text())
        prepared = json.loads((bundle/'data_prepared.json').read_text())
        self.ids = {k: torch.tensor(v, device='cuda') for k,v in split.items()}
        self.mean = torch.tensor(prepared['mean'], device='cuda')[None,:,None,None]
        self.std = torch.tensor(prepared['std'], device='cuda')[None,:,None,None]
        self.model = support.build_model(c).cuda()
        self.active = torch.compile(self.model, mode='default', fullgraph=True, dynamic=False)
        self.strong = StrongAugmentation(c, self.mean, self.std)
        self.crop = torch.Generator(device='cuda').manual_seed(c['seed']+1001)
        self.streams = RandomStreams(self.crop, self.strong, include_cuda=True)
        extra = 3*c['batch_size']*3*32*32*4 + 3*c['batch_size']*c['classes']*4
        self.snapshot = SharedInitialState(self.model, self.streams, extra_workspace_bytes=extra)
        per_epoch = len(split['train']) // c['batch_size']
        self.schedule_steps = per_epoch*c['epochs']; self.warmup_steps = per_epoch*c['warmup_epochs']
        assert self.schedule_steps == 35000 and self.warmup_steps == 875
        order_rng = np.random.default_rng(c['seed']+2001); batches = []
        for _ in range(math.ceil(self.protocol['steps_per_arm']/per_epoch)):
            order = order_rng.permutation(np.asarray(split['train']))
            batches.extend(order[:per_epoch*c['batch_size']].reshape(per_epoch,c['batch_size']))
        self.batches = [torch.tensor(v, device='cuda') for v in batches]
        self.last_labels = None
        self.emit('batch_plan', dict(count=len(batches),
            sha256=hashlib.sha256(np.asarray(batches).tobytes()).hexdigest(),
            gpu_bytes=sum(b.numel()*b.element_size() for b in self.batches)))
        self.budget(resources=True)

    def emit(self, event, payload):
        row = dict(event=event, all_in_seconds=time.monotonic()-self.started, payload=payload)
        with (self.bundle/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(event+' '+str(self.current_arm or ''), flush=True)
        if event == 'arm_result':
            write(self.bundle/('arm_'+payload['arm']+'.json'), payload)
        if event in ('started', 'preflight_passed', 'finished'):
            write(self.bundle/'execution_status.json', dict(status=payload['status'] if event == 'finished' else 'running',
                  event=event, all_in_seconds=row['all_in_seconds']))

    def budget(self, resources=False):
        if time.monotonic()-self.started >= self.protocol['bounds']['internal_wall_seconds']:
            raise RuntimeError('Continuous all-in wall bound exceeded')
        if resources:
            if torch.cuda.max_memory_allocated() >= self.protocol['bounds']['maximum_peak_allocated_bytes']:
                raise MemoryError('Allocated GPU memory bound exceeded')
            if sum(p.stat().st_size for p in self.bundle.rglob('*') if p.is_file()) >= self.protocol['bounds']['maximum_evidence_bytes']:
                raise OSError('Compact evidence byte bound exceeded')

    def make_batch(self, index):
        c = self.config; batch = self.batches[index]
        offsets = torch.randint(0,9,(c['batch_size'],2),device='cuda',generator=self.crop)
        flips = torch.rand(c['batch_size'],device='cuda',generator=self.crop)<.5
        y = self.labels[batch]
        x, target, _ = self.strong(self.images[batch], y, offsets, flips)
        self.last_labels = y
        return batch, x, target

    def autocast(self):
        return torch.autocast('cuda', dtype=torch.bfloat16)

    def marks(self):
        return [torch.cuda.Event(enable_timing=True) for _ in range(4)]

    def elapsed(self, marks):
        return [marks[0].elapsed_time(marks[1]), marks[1].elapsed_time(marks[2]),
                marks[2].elapsed_time(marks[3])]

    def synchronize(self):
        torch.cuda.synchronize()

    def smoke_parity(self, x):
        return dict(training=support.parity_check(self.model, self.active, x, self.last_labels),
                    evaluation=support.evaluation_parity_check(self.model, x, self.last_labels))

    def evaluate(self):
        return dict(validation=support.evaluate(self.active, self.images, self.labels,
            self.ids['validation'], self.mean, self.std, self.config['eval_batch_size']),
            train_panel=support.evaluate(self.active, self.images, self.labels,
            self.ids['train_panel'], self.mean, self.std, self.config['eval_batch_size']))

    def before_smoke(self):
        record = backend_record()
        self.emit('backend_before_smoke', dict(record=record, check=verify_backend_launch(
            self.bundle, record, 'backend_before_smoke')))

    def after_smoke(self):
        workers = [int(p.name) for p in Path('/proc/self/task').iterdir()
                   if (p/'comm').read_text().strip() == 'pt_autograd_0']
        if len(workers) != 1:
            raise RuntimeError('Exactly one existing autograd worker required for declared placement')
        for tid in [os.getpid(), workers[0]]:
            mask = os.sched_getaffinity(tid)
            if 116 not in mask:
                raise RuntimeError('Declared CPU116 unavailable; no alternative placement')
            self.original_affinities[tid] = mask
            os.sched_setaffinity(tid, {116})
            if os.sched_getaffinity(tid) != {116}:
                raise RuntimeError('Own thread placement failed')
        self.emit('placement', dict(main_cpu=116, autograd_cpu=116,
            original_masks={str(t): sorted(m) for t,m in self.original_affinities.items()}))
        self.retain_code('after_smoke')

    def retain_code(self, phase):
        records = loaded_code(); directory = self.bundle/'compiled_code'; directory.mkdir(exist_ok=True)
        for row in records:
            source = Path(row['path']).read_bytes()
            if hashlib.sha256(source).hexdigest() != row['source_sha256']:
                raise ValueError('Actually loaded code source changed')
            target = directory/(row['source_sha256']+'.py')
            if target.exists():
                if target.read_bytes() != source:
                    raise ValueError('Compiled provenance content mismatch')
            else:
                target.write_bytes(source)
        write(self.bundle/('code_'+phase+'.json'), records)

    def after_measured(self):
        record = backend_record()
        self.emit('backend_after_measured', dict(record=record, check=verify_backend_launch(
            self.bundle, record, 'backend_after_measured')))
        self.retain_code('after_measured')
        self.emit('resource_peak', dict(allocated_bytes=torch.cuda.max_memory_allocated(),
            snapshot_purpose_budget_bytes=self.snapshot.budget_bytes,
            optimizer_hash_chunk_bytes=2**20, all_in_seconds=time.monotonic()-self.started))

    def before_probe(self, label):
        self.budget()

    def before_training_step(self, arm, index):
        pass

    def before_optimizer_step(self, arm, index):
        pass

    def close(self):
        rows = []
        for tid, mask in self.original_affinities.items():
            try:
                os.sched_setaffinity(tid, mask)
                restored = os.sched_getaffinity(tid)
                rows.append(dict(tid=tid, expected=sorted(mask), actual=sorted(restored), passed=restored==mask))
            except OSError as exc:
                rows.append(dict(tid=tid, passed=False, error=repr(exc)))
        return dict(passed=all(r['passed'] for r in rows), threads=rows)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(); parser.add_argument('--bundle', type=Path, required=True)
    bundle = parser.parse_args().bundle.resolve()
    if HERE.parent != bundle/'source':
        raise RuntimeError('Execute only frozen runnable source')
    if (bundle/'execution_status.json').exists() or (bundle/'events.jsonl').exists():
        raise RuntimeError('One attempt only; existing evidence cannot be overwritten')
    write(bundle/'execution_status.json', dict(status='initializing', attempt=1))
    runtime = None
    try:
        # Environment gate and all hashes precede CUDA device initialization.
        from prepare_matched_training import verify_runnable
        verify_runnable(bundle)
        verify_environment_launch(bundle)
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one lifetime-claimed GPU required')
        runtime = GPURuntime(bundle, started=started)
        runtime.initialize()
        result = run_shared(runtime, runtime.protocol)
        write(bundle/'summary.json', result)
        if result['status'] != 'completed':
            raise RuntimeError('Matched comparison did not qualify; no retry')
    except BaseException as exc:
        write(bundle/'execution_status.json', dict(status='failed', error=repr(exc)))
        if runtime is not None:
            try:
                runtime.retain_code('failure')
            except BaseException as provenance_error:
                write(bundle/'code_retention_failure.json', dict(error=repr(provenance_error)))
        if runtime is not None and runtime.snapshot is not None and not runtime.snapshot.closed:
            runtime.model.zero_grad(set_to_none=True); runtime.snapshot.close(); runtime.close()
        raise


if __name__ == '__main__':
    main()
