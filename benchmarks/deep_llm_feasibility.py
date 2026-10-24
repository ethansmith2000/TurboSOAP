"""Single native QR20 feasibility attempt on a deeper LM and the shared OWT cache."""
import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE / 'cifar_support')]
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from cifar_optimizers import build_optimizer, apply_lr_multiplier
from cifar100_qualification import lr_at
from cadence_training_contract import validate_counters
from deep_llm_data import CACHE, plan, sha, shifted_loss, tokens_from_rows
from execution_fingerprints import backend_record
from llm_feasibility_runtime import build_model
from llm_sparse_screen_kernel import common_geometry_guard, qualify_actual_m
from matched_training_runtime import streaming_digest
from tensor_ownership import TensorOwnership


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def event_record(event, elapsed, row):
    return dict(event=event, elapsed_seconds=elapsed, **row)


def verify(bundle):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(bundle / name) != digest:
            raise RuntimeError('Frozen input/source changed: ' + name)
    audit = json.loads((bundle / 'data_audit.json').read_text())
    for name, record in json.loads((bundle / 'data_stat.json').read_text()).items():
        stat = (CACHE / name).stat()
        if [stat.st_size, stat.st_mtime_ns, stat.st_ino] != record:
            raise RuntimeError('Shared input file changed: ' + name)
    return manifest, audit


@contextmanager
def forward_context():
    # Retain the existing exact-repeatable model reference for the first depth test.
    with sdpa_kernel([SDPBackend.MATH]), torch.autocast('cuda', dtype=torch.bfloat16):
        yield


def run(bundle):
    started = time.monotonic()
    manifest, audit = verify(bundle)
    c = json.loads((bundle / 'config.json').read_text())
    def emit(event, row):
        with (bundle / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event_record(event,time.monotonic()-started,row), allow_nan=False) + '\n')
        if event in ('started', 'preflight', 'evaluation', 'phase_complete', 'finished'):
            print(json.dumps(dict(event=event, **row)), flush=True)
    def budget():
        if time.monotonic()-started > c['max_wall_seconds']:
            raise TimeoutError('Declared internal wall limit')
        if torch.cuda.max_memory_allocated() > c['maximum_allocated_bytes']:
            raise MemoryError('Declared GPU allocation limit')
        import psutil
        if psutil.Process().memory_info().rss > 12*2**30:
            raise MemoryError('Declared process RSS limit')
    launch = json.loads((bundle / 'launch.json').read_text())
    if os.environ.get('GPU_CLAIM_INDICES') != '7' or os.environ.get('CUDA_VISIBLE_DEVICES') != launch['uuid']:
        raise RuntimeError('Missing physical single-GPU lifetime claim')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one visible GPU')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_per_process_memory_fraction(c['maximum_allocated_bytes']/torch.cuda.get_device_properties(0).total_memory)
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    model = build_model(c)
    initial = streaming_digest(model.state_dict())
    reference = json.loads((bundle / 'initial_reference.json').read_text())
    if initial != reference['model_sha256'] or sum(p.numel() for p in model.parameters()) != 109396992:
        raise RuntimeError('CPU model initialization differs from frozen reference')
    snapshot = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    snapshot_owner = TensorOwnership(); snapshot_owner.watch_tree(snapshot, 'initial_cpu_snapshot')
    model = model.cuda()
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
    def reset():
        model.load_state_dict(snapshot); model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1]); model.train()
        if streaming_digest(model.state_dict()) != initial:
            raise RuntimeError('Full initial reset differs')
    import datasets
    data = datasets.load_from_disk(str(CACHE), keep_in_memory=False)
    for split in ('train', 'validation'):
        if len(data[split]) != audit['splits'][split]['rows'] or data[split]._fingerprint != audit['splits'][split]['fingerprint']:
            raise RuntimeError('Loaded shared dataset differs from audit')
    order, train_ids = plan(len(data['train']), c['seed'], c['schedule_steps'], c['batch_size'])
    validation_ids = np.random.default_rng(c['seed']+4001).choice(len(data['validation']), 64, replace=False)
    write(bundle/'data_plan.json', dict(training_plan_sha256=streaming_digest(order), shape=list(order.shape),
        total_unique_blocks=int(order.size), no_wrap=True, train_panel_ids=train_ids.tolist(),
        validation_ids=validation_ids.tolist(), prefix_125_sha256=streaming_digest(order[:125]),
        presented_tokens_per_call=c['batch_size']*1024, target_tokens_per_call=c['batch_size']*1023,
        boundary='Each cached row predicts its own positions1..1023 from positions0..1022; no target crosses a row boundary; no EOS insertion'))
    backend = backend_record(); write(bundle/'backend_before.json', backend)
    emit('started', dict(pid=os.getpid(), parameters=reference['parameters'], gpu=torch.cuda.get_device_name(0),
        cpu_snapshot_bytes=reference['snapshot_bytes'], primary_calls=c['primary_calls'], disposable_calls=25))

    def backward(call):
        cpu_tokens = tokens_from_rows(data['train'], order[call-1])
        if hashlib.sha256(cpu_tokens.numpy().tobytes()).hexdigest()!=audit['consumed_inputs']['train_batches_sha256'][call-1]:
            raise RuntimeError('Actual training tokens differ from pinned batch')
        tokens = cpu_tokens.cuda()
        model.zero_grad(set_to_none=True)
        with forward_context():
            logits = model(tokens); loss = shifted_loss(logits, tokens)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['clip_norm'], foreach=True)
        return loss.detach(), norm.detach()

    preflight = []
    for repeat in range(2):
        reset(); loss, norm = backward(1)
        record = dict(loss=float(loss), gradient_norm=float(norm), gradients_sha256=streaming_digest([p.grad for p in model.parameters()]), model_sha256=streaming_digest(model.state_dict()))
        if record['model_sha256'] != initial or (preflight and record != preflight[0]):
            raise RuntimeError('Exact repeated initial model/backward gate failed')
        preflight.append(record); del loss, norm
        budget()
    emit('preflight', dict(exact_repeated_backward=True, optimizer_calls=0, observation=preflight[0]))
    model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def evaluate(opt, call, phase):
        before = streaming_digest(dict(model=model.state_dict(), optimizer=opt.state_dict(),
            gradients=[p.grad for p in model.parameters()], rng=(torch.get_rng_state(),torch.cuda.get_rng_state())))
        flags = [(module, module.training) for module in model.modules()]
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        rows = {}
        try:
            model.eval()
            for name, split, ids in [('train_probe','train',train_ids), ('validation','validation',validation_ids)]:
                total = 0.; count = 0
                for start in range(0,len(ids),c['batch_size']):
                    cpu_tokens = tokens_from_rows(data[split],ids[start:start+c['batch_size']])
                    key='train_probe_batches_sha256' if split=='train' else 'validation_batches_sha256'
                    if hashlib.sha256(cpu_tokens.numpy().tobytes()).hexdigest()!=audit['consumed_inputs'][key][start//c['batch_size']]:
                        raise RuntimeError('Actual evaluation tokens differ from pinned batch')
                    tokens = cpu_tokens.cuda()
                    with forward_context():
                        logits = model(tokens); loss = shifted_loss(logits,tokens)
                    n = tokens.shape[0]*1023
                    total += float(loss)*n; count += n
                    del tokens, logits, loss
                rows[name] = dict(ce=total/count, valid_tokens=count, blocks=len(ids))
                if not math.isfinite(rows[name]['ce']):raise FloatingPointError('Nonfinite evaluation')
        finally:
            for module, flag in flags:module.training=flag
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
        after = streaming_digest(dict(model=model.state_dict(), optimizer=opt.state_dict(),
            gradients=[p.grad for p in model.parameters()], rng=(torch.get_rng_state(),torch.cuda.get_rng_state())))
        if before != after:raise RuntimeError('Evaluation changed source state/gradients/RNG')
        row = dict(phase=phase,call=call,metrics=rows,restoration_passed=True)
        emit('evaluation', row); budget(); return row

    phases = []; disposable_prefix = None; evaluations = []; releases = []
    for phase, calls in [('disposable',25),('primary',c['primary_calls'])]:
        reset(); opt = build_optimizer(model,c,'qr20')
        owner = TensorOwnership(); samples=[]; guards=[]
        router = SimpleNamespace(rng_digest=lambda:streaming_digest((torch.get_rng_state(),torch.cuda.get_rng_state())),
                                 emit=emit,moments=[])
        if phase == 'disposable':qualify_actual_m(opt.matrix,router,'qr20')
        if phase == 'primary':evaluations.append(evaluate(opt,0,phase))
        try:
            for call in range(1,calls+1):
                budget(); torch.cuda.synchronize(); iteration_started=time.perf_counter()
                apply_lr_multiplier(opt,lr_at(call,c['schedule_steps'],c['warmup_steps'],1.,c['min_lr_ratio']))
                loss,norm=backward(call)
                if call==1:
                    observation=dict(loss=float(loss),gradient_norm=float(norm),gradients_sha256=streaming_digest([p.grad for p in model.parameters()]),model_sha256=streaming_digest(model.state_dict()))
                    if observation != preflight[0]:raise RuntimeError('First actual backward differs from preflight')
                torch.cuda.synchronize(); marks=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                optimizer_started=time.perf_counter(); marks[0].record(); opt.step(); marks[1].record(); torch.cuda.synchronize()
                optimizer_ms=1000*(time.perf_counter()-optimizer_started)
                iteration_ms=1000*(time.perf_counter()-iteration_started)
                sample=dict(phase=phase,call=call,kind='cold' if call==1 else ('qr' if (call-1)%20==0 else 'ordinary'),
                    loss=float(loss),gradient_norm=float(norm),lr=opt.matrix.param_groups[0]['lr'],
                    optimizer_host_ms=optimizer_ms,optimizer_event_ms=marks[0].elapsed_time(marks[1]),iteration_host_ms=iteration_ms)
                if not all(math.isfinite(v) for v in sample.values() if isinstance(v,float)):
                    raise FloatingPointError('Nonfinite actual update/timing')
                del loss,norm,marks
                samples.append(sample);emit('update',sample)
                if call==1 or (call-1)%20==0 or call==calls:
                    check_started=time.perf_counter()
                    factors=common_geometry_guard(opt,model)
                    counts=validate_counters(opt,call)
                    if len(factors)!=96 or counts['parameter_count']!=48:raise RuntimeError('Deeper SOAP coverage differs')
                    guards.append(dict(call=call,factors=factors,counts=counts,seconds=time.perf_counter()-check_started))
                    emit('guard',dict(phase=phase,**guards[-1]))
                if call==25:
                    prefix=dict(loss_norm=[(r['loss'],r['gradient_norm']) for r in samples],
                        state_sha256=streaming_digest(dict(model=model.state_dict(),optimizer=opt.state_dict(),
                            rng=(torch.get_rng_state(),torch.cuda.get_rng_state()))))
                    if phase=='disposable':disposable_prefix=prefix
                    elif prefix!=disposable_prefix:raise RuntimeError('Exact disposable/primary 25-call prefix differs')
                if phase=='primary' and call in (25,calls):evaluations.append(evaluate(opt,call,phase))
            if phase=='disposable' and (len(router.moments)!=48 or not all(row['passed'] for row in router.moments)):
                raise RuntimeError('Missing actual first-refresh momentum qualification')
            rows=[r for r in samples if r['call']>50]
            means={kind:dict(samples=len(selected),**{key:statistics.mean(r[key] for r in selected)
                for key in ['optimizer_host_ms','optimizer_event_ms','iteration_host_ms']})
                for kind in ('ordinary','qr') if (selected:=[r for r in rows if r['kind']==kind])}
            record=dict(phase=phase,calls=calls,samples=samples,guards=guards,actual_m=router.moments,steady_means=means)
            phases.append(record);emit('phase_complete',dict(phase=phase,calls=calls,steady_means=means))
        finally:
            owner.watch_tree(opt.state_dict(),'optimizer_state'); owner.watch_tree([p.grad for p in model.parameters()],'gradients')
            opt.matrix.state.clear(); opt.auxiliary.state.clear(); model.zero_grad(set_to_none=True)
            del opt,router; gc.collect();torch.cuda.synchronize()
            release=owner.report();owner.close();releases.append(release);emit('release',dict(phase=phase,**release))
            if not release['all_registered_released']:raise RuntimeError('Optimizer/gradient storage retained')
    verify(bundle); write(bundle/'backend_after.json',backend_record())
    model_owner=TensorOwnership();model_owner.watch_tree(model.state_dict(),'model')
    del model; snapshot.clear(); del snapshot; gc.collect();torch.cuda.synchronize()
    snapshot_release=snapshot_owner.report();snapshot_owner.close()
    model_release=model_owner.report();model_owner.close()
    if not snapshot_release['all_registered_released'] or not model_release['all_registered_released']:
        raise RuntimeError('Model or initial snapshot storage retained')
    result=dict(status='completed',optimizer_calls=150,primary_calls=c['primary_calls'],
        preflight_backward_calls=2,all_in_seconds=time.monotonic()-started,phases=phases,evaluations=evaluations,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),releases=releases,
        snapshot_release=snapshot_release,model_release=model_release,exact_25_prefix=True,
        scope='Native deeper-model feasibility only; short early learning observations do not qualify candidate learning, convergence or production throughput')
    write(bundle/'summary.json',result)
    emit('finished',{key:result[key] for key in ['status','optimizer_calls','all_in_seconds','peak_allocated_bytes','exact_25_prefix']})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);args=parser.parse_args()
    write(args.bundle/'execution_status.json',dict(status='running',pid=os.getpid()))
    try:
        result=run(args.bundle)
    except BaseException as exc:
        import traceback
        write(args.bundle/'execution_status.json',dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc()))
        raise
    else:write(args.bundle/'execution_status.json',dict(status='completed',pid=os.getpid(),optimizer_calls=result['optimizer_calls']))
