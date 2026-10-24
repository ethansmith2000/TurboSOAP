"""Trial194: BF16 projection learning matched to completed native trial193/seed193."""
import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE),str(HERE/'cifar_support')]
import torch
import deep_llm_learning as native
from deep_llm_data import sha
from deep_llm_feasibility import write
from deep_projection_candidate_kernel import install_counted,CandidateObserver,projection_metrics
from tensor_ownership import TensorOwnership
from qr_reuse_profile_core import flags

BASELINE=Path('/workspace/optimizer_replay_results/llm_soap_deep_learning_seed193_frozen_20261019')
QUALITY_CALLS=(1000,3500,10000,20000,30000,35000)
MARGIN=.01


def quality_row(call,candidate,control):
    delta=candidate-control
    return dict(call=call,candidate_validation_ce=candidate,native_validation_ce=control,delta_ce=delta,
        margin_ce=MARGIN,postwarmup=call in QUALITY_CALLS,passes_panel=candidate<=control+MARGIN)


def recipe(control):
    return {**control,'matched_comparison':dict(trial=194,arms=['bf1620'],native_control_trial=193,
        matched_seed=193,disposable_calls=25,primary_calls=35000,
        scope='Candidate-owned BF16 projection learning matched to completed native193; QR20 unchanged')}


def verify_control(bundle):
    comparison=json.loads((bundle/'comparison.json').read_text())
    if comparison['trial']!=194 or comparison['matched_seed']!=193 or comparison['quality_margin_ce']!=MARGIN or comparison['quality_calls']!=list(QUALITY_CALLS):
        raise ValueError('Undeclared candidate/comparison criterion')
    for name,digest in comparison['native_files'].items():
        if sha(BASELINE/name)!=digest:raise RuntimeError('Native comparison evidence changed: '+name)
    config=json.loads((bundle/'config.json').read_text());old=json.loads((BASELINE/'config.json').read_text())
    if config!=recipe(old):raise ValueError('Undeclared model/data/schedule/resource change')
    if (bundle/'data_audit.json').read_bytes()!=(BASELINE/'data_audit.json').read_bytes():raise ValueError('Full prospective data hashes differ')
    if (bundle/'initial_reference.json').read_bytes()!=(BASELINE/'initial_reference.json').read_bytes():raise ValueError('Matched initial reference differs')
    return comparison


class CandidateRun(native.Run):
    def __init__(self,bundle):
        super().__init__(bundle);self.projection_counts={};self.projection_scope={'value':'training'}
        self.quality=[];self.adapter_qualification=None;self.finalized=False

    def setup(self):
        self.comparison=verify_control(self.bundle)
        self.control=json.loads((BASELINE/'summary.json').read_text())
        self.control_evaluations={row['call']:row for row in self.control['evaluations']}
        self.control_preflight=next(json.loads(line)['observation'] for line in (BASELINE/'events.jsonl').read_text().splitlines() if json.loads(line)['event']=='preflight')
        super().setup()

    def status(self,status='running',**extra):
        if status=='completed' and not self.finalized:status='finalizing'
        super().status(status,trial=194,arm='bf1620',matched_seed=193,**extra)

    def preflight(self):
        super().preflight()
        if self.preflight_record!=self.control_preflight:raise RuntimeError('Exact pre-update backward differs from completed native control')
        self.emit('native_preflight_matched',dict(exact_full_model_backward=True,matched_seed=193))

    @contextmanager
    def candidate_construction(self,phase):
        old_builder=native.build_optimizer;old_observer=native.LayerObserver
        counts=Counter();self.projection_counts[phase]=counts
        def builder(model,config,policy):
            opt=old_builder(model,config,policy)
            install_counted(opt.matrix,counts,self.projection_scope)
            return opt
        native.build_optimizer=builder;native.LayerObserver=CandidateObserver
        try:yield
        finally:
            native.build_optimizer=old_builder;native.LayerObserver=old_observer

    def trajectory(self,phase,calls):
        with self.candidate_construction(phase):super().trajectory(phase,calls)

    def check_dispatch_counts(self,phase,calls):
        counts=self.projection_counts[phase];refreshes=48*((calls-1)//20)
        mchecks=48 if phase=='disposable' else 384
        expected=dict(bf16_forward=48*(calls-1),bf16_back=48*(calls-1),highest_forward=refreshes,highest_back=refreshes+2*mchecks)
        actual={name:counts['training.'+name] for name in expected}
        if actual!=expected:raise RuntimeError('Actual projection/transport dispatch count differs: '+str((actual,expected)))
        return dict(training=actual,qualification={k:v for k,v in counts.items() if k.startswith('qualification.')})

    @torch.no_grad()
    def qualify_adapter(self):
        owner=TensorOwnership();before=self.endpoint(self.opt);rows=[]
        before_flags=flags();gradient_ids=[id(p.grad) for p in self.model.parameters()]
        old_scope=self.projection_scope['value'];self.projection_scope['value']='qualification'
        try:
            for group in self.opt.matrix.param_groups:
                for name,p in zip(group['param_names'],group['params']):
                    row=dict(parameter=name,shape=list(p.shape),**projection_metrics(self.opt.matrix,self.opt.matrix.state[p],group,p.grad,owner,check_live=True))
                    rows.append(row)
            if len(rows)!=48 or not all(row['passes_187_screen'] for row in rows):
                raise FloatingPointError('All48-matrix projection transfer screen failed')
        finally:
            self.projection_scope['value']=old_scope;gc.collect()
            release=owner.report() if owner._tensors else dict(all_registered_released=True,tracked_tensors=0);owner.close()
            after=self.endpoint(self.opt)
            unchanged=before==after and flags()==before_flags and gradient_ids==[id(p.grad) for p in self.model.parameters()]
            self.emit('adapter_qualification_restoration',dict(source_unchanged=unchanged,release=release))
            if not unchanged or not release['all_registered_released']:raise RuntimeError('Projection qualification mutated source or leaked workspace')
        self.adapter_qualification=dict(passed=True,rows=rows,source_unchanged=True,release=release)
        write(self.bundle/'adapter_qualification.json',self.adapter_qualification)
        self.emit('adapter_qualified',dict(matrices=48,largest_relative=max(row['relative'] for row in rows),smallest_cosine=min(row['cosine'] for row in rows),source_unchanged=True))

    def emit(self,event,row):
        if event=='started':row={**row,'trial':194,'source':'BF16 gradient/update projections; nativeQR20/storedFP32/highestMtransport; candidate-owned history','native_control':str(BASELINE)}
        if event=='phase_complete':
            if row['phase']=='disposable':self.qualify_adapter()
            row['projection_dispatch']=self.check_dispatch_counts(row['phase'],row['calls'])
        if event=='finished':event='base_runner_finished'
        super().emit(event,row)

    def evaluate(self,call):
        super().evaluate(call)
        latest=self.evaluations[-1];control=self.control_evaluations[call]
        if call==0 and latest['metrics']!=control['metrics']:raise RuntimeError('Exact initial evaluation differs from native control')
        comparison=quality_row(call,latest['metrics']['validation']['ce'],control['metrics']['validation']['ce'])
        self.quality.append(comparison);self.emit('native_quality_comparison',comparison)

    def run(self):
        result=super().run();verify_control(self.bundle)
        if self.adapter_qualification is None or not self.adapter_qualification['passed']:raise RuntimeError('Adapter never qualified')
        measured=result['phases'][1]['steady_means'];control=self.control['phases'][1]['steady_means']
        own_iteration=.95*measured['ordinary']['iteration_host_ms']+.05*measured['qr']['iteration_host_ms']
        native_iteration=.95*control['ordinary']['iteration_host_ms']+.05*control['qr']['iteration_host_ms']
        own_optimizer=.95*measured['ordinary']['optimizer_host_ms']+.05*measured['qr']['optimizer_host_ms']
        native_optimizer=.95*control['ordinary']['optimizer_host_ms']+.05*control['qr']['optimizer_host_ms']
        quality_pass=all(row['passes_panel'] for row in self.quality if row['postwarmup'])
        decision=dict(all_postwarmup_panels_pass=quality_pass,optimizer_lower=own_optimizer<native_optimizer,
            iteration_lower=own_iteration<native_iteration,candidate_iteration_ms=own_iteration,native_iteration_ms=native_iteration,
            candidate_optimizer_ms=own_optimizer,native_optimizer_ms=native_optimizer,
            provisional_screen_pass=quality_pass and own_optimizer<native_optimizer and own_iteration<native_iteration,
            interpretation='One matched initialization/data trajectory with sequential native control; exploratory cost/quality screen, not population equivalence, repeated wall timing or production speed.')
        result.update(trial=194,matched_seed=193,arm='bf1620',scope='Candidate-owned BF16 projection learning matched to native193; QR20 unchanged',
            quality_comparisons=self.quality,decision=decision,adapter_qualification_passed=True)
        write(self.bundle/'summary.json',result);write(self.bundle/'comparison_analysis.json',decision)
        self.finalized=True;native.Run.emit(self,'finished',dict(status='completed',trial=194,optimizer_calls=self.calls,decision=decision))
        self.status('completed');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);args=parser.parse_args()
    runtime=CandidateRun(args.bundle);runtime.status()
    try:runtime.run()
    except BaseException as exc:
        import traceback
        runtime.status('failed',error=repr(exc),traceback=traceback.format_exc());raise
