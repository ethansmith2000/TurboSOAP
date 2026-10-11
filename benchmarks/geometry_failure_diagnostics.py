"""Scalar-only geometry/refresh evidence. No model forwards or algorithm writes."""
import math
import torch
try:
    from .matched_training_runtime import optimizer_extras, assert_scalar_finite, tensor_values
except ImportError:
    from matched_training_runtime import optimizer_extras, assert_scalar_finite, tensor_values

BRACKET_CALLS = (251, 261, 271, 281)


def number(value):
    value = float(value)
    return value if math.isfinite(value) else {'nonfinite': 'nan' if math.isnan(value) else ('+inf' if value > 0 else '-inf')}


class GeometryFailure(FloatingPointError):
    def __init__(self, details):
        super().__init__('Predeclared basis Gram error exceeded')
        self.details = details  # JSON primitives only; never tensors/optimizer refs.


@torch.no_grad()
def compact_existing(value):
    if torch.is_tensor(value):
        t = value.detach()
        result = dict(shape=list(t.shape), dtype=str(t.dtype), elements=t.numel(), finite=bool(torch.isfinite(t).all()))
        if t.numel():
            result.update(minimum=number(t.min()), maximum=number(t.max()))
            if t.numel() == 1:
                result['scalar'] = number(t)
        return result
    if isinstance(value, dict):
        return {str(k): compact_existing(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact_existing(v) for v in value]
    if isinstance(value, float):
        return number(value)
    if value is None or type(value) in (str, int, bool):
        return value
    raise TypeError('Unsupported diagnostic state type: '+str(type(value)))


def refresh_branch(state, group, *, next_call=False):
    """Static branch eligibility, never claim a fallback actually occurred."""
    cold = not state or state.get('Q') is None
    internal = 0 if cold else int(state.get('step', 0)) + int(next_call)
    period = int(state.get('precondition_frequency', group['precondition_frequency']))
    warm = int(group.get('basis_warmup_steps', 0))
    method = group.get('basis_stage1_method')
    if method is None:
        return dict(internal_step=internal, cold=cold, policy='current SOAP', tracker_branch='not literal May')
    due = not cold and internal > 0 and internal % period == 0
    force = internal <= warm
    frequency = int(group['basis_metric_frequency'])
    refresh_index = max(1, (max(0, internal-warm)+max(1,period)-1)//max(1,period))
    metrics = due and method in ('tangent','brockett') and not force and (frequency <= 1 or (refresh_index-1)%frequency == 0)
    pipeline = due and not (force or method == 'qr' or (method in ('tangent','brockett') and not metrics)
        or (method in ('tangent','brockett') and float(group['basis_fallback_diag_err_threshold']) <= 0))
    return dict(internal_step=internal,cold=cold,refresh_due=due,force_warmup_fallback=due and force,
        stage1_method=method,metrics_enabled=metrics,pipeline_fallback_eligible=pipeline,
        basis_metric_frequency=frequency,refresh_index=refresh_index,
        scope='Static eligibility from existing state/settings; not observed fallback execution')


def parameter_context(model, opt):
    names={id(p): n for n,p in model.named_parameters()}
    groups={id(p): g for g in opt.matrix.param_groups for p in g['params']}
    return names, groups


def factor_identity(parameter, axis, q, state, names, groups, call, phase):
    group=groups.get(id(parameter), {})
    return dict(parameter_name=names.get(id(parameter),'<unnamed>'),factor_axis=axis,
        parameter_shape=list(parameter.shape),shape=list(q.shape),dtype=str(q.dtype),device=str(q.device),
        internal_step=int(state.get('step',0)),external_call=call,phase=phase,
        branch=refresh_branch(state,group,next_call=phase=='before_refresh') if group else None,
        controller_settings={k:compact_existing(v) for k,v in group.items() if k.startswith('basis_')},
        existing_controller_normalizer={k:compact_existing(v) for k,v in state.items()
            if k.startswith(('basis_', 'controller', 'normalizer'))})


@torch.no_grad()
def gram_row(q, context):
    q32=q.float()
    residual=q32.T @ q32 - torch.eye(len(q32), device=q.device)
    error=residual.norm()/len(q32)**.5  # Exact original highest-FP32 gate math.
    value=float(error)
    columns=residual.norm(dim=0)
    row=dict(context,finite_input=bool(torch.isfinite(q).all()),finite_gram=bool(torch.isfinite(residual).all()),
        normalized_gram_error=number(value),maximum_absolute_gram_error=number(residual.abs().max()),
        columns=dict(count=len(columns),minimum=number(columns.min()),maximum=number(columns.max()),
            mean=number(columns.mean()),worst_column=int(columns.argmax())))
    return value,row


@torch.no_grad()
def check_state(model, opt, maximum_gram_error=.05, *, external_call=None, phase='guard'):
    """Original finite checks, guard order/math/threshold; richer JSON evidence."""
    prior=torch.get_float32_matmul_precision();errors=[];tensors=0;skipped=0;rows=[]
    names,groups=parameter_context(model,opt)
    try:
        torch.set_float32_matmul_precision('highest')
        for tensor in model.state_dict().values():
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError('Nonfinite model parameter/buffer')
        for part in (opt.matrix,opt.auxiliary):
            assert_scalar_finite(optimizer_extras(part))
            for tensor in tensor_values(optimizer_extras(part)):
                tensors+=1
                if not bool(torch.isfinite(tensor).all()):
                    raise FloatingPointError('Nonfinite optimizer cache/diagnostic tensor')
            for state in part.state.values():
                assert_scalar_finite(state)
                for tensor in tensor_values(state):
                    tensors+=1
                    if not bool(torch.isfinite(tensor).all()):
                        raise FloatingPointError('Nonfinite optimizer state, including covariance/controller')
        for parameter,state in opt.matrix.state.items():
            factors=state.get('Q')
            if factors is None:continue
            for axis,q in enumerate(factors):
                if q is None or (isinstance(q,list) and not q):
                    skipped+=1;continue
                if not torch.is_tensor(q) or q.ndim!=2 or q.shape[0]!=q.shape[1] or not q.numel():
                    raise ValueError('Unsupported basis factor/sentinel')
                context=factor_identity(parameter,axis,q,state,names,groups,external_call,phase)
                value,row=gram_row(q,context);rows.append(row)
                if not math.isfinite(value) or value>=maximum_gram_error:
                    raise GeometryFailure(dict(failing_factor=row,previous_factors=rows[:-1],threshold=maximum_gram_error,
                        threshold_rule='finite normalized_gram_error strictly less than threshold',
                        existing_algorithm_diagnostics=compact_existing(optimizer_extras(opt.matrix))))
                errors.append(value)
        return dict(factor_count=len(errors),skipped_factors=skipped,checked_state_tensors=tensors,
            maximum_gram_error=max(errors,default=0.),factor_diagnostics=rows,
            existing_algorithm_diagnostics=compact_existing(optimizer_extras(opt.matrix)))
    finally:
        torch.set_float32_matmul_precision(prior)


@torch.no_grad()
def refresh_snapshot(model,opt,external_call,phase='before_refresh'):
    prior=torch.get_float32_matmul_precision();rows=[];names,groups=parameter_context(model,opt)
    try:
        torch.set_float32_matmul_precision('highest')
        for parameter,state in opt.matrix.state.items():
            for axis,q in enumerate(state.get('Q') or []):
                if q is None or (isinstance(q,list) and not q):continue
                _,row=gram_row(q,factor_identity(parameter,axis,q,state,names,groups,external_call,phase));rows.append(row)
        return dict(external_call=external_call,phase=phase,factors=rows,
            existing_algorithm_diagnostics=compact_existing(optimizer_extras(opt.matrix)),
            threshold_enforced=False,scope='Read-only pre-refresh observer; existing post-refresh guard remains authoritative')
    finally:
        torch.set_float32_matmul_precision(prior)


def collect_abort_audit(runtime):
    """Collect every independent component; audit failures never replace trigger."""
    result=dict(phase='abort',completed_after_measured_gate=False,components={})
    for name,method in [('resources','capture_abort_resources'),('backend','capture_abort_backend'),('code','capture_abort_code')]:
        try:
            result['components'][name]=dict(status='captured',value=getattr(runtime,method)())
        except BaseException as exc:
            result['components'][name]=dict(status='failed',error=repr(exc))
    return result
