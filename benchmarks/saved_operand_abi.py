"""Read-only verification of the original classifier operand's compiled ABI."""
import ast
import hashlib
from pathlib import Path
import torch


class OperandUnavailable(RuntimeError):pass


def classifier_abi(rows,batch,width,classes):
    modules={}
    for row in rows:
        p=Path(row['path']);data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['source_sha256']:raise OperandUnavailable('Loaded source hash changed')
        tree=ast.parse(data.decode());fn=next((n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='call'),None)
        if fn is not None:modules[row['source_sha256']]=fn
    forward=[]
    for h,fn in modules.items():
        returns=[n.value for n in ast.walk(fn) if isinstance(n,ast.Return) and isinstance(n.value,ast.Tuple)]
        if len(returns)!=1:continue
        ret=returns[0]
        if not ret.elts or not isinstance(ret.elts[0],ast.Name):continue
        logits=ret.elts[0].id
        alloc={n.targets[0].id:n.value for n in ast.walk(fn) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='empty_strided_cuda'}
        for call in ast.walk(fn):
            if not isinstance(call,ast.Call) or ast.unparse(call.func)!='extern_kernels.addmm' or len(call.args)!=3:continue
            if not any(k.arg=='out' and isinstance(k.value,ast.Name) and k.value.id==logits for k in call.keywords):continue
            if not isinstance(call.args[1],ast.Name):continue
            operand=call.args[1].id;allocation=alloc.get(operand)
            if allocation is None or len(allocation.args)!=3:continue
            if ast.literal_eval(allocation.args[0])!=(batch,width) or ast.literal_eval(allocation.args[1])!=(width,1) or ast.unparse(allocation.args[2])!='torch.bfloat16':continue
            logit_alloc=alloc.get(logits)
            if logit_alloc is None or ast.literal_eval(logit_alloc.args[0])!=(batch,classes):continue
            indexes=[i for i,n in enumerate(ret.elts) if isinstance(n,ast.Name) and n.id==operand]
            if len(indexes)!=1 or indexes[0]==0:continue
            forward.append(dict(forward_sha256=h,forward_operand=operand,saved_index=indexes[0]-1,saved_count=len(ret.elts)-1))
    pairs=[]
    for f in forward:
        for h,fn in modules.items():
            unpack=[n.targets[0] for n in ast.walk(fn) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Name) and n.value.id=='args' and len(n.targets)==1 and isinstance(n.targets[0],ast.Tuple)]
            if len(unpack)!=1 or len(unpack[0].elts)!=f['saved_count']+1:continue
            item=unpack[0].elts[f['saved_index']]
            if not isinstance(item,ast.Name):continue
            name=item.id;verified=False
            for call in ast.walk(fn):
                if not isinstance(call,ast.Call) or ast.unparse(call.func)!='extern_kernels.mm' or len(call.args)!=2 or not isinstance(call.args[1],ast.Name) or call.args[1].id!=name:continue
                lhs=call.args[0]
                if not isinstance(lhs,ast.Call) or ast.unparse(lhs.func)!='reinterpret_tensor' or len(lhs.args)!=4:continue
                if ast.literal_eval(lhs.args[1])!=(classes,batch):continue
                stride=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='assert_size_stride' and len(n.args)==3 and isinstance(n.args[0],ast.Name) and n.args[0].id==name]
                if len(stride)==1 and ast.literal_eval(stride[0].args[1])==(batch,width) and ast.literal_eval(stride[0].args[2])==(width,1):verified=True
            if verified:pairs.append(dict(**f,backward_sha256=h,backward_operand=name,shape=[batch,width],stride=[width,1],dtype='torch.bfloat16'))
    if len(pairs)!=1:raise OperandUnavailable(f'Expected one verified classifier forward/backward ABI, found {len(pairs)}')
    return pairs[0]


def select_operand(node,abi):
    if node is None or type(node).__name__!='CompiledFunctionBackward':raise OperandUnavailable('Output lacks direct CompiledFunctionBackward')
    try:
        checked=list(node.saved_tensors);unchecked=list(getattr(node,'_tensors_no_vc_check',[]))
    except (AttributeError,RuntimeError,TypeError) as exc:raise OperandUnavailable('Saved tensor access unavailable: '+str(exc)) from exc
    tensors=checked+unchecked
    if len(tensors)!=abi['saved_count']:raise OperandUnavailable('Saved tensor count disagrees with verified ABI')
    metadata=[dict(index=i,channel='version_checked' if i<len(checked) else 'no_version_check',dtype=str(t.dtype),shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset()) for i,t in enumerate(tensors)]
    candidates=[i for i,t in enumerate(tensors) if t.dtype==torch.bfloat16 and list(t.shape)==abi['shape'] and list(t.stride())==abi['stride'] and t.storage_offset()==0]
    if len(candidates)!=1:raise OperandUnavailable(f'Expected one classifier-shaped operand, found {len(candidates)}')
    if candidates[0]!=abi['saved_index']:raise OperandUnavailable('Unique operand index disagrees with verified backward ABI')
    return tensors[candidates[0]],dict(node_type=type(node).__name__,version_checked_count=len(checked),no_version_check_count=len(unchecked),selected_index=candidates[0],saved_metadata=metadata)
