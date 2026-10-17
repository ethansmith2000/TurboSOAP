"""Opt-in BF16 ordinary projections; highest FP32 transport stays native."""
import weakref
import torch
from soap import SOAP

def project(tensor,basis,back=False):
    if tensor.ndim!=2 or tensor.dtype!=torch.float32 or len(basis)!=2 or any(q is None or q.dtype!=torch.float32 for q in basis):
        raise ValueError('Declared two-sided FP32matrix/bases required')
    left,right=basis
    x=tensor.to(torch.bfloat16);l=left.to(torch.bfloat16);r=right.to(torch.bfloat16)
    return ((l@x)@r.T if back else (l.T@x)@r).float()

def install(matrix,policy,outputs=None):
    if policy not in ('native','bf16'):raise ValueError('Undeclared projection policy')
    if any(g['merge_dims'] for g in matrix.param_groups):raise ValueError('No merged dimensions declared')
    ref=weakref.ref(matrix)
    def forward(tensor,basis,group):
        obj=ref()
        if obj is None:raise RuntimeError('Released projection owner')
        if policy=='native' or torch.get_float32_matmul_precision()=='highest':return SOAP._project_with_basis(obj,tensor,basis,group)
        return project(tensor,basis)
    def back(tensor,basis,group):
        obj=ref()
        if obj is None:raise RuntimeError('Released projection owner')
        highest=torch.get_float32_matmul_precision()=='highest'
        value=SOAP._project_back_with_basis(obj,tensor,basis,group) if policy=='native' or highest else project(tensor,basis,True)
        if outputs is not None and not highest:outputs.append(value)
        return value
    matrix._project_with_basis=forward;matrix._project_back_with_basis=back
