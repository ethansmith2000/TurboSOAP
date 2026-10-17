from pathlib import Path
import ast,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import torch
from full_directional_curvature import single_product_terms,explicit_commutator_terms,quadratic_alpha

def test_one_product_formula_and_decomposition_with_nondiagonal_inputs():
    torch.manual_seed(161)
    for n in (3,7,16):
        z=torch.randn(n,n,dtype=torch.float64);b=z@z.T
        y=torch.randn(n,n,dtype=torch.float64);a=.1*(y-y.T)
        s,h=explicit_commutator_terms(b,a);p=single_product_terms(b,a)
        assert torch.allclose(p['slope'],s,rtol=1e-12,atol=1e-12)
        assert torch.allclose(p['curvature'],h,rtol=1e-12,atol=1e-12)
        assert torch.allclose(p['curvature'],p['diagonal_model']+p['offdiagonal_commutator']+p['diagonal_drift'],rtol=1e-12,atol=1e-12)
    source=(Path(__file__).resolve().parents[1]/'benchmarks/full_directional_curvature.py').read_text()
    f=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='single_product_terms')
    assert sum(isinstance(n,ast.MatMult) for n in ast.walk(f))==1

def test_equal_diagonal_covariance_exposes_missing_negative_curvature():
    b=torch.ones(4,4,dtype=torch.float64);a=torch.zeros_like(b);a[0,1]=.2;a[1,0]=-.2
    p=single_product_terms(b,a)
    assert float(p['slope'])==0 and float(p['diagonal_model'])==0
    assert float(p['curvature'])<0 and torch.equal(p['curvature'],p['diagonal_drift'])

def test_bounded_quadratic_minimum_handles_concave_zero_and_positive():
    for s,h in ((-3.,2.),(-.1,2.),(-1.,-2.),(1.,-4.),(1.,1.),(0.,0.),(0.,-1.)):
        alpha=quadratic_alpha(s,h);value=s*alpha+.5*h*alpha**2
        assert 0<=alpha<=1
        for t in (0.,.1,.25,.5,.75,1.):assert value<=s*t+.5*h*t*t+1e-12
