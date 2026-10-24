import pytest
import torch
from reference_llama import ReferenceLlama
from reference_execution import ExecutionLlama,token_loss


@pytest.mark.parametrize('compiled,stride',[(False,0),(False,1),(False,2),(True,1),(True,2)])
def test_execution_preserves_parameters_outputs_and_full_gradients(compiled,stride):
    torch.manual_seed(8)
    reference=ReferenceLlama(width=8,depth=3,head_dim=4,vocab=19,sequence=5)
    with torch.no_grad():
        for name,p in reference.named_parameters():
            if name=='readout' or name.endswith(('.output','.fc2')):p.normal_(std=.01)
    candidate=ExecutionLlama(width=8,depth=3,head_dim=4,vocab=19,sequence=5)
    candidate.load_state_dict(reference.state_dict())
    candidate.configure_execution(compiled,stride,backend='eager')
    tokens=torch.arange(12).reshape(2,6)%19
    a,b=reference(tokens[:,:-1]),candidate(tokens[:,:-1])
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    token_loss(a,tokens[:,1:]).backward();token_loss(b,tokens[:,1:]).backward()
    assert list(reference.state_dict())==list(candidate.state_dict())
    for (n,p),(m,q) in zip(reference.named_parameters(),candidate.named_parameters()):
        assert n==m
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


def test_loss_matches_full_token_mean_and_keeps_token_zero():
    logits=torch.randn(2,5,19,requires_grad=True);labels=torch.arange(10).reshape(2,5)
    loss=token_loss(logits,labels)
    torch.testing.assert_close(loss,torch.nn.functional.cross_entropy(logits.float().reshape(-1,19),labels.flatten()),rtol=0,atol=0)
    loss.backward();assert torch.isfinite(logits.grad).all()
