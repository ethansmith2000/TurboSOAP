import hashlib
import numpy as np
import pytest
import torch
from reference_fineweb_data import FineWebWindows, content_hash, sequence_losses, evaluation_mode
from reference_llama import ReferenceLlama


def shard(path, tokens, version=1):
    header = np.zeros(256, dtype='<i4')
    header[:3] = [20240520, version, len(tokens)]
    path.write_bytes(header.tobytes() + np.asarray(tokens, dtype='<u2').tobytes())
    return path


def test_windows_have_no_overlap_and_keep_last_target(tmp_path):
    a = shard(tmp_path/'a', range(15))
    b = shard(tmp_path/'b', range(20, 32))
    data = FineWebWindows([a, b], sequence=3, vocab=50)
    assert data.counts == [3, 3] and len(data) == 6
    windows = data.windows([5, 0, 3, 2])
    np.testing.assert_array_equal(windows, [[28,29,30,31],[0,1,2,3],[20,21,22,23],[8,9,10,11]])
    assert len(set(data.windows(range(6)).ravel())) == 24
    assert content_hash(windows) == hashlib.sha256(windows.astype('<u2').tobytes()).hexdigest()
    for bad in [[-1], [6], [[0]]]:
        with pytest.raises(ValueError): data.windows(bad)


@pytest.mark.parametrize('fault', ['header','version','length','vocab'])
def test_corrupt_shards_rejected(tmp_path, fault):
    p = shard(tmp_path/'a', [0,1,2,3], version=2 if fault=='version' else 1)
    if fault=='header':p.write_bytes(b'bad')
    if fault=='length':p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError): FineWebWindows([p], sequence=3, vocab=3 if fault=='vocab' else 10)


def test_cross_entropy_includes_zero_and_uses_all_targets():
    logits = torch.tensor([[[2.,0.],[0.,2.],[2.,0.]]], requires_grad=True)
    windows = torch.tensor([[1,0,1,0]])
    loss = sequence_losses(logits, windows)
    torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits.reshape(-1,2), windows[:,1:].reshape(-1)).reshape(1))
    loss.sum().backward()
    assert torch.count_nonzero(logits.grad) == 6


@pytest.mark.parametrize('fail', [False, True])
def test_eval_restores_state_modes_gradients_and_rng(fail):
    torch.manual_seed(9)
    model = ReferenceLlama(width=8, depth=2, head_dim=4, vocab=19, sequence=3)
    model.blocks[0].eval()  # Preserve mixed submodule modes too.
    tokens = torch.arange(8).reshape(2,4)%19
    sequence_losses(model(tokens[:,:-1]),tokens).mean().backward()
    params = [p.detach().clone() for p in model.parameters()]
    grads = [p.grad.clone() for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    rng = torch.get_rng_state().clone()
    try:
        with evaluation_mode(model):
            assert not model.recompute and not model.training
            losses = sequence_losses(model(tokens[:,:-1]),tokens)
            assert not losses.requires_grad
            torch.rand(4)
            if fail:raise RuntimeError('test')
    except RuntimeError:
        assert fail
    assert model.recompute and [m.training for m in model.modules()] == modes
    assert torch.equal(rng,torch.get_rng_state())
    for p, before, grad in zip(model.parameters(),params,grads):
        torch.testing.assert_close(p,before,rtol=0,atol=0)
        torch.testing.assert_close(p.grad,grad,rtol=0,atol=0)
