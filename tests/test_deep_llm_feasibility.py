"""Boundary and budget checks for the new shared-cache training adapter."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

spec = importlib.util.spec_from_file_location('deep_llm_data', Path(__file__).resolve().parents[1] / 'benchmarks/deep_llm_data.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_shifted_loss_drops_final_logit_and_never_predicts_across_rows():
    tokens = torch.tensor([[0, 1, 2], [3, 2, 1]])
    logits = torch.zeros(2, 3, 4, requires_grad=True)
    loss = adapter.shifted_loss(logits, tokens)
    expected = F.cross_entropy(logits[:, :2].reshape(-1, 4), torch.tensor([1, 2, 2, 1]))
    assert torch.equal(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, -1]) == 0
    assert torch.count_nonzero(logits.grad[:, :-1]) > 0


def test_plan_has_no_wrap_disjoint_probe_and_repeatable_prefix():
    first, panel = adapter.plan(200, 191, 10, 8, 16)
    longer, same_panel = adapter.plan(200, 191, 20, 8, 16)
    assert (first == longer[:10]).all()
    assert (panel == same_panel).all()
    assert len(set(longer.reshape(-1))) == longer.size
    assert not set(panel) & set(longer.reshape(-1))


def test_insufficient_plan_is_rejected_instead_of_wrapping():
    with pytest.raises(ValueError):
        adapter.plan(100, 191, 12, 8, 16)


def test_adapter_rejects_wrong_width_and_vocabulary():
    class Fake:
        def __getitem__(self, ids):
            return {'input_ids': [[0, 50257] for _ in ids]}
    with pytest.raises(ValueError):
        adapter.tokens_from_rows(Fake(), [0], width=2)
    with pytest.raises(ValueError):
        adapter.tokens_from_rows(Fake(), [0], width=3)


def test_guard_duration_can_be_journaled_with_elapsed_time():
    runner_spec = importlib.util.spec_from_file_location('deep_llm_feasibility', Path(__file__).resolve().parents[1] / 'benchmarks/deep_llm_feasibility.py')
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    row = runner.event_record('guard',12.,dict(phase='disposable',seconds=.25))
    assert row['seconds']==.25 and row['elapsed_seconds']==12.
