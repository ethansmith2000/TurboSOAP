import copy
import gc
from types import SimpleNamespace
import weakref
import pytest
import torch
from soap_reference import ResearchSOAP
from benchmarks.cifar_optimizer_benchmark import state_digest
from benchmarks.movement_retraction import configure_matrix
from benchmarks.owned_matrix_hooks import OwnedMatrixHooks
from benchmarks.wide_probe import WideTransportProbe


@pytest.mark.parametrize('fro', [False, True])
def test_closed_adapters_release_matrix_without_collecting_gc(fro):
    enabled = gc.isenabled()
    gc.disable()  # This isolated CPU test only; production GC remains unchanged.
    try:
        p = torch.nn.Parameter(torch.zeros(4, 6))
        matrix = ResearchSOAP([p])
        lease = OwnedMatrixHooks(matrix)
        if fro:
            configure_matrix(matrix, 'fro_cubic2_cap05')
        opt = SimpleNamespace(matrix=matrix)
        probe = WideTransportProbe(opt, [20])
        ref = weakref.ref(matrix)
        probe.close()
        lease.close()
        lease.close()
        assert '_research_refresh' not in matrix.__dict__
        assert '_gauge_step_one' not in matrix.__dict__
        del probe, opt, matrix
        assert ref() is None
        assert lease.matrix is None and not lease.previous and not lease.groups
    finally:
        if enabled:
            gc.enable()


def test_restores_existing_instance_hook_and_group_values():
    matrix = ResearchSOAP([torch.nn.Parameter(torch.zeros(3, 4))])
    custom = lambda *args: None
    matrix._research_refresh = custom
    matrix.param_groups[0]['basis_rotation_cap'] = .37
    before = {k: v for k, v in matrix.param_groups[0].items() if k != 'params'}
    lease = OwnedMatrixHooks(matrix)
    configure_matrix(matrix, 'fro_cubic2_cap05')
    matrix._research_refresh = lambda *args: 'temporary'
    lease.close()
    assert matrix._research_refresh is custom
    assert '_gauge_step_one' not in matrix.__dict__
    assert {k: v for k, v in matrix.param_groups[0].items() if k != 'params'} == before


@pytest.mark.parametrize('fro', [False, True])
def test_lease_does_not_change_actual_trajectory_probes_or_rng(fro):
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('high')
        torch.manual_seed(71)
        initial = torch.randn(4, 6)
        gradients = [torch.randn_like(initial) for _ in range(7)]
        records = []
        for owned in [False, True]:
            p = torch.nn.Parameter(initial.clone())
            matrix = ResearchSOAP([dict(params=[p],param_names=['matrix'])],
                basis_method='warm' if fro else 'qr', precondition_frequency=2,
                hard_reset_interval=4, transport_precision='highest')
            lease = OwnedMatrixHooks(matrix) if owned else None
            if fro:
                configure_matrix(matrix, 'fro_cubic2_cap05')
            probe = WideTransportProbe(SimpleNamespace(matrix=matrix), [2, 4, 6])
            rows = []
            for gradient in gradients:
                p.grad = gradient.clone()
                matrix.step()
                rows.extend(probe.flush())
            records.append(dict(parameter=state_digest(p),state=state_digest(matrix.state_dict()),probes=rows))
            rng = torch.get_rng_state().clone()
            probe.close()
            if lease:
                lease.close()
            assert torch.equal(rng, torch.get_rng_state())
        assert records[0] == records[1]
    finally:
        torch.set_float32_matmul_precision(previous)
