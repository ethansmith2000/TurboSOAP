import pytest

from fast_soap_qualification import ARMS, DIMENSIONS, SHAPES, config, kind, weighted


def test_shapes_cover_deeper_fused_attention_and_swiglu_factors():
    assert sorted({dimension for shape in SHAPES for dimension in shape}) == list(DIMENSIONS)
    assert sum(len(shape) for shape in SHAPES) == 8


@pytest.mark.parametrize('arm,counts', [
    ('qr20', {'ordinary': 404, 'qr': 21}),
    ('hold200', {'ordinary': 423, 'qr': 2}),
    ('j1', {'ordinary': 404, 'warm': 19, 'qr': 2}),
    ('fast', {'ordinary': 404, 'warm': 19, 'qr': 2}),
])
def test_declared_horizon_covers_repair_and_clean_repair_after_diagnostics(arm, counts):
    from collections import Counter
    assert Counter(kind(arm, age) for age in range(1, config()['calls']+1)) == counts
    assert kind(arm, 400) == 'qr'
    assert kind(arm, 220) == ('warm' if arm in ('j1', 'fast') else
                              'qr' if arm == 'qr20' else 'ordinary')


def test_cadence_weighting_charges_warm_and_periodic_repairs():
    means = dict(ordinary=2., warm=20., qr=100.)
    assert weighted('qr20', means) == pytest.approx(6.9)
    assert weighted('hold200', means) == pytest.approx(2.49)
    assert weighted('j1', means) == weighted('fast', means) == pytest.approx(3.3)
    assert set(config()['arms']) == set(ARMS)


def test_undeclared_arm_is_rejected():
    with pytest.raises(ValueError):
        kind('fast_variant', 20)
