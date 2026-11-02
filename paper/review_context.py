"""Calculate descriptive review context from saved summaries, using only CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = {
    'seed205': 'reports/bf16_full_horizon_20261026_seed205/comparison.json',
    'seed204': 'reports/bf16_replicate_20261027_seed204/comparison.json',
    'pilot': 'reports/bf16_learning_20261026/comparison.json',
    'controls': 'reports/bf16_refresh_controls_20261027/comparison.json',
    'calibration': 'reports/owt_restart_20261026/calibration.json',
    'geometry_stock205': 'reports/bf16_full_horizon_20261026_seed205/runs/qr10/geometry.json',
    'geometry_bf16205': 'reports/bf16_full_horizon_20261026_seed205/runs/bf16/geometry.json',
    'cuda_parity205': 'reports/bf16_full_horizon_20261026_seed205/runs/bf16/cuda_parity.json',
}


def calculate():
    data = {key: json.loads((ROOT / path).read_text())
            for key, path in SOURCES.items()}
    stock, warm = data['seed205']['results']
    a, b = stock['optimizer_by_kind'], warm['optimizer_by_kind']
    ordinary = .9 * (a['ordinary'] - b['ordinary'])
    refresh = .1 * a['qr'] - .08 * b['warm'] - .02 * b['qr']
    total = stock['mean_optimizer_ms'] - warm['mean_optimizer_ms']
    assert abs(ordinary + refresh - total) < 1e-10
    assert stock['clean_cycle_calls'] == warm['clean_cycle_calls'] == 9900
    pilot = data['pilot']['results']
    reference = {row['call']: row['ce'] for row in pilot[0]['validation']}
    gaps = {arm['case']: {str(row['call']): row['ce'] - reference[row['call']]
                         for row in arm['validation']}
            for arm in pilot}
    controls = data['controls']
    cached, qr20, qr50 = controls['cached_warm_prefix'], *controls['results']
    assert [qr20['case'], qr50['case']] == ['bf16_qr20', 'bf16_qr50']
    bands = {}
    for seed, (reference_arm, candidate) in {
        204: (data['seed204']['cached_quality_reference'], data['seed204']['candidate']),
        205: (stock, warm),
    }.items():
        reference_ce = {row['call']: row['ce'] for row in reference_arm['validation']}
        values = [row['ce'] - reference_ce[row['call']]
                  for row in candidate['validation'] if row['call'] >= 2000]
        bands[str(seed)] = dict(minimum=min(values), maximum=max(values))
    calibration = data['calibration']
    assert calibration['selected']['stock'] == .0015
    assert len([r for r in calibration['runs'] if r['profile'] == 'stock']) == 3
    prefix_events = []
    for arm, label, ordinary_fraction, warm_fraction, qr_fraction in [
        (pilot[0], 'Stock QR10', .9, 0, .1),
        (pilot[1], 'Original warm', .9, .08, .02),
        (pilot[2], 'Compiled FP32 state', .9, .08, .02),
        (pilot[3], 'Full BF16 warm', .9, .08, .02),
        (qr20, 'BF16 QR20', .95, 0, .05),
        (qr50, 'BF16 QR50', .98, 0, .02),
    ]:
        means = arm['optimizer_by_kind']
        parts = dict(ordinary=ordinary_fraction * means['ordinary'],
                     warm=warm_fraction * means.get('warm', 0),
                     qr=qr_fraction * means['qr'])
        assert abs(sum(parts.values()) - arm['mean_optimizer_ms']) < 1e-10
        prefix_events.append(dict(label=label, case=arm['case'],
                                  amortized_ms=parts, calls=[101, 2000],
                                  source='pilot' if arm in pilot else 'controls'))
    coordinate = {}
    for label, source in [('stock', 'geometry_stock205'), ('bf16', 'geometry_bf16205')]:
        rows = []
        for observation in data[source]:
            if observation['call'] not in (1991, 9991):
                continue
            values = [r['relative_coordinate_offdiagonal']
                      for r in observation['coordinate_residuals']]
            assert len(values) == 96
            rows.append(dict(call=observation['call'], age=observation['call'] - 1,
                             factors=len(values), mean=statistics.mean(values),
                             maximum=max(values)))
        coordinate[label] = rows
    parity = data['cuda_parity205']['eager_compiled']
    assert len(parity) == 3
    assert all(row['parameter_max_absolute'] == 0 and
               all(v == 0 for v in row['state_relative_errors'].values())
               for row in parity)
    return dict(
        interpretation='Descriptive accounting and development context, not a causal partition.',
        inputs={key: dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
                for key, path in SOURCES.items()},
        seed205_optimizer_event_means_ms=dict(stock=a, full_bf16=b),
        seed205_optimizer_mean_ms=dict(stock=stock['mean_optimizer_ms'],
                                      full_bf16=warm['mean_optimizer_ms']),
        seed205_optimizer_time_ratio=stock['mean_optimizer_ms'] / warm['mean_optimizer_ms'],
        seed205_event_accounting=dict(
            window=[101, 10000], count=9900,
            fractions=dict(stock_ordinary=.9, stock_qr=.1,
                           candidate_ordinary=.9, candidate_warm=.08, candidate_qr=.02),
            ordinary_saving_ms_per_call=ordinary,
            refresh_saving_ms_per_call=refresh,
            optimizer_saving_ms_per_call=total,
            refresh_share_percent=100 * refresh / total,
            ordinary_share_percent=100 * ordinary / total,
            complete_step_saving_ms_per_call=stock['mean_step_ms'] - warm['mean_step_ms']),
        pilot_ce_deltas_vs_stock=gaps,
        prefix_amortized_optimizer_events=prefix_events,
        coordinate_offdiagonal_diagnostics=coordinate,
        seed205_later_sampled_m_error_max=max(
            row['max_physical_m_error'] for row in data['geometry_bf16205']
            if row['call'] > 11),
        hypothetical_stock_calls_at_equal_steady_cost=(
            10000 * warm['mean_step_ms'] / stock['mean_step_ms']),
        pilot_full_bf16_minus_compiled_fp32_endpoint=(
            pilot[3]['validation'][-1]['ce'] - pilot[2]['validation'][-1]['ce']),
        control_qr20_step_increase_vs_cached_warm_percent=(
            100 * (qr20['mean_step_ms'] / cached['mean_step_ms'] - 1)),
        late_observed_gap_bands=bands,
        learning_rate_selection=dict(
            selected_matrix_peak_lr=.0015, arm='Stock QR10 baseline only',
            model_seed=204, candidate_rates=[.0005, .0015, .003], calls=500,
            full_schedule_horizon=10000, criterion='Lowest call-500 validation CE',
            candidate_specific_retuning=False),
        gpu_execution=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    value = json.dumps(calculate(), indent=2, allow_nan=False) + '\n'
    path = HERE / 'review_context.json'
    if args.check:
        assert path.read_text() == value, 'Review context changed'
    else:
        path.write_text(value)
    print('PASS: descriptive event accounting, pilot/control context and LR selection.')


if __name__ == '__main__':
    main()
