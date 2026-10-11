"""New prospective backend/device policy; old summary and gates stay frozen."""
import json
from pathlib import Path
import sys

try:
    from .launch_environment_contract import PHASES, check_backend, read_launch
    from .summarize_normalized_input import qualify
    from .normalized_diagnostic_contract import validate_config
except ImportError:
    from launch_environment_contract import PHASES, check_backend, read_launch
    from summarize_normalized_input import qualify
    from normalized_diagnostic_contract import validate_config


def apply_backend_gate(result, runs, contract, device):
    rows = []
    for run in runs:
        for phase in PHASES:
            row = check_backend(contract, device, run.get(phase, {}), phase)
            saved = run.get('environment_'+('before_smoke' if phase == PHASES[0] else 'after_measured')+'_check')
            row['early_check_exact'] = saved == row
            row['label'] = run['context']['label']
            rows.append(row)
    ok = len(runs) == 4 and len(rows) == 8 and all(r['passed'] and r['early_check_exact'] for r in rows)
    result.update(prospective_baseline_backend_match=ok, baseline_backend_checks=rows,
                  declared_device_binding=device)
    result['localization_supported'] = result.get('localization_supported', False) and ok
    if not ok:
        for pair in result.get('pairs', []):
            pair['classification'] = 'INCONCLUSIVE_PARITY_ABI_OR_RESOURCE_GATE'
    return result


def main(bundle):
    bundle = Path(bundle); contract, launch = read_launch(bundle)
    config = json.loads((bundle/'config.json').read_text()); validate_config(config)
    runs = [json.loads((bundle/'contexts'/c['label']/'result.json').read_text())
            for c in config['execution_diagnostic']['contexts'] if (bundle/'contexts'/c['label']/'result.json').exists()]
    result = apply_backend_gate(qualify(runs), runs, contract, launch['device_binding'])
    with (bundle/'summary.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    value = main(Path(sys.argv[1]))
    print(json.dumps({k: v for k, v in value.items() if k not in ('previous', 'contexts', 'pairs', 'baseline_backend_checks')}, indent=2))
