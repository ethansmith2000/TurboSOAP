"""Isolated movement experiment adapter, with optional offline attribution."""
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]


def main():
    import cifar_lr_screen as trainer
    import cifar_strict_transport as transport
    from movement_retraction import configure_matrix
    from cifar_movement_probe import MovementProbe
    from cifar_retraction_screen import configure_matrix as configure_old
    bundle = Path(sys.argv[sys.argv.index('--bundle')+1])
    study = json.loads((bundle/'config.json').read_text())['memory_screen']
    if study.get('movement_attribution'):
        transport.StrictTransportProbe = MovementProbe
    old_config, old_build = trainer.screen_arm_config, trainer.build_optimizer
    records = []
    def config(c, arm):
        result = old_config(c, arm)
        result['movement_label'] = arm['label']
        for key in ('movement_policy', 'retraction_variant'):
            if key in arm: result[key] = arm[key]
        return result
    def build(model, c, policy):
        opt = old_build(model, c, policy)
        if 'movement_policy' in c: configure_matrix(opt.matrix, c['movement_policy'])
        if 'retraction_variant' in c: configure_old(opt.matrix, c['retraction_variant'])
        records.append(dict(label=c.get('movement_label', 'smoke'), groups=[
            {k:v for k,v in g.items() if k.startswith('basis_') or k.startswith('research_') or
             k in ('transport_precision','shampoo_beta','betas','hard_reset_interval','precondition_frequency')}
            for g in opt.matrix.param_groups]))
        return opt
    original = transport.diagnose
    def diagnose(opt, old, group, age, actual):
        d = original(opt, old, group, age, actual)
        if not d['targeted_pass']:
            with (bundle/'failed_probe.jsonl').open('a') as f:
                f.write(json.dumps(dict(age=age, shape=list(old['m'].shape), diagnostic=d))+'\n')
        return d
    transport.diagnose = diagnose
    trainer.screen_arm_config, trainer.build_optimizer = config, build
    try: trainer.main()
    finally: (bundle/'experimental_groups.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__': main()
