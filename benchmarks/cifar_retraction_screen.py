"""Isolated entry point applying experimental fixed retractions to CIFAR arms."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE)]
from warm_retraction import VARIANTS, warm


def configure_matrix(matrix, name):
    by_name={v.name:v for v in VARIANTS}
    if name not in by_name:raise ValueError('Unknown experimental retraction')
    variant=by_name[name]
    for group in matrix.param_groups:
        group.update(research_retraction_variant=name,basis_ns_iterations=variant.iterations,
                     basis_rotation_cap_mode=variant.cap_mode,basis_rotation_cap=variant.cap)
    def gauge(cov,q,group):
        return warm(cov,q,group,by_name[group['research_retraction_variant']])
    matrix._gauge_step_one=gauge


def main():
    import cifar_lr_screen as trainer
    import cifar_strict_transport as probe_module
    original_config=trainer.screen_arm_config
    original_build=trainer.build_optimizer
    records=[]
    original_diagnose=probe_module.diagnose
    def diagnose(opt,old,group,age,actual):
        d=original_diagnose(opt,old,group,age,actual)
        if not d['targeted_pass']:
            record=dict(age=age,shape=list(old['m'].shape),
                        variant=group.get('research_retraction_variant'),diagnostic=d)
            bundle=Path(sys.argv[sys.argv.index('--bundle')+1])
            with (bundle/'failed_probe.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            print('FAILED_PROBE '+json.dumps(record),flush=True)
        return d
    def arm_config(config,arm):
        result=original_config(config,arm)
        result['experimental_arm_label']=arm['label']
        if 'retraction_variant' in arm:result['retraction_variant']=arm['retraction_variant']
        return result
    def build(model,config,policy):
        opt=original_build(model,config,policy)
        if 'retraction_variant' in config:
            assert policy=='warm20'
            configure_matrix(opt.matrix,config['retraction_variant'])
        records.append(dict(label=config.get('experimental_arm_label','compile_smoke'),
            groups=[{k:v for k,v in g.items() if k in ('basis_method','basis_ns_iterations',
                'basis_rotation_cap','basis_rotation_cap_mode','research_retraction_variant',
                'transport_precision','shampoo_beta','betas','hard_reset_interval','precondition_frequency')}
                for g in opt.matrix.param_groups]))
        return opt
    trainer.screen_arm_config=arm_config;trainer.build_optimizer=build
    probe_module.diagnose=diagnose
    try:trainer.main()
    finally:
        bundle=Path(sys.argv[sys.argv.index('--bundle')+1])
        (bundle/'experimental_groups.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
