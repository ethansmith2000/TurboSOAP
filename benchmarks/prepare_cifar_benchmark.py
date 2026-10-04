"""Freeze a new calibration bundle using the corrected cached CIFAR workload."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REPO=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare(bundle,base,*,matrix_lr=.001,auxiliary_lr=.0005,beta2=.999,
            covariance_beta=.999,seed=139,prefix_steps=128,branch_steps=512):
    if bundle.exists():raise ValueError('Use a new bundle directory')
    if prefix_steps<=50 or branch_steps<=50:raise ValueError('Need more than 50 updates for timing')
    if beta2 not in (.99,.999) or covariance_beta not in (.99,.999):raise ValueError('Use .99 or .999')
    if not (0<matrix_lr<1 and 0<auxiliary_lr<1):raise ValueError('Invalid LR')
    if shutil.disk_usage(bundle.parent).free<5*2**30:raise RuntimeError('Require 5 GiB free')
    prepared=json.loads((base/'prepared.json').read_text())
    for name in ['config','split']:
        assert sha(base/(name+'.json'))==prepared[name+'_sha256'],name
    for name in ['cifar100_qualification.py','cifar100_strong_augmentation.py','vit.py']:
        assert sha(REPO/'benchmarks/cifar_support'/name)==prepared['sources'][name],name
    c=json.loads((base/'config.json').read_text())
    if c.get('evaluation_policy')!='eager_bf16' or c.get('recipe')!='mixup_cutmix_randaugment':
        raise ValueError('Use the corrected strong-recipe base bundle')
    assert sha(Path(c['data_root'])/'cifar100/train-00000-of-00001.parquet')==c['data_sha256']
    c.update(lr=auxiliary_lr,matrix_lr=matrix_lr,betas=[c['betas'][0],beta2],
             shampoo_beta=covariance_beta,seed=seed,precondition_mode='all',
             covariance_compute_dtype='float32',prefix_steps=prefix_steps,branch_steps=branch_steps,
             optimizer='research_soap',max_wall_seconds=1800,checkpoint=False,official_test=False)
    assert prefix_steps+branch_steps<=prepared['steps_per_epoch']*c['epochs']
    (bundle/'source/benchmarks/cifar_support').mkdir(parents=True)
    (bundle/'upstream_reference').mkdir()
    for rel in ['soap.py','soap_reference.py','benchmarks/cifar_optimizers.py',
                'benchmarks/cifar_optimizer_benchmark.py','benchmarks/prepare_cifar_benchmark.py']:
        shutil.copy2(REPO/rel,bundle/'source'/rel)
    for directory,target in [(REPO/'benchmarks/cifar_support',bundle/'source/benchmarks/cifar_support'),
                             (REPO/'tests/reference',bundle/'upstream_reference')]:
        for p in directory.iterdir():
            if p.is_file():shutil.copy2(p,target/p.name)
    shutil.copy2(base/'split.json',bundle/'split.json')
    shutil.copy2(base/'prepared.json',bundle/'data_prepared.json')
    (bundle/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    manifest=dict(source_sha256={str(p.relative_to(bundle/'source')):sha(p) for p in (bundle/'source').rglob('*') if p.is_file()},
        config_sha256=sha(bundle/'config.json'),split_sha256=sha(bundle/'split.json'),
        base_bundle=str(base),base_prepared_sha256=sha(base/'prepared.json'),
        base_repository_head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        evidence_budget_bytes=20*2**20,checkpoint_policy='No saved weights; common branch state held in RAM and discarded.')
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--base-bundle',type=Path,default=Path('/workspace/SNRAdam/measurement_runs/cifar100_strong_20260930_v2'))
    p.add_argument('--matrix-lr',type=float,default=.001)
    p.add_argument('--auxiliary-lr',type=float,default=.0005)
    p.add_argument('--beta2',type=float,choices=[.99,.999],default=.999)
    p.add_argument('--covariance-beta',type=float,choices=[.99,.999],default=.999)
    p.add_argument('--seed',type=int,default=139)
    p.add_argument('--prefix-steps',type=int,default=128)
    p.add_argument('--branch-steps',type=int,default=512)
    a=p.parse_args()
    prepare(a.bundle.resolve(),a.base_bundle.resolve(),matrix_lr=a.matrix_lr,auxiliary_lr=a.auxiliary_lr,
            beta2=a.beta2,covariance_beta=a.covariance_beta,seed=a.seed,
            prefix_steps=a.prefix_steps,branch_steps=a.branch_steps)
    print(a.bundle.resolve())


if __name__=='__main__':main()
