"""Standalone early-learning curves; no time-to-quality or seed error bars."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle
    report=json.loads((b/'screen.json').read_text());s=json.loads((b/'summary.json').read_text())
    assert s['status']=='completed' and s['all_effective_groups_verified']
    colors={'qr20':'#245a96','qr40':'#579bd4','warm20':'#b54a19','warm40':'#de9a39'}
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True,layout='constrained')
    for ax,beta in zip(axes,[.999,.99]):
        for policy,color in colors.items():
            run=next(r for r in report['runs'] if r['policy']==policy and r['covariance_beta']==beta and not r.get('repeat_of'))
            ax.plot([o['epoch'] for o in run['observations']],[o['validation']['ce'] for o in run['observations']],
                marker='o',color=color,label=policy.upper() if policy.startswith('qr') else policy+' (NS6)',
                linestyle='--' if policy.endswith('40') else '-')
        ax.set_title(f'Covariance beta {beta}');ax.set_xticks([5,10,20]);ax.set_xlabel('Epoch')
        ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
    axes[0].set_ylabel('Validation cross-entropy (lower is better)')
    fig.suptitle('CIFAR-100: refresh cadence and covariance memory\nSeed 139; LR 0.0005; first 20 of 200 scheduled epochs',fontsize=12)
    for ext in ['png','svg']:fig.savefig(b/('learning_curves.'+ext),dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
