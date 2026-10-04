"""Plot paired learning trajectories and within-policy covariance effects."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
bundle=parser.parse_args().bundle
summary=json.loads((bundle/'summary.json').read_text())
assert summary['status']=='completed' and summary['early_anchor_bridge']['complete']
fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey='row',layout='constrained')
for column,policy in enumerate(['qr10','qr20','warm10']):
    for beta,color,style in [(.999,'#687485','--'),(.99,'#2167b2','-')]:
        row=next(r for r in summary['rows'] if r['policy']==policy and
                 r['covariance_beta']==beta and not r['repeated'])
        trajectory=row['trajectory']
        axes[0,column].plot([o['epoch'] for o in trajectory],
            [o['validation_ce'] for o in trajectory],style,marker='o',markersize=4,
            color=color,label=f'Covariance beta {beta}')
    effect=next(p for p in summary['paired_effects'] if p['policy']==policy)
    deltas=effect['trajectory_deltas']
    axes[1,column].plot([d['epoch'] for d in deltas],[d['ce_delta'] for d in deltas],
                       color='#2167b2',marker='o',markersize=4)
    axes[1,column].axhline(0,color='#555555',linewidth=1)
    axes[1,column].set_xlabel('Epoch (original 200-epoch schedule)')
    axes[0,column].set_title({'qr10':'QR10','qr20':'QR20','warm10':'Warm10 + QR200'}[policy])
    axes[0,column].legend(frameon=False,fontsize=9)
    axes[1,column].set_title(f"100-epoch CE change: {effect['endpoint']['ce_delta']:+.4f}",fontsize=11)
    for ax in axes[:,column]:
        ax.grid(alpha=.2);ax.set_axisbelow(True);ax.set_xticks([20,40,60,80,100])
axes[0,0].set_ylabel('Validation cross-entropy')
axes[1,0].set_ylabel('CE change: beta .99 minus .999\nNegative is better')
fig.suptitle('CIFAR-100: longer covariance-memory comparison\nSeed 139; matrix/auxiliary Adam beta2 .999; both peak LRs .0005',fontsize=13)
for suffix in ['png','svg']:fig.savefig(bundle/f'covariance_trajectories.{suffix}',dpi=170)
plt.close(fig)
