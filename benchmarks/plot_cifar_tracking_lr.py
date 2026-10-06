"""Endpoint LR bracket and same-state residual diagnostics; no uncertainty bars."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle
    s=json.loads((b/'summary.json').read_text())
    assert s['all_prior_anchors_exact'] and s['tracking_coverage_verified']
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    for policy,color in [('qr20','#245a96'),('warm20','#b54a19')]:
        rows=sorted([r for r in s['rows'] if r['policy']==policy and not r['repeated']],key=lambda r:r['matrix_lr'])
        ax.plot([r['matrix_lr'] for r in rows],[r['ce'] for r in rows],marker='o',color=color,
                label='QR20' if policy=='qr20' else 'Warm20 (NS6)')
    ax.set_xscale('log');ax.set_xticks([.00025,.0005,.001],['0.00025','0.0005','0.001'])
    ax.minorticks_off()
    ax.set_xlabel('Matrix peak learning rate (auxiliary LR stays 0.0005)')
    ax.set_ylabel('Validation CE at epoch 20 (lower is better)');ax.grid(alpha=.2);ax.legend(frameon=False)
    ax.set_title('CIFAR-100: bounded LR check\nCovariance beta 0.99; seed 139; original 200-epoch schedule')
    for ext in ['png','svg']:fig.savefig(b/('lr_endpoints.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharey=True,layout='constrained')
    colors=['#999999','#245a96','#b54a19','#dc9639'];keys=['before','qr','warm1','warm2']
    labels=['Incoming basis','QR shadow','One NS6 step','Two NS6 steps']
    for ax,policy,event in zip(axes,['qr20','warm20'],['qr','warm']):
        rows=sorted([r for r in s['tracking'] if r['label']==policy+'_lr0.0005' and r['executed_method']==event],
                    key=lambda r:r['dimension']);x=np.arange(len(rows))
        for i,(key,color,label) in enumerate(zip(keys,colors,labels)):
            ax.bar(x+(i-1.5)*.2,[r['mean_residual'][key] for r in rows],width=.2,color=color,label=label)
        ax.set_xticks(x,[str(r['dimension']) for r in rows]);ax.set_xlabel('Factor dimension')
        ax.set_title(('QR20' if policy=='qr20' else 'Warm20')+' incoming states')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Mean normalized off-diagonal residual')
    handles,legend_labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='outside lower center',ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Local tracking alternatives on identical incoming states\nWarm panel excludes the scheduled QR reset; shadows are not executed',fontsize=12)
    for ext in ['png','svg']:fig.savefig(b/('tracking_residuals.'+ext),dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
