"""Paired early loss and distinct synthetic event costs; no pooled speedup."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig,b,name):
    for ext in ['png','svg']:fig.savefig(b/(name+'.'+ext),dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle
    s=json.loads((b/'summary.json').read_text());assert s['all_prior_anchors_exact']
    primary=[r for r in s['rows'] if not r['repeated']]
    methods=[('qr20',.1),('warm20',.1),('warm20',.2)]
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharey=True,layout='constrained')
    for ax,beta in zip(axes,[.99,.999]):
        for i,(method,cap) in enumerate(methods):
            vals=[next(r['ce'] for r in primary if (r['policy'],r['cap'],r['precision'],r['beta'])==(method,cap,precision,beta))
                  for precision in ['inherit','highest']]
            ax.plot([i-.08,i+.08],vals,color='#aaaaaa',linewidth=1)
        for offset,precision,label,color in [(-.08,'inherit','Inherited transport','#245a96'),(.08,'highest','Strict FP32 transport','#b54a19')]:
            vals=[next(r['ce'] for r in primary if (r['policy'],r['cap'],r['precision'],r['beta'])==(method,cap,precision,beta)) for method,cap in methods]
            ax.scatter(np.arange(3)+offset,vals,label=label,color=color,s=48,zorder=3)
        ax.set_xticks(range(3),['QR20','Warm20\ncap .1','Warm20\ncap .2']);ax.set_title(f'Covariance beta {beta}')
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Validation CE at epoch 20 (lower is better)')
    fig.suptitle('Paired precision/cap screen — seed 139, fixed peak LR .0005\nOriginal 200-epoch schedule; no independent-seed uncertainty estimate',fontsize=11)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    save(fig,b,'learning_endpoints')
    cost=s['cost'];shapes=list(dict.fromkeys(tuple(r['shape']) for r in cost));x=np.arange(len(shapes))
    fig,axes=plt.subplots(2,1,figsize=(10,8),layout='constrained')
    for i,(method,cap,color,label) in enumerate([('qr',.1,'#245a96','QR'),('warm',.1,'#b54a19','NS6 warm cap .1'),('warm',.2,'#32855c','NS6 warm cap .2')]):
        values=[next(r for r in cost if tuple(r['shape'])==shape and r['method']==method and r['cap']==cap) for shape in shapes]
        axes[0].bar(x+(i-1)*.24,[r['strict_ms'] for r in values],width=.24,color=color,label=label)
        axes[1].scatter(x+(i-1)*.12,[100*r['relative_change'] for r in values],color=color,label=label,s=40)
    for ax in axes:
        ax.set_xticks(x,[f'{a} × {c}' for a,c in shapes],rotation=15);ax.set_xlabel('Matrix shape, both factors active')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_yscale('log');axes[0].set_ylabel('Strict-m full refresh cost (ms) — log scale')
    axes[1].axhline(0,color='#666666',linestyle='--',linewidth=1);axes[1].set_ylabel('Strict vs inherited event cost change (%)')
    axes[0].set_title('Separate synthetic event test: identity bases, rank-limited covariance\nThree short timing rounds; no trainer throughput inference',fontsize=11)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    save(fig,b,'shape_refresh_cost')


if __name__=='__main__':main()
