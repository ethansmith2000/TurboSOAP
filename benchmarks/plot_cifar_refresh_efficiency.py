"""Local tracking/cost panels; no learning or whole-step speedup inference."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS=['#245a96','#b54a19','#dc9639','#8856a7','#32855c']
VARIANTS=['qr','warm1','warm2_final','warm2_sequential','warm1_cap02']
LABELS=['QR','Warm ×1, cap .1','Warm ×2, final m','Warm ×2, sequential m','Warm ×1, cap .2']


def save(fig,b,name):
    for ext in ['png','svg']:fig.savefig(b/(name+'.'+ext),dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle
    s=json.loads((b/'summary.json').read_text());assert s['all_prior_anchors_exact']
    gate_note = ('No variant passes every precision check; rejected samples retained'
                 if all(s['candidate_rejections'].values()) else 'Includes rejected samples; see numerical gate report')
    factors=[f for f in s['factors'] if f['label']=='warm20_lr0.0005' and f['executed_method']=='warm']
    factors.sort(key=lambda f:f['dimension']);x=np.arange(len(factors))
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for i,k in enumerate([0,1,2,4]):
        ax.bar(x+(i-1.5)*.2,[f['residual'][VARIANTS[k]] for f in factors],width=.2,
               color=COLORS[k],label=LABELS[k])
    ax.set_xticks(x,[str(f['dimension']) for f in factors]);ax.set_xlabel('Factor dimension')
    ax.set_ylabel('Mean normalized off-diagonal residual');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.set_title('Local alternatives on warm-trajectory incoming states\nSelected warm events; lower residual is better\n'+gate_note,fontsize=10)
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    save(fig,b,'local_residuals')
    rows=[r for r in s['rows'] if r['label']=='warm20_lr0.0005' and r['executed_method']=='warm']
    shapes=sorted({tuple(r['shape']) for r in rows});x=np.arange(len(shapes))
    fig,axes=plt.subplots(2,1,figsize=(9,8),layout='constrained')
    for i,(name,color,label) in enumerate(zip(VARIANTS,COLORS,LABELS)):
        selected=[next(r for r in rows if tuple(r['shape'])==shape and r['variant']==name) for shape in shapes]
        for ax,key in zip(axes,['refresh_ms','reduction_per_ms']):
            ax.bar(x+(i-2)*.16,[r[key] for r in selected],width=.16,color=color,label=label)
    for ax in axes:
        ax.set_xticks(x,[f'{a} × {c}' for a,c in shapes]);ax.set_xlabel('Matrix shape')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Complete local refresh (ms)')
    axes[1].set_ylabel('Mean residual decrease / refresh ms')
    axes[0].set_title('Warm incoming states: local refresh cost and tracking efficiency\nIncludes m transport and v handling; excludes the rest of the optimizer step\n'+gate_note,fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    save(fig,b,'refresh_cost_efficiency')


if __name__=='__main__':main()
