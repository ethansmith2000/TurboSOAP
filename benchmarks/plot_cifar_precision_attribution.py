"""Descriptive local precision differences, without additive attribution claims."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

VARIANTS=['qr','warm1','warm2_final','warm2_sequential','warm1_cap02']
LABELS=['QR','Warm ×1\ncap .1','Warm ×2\nfinal m','Warm ×2\nsequential m','Warm ×1\ncap .2']


def save(fig,b,name):
    for ext in ['png','svg']:fig.savefig(b/(name+'.'+ext),dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle
    s=json.loads((b/'summary.json').read_text());assert s['all_prior_anchors_exact']
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,layout='constrained')
    keys=['full_update_difference','fixed_basis_transport_difference','strict_transport_remaining_difference']
    legends=['Full high vs strict refresh','Transport arithmetic: same Q/v','Remaining after strict m transport']
    colors=['#245a96','#b54a19','#32855c']
    for ax,policy,event in zip(axes,['qr20','warm20'],['qr','warm']):
        rows=[next(r for r in s['rows'] if r['label']==policy+'_lr0.0005' and r['executed_method']==event and r['variant']==v) for v in VARIANTS]
        x=np.arange(len(rows))
        for i,(key,label,color) in enumerate(zip(keys,legends,colors)):
            ax.bar(x+(i-1)*.24,[100*r['metrics'][key]['mean'] for r in rows],width=.24,label=label,color=color)
        ax.set_xticks(x,LABELS,fontsize=9);ax.set_yscale('log');ax.axhline(1,color='#555555',linestyle='--',linewidth=1)
        ax.set_title(policy.upper()+' incoming states');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Mean relative implied-update difference (%) — log scale')
    fig.suptitle('Precision attribution on unchanged incoming optimizer states\nWarm panel excludes QR reset; means do not indicate every sample passes',fontsize=11)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    save(fig,b,'precision_paths')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    stages=['transport_arithmetic','basis_arithmetic_fixed_order','basis_ordering_fixed_v','v_permutation_fixed_basis']
    labels=['Transport\narithmetic','Basis arithmetic\nfixed ordering','Basis ordering\nfixed v','v permutation\nfixed basis']
    x=np.arange(len(stages))
    for i,(policy,event,color) in enumerate([('qr20','qr','#245a96'),('warm20','warm','#b54a19')]):
        row=next(r for r in s['rows'] if r['label']==policy+'_lr0.0005' and r['executed_method']==event and r['variant']=='qr')
        ax.bar(x+(i-.5)*.32,[100*row['stages'][k]['mean'] for k in stages],width=.32,label=policy.upper()+' incoming states',color=color)
    ax.set_xticks(x,labels);ax.set_ylabel('Mean stage-vector norm / strict update norm (%)')
    ax.set_title('QR precision path: stage magnitudes do not add\nOrdering and v changes are diagnostic counterfactuals, not proposed policies',fontsize=11)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    save(fig,b,'qr_precision_stages')


if __name__=='__main__':main()
