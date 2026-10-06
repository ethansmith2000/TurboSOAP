"""Static scientific figures; runtime and learning use separate panels/axes."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;out=root/'soap_retraction_bounded_20261005'
    s=json.loads((out/'summary.json').read_text())
    kinds=['qr20','scaled_ns6','spectral_cubic1_cap05','spectral_quintic1_cap05']
    names=['QR20','NS6 / average cap .1','Cubic1 / row cap .5','Quintic1 / row cap .5']
    colors=['#465c78','#c97a37','#24928d','#755ab2']
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for ax,beta in zip(axes,[.99,.999]):
        for k,(name,color) in enumerate(zip(kinds,colors)):
            vals=[next(x['optimizer_ms'] for x in s['rows'] if x['width']==width and x['beta']==beta and x['name']==name and not x['repeat']) for width in [768,1536]]
            ax.bar(np.arange(2)+(k-1.5)*.19,vals,.18,color=color,label=names[k])
        ax.set_xticks([0,1],['3072','6144']);ax.set_xlabel('Largest factor dimension');ax.set_title(f'Covariance beta {beta}')
        ax.set_ylabel('Optimizer ms / update');ax.set_ylim(0,23);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Synthetic natural cycles: fixed retraction policies\n420 updates; warm20 / QR200; diagnostics excluded',fontsize=12)
    for ext in ('png','svg'):fig.savefig(out/f'natural_cycle_cost.{ext}',dpi=160)
    plt.close(fig)
    b=root/'soap_retraction_diagnostic_20261005';d=json.loads((b/'summary.json').read_text())
    vals=[v for v in d['attribution'] if v['dimension']==3072]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].plot([x['step'] for x in vals],[x['candidate_spectrum']['maximum_gram_eigenvalue'] for x in vals],'o-',label='Candidate maximum')
    axes[0].plot([x['step'] for x in vals],[x['candidate_spectrum']['minimum_gram_eigenvalue'] for x in vals],'o-',label='Candidate minimum')
    axes[0].axhline(1,color='gray',linestyle=':');axes[0].set_ylabel('Eigenvalue of candidate Gram');axes[0].legend(frameon=False)
    axes[1].plot([x['step'] for x in vals],[x['scale'] for x in vals],'o-',color='#c97a37');axes[1].set_ylabel('Row-sum retraction input scale');axes[1].set_ylim(0,1.1)
    for ax in axes:ax.set_xlabel('Sampled update age');ax.grid(alpha=.2)
    fig.suptitle('NS6 reference trajectory: the first warm refresh is different\n3072-factor, average cap .1; offline same-state diagnostics',fontsize=12)
    for ext in ('png','svg'):fig.savefig(b/f'scaling_attribution.{ext}',dpi=160)
    plt.close(fig)
    b=root/'cifar_soap_retraction_v2_20261005'
    if (b/'summary.json').exists():
        s=json.loads((b/'summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
        for ax,beta in zip(axes,[.99,.999]):
            rows=[next(r for r in s['rows'] if r['kind']==k and r['beta']==beta and not r['repeated']) for k in ['qr20','ns6','cubic2','quintic1']]
            ax.scatter(np.arange(4),[r['ce'] for r in rows],c=colors,s=65)
            for i,r in enumerate(rows):ax.annotate(f"{r['ce']:.4f}",(i,r['ce']),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
            ax.set_xticks(np.arange(4),['QR20','NS6','Cubic2','Quintic1']);ax.set_ylabel('Validation cross-entropy (lower is better)');ax.set_title(f'Covariance beta {beta}');ax.grid(axis='y',alpha=.2)
            ax.margins(x=.2,y=.3)
        fig.suptitle('CIFAR-100: paired fixed-update learning\nOne seed, 3500 updates; joint cap / retraction intervention',fontsize=12)
        for ext in ('png','svg'):fig.savefig(b/f'learning_endpoints.{ext}',dpi=160)
        plt.close(fig)


if __name__=='__main__':main()
