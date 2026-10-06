"""Standalone scientific figures for movement attribution and paired learning."""
import argparse
import json
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root
    b=root/'cifar_soap_movement_attribution_20261006';r=json.loads((b/'screen.json').read_text())
    ages=[20,180,200,220,400,420,860]
    fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for key,label,color in [('row_bound','Row-sum bound','#8063ad'),('skew_fro_bound','Skew-Frobenius bound','#258b89')]:
        values=[]
        for age in ages:
            fs=[f for a in r['runs'] if not a.get('repeat_of') for p in a['tracking_probes']
                for m in p['matrices'] if m['age']==age for f in m['movement_factors']]
            values.append(st.median(f[key]/max(f['raw_spectral'],1e-30) for f in fs))
        ax[0].plot(range(len(ages)),values,'o-',label=label,color=color)
    ax[0].set(xticks=range(len(ages)),xticklabels=ages,xlabel='Optimizer age (QR at 200/400)',
        ylabel='Median bound / actual spectral norm',title='How conservative is clipping?')
    ax[0].axhline(1,color='gray',ls=':',lw=1);ax[0].legend(fontsize=9)
    summary=json.loads((b/'summary.json').read_text())
    for policy,label,color in [('row_cubic2','Row cubic2','#8063ad'),('fro_cubic2','Frobenius cubic2','#258b89'),('ns6','NS6','#ce8335')]:
        xs=[x for x in summary['by_age'] if x['policy']==policy]
        ax[1].plot(range(len(ages)),[-x['mean_residual_change'] for x in xs],'o-',label=label,color=color)
    ax[1].set(xticks=range(len(ages)),xticklabels=ages,xlabel='Optimizer age',
        ylabel='Mean reduction in covariance residual',title='Alternative refreshes on identical states')
    ax[1].legend(fontsize=9)
    for a in ax:a.grid(alpha=.2)
    fig.suptitle('CIFAR common-state movement attribution\nBoth covariance betas; repeats excluded')
    for ext in ['png','svg']:fig.savefig(b/f'movement_attribution.{ext}',dpi=180)
    plt.close(fig)
    b=root/'cifar_soap_movement_budget_20261006'
    if not (b/'summary.json').exists():return
    result=json.loads((b/'summary.json').read_text());rows=result['rows']
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    kinds=['qr20','ns6','row_cubic2','fro_cubic2'];labels=['QR20','NS6','Row cubic2','Fro cubic2']
    colors=['#49637e','#ce8335','#8063ad','#258b89']
    for col,beta in enumerate([.99,.999]):
        for i,kind in enumerate(kinds):
            matches=[r for r in rows if r['beta']==beta and r['kind']==kind]
            for r in matches:
                offset=.09 if r['repeat'] else 0
                marker='x' if r['repeat'] else 'o'
                axes[0,col].scatter(i+offset,r['ce'],color=colors[i],marker=marker,s=55)
                if not r['repeat']:axes[0,col].annotate(f"{r['ce']:.3f}",(i,r['ce']),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
                if r['budget']:
                    axes[1,col].scatter(i+offset,r['budget']['ce'],color=colors[i],marker=marker,s=55)
                    if not r['repeat']:axes[1,col].annotate(f"{r['budget']['ce']:.3f}\n{r['budget']['step']} steps",(i,r['budget']['ce']),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
        for row in range(2):
            a=axes[row,col];a.set(xticks=range(4),xticklabels=labels,ylabel='Validation CE (lower is better)')
            a.grid(axis='y',alpha=.2);a.margins(y=.25)
        axes[0,col].set_title(f'Covariance beta {beta}: fixed 3500 updates')
        axes[1,col].set_title('90-second measured training budget')
    for row in range(2):
        values=[r['ce'] if row==0 else r['budget']['ce'] for r in rows if row==0 or r['budget']]
        low,high=min(values),max(values);padding=.2*(high-low)
        for col in range(2):
            axes[row,col].set_xlim(-.45,3.55)
            axes[row,col].set_ylim(low-padding,high+padding)
    qualifier='Wall endpoints repeat-qualified' if result['repeat_qualified_wall_interpretation'] else 'Wall endpoints exploratory: timing repeat criterion failed'
    fig.suptitle('CIFAR movement bound intervention; one seed, early learning\n'+qualifier+'; × marks repeated controls')
    for ext in ['png','svg']:fig.savefig(b/f'learning_budget.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
