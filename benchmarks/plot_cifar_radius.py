"""Compact common-state radius results; shadows are not learning outcomes."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    b=Path(sys.argv[1]);s=json.loads((b/'summary.json').read_text());by={x['policy']:x for x in s['rows']}
    keys=['base05_cubic2','radius06_cubic2','radius06_cubic3','radius06_alternate']
    rows=[by[k] for k in keys];labels=['0.5 / cubic2','0.6 / cubic2','0.6 / cubic3','0.6 / alternating']
    x=list(range(4));colors=['#557799','#dd8833','#338877','#8877aa']
    fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    base=rows[0]
    axs[0,0].bar(x,[100*(r['mean_movement']/base['mean_movement']-1) for r in rows],color=colors)
    axs[0,0].set(title='Mean movement gain over cap 0.5',ylabel='%')
    axs[0,1].bar(x,[100*(r['mean_residual_change']/base['mean_residual_change']-1) for r in rows],color=colors)
    axs[0,1].set(title='Mean residual-reduction gain',ylabel='%')
    axs[1,0].bar([i-.18 for i in x],[100*r['maximum_world_m'] for r in rows],width=.36,label='Single refresh',color='#447799')
    axs[1,0].bar([i+.18 for i in x],[100*r['maximum_chain_world_m'] for r in rows],width=.36,label='Nine-refresh frozen covariance',color='#cc8833')
    axs[1,0].axhline(1,color='#aa4444',linestyle='--',label='Declared 1% limit')
    axs[1,0].set(title='Maximum original-coordinate momentum drift',ylabel='%')
    axs[1,0].legend(fontsize=8)
    axs[1,1].bar(x,[r['mean_sampled_refresh_ms'] for r in rows],color=colors)
    axs[1,1].axhline(by['qr']['mean_sampled_refresh_ms'],color='#555555',linestyle='--',label='Matched QR refresh')
    axs[1,1].set(title='Sampled complete-refresh cost',ylabel='ms / matrix refresh')
    axs[1,1].legend(fontsize=8)
    for ax in axs.flat:ax.set_xticks(x,labels,rotation=12,ha='right')
    fig.suptitle('Same incoming CIFAR states: radius and correction work\nOffline shadows; no candidate learning or whole-optimizer throughput claim',fontsize=12)
    fig.savefig(b/'radius_attribution.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
