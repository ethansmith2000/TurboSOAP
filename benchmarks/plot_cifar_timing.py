"""Static diagnostic plots from completed timing evidence, with no training."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    b=Path(sys.argv[1]);r=json.loads((b/'screen.json').read_text())
    fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    colors={'qr':'#3366aa','fro':'#c46a23'}
    runs=r['runs']
    for a in runs:
        x=[h['step'] for h in a['history']]
        ax[0,0].plot(x,[h['model_backward_cuda_ms'] for h in a['history']],label=a['label'],alpha=.85)
        ax[0,1].plot(x,[h['optimizer_ms'] for h in a['history']],label=a['label'],alpha=.85)
    ax[0,0].set(title='Model/backward CUDA-event interval',xlabel='Update',ylabel='ms / update')
    ax[0,1].set(title='Optimizer CUDA-event interval',xlabel='Update',ylabel='ms / update')
    ax[0,0].legend(ncol=3,fontsize=8)
    pos=list(range(len(runs)))
    ax[1,0].bar(pos,[a['training_seconds'] for a in runs],color=[colors[a['kind']] for a in runs])
    ax[1,0].set(xticks=pos,xticklabels=[a['label'] for a in runs],title='Full 3500-update training wall',ylabel='seconds')
    for a,i in zip(runs,pos):
        host=sum(h['host']['phases']['model_backward']['host_wall_ms'] for h in a['history'])/3500
        main=sum(h['host']['phases']['model_backward']['thread_cpu_ms'] for h in a['history'])/3500
        process=sum(h['host']['phases']['model_backward']['process_cpu_ms'] for h in a['history'])/3500
        ax[1,1].bar(i-.24,host,width=.24,color='#3366aa',label='Host wall' if i==0 else None)
        ax[1,1].bar(i,main,width=.24,color='#44aa88',label='Main-thread CPU' if i==0 else None)
        ax[1,1].bar(i+.24,process,width=.24,color='#c46a23',label='Process CPU' if i==0 else None)
    ax[1,1].set(xticks=pos,xticklabels=[a['label'] for a in runs],title='Model/backward host accounting',ylabel='ms / update')
    ax[1,1].legend(fontsize=8)
    fig.suptitle('Interleaved identical-work timing — seed 139, covariance .99\nEvent intervals include dispatch gaps; host and GPU intervals overlap',fontsize=12)
    fig.savefig(b/'timing_diagnostic.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
