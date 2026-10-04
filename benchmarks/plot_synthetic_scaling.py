"""Show dimension/workload sensitivity of estimated warm-to-QR cost ratios."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle
s=json.loads((b/'summary.json').read_text());assert s['status'] in ['completed','completed_with_rejections']
grid=s['comparisons']+s['unavailable']
widths=sorted({r['width'] for r in grid});tokens_grid=sorted({r['tokens'] for r in grid})
fig,axes=plt.subplots(2,len(tokens_grid),figsize=(5.5*len(tokens_grid),7.5),sharex=True,sharey=True,squeeze=False,layout='constrained')
for i,mode in enumerate(['all','smaller_side']):
    for j,tokens in enumerate(tokens_grid):
        ax=axes[i,j]
        for baseline,color in [('qr10','#2864b4'),('qr20','#cc7722')]:
            rows=sorted([r for r in s['comparisons'] if r['mode']==mode and r['tokens']==tokens and
                         r['baseline']==baseline],key=lambda r:r['width'])
            for metric,style,label in [('optimizer_cuda_ms','-','optimizer'),('whole_step_wall_ms','--','whole step')]:
                by_width={r['width']:r for r in rows}
                ax.plot(widths,[by_width[w]['ratio_warm_over_baseline'][metric]['median'] if w in by_width else float('nan') for w in widths],
                        style,color=color,marker='o',label=f'Warm/{baseline.upper()} {label}')
        ax.axhline(1,color='#555555',linewidth=1)
        ax.set_title(f"{'Two-sided' if mode=='all' else 'Smaller side'}; {tokens} activation rows")
        ax.grid(alpha=.2);ax.set_axisbelow(True);ax.set_xticks(widths)
        ax.set_xlabel('Block width (rectangular weights expand 4×)')
        if j==0:ax.set_ylabel('Estimated warm / QR cost\nBelow 1 means cheaper')
        rejected=sorted({r['width'] for r in s['unavailable'] if r['mode']==mode and r['tokens']==tokens})
        if rejected:ax.text(.03,.04,'Guard rejected at width '+', '.join(map(str,rejected)),transform=ax.transAxes,color='#a12626',fontsize=9)
axes[0,0].legend(frameon=False,fontsize=8)
fig.suptitle('Synthetic SOAP scaling: cadence-weighted event estimates\nWarm10 + QR200; eager BF16 mock block; rejected cases have no ratio',fontsize=13)
natural=b/'natural_cycle/results.json'
if natural.exists():
    d=json.loads(natural.read_text())
    failures=[c for c in d['cases'] if c['status']=='guard_rejected']
    if failures:fig.supxlabel('Separate natural-cycle check also rejected warm at width 768; passing event probes do not establish sustained stability.',fontsize=10,color='#a12626')
for suffix in ['png','svg']:fig.savefig(b/f'scaling_ratios.{suffix}',dpi=170)
plt.close(fig)
