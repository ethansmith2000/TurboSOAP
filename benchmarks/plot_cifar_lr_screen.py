"""Static LR-screen figure; no inferred uncertainty from a single seed."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--combined',action='store_true');args=p.parse_args();b=args.bundle
s=json.loads((b/('combined_lr_summary.json' if args.combined else 'summary.json')).read_text())
assert s['status']==('verified' if args.combined else 'completed')
fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
for policy,color in [('qr10','#2864b4'),('warm10','#d87922'),('qr20','#22875a')]:
    rows=sorted([r for r in s['rows'] if r['policy']==policy and not r.get('repeated',False)],key=lambda r:r['matrix_lr'])
    ax.plot([r['matrix_lr'] for r in rows],[r['validation_ce'] for r in rows],
            marker='o',label=policy,color=color,linewidth=2)
rates=sorted({r['matrix_lr'] for r in s['rows'] if not r.get('repeated',False)})
ax.set_xscale('log');ax.set_xticks(rates,[f'{rate:g}' for rate in rates])
ax.minorticks_off();ax.set_xlabel('Matrix peak learning rate (auxiliary LR fixed at 0.0005)')
ax.set_ylabel('Validation cross-entropy at 3,500 updates (lower is better)')
ax.set_title(('CIFAR-100: verified combined LR bracket' if args.combined else 'CIFAR-100: bounded SOAP LR screen')+
             '\nOne seed; 20 epochs on the original 200-epoch LR schedule')
ax.grid(alpha=.2);ax.legend(frameon=False)
for suffix in ['png','svg']:fig.savefig(b/(('combined_lr_screen.' if args.combined else 'lr_screen.')+suffix),dpi=170)
plt.close(fig)
