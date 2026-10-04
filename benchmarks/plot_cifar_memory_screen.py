"""Plot fixed-update, within-policy effects without inferring seed uncertainty."""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
b=p.parse_args().bundle;s=json.loads((b/'summary.json').read_text())
assert s['status']=='completed' and s['effective_group_memory_checks_passed']
policies=['qr10','qr20','warm10'];locations=np.arange(len(policies))
fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
for variant,shift,color,label in [('cov99',-.18,'#2864b4','Covariance beta → .99'),
                                  ('v99',.18,'#d87922','Matrix Adam beta2 → .99')]:
    values=[next(x['ce_delta_vs_own_baseline'] for x in s['rows']
                 if x['policy']==policy and x['variant']==variant and not x['repeated']) for policy in policies]
    bars=ax.bar(locations+shift,values,width=.34,color=color,label=label)
    ax.bar_label(bars,labels=[f'{v:+.4f}' for v in values],padding=4,fontsize=10)
ax.axhline(0,color='#444444',linewidth=1)
ax.set_xticks(locations,['QR10','QR20','Warm10 + QR200'])
ax.set_ylabel('Validation CE change vs own baseline (negative is better)')
ax.set_title('CIFAR-100: independent memory changes\nOne seed, 20 epochs; matrix/auxiliary LR .0005')
ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.22)
for suffix in ['png','svg']:fig.savefig(b/('memory_effects.'+suffix),dpi=170)
plt.close(fig)
