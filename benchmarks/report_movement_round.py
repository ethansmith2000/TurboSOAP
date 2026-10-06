"""Connect common-state mechanism, synthetic cost and separately measured learning."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root
    load=lambda n:json.loads((root/n/'summary.json').read_text())
    a=load('cifar_soap_movement_attribution_20261006');n=load('soap_movement_natural_20261006')
    l=load('cifar_soap_movement_budget_20261006')
    assert a['status']==n['status']==l['status']=='completed'
    by={x['policy']:x for x in a['rows']};f=by['fro_cubic2'];r=by['row_cubic2']
    rows=[x for x in l['rows'] if not x['repeat']]
    qualified=l['repeat_qualified_wall_interpretation']
    lines=['# Movement bound research round','',
        'Completed 2026-10-06 (America/New_York). Three sequential stages: actual CIFAR common-state attribution, natural synthetic cycles through large factors, then paired fixed-update and measured-wall learning. Production/defaults unchanged; all states stayed in RAM, with no retained weights.','',
        '## What changed','',
        'The new candidate replaces the maximum-row-sum rotation cap with `||A||F / sqrt(2)`, an upper bound on the spectral norm of a real skew matrix. It retains cap .5 and two unscaled cubic polar corrections. Both factors, refresh cadence, QR resets, Adam memory, covariance beta, LR and strict M transport remain matched. The deployed policy adds no eigensolve, power iteration or online error controller.','',
        '## Mechanism and qualification','',
        f"On identical CIFAR incoming states, Frobenius-cubic2 has {f['mean_movement']/r['mean_movement']:.2f}× the movement and {f['mean_residual_change']/r['mean_residual_change']:.2f}× the covariance-residual reduction of row-cubic2. Maximum sampled world-M error is {100*f['maximum_world_m']:.3f}%. This is direct local clipping attribution; it does not establish a learning benefit.",'',
        'The Frobenius bound was tighter in 1,343 of 1,344 sampled factor states; this is not a universal ordering. It remains conservative for high-rank rotations. The exact-spectral oracle retains more movement but requires an offline eigensolve. Frobenius-quintic1 cap .4 also passed shadow checks but moved less; only cubic2 advances in the trained intervention.','',
        '![Common-state attribution](../cifar_soap_movement_attribution_20261006/movement_attribution.png)','',
        'All 18 natural-cycle cases qualify through QR resets at 200/400, including both exact repeats. Largest factor is 6144. Whole-history and per-refresh M errors remain below the predeclared 1% criterion. At width 1536/beta .999:','',
        '| Policy | Mean optimizer ms/update |','|---|---:|']
    for x in n['rows']:
        if x['width']==1536 and x['beta']==.999 and not x['repeat']:
            lines.append(f"| {x['policy']} | {x['optimizer_ms']:.3f} |")
    lines+=['',
        'The two cubic2 bounds cost essentially the same here. These are instrumented synthetic cycles with offline probes excluded; they are not end-to-end CIFAR timing.','',
        '## Paired learning','',
        'One seed, 3500 fixed updates on the original 200-epoch LR schedule, both covariance betas and fixed LR .0005. Secondary endpoint: first 25-step boundary reaching 90 measured training seconds, excluding init/evaluation/offline probes. Original update-indexed schedule is retained; it is not independently tuned for each wall horizon.','',
        '| Cov beta | Policy | CE at 3500 | Optimizer ms | 90s steps | 90s CE |',
        '|---:|---|---:|---:|---:|---:|']
    for x in rows:
        w=x['budget'];tail=f"{w['step']} | {w['ce']:.6f}" if w else 'capped | unavailable'
        lines.append(f"| {x['beta']} | {x['kind']} | {x['ce']:.6f} | {x['optimizer_ms']:.3f} | {tail} |")
    lines+=['',
        '![Fixed-update and measured-wall learning](../cifar_soap_movement_budget_20261006/learning_budget.png)','',
        'More movement improves fixed-update CE modestly: Frobenius beats row cubic2 by .00344 at beta .99 and .02721 at .999, while still trailing QR and NS6. This is a one-seed effect, not a replicated practical improvement. The large immediate tracking gain did not yield a comparably large learning gain.','',
        f"The predeclared timing-repeat gate **{'passed' if qualified else 'failed'}**. It requires <=3% fixed 3500 training-wall and <=2% optimizer range/mean for both repeated .99 controls. Both controls reproduce fixed-update weights, metrics and numerical probes exactly, but that alone does not qualify timing.",'',
        '| Repeated control | Training-wall range/mean | Optimizer range/mean | Gate |',
        '|---|---:|---:|---|']
    for c in l['controls']:lines.append(f"| {c['label']} | {100*c['ranges']['training_seconds']:.2f}% | {100*c['ranges']['optimizer_ms']:.2f}% | {c['qualified']} |")
    if not qualified:
        lines+=['',
            '**Measured-wall endpoints remain exploratory. No time-to-quality winner is promoted.** Model/backward timing varies substantially across arms; similar optimizer costs can coexist with different update counts. Fixed arm order and only two timing repeats do not identify the cause. Do not reinterpret this as a hardware effect or combine synthetic savings with CIFAR losses.']
    lines+=['','## Decision and next steps','',
        'Keep Frobenius-cubic2 as an experimental mechanism control. It improves movement at almost unchanged cubic2 cost, but it does not close the fixed-update quality gap to QR/NS6. Do not promote new defaults or launch a broad cap/coefficient grid.','',
        'Before interpreting small wall-quality differences, qualify timing with repeated/interleaved identical work and separately account for model, optimizer and host dispatch. Preserve the failed timing criterion and raw endpoints. A larger-shape trainer is still required for target end-to-end gains.','',
        'For the algorithm, separate bound conservatism from allowed movement radius. On reconstructed common states, compare a slightly larger radius paired with the minimum fixed correction work that meets a prospectively declared approximation target, versus a scheduled stronger correction. Use the derived scalar Gram-error maps to choose a narrow candidate set; include every extra reduction/GEMM and moment transport in cost. Residual decrease remains a diagnostic, not the selection objective.','',
        'Replicate surviving learning effects on seeds 271/811 and longer horizons before promotion. Explicit BF16 execution, natural-cycle compilation/state ownership, equal cumulative optimizer budgets and exact original H200 provenance remain open. Current results do not supersede earlier cubic1 or QR/full-reference failures.','',
        '## Verification and evidence','',
        f"189 CPU tests passed. Attribution: 250 actual-control refresh guards, 840 sampled matrices including an exact repeat, four exact historical observations. Natural qualification: {n['updates']} updates and two exact model/loss/diagnostic repeats. Learning: {l['trainer_steps']} trainer steps, {l['refresh_guards']} refresh guards, {l['matrix_probes']} targeted probes and {len(l['anchors'])} exact historical observations; both fixed-update repeats/probes match.",'',
        'The selected-basis implementation checks remain separate from full strict-refresh differences; QR/reset disagreements are preserved. A 1% M criterion is a predeclared qualification target, not proof of optimal learning.','',
        '- [Common-state attribution](../cifar_soap_movement_attribution_20261006/README.md)',
        '- [Natural-cycle qualification](../soap_movement_natural_20261006/README.md)',
        '- [Paired learning and timing repeats](../cifar_soap_movement_budget_20261006/README.md)',
        '- [Provenance and retention](PROVENANCE.md)','',
        'All GPU work is complete and the claim is released. Compact evidence is retained; no weights/checkpoints/tensor snapshots were saved. Prior evidence and archives are unchanged.']
    b=root/'soap_movement_round_20261006';(b/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
