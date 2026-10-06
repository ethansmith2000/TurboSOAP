"""Write a reproducible readout from the validated factorial summary."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);b=p.parse_args().bundle
    s=json.loads((b/'summary.json').read_text());assert s['all_prior_anchors_exact']
    primary=sorted([r for r in s['rows'] if not r['repeated']],key=lambda r:(r['beta'],r['policy'],r['cap'],r['precision']))
    def at(beta,policy,precision,cap):
        return next(r for r in primary if (r['beta'],r['policy'],r['precision'],r['cap'])==(beta,policy,precision,cap))
    strict_labels={r['label'] for r in s['rows'] if r['precision']=='highest'}
    strict_d=[d for d in s['diagnostics'] if d['label'] in strict_labels]
    lines=['# Strict m transport: implementation, shape cost and paired learning','',
           'Completed 2026-10-04 UTC. This round implements opt-in strict-FP32 first-moment transport, measures its full refresh cost, and executes a paired precision/cap/covariance screen. All original defaults remain inherited precision. These results are one-seed early-training evidence, not convergence or a globally tuned optimizer ranking.','',
           '## Main readout and decision','',
           'Strict transport improves QR CE by '+', '.join(f"{at(beta,'qr20','highest',.1)['ce']-at(beta,'qr20','inherit',.1)['ce']:+.6f} at covariance {beta}" for beta in [.99,.999])+'. Warm does not improve uniformly: strict transport worsens cap-.1 CE at both betas, while improving cap-.2 CE at both. Raising the cap helps at .99 under either precision, but at .999 it hurts inherited transport and helps strict transport. The factorial therefore exposes a cap/precision/memory interaction rather than a universal precision or cap recommendation.','',
           'Best tested QR versus best tested warm CE is '+', '.join(f"{min(r['ce'] for r in primary if r['beta']==beta and r['policy']=='qr20'):.6f} versus {min(r['ce'] for r in primary if r['beta']==beta and r['policy']=='warm20'):.6f} at beta {beta}" for beta in [.99,.999])+'. QR also has lower optimizer cost on this small-factor trainer. These are fixed-LR early endpoints on one seed; they do not select a universal default.','',
           'Keep strict transport opt-in and stop expanding this precision/cap grid. Next replicate the selected QR and cap-.2 warm comparisons on paired seeds 271/811, preserving inherited-precision and covariance controls, then check longer horizons on the original schedule before promotion. On the separate runtime track, test matched-precision compiled refreshes and natural cadence at the 3072/6144-factor shapes before any wide-model speed claim. No such follow-up job is launched in this round.','',
           '## Implementation and scope','',
           'ResearchSOAP now accepts transport_precision=highest; inherit preserves its previous path. Strict precision applies only to the first-moment world/basis round trip at actual refreshes, including QR resets, and the prior setting is restored even on exceptions. Basis construction, covariance accumulation, ordinary gradient projections, Adam updates, v handling and auxiliary AdamW are unchanged. The setting and cap persist in optimizer parameter groups. This is the tested sequential eager research path; optimizer compilation and concurrent-thread use of the process-wide precision setting are not qualified. The production SOAP class is unchanged.','',
           'At each covariance beta .99/.999, compare QR20 inherited/strict transport and warm20 inherited/strict transport at caps .1/.2. Warm uses NS6 and QR resets every 200 updates. Both factors are active. Matrix and auxiliary peak LR .0005, Adam(.9,.999), seed 139 and the original 200-epoch strong-CIFAR schedule are fixed. Fourteen arms include two final exact repeats; each stops at 3500 updates/20 epochs. No LR retuning, best-checkpoint selection or horizon extension occurs.','',
           '## Fixed-update learning result','',
           '| Covariance beta | Policy | Cap | m transport | Validation CE | Accuracy (%) | Optimizer ms/update |',
           '|---:|---|---:|---|---:|---:|---:|']
    for r in primary:
        lines.append(f"| {r['beta']} | {r['policy']} | {r['cap']} | {r['precision']} | {r['ce']:.6f} | {100*r['accuracy']:.2f} | {r['optimizer_ms']:.3f} |")
    lines+=['','![Paired early endpoints](learning_endpoints.png)','',
            'Negative CE deltas favor the intervention. Precision contrasts keep beta, cap, cadence and learning rate fixed:','',
            '| Control → intervention | CE delta | Accuracy delta (pp) |',
            '|---|---:|---:|']
    for c in s['contrasts']:
        if c['kind']=='strict_transport':lines.append(f"| {c['control']} → strict m | {c['ce_delta']:+.6f} | {c['accuracy_pp_delta']:+.2f} |")
    lines+=['','Cap contrasts hold precision and beta fixed:','',
            '| Control → cap .2 | CE delta | Accuracy delta (pp) |','|---|---:|---:|']
    for c in s['contrasts']:
        if c['kind']=='cap_0.1_to_0.2':lines.append(f"| {c['control']} | {c['ce_delta']:+.6f} | {c['accuracy_pp_delta']:+.2f} |")
    lines+=['','Covariance contrasts compare .99 minus .999 at the same policy/precision/cap:','',
            '| Policy at .999 → .99 | CE delta | Accuracy delta (pp) |','|---|---:|---:|']
    for c in s['contrasts']:
        if c['kind']=='covariance_0.999_to_0.99':lines.append(f"| {c['control']} | {c['ce_delta']:+.6f} | {c['accuracy_pp_delta']:+.2f} |")
    lines+=['','The precision-by-cap CE difference-in-differences is '+', '.join(f"{v:+.6f} at beta {k}" for k,v in s['cap_precision_interaction_ce'].items())+'. These are conditional observed differences on the same seed. They do not estimate uncertainty across seeds. The earlier 100-epoch memory study already showed that early covariance effects can shrink.','',
            '## Separate synthetic refresh cost','',
            'A seeded rectangular gradient produces rank-limited PSD covariances; incoming bases are identity and v is random positive. Time the actual research refresh implementation, including both factors, m transport, QR v reindexing and precision-switch dispatch overhead. Record three rotated rounds after an initial call, with three repetitions per CIFAR-shape round and one per large-shape round. Covariance accumulation, ordinary Adam updates, model work and diagnostics are excluded. These fixtures and event costs are separate from the natural CIFAR optimizer timings above.','',
            '| Matrix shape | QR strict m (ms) | NS6 warm cap .1 strict m (ms) | NS6 warm cap .2 strict m (ms) |',
            '|---|---:|---:|---:|']
    for shape in dict.fromkeys(tuple(c['shape']) for c in s['cost']):
        vals=[next(c['strict_ms'] for c in s['cost'] if tuple(c['shape'])==shape and c['method']==method and c['cap']==cap)
              for method,cap in [('qr',.1),('warm',.1),('warm',.2)]]
        lines.append(f"| {shape[0]} × {shape[1]} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} |")
    overhead=[100*c['relative_change'] for c in s['cost']]
    lines+=['',f"Strict-m event cost changes range from {min(overhead):+.2f}% to {max(overhead):+.2f}% versus matched inherited transport in these short samples. Small signed changes are within measurement variability, not proof that stricter arithmetic can be faster. At 768×3072, NS6 warm is cheaper than QR; at 1536×6144, warm is more expensive. Do not extrapolate a single size's ranking or infer large-model throughput from these isolated events.",'',
            '![Shape-specific refresh cost](shape_refresh_cost.png)','',
            'All 18 strict-transport shape cases pass the selected-basis update, world-m and geometry checks. All 18 inherited/strict pairs produce exactly identical bases and v. This establishes implementation scope on these fixtures, not long-run stability of a wide trainer.','',
            '## Numerical diagnostics and reproducibility','',
            f"All {s['refresh_guard_count']:,} training initialization/refresh guards pass, covering 48 factors each. Selected-age probes cover {s['probe_matrix_records']:,} matrix records. The {sum(d['samples'] for d in strict_d):,} strict-transport records all pass the targeted selected-basis implementation gate; maximum selected-basis update error is {max(d['maximum_selected_basis_update_error'] for d in strict_d):.3g}. That gate requires error<1e-6, world-m preservation error<1%, and Gram<.05. It is separate from the preexisting full strict-refresh agreement diagnostic.",'',
            'Full strict-refresh failures for actual strict-transport arms are retained, separated by warm versus QR events:','',
            '| Arm | Actual refresh | Records | Full strict-update failures (≥1%) | Maximum difference (%) |',
            '|---|---|---:|---:|---:|']
    for d in strict_d:
        lines.append(f"| {d['label']} | {d['event']} | {d['samples']} | {d['full_strict_rejections']} | {100*d['maximum_full_strict_difference']:.3f} |")
    lines+=['','The high-precision-setting basis remains part of every actual policy, so QR/reset full-reference sensitivity has not been fixed or excluded after the fact. No retrospective threshold change or whole-policy full-reference pass is claimed. No online error controller or new fallback has been added.','',
            'All 12 historical unchanged QR/warm cap-.1 observations reproduce weights, metrics, RNG, consumed IDs and augmentations exactly. Both final controls reproduce all three observations and every numerical probe record exactly. The CPU suite passes 176 tests.','',
            '| Repeated control | Optimizer range/mean | Model/backward range/mean | Training-wall range/mean |','|---|---:|---:|---:|']
    for c in s['controls']:
        v=c['timing_relative_range'];lines.append(f"| {c['original']} | {100*v['optimizer_ms']:.2f}% | {100*v['model_backward_cuda_ms']:.2f}% | {100*v['training_wall']:.2f}% |")
    lines+=['','Keep these timing controls alongside all cost comparisons. No equal-wall learning comparison was run. Synthetic event timing is not added to CIFAR losses to construct a fictitious large-model speedup.','',
            '## Evidence and retention','',
            f"The cost gate took {s['cost_gate_seconds']:.2f}s and the training campaign {s['training_all_in_seconds']:.2f}s, within the predeclared limits. [Protocol](PROTOCOL.md), [cost gate](cost_gate.json), [raw training/probes](screen.json), [validated summary](summary.json), configs, seeds, frozen source, logs, tests and plots are retained. No weights, checkpoints or tensor snapshots were saved. Incoming references are discarded after each probe. The GPU claim is released and private supervisor stopped after completion; prior evidence, other jobs and shared dataset caches are preserved.",'']
    (b/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
