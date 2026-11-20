"""Build the per-reviewer P2 prompt bundles (moderator bookkeeping).

Each bundle: header, Document 1 (P1 brief), Document 2 (own P1 answer), Documents 3-5 (others, labelled X/Y/Z in a
seeded random order per reviewer), Document 6 (reveal with labels substituted and the sanitized Part 2 text), then the
P2 task.  Raw reviewer texts come from transcripts/ (no provenance headers); Fable's own provenance section (which
quotes auto-injected memory-index lines) is stripped.
"""
import json
import random
import re
from pathlib import Path

PANEL = Path('/workspace/TurboSOAP/research/reviews/panel_20261120')
REPORT = Path('/workspace/TurboSOAP/reports/refresh_study_20261120')
SEED = 20261120

raw = {
    'astra': (PANEL / 'transcripts/P1_gpt-6.2-astra_final.md').read_text(),
    'fable': (PANEL / 'transcripts/P1_fable_handback.md').read_text(),
    'neon': (PANEL / 'transcripts/P1_gemini-4-neon-xhigh_response.md').read_text(),
    'carbon': (PANEL / 'transcripts/P1_gemini-4-carbon-high_response.md').read_text(),
}
# strip Fable's provenance block (everything up to and including the first '---' line)
lines = raw['fable'].splitlines()
cut = next(i for i, l in enumerate(lines) if l.strip() == '---')
assert lines[0].startswith('## Provenance'), lines[0]
raw['fable'] = '\n'.join(lines[cut + 1:]).lstrip('\n')

model_ids = dict(astra='gpt-6.2-astra', fable='fable', neon='gemini-4-neon-xhigh', carbon='gemini-4-carbon-high')

# ---- sanitized Part 2 text
rep = REPORT.joinpath('REPORT.md').read_text()
part2 = rep[rep.index('## Part 2'):]
part2 = part2.split('\nFiles:')[0]
subs = [
    ("1-layer Transformers from the repo's `transformer.py`:", '1-layer Transformers of the same architecture family:'),
    ("**Driver.** `soap_reference.ResearchSOAP` QR10 with the stock betas\n  (.95/.95/.95), built via `owt_restart.build_optimizer`.",
     '**Driver.** The FP32 SOAP reference with QR10 and the stock betas\n  (.95/.95/.95).'),
    ('An\noffline chain of ResearchSOAP\'s QR refresh', 'An\noffline chain of the reference\'s QR refresh'),
    ('The full table is in `tables.md` and `ablation.json`.', 'The full table follows this section.'),
    ('; see the per-dimension breakdown in `replay.json`.', '.'),
]
for a, b in subs:
    assert a in part2, a[:50]
    part2 = part2.replace(a, b)
part2 = re.sub(r'from the cached OpenWebText at\n  `[^`]+`\. ', 'from the cached OpenWebText. ', part2)
part2 = re.sub(r', in BF16, under\n  `[^`]+` \(≈40 GB\)\.', ', in BF16.', part2)
part2 = part2.replace('## Part 2: open-loop replay and warm-step ablation', '### Part 2 report (verbatim except paths and file names removed)')
part2 = part2.replace('\n### ', '\n#### ').replace('\n#### Part 2 report', '\n### Part 2 report')
tables = REPORT.joinpath('tables.md').read_text()
abl = tables[tables.index('## Ablation'):]
abl = abl.replace('## Ablation (late window; warm10/QR50 chains)', '#### Full ablation table (late window; warm10/QR50 chains; "_qr10"/"_qr50" are reference rows)')
part2 = part2.rstrip() + '\n\n' + abl.strip() + '\n'
for bad in ('/workspace', '.py', 'replay.json', 'ablation.json', 'tables.md', 'soap_reference', 'owt_restart'):
    assert bad not in part2, bad

template = (PANEL / 'P2_REVEAL_TEMPLATE.md').read_text().replace('⟦PART2⟧', part2.strip())
brief1 = (PANEL / 'P1_BRIEF.md').read_text()
task = (PANEL / 'P2_BRIEF.md').read_text()

rng = random.Random(SEED)
assignment = {}
for me in ['astra', 'fable', 'neon', 'carbon']:
    others = [o for o in ['astra', 'fable', 'neon', 'carbon'] if o != me]
    rng.shuffle(others)
    labels = dict(zip(others, ['X', 'Y', 'Z']))
    assignment[model_ids[me]] = {model_ids[o]: l for o, l in labels.items()}
    reveal = template.replace(f'⟦{me}⟧', 'you')
    for o, l in labels.items():
        reveal = reveal.replace(f'⟦{o}⟧', f'Reviewer {l}')
    assert '⟦' not in reveal
    parts = [
        '# Stage 2 of a review panel: reveal, cross-reading and ranking\n\n'
        'This prompt contains six documents followed by the stage-2 task. Read the documents first; the task is at the '
        'end. The ground rules inside Document 1 (no web, no tools) applied to stage 1 only; stage 2 has its own rules, '
        'stated in the task.\n',
        '# Document 1: the stage-1 brief (verbatim)\n\n' + brief1.strip(),
        '# Document 2: your stage-1 answer (verbatim)\n\n' + raw[me].strip(),
    ]
    for o, l in sorted(labels.items(), key=lambda kv: kv[1]):
        parts.append(f'# Document {3 + "XYZ".index(l)}: Reviewer {l}\'s stage-1 answer (verbatim, anonymized)\n\n'
                     + raw[o].strip())
    parts.append('# Document 6: stage-2 reveal\n\n' + reveal.strip())
    parts.append('# Stage-2 task\n\n' + task.strip())
    text = '\n\n---\n\n'.join(parts) + '\n'
    for bad in ('Gemini', 'GPT', 'Claude', 'Fable', 'Neon', 'Carbon', 'astra', 'ethan', 'gmail', '/workspace',
                'claude.ai'):
        assert bad not in text, (me, bad)
    (PANEL / f'prompts/P2_{model_ids[me]}.md').write_text(text)
    print(me, len(text.encode()), 'bytes', len(text.split()), 'words', labels)
(PANEL / 'prompts/P2_assignment.json').write_text(json.dumps(dict(seed=SEED, labels_seen_by=assignment), indent=1))
