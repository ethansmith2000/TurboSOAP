"""Typeset the working manuscript with Pandoc/TeX; no research execution.

Dependencies: pandoc, pdflatex, rsvg-convert and Poppler utilities.
Run from any directory. Generated LaTeX, vector figures and build logs are
retained under paper/typeset; the PDF is paper/TurboSOAP-review.pdf.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE / 'typeset'
PDF = HERE / 'TurboSOAP-review.pdf'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, *, stdin=None):
    result = subprocess.run(args, cwd=OUT, input=stdin, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed:\n{result.stdout[-8000:]}')
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    source_text = (HERE / 'DRAFT.md').read_text()
    figure_names = re.findall(r'!\[[^\]]*\]\((figures/[^)]+\.svg)\)', source_text)
    assert len(figure_names) == 2 and len(set(figure_names)) == 2
    inputs = [HERE / 'DRAFT.md', HERE / 'build_pdf.py', HERE / 'author.json',
              *[HERE / name for name in figure_names]]
    input_hashes = {str(p.relative_to(HERE)): sha(p) for p in inputs}
    record = OUT / 'build.json'
    if args.check:
        saved = json.loads(record.read_text())
        assert saved['inputs'] == input_hashes, 'PDF source inputs changed'
        for name, expected in saved['outputs'].items():
            assert sha(HERE / name) == expected, f'Output changed: {name}'
        print('PASS: PDF source and output hashes match the recorded build.')
        return
    for name in ['pandoc', 'pdflatex', 'rsvg-convert', 'pdfinfo', 'pdftotext']:
        if not shutil.which(name):
            raise SystemExit(f'Missing typesetting dependency: {name}')
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    for figure_name in figure_names:
        name = Path(figure_name).stem
        run(['rsvg-convert', '--format=pdf', '--output',
             str(OUT / 'figures' / (name + '.pdf')),
             str(HERE / 'figures' / (name + '.svg'))])
    text = source_text
    title, body = text.split('\n', 1)
    body = re.sub(r'^\s*Working manuscript,.*?\n\s*\n', '', body, count=1,
                  flags=re.S)
    abstract, body = body.split('## 1. Introduction', 1)
    abstract = abstract.replace('## Abstract', '').strip()
    body = '## 1. Introduction' + body
    body = re.sub(r'^(#{2,}) ', lambda m: m[1][1:] + ' ', body, flags=re.M)
    body = body.replace('.svg)', '.pdf)')
    # Caption paragraphs already supply the full description. Suppress the
    # short Markdown alt caption to avoid duplicate figure captions.
    body = re.sub(r'!\[[^\]]*\]\((figures/[^)]+)\)', r'![](\1)', body)
    body = re.sub(r'<!--.*?-->', '', body, flags=re.S)
    # Keep repository-local links readable as artifact names in a standalone PDF.
    body = re.sub(r'\[([^\]]+)\]\((?!https?://)([^)]+\.md)\)',
                  lambda m: f'{m[1]} (`{m[2]}`)', body)
    ast = json.loads(run(['pandoc', '-f', 'markdown+tex_math_single_backslash',
                          '-t', 'json'], stdin=body))
    blocks = []
    for block in ast['blocks']:
        if (block['t'] == 'Table' and blocks and blocks[-1]['t'] == 'Para'
                and len(blocks[-1]['c']) == 1
                and blocks[-1]['c'][0]['t'] == 'Emph'):
            caption = blocks.pop()
            blocks.extend([
                {'t': 'RawBlock', 'c': ['latex', r'\begin{minipage}{\linewidth}']},
                caption, block,
                {'t': 'RawBlock', 'c': ['latex', r'\end{minipage}']},
            ])
        else:
            blocks.append(block)
    ast['blocks'] = blocks
    abstract_ast = json.loads(run(['pandoc', '-f', 'markdown', '-t', 'json'],
                                  stdin=abstract))
    author = json.loads((HERE / 'author.json').read_text())
    assert author['affiliation'] is None and author['venue'] is None
    ast['meta'] = {
        'title': {'t': 'MetaString', 'c': title.lstrip('# ')},
        'author': {'t': 'MetaInlines', 'c': [
            {'t': 'Str', 'c': author['name']},
            {'t': 'RawInline', 'c': ['latex', r'\\{\small ']},
            {'t': 'Link', 'c': [['', [], []], [{'t': 'Str', 'c': author['email']}],
                                 ['mailto:' + author['email'], '']]},
            {'t': 'RawInline', 'c': ['latex', '}']},
        ]},
        'author-meta': {'t': 'MetaString', 'c': author['name']},
        'date': {'t': 'MetaString', 'c': 'Working review draft | 29 October 2026'},
        'abstract': {'t': 'MetaBlocks', 'c': abstract_ast['blocks']},
        'documentclass': {'t': 'MetaString', 'c': 'article'},
        'fontsize': {'t': 'MetaString', 'c': '10pt'},
        'geometry': {'t': 'MetaString', 'c': 'margin=0.85in'},
        'colorlinks': {'t': 'MetaBool', 'c': True},
        'linkcolor': {'t': 'MetaString', 'c': 'blue'},
        'urlcolor': {'t': 'MetaString', 'c': 'blue'},
    }
    header = OUT / 'header.tex'
    header.write_text(r'''\usepackage{microtype}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{0.35em}
\setcounter{secnumdepth}{0}
\AtBeginDocument{\hypersetup{pdfsubject={Working manuscript; empirical optimizer case study}}}
''')
    tex = run(['pandoc', '-f', 'json', '-t', 'latex', '--standalone',
               '--include-in-header', str(header)], stdin=json.dumps(ast))
    # Every manuscript table is short. Keep each table together instead of
    # allowing Pandoc's longtable output to split five-row tables across pages.
    def keep_table(match):
        block = match[0]
        footer = '\\endhead\n\\bottomrule\\noalign{}\n\\endlastfoot\n'
        assert footer in block, 'Unexpected Pandoc table footer'
        block = block.replace(footer, '', 1)
        block = block.replace(r'\begin{longtable}[]', r'\begin{tabular}', 1)
        block = block.replace(r'\end{longtable}',
                              '\\bottomrule\n\\end{tabular}', 1)
        return '\\begin{center}\n' + block + '\n\\end{center}'
    tex, count = re.subn(r'\\begin\{longtable\}\[\].*?\\end\{longtable\}',
                        keep_table, tex, flags=re.S)
    assert count == 5, f'Expected five short manuscript tables, found {count}'
    (OUT / 'manuscript.tex').write_text(tex)
    for i in range(2):
        log = run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                   '-no-shell-escape', 'manuscript.tex'])
        (OUT / f'pdflatex-pass{i + 1}.txt').write_text(log)
    shutil.copyfile(OUT / 'manuscript.pdf', PDF)
    info = run(['pdfinfo', str(PDF)])
    extracted = run(['pdftotext', '-layout', str(PDF), '-'])
    (OUT / 'pdfinfo.txt').write_text(info)
    (OUT / 'extracted.txt').write_text(extracted)
    warnings = [s for s in (OUT / 'manuscript.log').read_text().splitlines()
                if any(x in s for x in ['Overfull', 'Missing character',
                                       'LaTeX Warning', 'undefined'])]
    outputs = [PDF, OUT / 'manuscript.tex', OUT / 'header.tex',
               *[OUT / Path(name).with_suffix('.pdf') for name in figure_names]]
    record.write_text(json.dumps({
        'built_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'inputs': input_hashes,
        'outputs': {str(p.relative_to(HERE)): sha(p) for p in outputs},
        'tools': {n: run([n, '--version']).splitlines()[0]
                  for n in ['pandoc', 'pdflatex', 'rsvg-convert']},
        'pdfinfo': info, 'warnings': warnings,
        'authors_or_venue_invented': False, 'gpu_execution': False,
    }, indent=2) + '\n')
    print(info)
    print(json.dumps({'warnings': warnings, 'pdf': str(PDF)}, indent=2))


if __name__ == '__main__':
    main()
