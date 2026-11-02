"""Create a standalone, editable Overleaf project and compile its upload ZIP.

Export is intentionally one-shot: an existing editing folder is never replaced.
Use --check to verify the delivered project without changing it.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE / 'overleaf'
ZIP = HERE / 'TurboSOAP-Overleaf.zip'
PDF = HERE / 'TurboSOAP-Overleaf.pdf'
RECORD = HERE / 'overleaf_export.json'
SECTION_NAMES = [
    '01_introduction', '02_related_work', '03_method', '04_experimental_design',
    '05_results', '06_discussion', '07_conclusion', '08_pilot_appendix',
    '09_reproducibility_appendix',
]
CITATIONS = {
    '2409.11321v2': 'vyas2024soap',
    '2605.26327v1': 'milligan2026reparametrizing',
    '2506.03595v2': 'eschenhagen2025purifying',
    '2405.18144v3': 'wang2024fourbit',
    '10.2307/1993275': 'forsythe1960cyclic',
    '10.1137/0708036': 'bjorck1971iterative',
}
PREAMBLE = r'''\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{microtype}
\usepackage{parskip}
\usepackage[margin=0.85in]{geometry}
\usepackage{booktabs,array,calc}
\usepackage{graphicx}
\usepackage[authoryear,round]{natbib}
\usepackage{xcolor}
\PassOptionsToPackage{hyphens}{url}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\IfFileExists{xurl.sty}{\usepackage{xurl}}{}
\urlstyle{same}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{0.35em}
% Section titles already contain the reviewed manuscript's numbers.
\setcounter{secnumdepth}{0}
\setkeys{Gin}{width=\linewidth,keepaspectratio}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
'''
README = '''# TurboSOAP: editable Overleaf project

Author: Ethan Smith
Contact: ethansmith2000@gmail.com
No affiliation, organization, or venue is specified.

Upload TurboSOAP-Overleaf.zip using New Project > Upload Project in Overleaf.
Select main.tex as the main document and pdfLaTeX as the compiler, then Recompile.
Overleaf handles the BibTeX bibliography automatically.
Official instructions: https://docs.overleaf.com/managing-projects-and-files/uploading-a-project

## Where to edit

- sections/00_abstract.tex: abstract.
- sections/01_introduction.tex through 07_conclusion.tex: paper sections.
- sections/08_pilot_appendix.tex and 09_reproducibility_appendix.tex: appendices.
- metadata.tex: title, author, contact and draft date.
- references.bib: bibliography; use citep/citet commands for citations.
- preamble.tex: page layout and packages.
- figures/: the two vector PDF figures.
- main.tex: document order and bibliography selection.

All prose, equations, tables and captions are ordinary editable LaTeX.
The project has no Python, Pandoc, GPU, shell-escape or external-file dependency.
The PDF preview is supplied separately from the upload ZIP.

## Local compilation, if desired

    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex

The scientific content and table values come from the reviewed 29 October 2026
manuscript. Citation formatting uses natbib/plainnat with a live BibTeX database.
Once you start editing in Overleaf, treat that LaTeX project as your editing copy.
The repository's export script refuses to overwrite an existing project folder.
Later Markdown changes and Overleaf edits must be merged deliberately.
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return [HERE / 'DRAFT.md', HERE / 'author.json', HERE / 'references.bib',
            HERE / 'typeset/manuscript.tex', Path(__file__).resolve(),
            HERE / 'typeset/figures/full_horizon.pdf',
            HERE / 'typeset/figures/review_tradeoff.pdf']


def compile_project(project):
    logs = []
    for command in [
        ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-no-shell-escape', 'main.tex'],
        ['bibtex', 'main'],
        ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-no-shell-escape', 'main.tex'],
        ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-no-shell-escape', 'main.tex'],
    ]:
        result = subprocess.run(command, cwd=project, text=True, capture_output=True)
        logs.append(' '.join(command) + '\n' + result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(logs[-1][-9000:])
    final_log = (project / 'main.log').read_text()
    warnings = [s for s in final_log.splitlines()
                if any(x in s for x in ['Overfull', 'Missing character',
                                       'LaTeX Warning', 'Package natbib Warning', 'undefined'])]
    assert not warnings, warnings
    assert (project / 'main.bbl').read_text().count(r'\bibitem') == 6
    return '\n\n'.join(logs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        saved = json.loads(RECORD.read_text())
        for name, expected in saved['inputs'].items():
            assert sha(HERE / name) == expected, f'Export input changed: {name}'
        for name, expected in saved['project_files'].items():
            assert sha(PROJECT / name) == expected, f'Project edited: {name}'
        assert sha(ZIP) == saved['zip_sha256']
        assert sha(PDF) == saved['pdf_sha256']
        print('PASS: authored Overleaf sources, preview PDF and upload ZIP match the export.')
        return
    if PROJECT.exists() or ZIP.exists():
        raise SystemExit('Overleaf editing copy already exists; refusing to overwrite it.')
    author = json.loads((HERE / 'author.json').read_text())
    assert author['affiliation'] is None and author['venue'] is None
    tex = (HERE / 'typeset/manuscript.tex').read_text()
    title = re.search(r'\\title\{([^{}]+)\}', tex)[1]
    abstract = tex.split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0].strip()
    body = tex.split(r'\end{abstract}', 1)[1].split(r'\hypertarget{references}', 1)[0]
    cited = set()

    def cite(match):
        url, label = match.groups()
        key = next((value for fragment, value in CITATIONS.items() if fragment in url), None)
        if key is None:
            return match[0]
        cited.add(key)
        citation = r'\citep{' + key + '}'
        return 'SOAP~' + citation if label.strip() == 'SOAP' else citation

    body = re.sub(r'\\href\{([^{}]+)\}\{([^{}]*)\}', cite, body, flags=re.S)
    body = re.sub(r'\(\s*(\\citep\{[^}]+\})\s*\)', r'\1', body)
    assert cited == set(CITATIONS.values()), cited
    heads = list(re.finditer(r'\\hypertarget\{[^}]+\}\{%\s*\\section\{', body))
    assert len(heads) == len(SECTION_NAMES), len(heads)
    assert not body[:heads[0].start()].strip()
    (PROJECT / 'sections').mkdir(parents=True)
    (PROJECT / 'figures').mkdir()
    (PROJECT / 'sections/00_abstract.tex').write_text(abstract + '\n')
    for i, name in enumerate(SECTION_NAMES):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        (PROJECT / 'sections' / (name + '.tex')).write_text(body[heads[i].start():end].strip() + '\n')
    (PROJECT / 'preamble.tex').write_text(PREAMBLE)
    metadata = (
        '\\title{' + title + '}\n'
        '\\author{' + author['name'] + r'\\{\small \href{mailto:' +
        author['email'] + r'}{\nolinkurl{' + author['email'] + '}}}}\n'
        r'\date{Working draft \textbar{} 29 October 2026}' + '\n' +
        '\\hypersetup{pdftitle={' + title + '},pdfauthor={' + author['name'] + '}}\n')
    (PROJECT / 'metadata.tex').write_text(metadata)
    main_tex = '% Main document: select this file in Overleaf.\n'
    main_tex += '\\documentclass[10pt]{article}\n'
    main_tex += '\\input{preamble.tex}\n\\input{metadata.tex}\n\\begin{document}\n\\maketitle\n'
    main_tex += '\\begin{abstract}\n\\input{sections/00_abstract.tex}\n\\end{abstract}\n'
    main_tex += ''.join('\\input{sections/' + name + '.tex}\n' for name in SECTION_NAMES)
    main_tex += '\\bibliographystyle{plainnat}\n\\bibliography{references}\n\\end{document}\n'
    (PROJECT / 'main.tex').write_text(main_tex)
    bib = (HERE / 'references.bib').read_text().replace('Björck, Å.', r'Bj{\"o}rck, {\AA}.')
    (PROJECT / 'references.bib').write_text(bib)
    (PROJECT / 'README.md').write_text(README)
    for name in ['full_horizon', 'review_tradeoff']:
        shutil.copyfile(HERE / 'typeset/figures' / (name + '.pdf'),
                        PROJECT / 'figures' / (name + '.pdf'))
    files = {str(p.relative_to(PROJECT)): sha(p)
             for p in sorted(PROJECT.rglob('*')) if p.is_file()}
    # Build from the exact upload package rather than relying on repository paths.
    with zipfile.ZipFile(ZIP, 'x', compression=zipfile.ZIP_DEFLATED) as output:
        for name in files:
            output.write(PROJECT / name, arcname=name)
    with tempfile.TemporaryDirectory(prefix='soap-overleaf-') as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(ZIP) as zipped:
            zipped.extractall(extracted)
        log = compile_project(extracted)
        shutil.copyfile(extracted / 'main.pdf', PDF)
        (HERE / 'overleaf_compile.log').write_text(log)
        info = subprocess.check_output(['pdfinfo', str(PDF)], text=True)
        text = subprocess.check_output(['pdftotext', '-layout', str(PDF), '-'], text=True)
        assert author['name'] in text and author['email'] in text
        assert all(key in (extracted / 'main.bbl').read_text() for key in cited)
        (HERE / 'overleaf_extracted.txt').write_text(text)
    record = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  author=author, main_document='main.tex', compiler='pdfLaTeX',
                  bibliography='natbib/plainnat + BibTeX', citations=sorted(cited),
                  inputs={str(p.relative_to(HERE)): sha(p) for p in sources()},
                  project_files=files, zip_sha256=sha(ZIP), pdf_sha256=sha(PDF),
                  pdfinfo=info, final_tex_warnings=[], fresh_zip_compilation='passed',
                  gpu_execution=False, external_upload=False)
    RECORD.write_text(json.dumps(record, indent=2) + '\n')
    print(info)
    print('PASS: standalone ZIP compiles with six resolved bibliography entries.')
    print(ZIP)


if __name__ == '__main__':
    main()
