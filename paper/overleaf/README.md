# TurboSOAP: editable Overleaf project

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
