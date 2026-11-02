# Review PDF build

`../TurboSOAP-review.pdf` is the current working proof, prepared on 29 October
2026. It incorporates the verified findings from all four model reviews,
documented in `../reviews_20261029/RESPONSES.md`.
The first proof is preserved as `../reviews_20261029/proof_before_astra.pdf`.
The current PDF includes the supplied Ethan Smith name/email. No affiliation
or venue is specified. The separate Overleaf version is the editable delivery;
see ../overleaf/README.md. Historical review validation describes the earlier
pre-author proof; current authorship/export checks are in ../overleaf_validation.json.

From the repository root:

```bash
python3 paper/build_artifacts.py --check
python3 paper/build_pdf.py
python3 paper/build_pdf.py --check
```

The PDF build uses Pandoc, pdfLaTeX, Latin Modern fonts, `rsvg-convert`, and
Poppler. This machine installed Ubuntu packages `pandoc`,
`texlive-latex-recommended`, `texlive-fonts-recommended`, `lmodern`,
`librsvg2-bin`, and `poppler-utils` for document preparation. Exact tool versions
and input/output hashes are in `build.json`.

The generated `manuscript.tex` and vector PDF figures remain here for inspection.
Two TeX passes resolve document links. Build logs, PDF metadata, and extracted
text are retained. The first proof has eight pages; the expanded reviewed
draft has nine. The final build and visual inspection are recorded in
`../reviewed_validation_20261029.json`. These are layout checks, distinct
from independent scientific review.

The Markdown manuscript is the authoring source. Regenerate this proof after
any manuscript change; `--check` fails on changed source or output hashes.
This build neither imports the optimizer nor uses a GPU. The old draft manifest
and 28 October archive describe the first Markdown package and remain historical.
