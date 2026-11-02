# TurboSOAP paper working package

The [Overleaf upload ZIP](TurboSOAP-Overleaf.zip) and [its compiled PDF](TurboSOAP-Overleaf.pdf)
are ready. The editable source is in [overleaf](overleaf), with separate LaTeX
sections and a live BibTeX bibliography. Author: Ethan Smith; contact is in
[author.json](author.json). No affiliation or venue is specified. Upload the ZIP
as a project and compile main.tex with pdfLaTeX; see the included README.
After starting edits in Overleaf, treat that LaTeX project as the editing copy.

The [current PDF proof](TurboSOAP-review.pdf) and [review responses](reviews_20261029/RESPONSES.md)
are available. Astra, Sol, Fable and Opus reviews are complete and addressed.
The [typesetting guide](typeset/README.md) records
CPU-only PDF build commands and layout checks.

Start with [the manuscript](DRAFT.md) and [outline/editorial plan](OUTLINE.md).
This is a first technical draft, dated2026-10-28, using the completed bounded
study. Author metadata is now supplied, with no affiliation or venue.

- [DRAFT.md](DRAFT.md): abstract, introduction, related work, explicit method,
  experimental design, results, limitations and appendices.
- [CLAIMS.md](CLAIMS.md): what each claim is supported by and its scope.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): entry point, exact configuration,
  data/runtime versions, CPU regeneration and historical launch records.
- [evidence.json](evidence.json): calculated results and source SHA256 inventory.
- [configuration.json](configuration.json): recorded training/method/data recipe.
- [review_context.json](review_context.json): descriptive event accounting,
  pilot/control contrasts and LR selection, with its own CPU checker.
- [tables.md](tables.md): five generated tables, also embedded in the manuscript.
- [references.bib](references.bib): versioned primary-reference bibliography.
- [figures](figures): preserved report exports and a new review-derived figure.
- [ARTIFACTS.md](ARTIFACTS.md): reviewed package contents and regeneration.

From the repository root, using ordinary Python3:

```bash
python3 paper/build_artifacts.py --check
```

To rebuild the generated table blocks, evidence inventory and configuration:

```bash
python3 paper/build_artifacts.py
```

These commands use only the Python standard library and saved scalar evidence.
They do not import Torch, request a GPU, launch experiments, or change frozen
results. The check also verifies four production optimizer pins, two frozen
report figure copies, and the review figure's source/output manifest.
The earlier completion report can regenerate its plots with Matplotlib as
described in the reproduction guide.

The main precision option is fixed fullBF16 M/V/C/Q with FP32 master parameters,
warm10/QR50 and compiled matrix groups. The manuscript reports a loss gap rather
than claiming exact matching. Primary training speed is measured against the
original stock-beta QR10 implementation. Development-pilot and cached-prefix
records remain distinct from independent seed replication.

The experiment ledger is completed and its monitor remains disabled. Creating
this draft package did not change production optimizer code or run new training.
The preexisting completion backup and first Markdown archive remain unchanged.
The reviewed working package is saved separately as
`/workspace/soap-paper-reviewed-20261029.tar.gz` after validation.
That archive and its validation records preserve the pre-author review package.
The authored Overleaf delivery has its own input/output inventory in
overleaf_export.json and content/layout checks in overleaf_validation.json.
