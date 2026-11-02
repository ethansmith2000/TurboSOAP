# TurboSOAP paper working package

Start with [the manuscript](DRAFT.md) and [outline/editorial plan](OUTLINE.md).
This is a first technical draft, dated2026-10-28, using the completed bounded
study. Author/affiliation and venue formatting remain editorial inputs.

- [DRAFT.md](DRAFT.md): abstract, introduction, related work, explicit method,
  experimental design, results, limitations and appendices.
- [CLAIMS.md](CLAIMS.md): what each claim is supported by and its scope.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): entry point, exact configuration,
  data/runtime versions, CPU regeneration and historical launch records.
- [evidence.json](evidence.json): calculated results and source SHA256 inventory.
- [configuration.json](configuration.json): recorded training/method/data recipe.
- [tables.md](tables.md): five generated tables, also embedded in the manuscript.
- [references.bib](references.bib): versioned primary-reference bibliography.
- [figures](figures): byte-identical SVG copies from the completed report.

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
results. The check also verifies four production optimizer pins and both figures.
The earlier completion report can regenerate its plots with Matplotlib as
described in the reproduction guide.

The main precision option is fixed fullBF16 M/V/C/Q with FP32 master parameters,
warm10/QR50 and compiled matrix groups. The manuscript reports a loss gap rather
than claiming exact matching. Primary training speed is measured against the
original stock-beta QR10 implementation. Development-pilot and cached-prefix
records remain distinct from independent seed replication.

The experiment ledger is completed and its monitor remains disabled. Creating
this draft package did not change production optimizer code or run new training.
The preexisting completion backup remains unchanged; the paper work is saved
separately in `/workspace/soap-paper-draft-20261028.tar.gz` after validation.
