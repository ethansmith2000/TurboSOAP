# Reviewed working artifact package

The latest editing delivery is [TurboSOAP-Overleaf.zip](TurboSOAP-Overleaf.zip),
with [compiled preview](TurboSOAP-Overleaf.pdf) and [editable source](overleaf).
It lists Ethan Smith and the supplied email, with no affiliation or venue.
This source-only ZIP was compiled from a fresh extraction; Overleaf needs no
Python/Pandoc installation. The delivery manifest is overleaf_export.json.
The larger reviewed archive below remains a historical pre-author snapshot.

The reviewed PDF is [TurboSOAP-review.pdf](TurboSOAP-review.pdf).
Its authoring source is [DRAFT.md](DRAFT.md); generated LaTeX and vector PDF
figures are in [typeset](typeset). The four independent model reviews, original
snapshot and finding-by-finding responses are in [reviews_20261029](reviews_20261029).
The supplied author metadata is in author.json. Public artifact hosting remains
open; no affiliation or venue has been added. Nothing has been submitted,
committed or published.

## Local archive

The separate archive is `/workspace/soap-paper-reviewed-20261029.tar.gz`,
with an adjacent SHA256 file. It contains the working TurboSOAP research code,
tests, benchmark runners, notes, compact reports, paper package and six frozen
campaign bundles under `optimizer_replay_results/`:

- `soap_owt_restart_20261026_v4`: original reference and LR calibration.
- `soap_bf16_learning_20261026_v1`: five-arm development pilot.
- `soap_bf16_full_20261026_seed205_v1`: full seed205 pair.
- `soap_bf16_replicate_20261027_seed204_v1`: seed204 replication/replay.
- `soap_bf16_refresh_controls_20261027_seed204_v1`: short QR controls.
- `soap_bf16_qr_profile_20261027_v1`: synthetic local QR profile.

The bundles retain their frozen sources, scalar logs, geometry, manifests or
provenance, protocols, and recorded launch files. Dataset arrays, compiler
caches, Python bytecode and Git metadata are omitted. No trained tensor
checkpoints were saved by these studies. The package supports scalar/paper
reconstruction and source inspection; training reproduction separately needs
the pinned dataset and an appropriate admitted GPU.

## CPU checks and builds

From the extracted TurboSOAP directory:

```bash
python3 paper/build_artifacts.py --check
python3 paper/review_context.py --check
python3 paper/build_pdf.py --check
```

The first two commands use only the standard library. The PDF hash check also
uses only Python; regeneration requires the tools in [typeset/README.md](typeset/README.md).
To regenerate the review figure, evidence records and PDF:

```bash
python3 paper/review_context.py
python3 paper/plot_review_context.py
python3 paper/build_artifacts.py
python3 paper/build_pdf.py
```

Figure rendering requires Matplotlib and runs on CPU. Source and output hashes
are saved; PDF timestamps can change on rebuild. The exact recipe, historical
execution settings and archive filenames are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
The original completed-experiment archive and 28 October first-draft archive
are preserved separately. This reviewed package changes no production optimizer,
frozen result, protocol, GPU resource reservation or experiment schedule.
