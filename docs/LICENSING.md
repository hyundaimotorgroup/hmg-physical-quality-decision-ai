# Licensing and provenance

* **Code and weights**: Apache License 2.0 (see `LICENSE`, `NOTICE`; the license is also recorded in every weights manifest).
* **Third-party dependency licenses**: see [`THIRD_PARTY_LICENSES`](../THIRD_PARTY_LICENSES) for the full list of direct and transitive open source dependencies, their licenses, and copyright/attribution notices.
* **Architectures**: original plain-PyTorch implementations in `src/pqdai/models`; no third-party model code is included.
* **Initialisation**: all weights were trained from random initialisation; no pretrained checkpoints were used.
* **Training signal**: trained on the project's own labelled data.
* **Dependencies**: PyTorch, NumPy, OpenCV, SciPy, Shapely, ensemble-boxes, safetensors, Pillow; optional FastAPI. None of the inference code imports `ultralytics` or `sahi` (enforced by `tests/test_public_surface.py`).
* **Data**: no training images or customer media are distributed. The reference bank contains embeddings only, with relative basenames at most.

This document states facts about how the artefacts were produced; it is not legal advice. Downstream users who need legal certainty (for example for commercial redistribution) should perform their own review.
