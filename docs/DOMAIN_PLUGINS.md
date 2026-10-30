# Writing a domain plugin

A domain is a `DomainSpec` (see `src/pqdai/domains/__init__.py`):

| field | meaning |
|---|---|
| `name`, `description` | registry key and one-liner |
| `class_names` | classifier output order; **index 0 must be the unsafe class** (`unsafe_class`) - hard-vote ties and the "any unsafe object fails the item" rule rely on it |
| `detector_factory(weights_path, device, thresholds)` | returns an object with `detect(image_path) -> [Detection(polygon (4,2), confidence)]` and `last_rejected_candidates`; normally a `TiledDetector` |
| `classifier_factory(weights_path, device)` | returns an object with `predict_batch_with_uncertainty(list_of_bgr)` -> dicts with `probabilities`, `embedding`, ... |
| `preprocess_fn` | BGR crop -> BGR crop (lighting normalisation etc.) |
| `thresholds` | `conf`, `verify_conf`, `alpha`, `k`, `crop_pad_ratio` |
| `weights_dir`, `detector_file`, `classifier_file` | release layout (`*.safetensors` + `*.manifest.json` + `manifest.json`) |
| `reference_bank_path` | `.npz` bank in the classifier's embedding space |

Steps: copy `src/pqdai/domains/example_template/` to `src/pqdai/domains/<name>/`, fill in the spec, add the name to `BUILTIN` in `domains/__init__.py` (or publish a package exposing an entry point in group `pqdai.domains` that returns a `DomainSpec`), ship weights under `weights/<name>/`, then build a bank:

```bash
pqdai bank build --domain <name> --crops my_crops/ --out bank.npz   # my_crops/<CLASS>/*.jpg
pqdai image --domain <name> --image sample.jpg
```

The template ships stub models only (no weights); `tests/test_domains.py` shows how to unit-test a plugin with stubs. Video mode currently expects `['BAD', 'GOOD']` class names.

## See also
* [REFERENCE_BANK.md](REFERENCE_BANK.md) - building the per-domain reference bank that the plugin points to
* [MODEL_CARD.md](MODEL_CARD.md) - the weld domain's model card, a template for documenting a new domain
* [architecture.md](architecture.md) - how the core pipeline calls into a domain
* [VIDEO.md](VIDEO.md) and [WEBAPP.md](WEBAPP.md) - video tracking and the web app, both domain-selectable (`--domain`, `PQDAI_DOMAIN`)
* [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) - what to do before publishing a new domain's weights
