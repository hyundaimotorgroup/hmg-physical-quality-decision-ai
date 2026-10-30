# Model card - weld domain (v0.1 preview)

> ## v0.1 preview
> The weights in `weights/weld` (v0.1.0) are a **preview release**. The classifier is in a mature state; the **detector is still being improved** (see [Known limitations](#known-limitations)). Decision support only - not a certified or safety-certified inspection system.

## Models
| | classifier | detector |
|---|---|---|
| class | `CropClassifierNet` | `OBBDetectorNet` (`small`) |
| parameters | 7.06 M | 8.69 M |
| input | RGB, 224x224 crop (shorter side resize + centre crop), float [0,1], no mean/std | RGB, 640x640 letterbox (pad 114), float [0,1] |
| output | logits `[BAD, GOOD]` + 1280-d embedding | oriented boxes (cx,cy,w,h,theta), one class `weld`, NMS-free one-to-one head |
| format | `classifier.safetensors` + manifest | `detector.safetensors` + manifest |

## Provenance (factual)
* Both architectures are **original implementations written in plain PyTorch** for this project; no third-party model code is included.
* **No pretrained weights** were used for initialisation; all weights were trained from random initialisation.
* The weights were obtained by training on the project's own labelled weld images and crops.
* See [LICENSING.md](LICENSING.md).

## Intended use and safety
**Intended use:** assistive screening of weld photos and video - locate weld spots/beads and flag likely-BAD ones for a human reviewer. It is **decision support**.

**Not intended for:** sole basis of a safety-critical release or acceptance decision; replacement of a qualified inspector or of a certified inspection procedure; use on materials, processes, cameras or lighting that were not validated by you.

**Safety notes**
* This model is **not certified** and **not safety-certified**. No claim is made that it meets any inspection standard.
* A GOOD verdict is not proof of quality. The classifier under-calls BAD (see limitations); an image with zero detections is reported as GOOD by default, which only means that nothing was found.
* Validate on your own data before relying on it, rebuild the reference bank for your site, and review uncertain crops (`used_retrieval=True`) and low-confidence detections manually.
* The pipeline is configured for safety-first aggregation: one BAD frame makes an object BAD, one BAD object makes the item BAD. This increases sensitivity but also false alarms.

## Results (v0.1 preview, held-out data, full pipeline)
Values are for the v0.1.0 weights in `weights/weld` and are also recorded in the weight manifests. The classifier is evaluated on group-held-out crops, detection-label crops and held-out video crops. The BAD-class figures rest on very few samples (21 BAD crops in the original-label set) and are noisy.

| classifier (crops) | value |
|---|---|
| pipeline-level agreement (TTA + bank gate) | 96.6 % |
| accuracy vs. original labels | 93.3 % |

| detector (99 photos, tiled pipeline, conf 0.1 / verify 0.2) | value |
|---|---|
| recall @IoU 0.5 | 79.6 % |
| precision @IoU 0.5 | 51.2 % |
| recall @IoU 0.7 | 46.1 % |

Precision is low by design: the pipeline extracts candidates at a low threshold and a later verification step filters them.

## Known limitations
* **Detector boxes are loose at tight IoU.** Recall at IoU 0.5 is 79.6 %, dropping to 46.1 % at IoU 0.7, so localisation is roughly right but not tight. Thresholds may need re-tuning for your data.
* **BAD is under-called.** BAD recall is about 60 % on a small original-label validation subset with 21 BAD samples - a noisy estimate. Rely on the reference-bank gate and human review for BAD detection.
* **Data:** phone photos/video of a limited number of workpieces and lighting setups; expect degradation elsewhere. Rebuild the reference bank for your site.
* **Label scarcity:** BAD examples are about a quarter of the data.
* **Evaluation:** metrics are on a small held-out set whose data source overlaps with training data; treat them as indicative, not as guarantees.
