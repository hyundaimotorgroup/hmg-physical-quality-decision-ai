# Detector comparison for weld inspection

How the current object-detector families look from the point of view of this project's weld setting, and how to swap the detector if you want to.

> **Read this first**
> * This is a **literature-based** comparison. **No head-to-head benchmark of the other detectors on this project's weld data was run.** Nothing below says that another detector would beat (or lose to) the bundled one on your photos.
> * All external numbers were retrieved on **2026-10-20**. Each is labelled *vendor-reported* (the number comes from the vendor's own docs/README), *paper-reported*, or *third-party*. Numbers from different rows were measured on **different hardware, input sizes, datasets and protocols** and must not be compared across rows.
> * Licence column: the retrieval tooling available when this was written could not open `LICENSE` files directly (GitHub/arXiv fetches were blocked), so licence entries come from search-result excerpts of READMEs/docs/model cards. They are marked **reported** (a project page or README says so) or **unverified**. Read the actual `LICENSE` file before you ship anything. This is not legal advice.
> * The detector shipped in this repo is a **v0.1 preview** with the limitations described in [section 1](#1-our-setting).

Contents: [1 Our setting](#1-our-setting) - [2 Comparison table](#2-comparison-of-recent-families) - [3 Notes per family](#3-notes-per-family) - [4 What matters for weld](#4-what-matters-for-weld) - [5 Guidance and plug-in contract](#5-guidance) - [6 Benchmark protocol](#6-how-to-benchmark-your-detector-on-your-data) - [7 Sources](#7-sources)

---

## 1. Our setting

| Property | Value |
|---|---|
| Task | find individual weld nuggets/beads in phone photos and video frames |
| Image size | up to 5712x4284 photos, 1080p video frames |
| Object size | small: median about 6 % of the image side |
| Boxes | **oriented** - crops are cut along the rotated box and passed to a GOOD/BAD classifier, so the angle matters downstream |
| Stage-1 priority | **recall over precision**: extract all candidates at a low threshold, then a verification pass (zoomed re-detection) and the classifier filter |
| Labels | about 1,000 labelled photos, 7.2k weld boxes, single class |
| Compute | workstation GPU; CPU / Mac desirable |
| Bundled detector | from-scratch PyTorch, anchor-free, oriented: CSP backbone + PAN neck + dual one-to-many / one-to-one head, ProbIoU-style rotated losses, NMS-free inference path, **8.7 M parameters**, 640x640 letterbox input (`OBBDetectorNet`, `small`). Trained from random initialisation, no pretrained weights ([MODEL_CARD.md](MODEL_CARD.md)) |
| Inference wrapper | `TiledDetector` (`src/pqdai/core/tiled_detector.py`): tiling for images above 1600 px (1024 px slices, 0.35 overlap), optional TTA, ROI re-detection with `verify_conf` |

### Where the bundled detector stands (own measurements, honest)

From [MODEL_CARD.md](MODEL_CARD.md), 99 validation photos, tiled pipeline, conf 0.1 / verify 0.2:

| | bundled detector |
|---|---|
| recall @ IoU 0.5 vs labels | 79.6 % |
| precision @ IoU 0.5 | 51.2 % |
| recall @ IoU 0.7 | 46.1 % |

* Boxes are comparatively loose at tight IoU thresholds and confidence scores are on the low side, so thresholds need re-tuning on your data.
* Precision about 51 % is by design (low stage-1 threshold, filtered later).
* Evaluation set is small: indicative, not a guarantee.

---

## 2. Comparison of recent families

Legend: **Oriented** = native (the official release trains/predicts rotated boxes), adapt (needs code changes, e.g. an angle head and rotated loss), none (axis-aligned only). **NMS-free** = end-to-end head, no NMS post-processing needed in the official inference path. **Fit** is my judgement for *this* project from the literature only, not a measurement.

### 2.1 Capabilities, licences, practicality

| Family | Oriented | NMS-free | Code licence | Weights licence | Small objects | Data efficiency | CPU / edge | Fit for weld |
|---|---|---|---|---|---|---|---|---|
| **YOLO26** (Ultralytics, Jan 2026) | native (`-obb` models) | yes (default one-to-one head; one-to-many head + NMS optional) | AGPL-3.0 or paid Enterprise (**reported**) | AGPL-3.0 by default, Enterprise for closed use (**reported**); includes weights you train with it | P3-P5 head by default; no P2 variant in the official list I saw (**unverified**) | strong: COCO/DOTA-pretrained weights, fine-tunes well on small sets | good: ONNX/OpenVINO/CoreML export, n/s sizes | **high** technically (native OBB, pretrained, CPU-friendly); **licence is the gating issue** for closed-source use |
| **YOLO11-OBB** (Ultralytics) | native | no (NMS) | AGPL-3.0 / Enterprise (**reported**) | same as above (**reported**) | as above | strong (pretrained) | good | **medium-high**; older than YOLO26, needs NMS |
| **RF-DETR** (Roboflow) | none in the official release (detection + segmentation) | yes (DETR) | Apache-2.0 for Nano-Large; PML 1.0 for "plus" models Atto/Femto/Pico/XL/2XL detection (**reported**) | same split as code (**reported**) | DINOv2 backbone, patch-based; resolution 384-880 px by size; small objects at 640 px input are a risk (no stride-4 map) | very strong: designed for fine-tuning from a DINOv2 backbone; paper evaluates on RF100-VL | GPU-oriented; CPU is slow for transformers (**unverified** numbers) | **medium**: great data efficiency and licence for Nano-Large, but axis-aligned only |
| **D-FINE** | none | yes (DETR) | **unverified** (a Roboflow page reports Apache-2.0; the LICENSE file was not opened) | **unverified**; README reportedly warns that Objects365-pretrained checkpoints (`*_obj365`, `*_obj2coco`) may be subject to Objects365 terms | HGNetv2 + multi-scale decoder; no dedicated P2 | good (Objects365 pretrain helps +2 AP on S) | GPU-oriented | **medium**: strong accuracy, no oriented boxes |
| **DEIM / DEIMv2** | none | yes (DETR) | DEIM v1: Apache-2.0 (**reported**). **DEIMv2: custom "DEIMv2 License", reportedly changed from Apache-2.0 in Aug 2026 and said to restrict commercial use - check the repo** | DINOv3 backbones carry their own terms (**unverified**) | DINOv3 + spatial tuning adapter; small sizes down to 0.5 M params | good (DINOv3 features) | small variants (Pico 1.5 M, Atto 0.5 M) are edge-friendly | **medium**: no OBB, licence uncertain |
| **RT-DETR / RT-DETRv2** | none | yes (DETR) | Apache-2.0 (**reported** by HF model card and third-party; repo LICENSE not opened) | **unverified** (HF port lists Apache-2.0) | hybrid encoder, ResNet backbones, 640 px | good (ImageNet / Objects365 pretrain) | GPU (TensorRT) oriented; R18 is the lightest | **medium**: mature, widely used in weld papers, no OBB |
| **RTMDet / RTMDet-R** (MMDetection / MMRotate) | RTMDet-R native; RTMDet axis-aligned | no (NMS; the paper's latency excludes NMS) | Apache-2.0 (**reported** by the README) | **unverified**; DOTA/COCO-pretrained checkpoints inherit dataset terms | CSPNeXt backbone, strides 8-32; no P2 by default | good (COCO pretrain; DOTA pretrain for -R) | tiny/s variants export to ONNX/TensorRT; MMRotate setup is heavy | **medium-high**: native rotated boxes + permissive licence, but MMRotate tooling is dated and rotated NMS needs custom ops |
| **Oriented R-CNN** (MMRotate) | native | no | Apache-2.0 for MMRotate (**reported**); original authors' repo licence **unverified** | **unverified** | two-stage; RoI features help small objects somewhat | good (ImageNet backbone) | slow (15 FPS on 1024x1024 on a 2080Ti in the paper) | **medium**: accurate recipe, heavy for CPU |
| **LSKNet** (backbone, used inside Oriented R-CNN / ReDet in MMRotate) | native (via the detector it is plugged into) | no | Apache-2.0 (**reported** by a mirror; official repo not opened) | **unverified** | large selective kernels aimed at context-dependent remote-sensing objects | good (ImageNet-pretrained backbone) | heavier than CSP-style nets | **medium-low**: tuned for aerial context, our objects are small isolated nuggets |
| **RiO-DETR** (ECCV 2026) | native (oriented DETR) | yes | **unverified**; the repo is reported to be a **partial** release (the author says full research code cannot be released) | **unverified** | oriented dense one-to-one assignment, tested on DOTA | **unverified** for tiny datasets; no pretrained-weights story found | n: 4.0 M params, 2.7 ms on a T4 (paper) | **medium** as a design reference; **low** as a drop-in until the code is complete |
| **Weld-specific papers** (WELD-DETR, RT-DETR weld frameworks, DSF-YOLO, YOLOv8 variants) | none (axis-aligned, mostly **defect** detection, many on **X-ray**) | RT-DETR-based: yes; YOLOv8-based: no | mostly **unverified** / not released | **unverified** | multi-scale fusion, wavelet modules, P2 layers (SFW-YOLO) | small datasets (1k-3k images, augmented) | 30-67 FPS reported on unspecified GPUs | **low-medium** as drop-ins (different task and modality); useful as ideas |

### 2.2 Reported accuracy and speed (never compare across rows)

| Family / model | Dataset, protocol | Result | Hardware / runtime as stated by the source | Kind | Source |
|---|---|---|---|---|---|
| YOLO26n / s / m / l / x | COCO val2017, 640 px | mAP 40.9 / 48.6 / 53.1 / 55.0 / 57.5 (one-to-many head with NMS); end-to-end head 40.1 / 47.8 / 52.5 / 54.4 / 56.9 | n: 38.9 ms CPU ONNX (Intel Xeon 2.00 GHz) and 1.7 ms T4 TensorRT10; x: 525.8 ms CPU, 11.8 ms T4 | vendor-reported + paper (Ultralytics authors) | [Ultralytics README](https://github.com/ultralytics/ultralytics), [paper](https://arxiv.org/pdf/2509.25164) |
| YOLO26n-obb / s / m / l / x | DOTAv1 **test**, 1024 px, single model **multi-scale** | mAP50 78.9 / 80.9 / 81.0 / 81.6 / 81.7 | n: 97.7 ms CPU ONNX, 2.8 ms T4 TRT10; s: 218.0 ms, 4.9 ms; x: 1485.7 ms, 30.5 ms. Source text names both "T4" and an EC2 P4d host for the speed runs, so the hardware wording is inconsistent | vendor-reported | [Ultralytics README](https://github.com/ultralytics/ultralytics), [OBB docs](https://docs.ultralytics.com/tasks/obb) |
| YOLO11n-obb ... x | DOTAv1 test, 1024 px, multi-scale | mAP50 78.4 / 79.5 / 80.9 / 81.0 / 81.3 | n: 117.6 ms CPU ONNX, 4.4 ms T4 TRT10; x: 1408.6 ms, 28.6 ms | vendor-reported | [Ultralytics YOLO11 docs](https://docs.ultralytics.com/models/yolo11), [OBB docs](https://docs.ultralytics.com/tasks/obb) |
| Reproduction caveat | YOLO26n-obb T4 speed | a user reports not reproducing the published 2.8 ms (they measured 3.6 ms; setup not verified by me) | T4 TensorRT10 | third-party issue | [ultralytics#23689](https://github.com/ultralytics/ultralytics/issues/23689) |
| RF-DETR Nano | COCO val2017 | AP 48.0 (50:95) | T4, TensorRT FP16, batch 1 (Nano 2.3 ms); Roboflow measured accuracy in-house for every model it plots | paper-reported / vendor-measured | [paper](https://arxiv.org/abs/2511.09554), [Roboflow docs](https://rfdetr.roboflow.com/) |
| RF-DETR Medium / Large / 2XL | COCO val2017 | AP 54.7 @ 4.4 ms / 56.5 @ 6.8 ms / 60.1 @ 17.2 ms | T4 TensorRT FP16 batch 1 | vendor-reported | [Roboflow docs](https://rfdetr.roboflow.com/), [repo](https://github.com/roboflow/rf-detr) |
| D-FINE-S / L / X | COCO val2017 | AP 48.5 (3.49 ms, 10.2 M params, 25.2 GFLOPs) / 54.0 (124 FPS) / 55.8 (78 FPS); S with Objects365 pretrain: 50.7 | T4, TensorRT FP16, batch 1 (TensorRT 10.4.0 per the PyPI mirror) | paper-reported | [paper](https://arxiv.org/html/2410.13842v1), [repo](https://github.com/Peterande/D-FINE) |
| DEIMv2-S / X / Pico | COCO | AP 50.9 (9.71 M params) / 57.8 (50.3 M, 151.6 GFLOPs) / 38.5 (1.5 M) | latency hardware not captured by my search - **unverified** | paper-reported | [paper](https://arxiv.org/pdf/2509.20787), [repo](https://github.com/Intellindust-AI-Lab/DEIMv2) |
| RT-DETR-R18 / R34 | COCO val2017, 640 px | AP 46.5 (217 FPS) / 48.9 (161 FPS) | single T4, batch 1, FP16, TensorRT >= 8.5.1 | paper / repo-reported | [repo](https://github.com/lyuwenyu/RT-DETR) |
| RT-DETRv2-S / L / X | COCO val2017 | AP 48.1 (217 FPS) / 53.4 (108 FPS) / 54.3 (74 FPS). The paper's table gives 47.9 for S and the repo README 48.1; model labels also differ between repo and paper | same T4 setup | paper / repo-reported | [paper](https://arxiv.org/pdf/2407.17140), [repo](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch) |
| RTMDet | COCO | 52.8 AP, 300+ FPS | RTX 3090, TensorRT 8.4.3 FP16, batch 1, **NMS time not counted** | paper-reported | [paper](https://ar5iv.labs.arxiv.org/html/2212.07784) |
| RTMDet-R | DOTA-v1.0 | 78.9 mAP single-scale, 81.3 multi-scale, 121 FPS | RTX 3090, TensorRT FP16, batch 1 (as listed in the MMRotate README per search excerpt) | vendor-reported | [MMRotate](https://github.com/open-mmlab/mmrotate) |
| Oriented R-CNN (R50-FPN) | DOTA | 75.87 mAP (80.87 multi-scale) | 15.1 FPS at 1024x1024 on one RTX 2080 Ti | paper-reported | [paper](https://arxiv.org/abs/2108.05699) |
| LSKNet (in Oriented R-CNN-style detectors) | DOTA-v1.0 | about 81.6-81.9 mAP, multi-scale; sources disagree (81.64 in the paper abstract as quoted, 81.85 on a listing) - I did not reconcile them | not captured | paper-reported | [repo](https://github.com/zcablii/LSKNet) |
| RiO-DETR-n / x | DOTA-1.0, single-scale | AP50 78.4 (2.7 ms, 4.0 M params) / 81.8 (29.9 ms) | single T4, TensorRT 10 FP16; end-to-end models exclude pre/post-processing, others use BatchedNMSPlugin. A third-party review notes the YOLO26-obb baseline is a community implementation and that latency protocols are not like-for-like | paper-reported | [paper](https://arxiv.org/abs/2603.09411), [repo](https://github.com/RicePasteM/RiO-DETR), [review](https://pith.science/paper/2603.09411) |
| RT-DETR-based weld framework (Adv. Eng. Informatics 2025) | own weld defect dataset, 1,134 images | mAP50 0.996, mAP50:95 0.801, 67 FPS | GPU not captured | paper-reported | [link](https://www.sciencedirect.com/science/article/pii/S1474034625002113) |
| WELD-DETR (Sensors 2025) | own welding-defect set, 3,247 images augmented to 12,000 | reports gains over YOLOv8s and EfficientDet-D0 (the excerpt I saw is ambiguous about the exact deltas) | not captured | paper-reported | [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC12656187) |
| CGRT-DETR (J. Failure Analysis & Prevention 2026) | class-imbalanced weld defects | 96.2 % mAP, 30 FPS, 10.49 M params | not captured | paper-reported | [link](https://link.springer.com/article/10.1007/s11668-026-02383-z) |
| DSF-YOLO (Sci. Reports 2025) | X-ray weld defects, 8 classes | mAP50 99.1 %, mAP50:95 74.7 %; 35.8 M params, 64.8 GFLOPs | x0.75 variant: 18.3 ms / 54.6 FPS, GPU not captured | paper-reported | [link](https://www.nature.com/articles/s41598-025-06811-2) |

**Reading these numbers**

* DOTA is **aerial imagery** with 15 classes and mostly larger, denser, differently-shaped objects than weld nuggets in phone photos. DOTA mAP50 only says a model can do oriented detection; it does not predict weld accuracy. COCO AP is on 640 px images of everyday objects.
* Speeds are for fixed-size inputs. For a 5712x4284 photo with 1024-px tiles, 0.35 overlap and a full-image pass you run **tens of forward passes per image**; per-tile latency multiplied by tile count is the number that matters.
* The weld papers solve a **different task**: defects (cracks, porosity, slag) inside a weld, mostly in X-ray images, with axis-aligned boxes. They are not evidence about finding weld nuggets in phone photos. Their datasets are also small (1-3k images), which supports "pretrained transformers work with few images" but nothing more.

---

## 3. Notes per family

* **YOLO26 / YOLO11-OBB.** The only mainstream family in this list that ships *native* oriented models with pretrained weights, a one-command fine-tune, and ONNX/CoreML/OpenVINO export. The YOLO26 docs describe an end-to-end (NMS-free) head and an angle-aware loss for OBB. The obstacle is the licence: AGPL-3.0 (source disclosure for derived works, including network use per third-party commentary) or a paid Enterprise licence, and the licence applies to models you train with it. This repo is Apache-2.0 and its public-surface test forbids importing `ultralytics`; therefore any Ultralytics detector must be an **optional, user-side adapter**, never bundled.
* **RF-DETR.** Best "few labelled images" story (DINOv2 backbone, fine-tuning is the design goal) and a permissive licence for Nano-Large. But no oriented boxes, and inputs are 384-880 px, so weld nuggets at ~6 % of the image side need tiling to stay resolvable. If you adopt it, the usable output is an axis-aligned box; crops cut from it contain more background than crops cut along a rotated box, which may hurt the classifier. A cheap workaround: use RF-DETR as stage 1 and estimate the angle in a second step (for example a small angle regressor or a minimum-area rectangle on a segmentation mask) - untested here.
* **D-FINE, DEIM/DEIMv2, RT-DETR/v2.** Strong COCO accuracy-per-latency on a T4, all axis-aligned. Check the licence file for each (DEIMv2 in particular) and check the Objects365 caveat on D-FINE checkpoints. They would need an angle head and rotated loss to be useful here.
* **RTMDet-R / Oriented R-CNN / LSKNet (MMRotate).** The most complete open recipes for rotated boxes and Apache-2.0 reported. The cost is engineering: MMRotate 1.x depends on MMEngine/MMCV/MMDetection version pinning and custom rotated ops, and I did not verify its current maintenance status. LSKNet's large-kernel design targets contextual remote-sensing objects; there is no evidence it helps for isolated weld nuggets.
* **RiO-DETR.** Interesting because it is a real-time oriented DETR with explicit handling of angle periodicity. As of the retrieval date its public code is reported incomplete, so it is a reference rather than an option.
* **Weld-specific literature.** Useful as a source of ideas (multi-scale fusion, wavelet modules, P2 layers for small defects, augmentation schedules). None targets "locate weld nuggets in phone photos with oriented boxes", and most code/weights are unreleased or of unknown licence.

---

## 4. What matters for weld

### 4.1 Small objects in huge images
* Median object is about 6 % of the image side. On a 5712x4284 photo that is roughly 250-340 px; after a naive downscale to 640 px it would be about 40 px, and at a stride of 32 only a ~1-2 cell footprint. Thin or small beads suffer first.
* **Tiling + ROI re-detection** (what `TiledDetector` does): detect on overlapping tiles, pool candidates, then re-detect on a zoomed crop around each candidate and drop candidates that are not reconfirmed at `verify_conf`. This turns "tiny object in a huge image" into "medium object in a 640-px window".
* **Stride-4 / P2 heads** keep small-object detail in the feature pyramid. They cost compute (4x the cells of P3) but are the standard fix when tiling is not enough. SFW-YOLO's weld paper adds a P2 layer for this reason ([link](https://www.sciencedirect.com/science/article/abs/pii/S0263224125009674)). Whether it helps here is untested.
* Tile boundaries cut objects; overlap (0.35 here) and the extra full-image pass compensate, at the price of duplicates that must be merged.

### 4.2 Rotated boxes and angle periodicity
* The crop for the classifier is cut along the oriented box, so angle error directly changes what the classifier sees.
* Angle regression has discontinuities (periodicity, and width/height swap ambiguity). Losses that model the box as a **2D Gaussian** (GWD / KLD / ProbIoU family) are smooth across the boundary and are what this repo's ProbIoU-style loss relies on. Native OBB heads in YOLO26 and RiO-DETR also address this explicitly.
* The bundled detector's IoU >= 0.9 agreement is low (18 %) even though IoU 0.5 agreement is high (75 %): the localisation is roughly right but loose. If your classifier is sensitive to crop tightness, evaluate at IoU 0.7 as well as 0.5.

### 4.3 Few labelled images
* About 1,000 photos / 7.2k boxes is small. Families that start from a **pretrained backbone** (ImageNet, COCO, DOTA, DINOv2/v3) usually need far fewer labels than a from-scratch network. The bundled detector is from-scratch, which is why data efficiency matters more here than for a pretrained-backbone family.
* Practical options: (a) initialise a backbone from a public pretrained model if its licence fits; (b) stronger augmentation (below); (c) hard-negative mining from `last_rejected_candidates` (the `collect_rejected_candidates` hook in `TiledDetector`).

### 4.4 Glare, low contrast, metal surfaces
* Photometric augmentation (brightness/contrast/gamma, CLAHE, specular highlights, blur, JPEG, white balance) and test-time augmentation (`tta_scales`, `tta_flip` in `TiledDetector`) are the cheap, model-agnostic remedies.
* Weld papers consistently report that very faint / low-contrast targets remain the hardest case (e.g. the WELD-DETR authors say so for faint defects). Do not expect a bigger detector to fix lighting problems that the capture setup creates.

### 4.5 Recall-first cascades
* Stage 1 should run at a **low** `conf` and favour recall; `verify_conf` (zoomed re-detection) and the classifier absorb the false positives. That is why precision is about 51 % in the table above.
* Consequence for model selection: compare detectors by **recall at the operating point where the cascade's precision is acceptable**, not by COCO/DOTA mAP. A model with a better mAP can still have worse recall at a fixed low threshold.

### 4.6 Score calibration
* Detector scores differ systematically between models and training recipes. `conf` and `verify_conf` therefore **do not transfer** when you swap detectors. Re-derive them on a validation set, for example by choosing the threshold that gives the target recall, then check the number of candidates per image.
* One-to-one (NMS-free) heads and DETR-style heads tend to produce sparse, peaky score distributions; one-to-many heads with NMS give denser scores. A low threshold behaves very differently in the two cases, so recheck the candidate count after swapping.
* Note that the bundled `DetRunner` keeps an optional rotated NMS (`nms_iou`, default 0.7) on top of the one-to-one head; the NMS-free claim refers to the head design, and the NMS can be switched off with `nms_iou=None`.

---

## 5. Guidance

### 5.1 When to use this repo's detector
* You need an **Apache-2.0, dependency-light, CPU/Mac-runnable** oriented detector with no third-party model code or pretrained weights.
* You can accept a v0.1 preview: recall and box tightness as reported in the model card, thresholds to retune.
* You will fine-tune it on your own site data (it is small: 8.7 M parameters), and you keep a human in the loop.

### 5.2 When to plug in another detector
* You have **a licence-compatible pretrained oriented detector** (for example RTMDet-R or an Ultralytics OBB model under an Enterprise licence, or your own) and want better recall / tighter boxes quickly.
* You need to work on very different imagery (other materials, camera, scale) and prefer to start from a pretrained backbone.
* You only need axis-aligned boxes downstream (the classifier does not depend on the angle): then RF-DETR, D-FINE or RT-DETR become viable.
* If you do plug one in, **re-tune** `conf`, `verify_conf`, tile size and overlap, and rebuild the reference bank (its embeddings depend on the crops).

### 5.3 Plug-in contract (`TiledDetector` backend)
`TiledDetector(backend=...)` accepts any object with:

```python
class MyBackend:
    def predict(self, bgr: np.ndarray, conf: float) -> tuple[np.ndarray, np.ndarray]:
        """bgr: HxWx3 uint8 (OpenCV order), any size up to the tile size, or the full image.
        returns (polygons, confs)
          polygons: (N, 4, 2) float32, the 4 corners of each oriented box, in the pixel
                    coordinates of the *input* image, corners in order around the box
          confs:    (N,) float32 in [0, 1], one score per polygon, all >= conf
        """

    # optional: faster batch path; images in a chunk share one shape
    def predict_batch(self, images: list[np.ndarray], conf: float) -> list[tuple[np.ndarray, np.ndarray]]:
        ...
```

Facts about how it is called (from `src/pqdai/core/tiled_detector.py`):
* `TiledDetector` passes its own `conf` (may be as low as 0.1 for extract-all) as the threshold. The backend must return **everything above `conf`**, not a pre-thresholded top-k at its own default.
* It reshapes whatever you return with `reshape(-1, 4, 2)` / `reshape(-1)`, casts to float32, so empty detections can be `np.zeros((0, 4, 2))` and `np.zeros(0)`.
* Inputs are: the whole image if both sides are <= `tile_threshold` (1600 px), otherwise `slice_size` (1024) px tiles, plus one full-image pass, plus zoomed ROI crops for verification. **Resizing/letterboxing to the model's input size is the backend's job**, and the polygons must be mapped back to the input image's coordinates (see `DetRunner._run` for the letterbox inverse).
* Cross-tile duplicates are merged downstream (NMS at IoU 0.99 on axis-aligned envelopes, then WBF), but NMS **within** a tile is the backend's job unless the model is NMS-free.
* An axis-aligned detector can satisfy the contract by returning the four corners of each box; the crop will then not be rotated.

Wiring (no change to the pipeline code is required):

```python
from pqdai.core.tiled_detector import TiledDetector
det = TiledDetector(backend=MyBackend(), conf=0.1, verify_conf=0.2)
detections = det.detect("photo.jpg")   # [Detection(polygon (4,2), confidence)]
```

To use it through a named domain, return that `TiledDetector` from the domain's `detector_factory`; see [DOMAIN_PLUGINS.md](DOMAIN_PLUGINS.md).

### 5.4 What experiment is still missing
A fair answer to "which detector is best for our weld photos" needs the following, none of which has been run:

1. The same **split** (train / val / test by workpiece, not by photo - photos of the same part leak otherwise) and the same labels for every candidate.
2. Each candidate **fine-tuned with its own recommended recipe** at a matched compute budget, from its public pretrained weights where licensing allows, plus the from-scratch bundled detector.
3. Each candidate run **through `TiledDetector`** with identical tiling, so the tiling/ROI logic is held constant.
4. Metrics from section 6, at a threshold chosen on the validation split to reach the same recall target, with several seeds (the v0.1 numbers above come from one run on 99 photos and carry sampling noise of a few points).
5. End-to-end effect on the **classifier**: BAD recall and false-alarm rate of the full pipeline, since the detector only matters through what it hands to the classifier.

---

## 6. How to benchmark your detector on your data

This is the evaluation idea used for the numbers in the model card: match detections to labels, report recall/precision at IoU 0.5 and 0.7, and time it.

**Setup**
* Hold out photos by **workpiece/session**. Keep at least ~100 photos. Report the count of photos and boxes.
* Labels as 4-corner polygons in the original image coordinates. Run every detector through the same `TiledDetector` settings.
* Use a **fixed, low `conf`** (for example 0.1) so recall is the operating point you actually deploy; also report the number of detections per image.

**Metrics** (per detector, over all held-out photos)

| Metric | Definition |
|---|---|
| recall@IoU t | share of label boxes matched by a detection with polygon IoU >= t (t = 0.5 and 0.7; optionally 0.9) |
| precision@IoU t | share of detections matched to a label |
| count error | per image, `N_det - N_label`: report mean absolute error and signed bias (important if the number of welds is itself a result) |
| latency | median and p95 seconds per image at a **fixed input resolution** (for example one 5712x4284 photo and one 1920x1080 frame), batch size as deployed, after warm-up, with `torch.cuda.synchronize()` around the timing; state hardware, precision, runtime |

**Matching:** one-to-one greedy matching sorted by descending confidence; a label can be used once. Use polygon IoU (Shapely is already a dependency of this repo).

```python
import numpy as np, time
from shapely.geometry import Polygon

def piou(a, b):
    pa, pb = Polygon(a), Polygon(b)
    if not (pa.is_valid and pb.is_valid): return 0.0
    u = pa.union(pb).area
    return pa.intersection(pb).area / u if u > 0 else 0.0

def match(dets, confs, gts, thr):
    """greedy one-to-one matching; returns (#matched, #dets, #gts)"""
    used, tp = set(), 0
    for i in np.argsort(-np.asarray(confs)):
        best, bj = 0.0, -1
        for j, g in enumerate(gts):
            if j in used: continue
            v = piou(dets[i], g)
            if v > best: best, bj = v, j
        if best >= thr: used.add(bj); tp += 1
    return tp, len(dets), len(gts)

# per photo: recall = tp / n_gt, precision = tp / n_det, count error = n_det - n_gt
# latency:   warm up 3 runs, then time det.detect(path) over >= 20 runs; report median and p95
```

**Reporting checklist**
* detector name and version, weights licence, parameter count, input size, `TiledDetector` settings (`tile_threshold`, `slice_size`, `overlap`, TTA, `conf`, `verify_conf`);
* recall/precision at 0.5 and 0.7, detections per image, count error;
* latency with hardware and precision, **per image at fixed resolution** (not per tile) and per tile;
* number of held-out photos / workpieces, number of seeds, and whether the held-out set overlaps training data in source (as in the v0.1 model card, where it does);
* the downstream pipeline result (classifier BAD recall, false-alarm rate) if you swap the detector in the full pipeline.

---

## 7. Sources

All retrieved **2026-10-20** through web search excerpts (direct page fetches were blocked in the authoring environment, so tables were read from search summaries; re-check the linked originals before relying on a number). Classification: V = vendor-reported, P = paper-reported, T = third-party.

| Item | Link | Kind |
|---|---|---|
| Ultralytics repo / YOLO26, YOLO11 tables | <https://github.com/ultralytics/ultralytics> | V |
| YOLO26 docs | <https://docs.ultralytics.com/models/yolo26> | V |
| Ultralytics OBB docs | <https://docs.ultralytics.com/tasks/obb> | V |
| YOLO26 paper | <https://arxiv.org/pdf/2509.25164> | P |
| Ultralytics licence page | <https://www.ultralytics.com/license> | V |
| YOLO26-obb speed reproduction issue | <https://github.com/ultralytics/ultralytics/issues/23689> | T |
| RF-DETR repo and docs | <https://github.com/roboflow/rf-detr>, <https://rfdetr.roboflow.com/> | V |
| RF-DETR paper | <https://arxiv.org/abs/2511.09554> | P |
| D-FINE repo and paper | <https://github.com/Peterande/D-FINE>, <https://arxiv.org/html/2410.13842v1> | V / P |
| DEIMv2 repo and paper | <https://github.com/Intellindust-AI-Lab/DEIMv2>, <https://arxiv.org/pdf/2509.20787> | V / P |
| DEIMv2 licence comparison | <https://robotinaction.tech/model-licenses.html> | T |
| RT-DETR / RT-DETRv2 repo and paper | <https://github.com/lyuwenyu/RT-DETR>, <https://arxiv.org/pdf/2407.17140> | V / P |
| RT-DETR HF model card (licence) | <https://huggingface.co/PekingU/rtdetr_r50vd/raw/main/README.md> | V |
| RTMDet paper | <https://ar5iv.labs.arxiv.org/html/2212.07784> | P |
| MMRotate (RTMDet-R, Oriented R-CNN) | <https://github.com/open-mmlab/mmrotate> | V |
| Oriented R-CNN paper | <https://arxiv.org/abs/2108.05699> | P |
| LSKNet repo | <https://github.com/zcablii/LSKNet> | V |
| RiO-DETR paper, repo, third-party review | <https://arxiv.org/abs/2603.09411>, <https://github.com/RicePasteM/RiO-DETR>, <https://pith.science/paper/2603.09411> | P / V / T |
| WELD-DETR | <https://pmc.ncbi.nlm.nih.gov/articles/PMC12656187> | P |
| RT-DETR weld framework (AEI 2025) | <https://www.sciencedirect.com/science/article/pii/S1474034625002113> | P |
| CGRT-DETR | <https://link.springer.com/article/10.1007/s11668-026-02383-z> | P |
| DSF-YOLO | <https://www.nature.com/articles/s41598-025-06811-2> | P |
| SFW-YOLO (P2 layer, steel bridge welds) | <https://www.sciencedirect.com/science/article/abs/pii/S0263224125009674> | P |

Internal references: [MODEL_CARD.md](MODEL_CARD.md), [LICENSING.md](LICENSING.md), [DOMAIN_PLUGINS.md](DOMAIN_PLUGINS.md), [architecture.md](architecture.md).
