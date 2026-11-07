# Changelog

## 0.1.1
- Detector weights updated (`weights/weld/detector.safetensors`): fine-tuned from the v0.1.0 detector with Pearson-correlation feature distillation (same `small` architecture, 8.69M parameters). Box agreement with the previous pipeline improved (IoU 0.7: 64.0 % -> 71.6 %, IoU 0.9: 18.1 % -> 29.2 %); recall at IoU 0.5 is 80.8 % (was 79.6 %, within seed noise) and precision is unchanged (51.2 %). The detector is still not at parity (reference recall 82.5 %).
- Classifier weights are unchanged from v0.1.0.
- No API or CLI changes.

## 0.1.0 (unreleased)
- First public structure: domain-agnostic core harness + `weld` as the first domain plugin.
- Pure-PyTorch classifier and oriented-box detector, released as safetensors with JSON manifests (interim weights).
- Reference bank module (`pqdai bank build|eval|info|merge`), unified `pqdai` CLI, image/video pipelines, web app.
