# Released weights

`weights/<domain>/` contains `classifier.safetensors`, `detector.safetensors`, one `*.manifest.json` per model
(architecture, input format, class names, metrics, sha256, license, version) and an index `manifest.json`.

Files are stored with git-lfs (`git lfs pull`). Verify integrity with `pqdai verify-weights --domain weld`.
Override locations with `PQDAI_WELD_WEIGHTS=<dir>`, or per model with `--detector-weights` / `--classifier-weights`.

`weights/weld` currently holds an **interim** export (`"status": "interim"` in the manifests); final weights replace it.
