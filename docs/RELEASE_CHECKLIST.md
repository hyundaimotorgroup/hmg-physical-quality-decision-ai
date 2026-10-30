# Release checklist (v0.1.0-preview)

Work through this list before tagging a release. Items marked `<<FILL>>` are placeholders in the tree that still need real values.

## Weights and metrics
- [x] Re-export the final weights with the private export tool (`tools/export_release.py`, `--status final --version 0.1.0`) into `weights/weld/`, replacing the interim files.
- [x] `pqdai verify-weights --domain weld` prints `OK` for every model (sha256 matches the manifests).
- [x] Refresh the numbers in `README.md`, `README.ko.md` and `docs/MODEL_CARD.md` (search for `<<FILL AFTER FINAL EXPORT>>`) and the metrics inside `weights/weld/*.manifest.json`.
- [ ] If the final detector uses an architecture preset that is not in `src/pqdai/models`, add the architecture definition so the manifest's `arch` name can be loaded.
- [ ] Run `pqdai video` with the final weights on a sample clip and compare the object count and GOOD / BAD result with the previous pipeline (not yet verified with real weights).

## Repository metadata
- [ ] Set the real repository owner: `OWNER` in `CITATION.cff` (`repository-code`), README badges, and the URLs in `pyproject.toml`.
- [ ] Set the copyright holder in `LICENSE` / `NOTICE` if it should name a specific legal entity (currently generic contributors).
- [ ] Fill in the security contact in `SECURITY.md` and the conduct contact in `CODE_OF_CONDUCT.md` if they are still placeholders.

## Packaging and distribution
- [ ] `git lfs install && git lfs track "*.safetensors"` (see `.gitattributes`); confirm `git lfs ls-files` lists both weight files.
- [ ] `git add` / first commit, then tag: `git tag v0.1.0-preview`.
- [ ] CI is green on the default branch (ruff + pytest on CPU).
- [ ] Fresh-clone check: clone into an empty directory with git-lfs installed, `pip install -e ".[web]"`, run the "Getting started in 5 minutes" commands from the README.

## Legal and safety
- [ ] Legal review of the licensing position in `docs/LICENSING.md`. The repository documents provenance factually (original architectures, no pretrained initialisation) but makes **no legal conclusion**. Have counsel confirm before publishing.
- [ ] Confirm that no customer imagery, internal paths, employee identifiers or internal project names are in the tree (`tests/test_public_surface.py` plus a manual look at `docs/` and `webapp/`).
- [ ] Re-read the intended-use and limitation text in `README.md` and `docs/MODEL_CARD.md`; the release must stay labeled as decision support, not a certified inspection system.
