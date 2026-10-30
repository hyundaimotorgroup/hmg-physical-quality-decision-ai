# Reference bank and the uncertainty gate

**Why.** A classifier can be confidently wrong on lighting/surface conditions it saw rarely. The harness routes only *uncertain* crops to a memory of labelled examples instead of trusting a single softmax.

**How it works** (`pqdai.core.pipeline`, `pqdai.core.reference_bank`):
1. The crop is preprocessed and shown to the classifier under five photometric views (gamma 0.85/1.15, brightness +-15); probabilities are averaged. This turns lighting fragility into entropy.
2. If entropy <= `alpha` (default 0.1; binary max is 0.693) the classifier's argmax is the verdict.
3. If entropy > `alpha` the crop's embedding (the un-jittered view) is compared by cosine similarity with every bank embedding; the top `k` (default 3) vote by hard majority. Ties go to class index 0, the unsafe class. The vote fraction is reported as the verdict's confidence.
4. `--no-retrieval` disables step 3 so you can see the classifier on its own.

**File format** - `.npz` schema 2: `embeddings (N,D) float32`, `labels (N,)`, `class_names`, `names` (relative basenames only, never paths), `schema`, `meta` (JSON: provenance, `built_for_model` = sha256 of the classifier it was built with, thresholds used). Legacy files (`embeddings`, `labels`, `image_paths`, `metadata`) still load; paths are reduced to basenames.

**Compatibility.** The bank must live in the deployed classifier's embedding space. The pipeline validates dimension and class names on start and warns when `built_for_model` differs from the deployed classifier. The bundled weld bank (494 embeddings, 247 BAD / 247 GOOD, 1280-d) was created with an earlier classifier build; the released classifier was trained to reproduce that embedding space (cosine ~0.95 on held-out crops), so it keeps working, but rebuilding the bank for your own site is recommended.

**Rebuild for your site**
```bash
# crops/BAD/*.jpg  crops/GOOD/*.jpg   (curated, certain examples only)
pqdai bank build --domain weld --crops crops/ --out my_bank.npz --balance
pqdai bank eval  --domain weld --bank my_bank.npz       # leave-one-out k-NN accuracy per class
pqdai bank merge a.npz b.npz --out merged.npz
pqdai image --domain weld --image photo.jpg --reference-bank my_bank.npz
```
Tips: use only examples you are sure of (the bank *overrides* the classifier on uncertain crops), keep classes balanced, near-duplicates are dropped at cosine >= 0.995 (`--dedup`, `--no-dedup`), and re-evaluate after every classifier update. No images are stored in the bank.
