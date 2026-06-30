# 01 — Repository Structure & Infrastructure
## Drive layout, central config system, Colab boilerplate, MLflow, checkpointing, parallel-run safety

> Read `00_overview_and_task_definition.md` first. This document specifies the **physical and logical structure** everything else plugs into. Treat every path and filename below as fixed, not illustrative — consistency of paths is what makes the MLflow + Drive + multi-account setup work (see Section 5).

---

## 1. Environment model

- **Local Ubuntu device**: where the repository is authored/edited (e.g. via an AI coding agent + git). This repo lives **inside the Google Drive–synced folder** on disk, so any edit made locally appears in Drive automatically, with no separate push/pull step.
- **Google Drive (5TB, shared across all Google accounts used)**: the single source of truth for code, data, checkpoints, and logs. Every Colab account mounts the *same* underlying Drive.
- **Google Colab (free tier, T4-class GPU, multiple accounts for parallel runs)**: pure compute. Colab's local disk (`/content/...` outside the Drive mount) is treated as **ephemeral scratch only** — nothing that needs to survive a disconnect is ever written there as its final location.

**Canonical project root (fixed, non-negotiable):**
```
/content/drive/MyDrive/az-asr-align-demo/
```
This exact string must be used in every notebook, every config file, and every MLflow tracking URI. Do not let it vary between notebooks or sessions — MLflow's file-based backend records absolute paths inside run metadata at creation time, and a mismatched mount path later breaks the ability to read back old runs.

---

## 2. Repository layout

```
az-asr-align-demo/
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── data_pipeline.yaml
│   ├── models/
│   │   ├── whisper.yaml
│   │   └── mms.yaml
│   └── experiments/
│       ├── whisper_frozen.yaml
│       ├── whisper_lora.yaml
│       ├── mms_frozen.yaml
│       └── mms_lora.yaml
├── src/
│   ├── config_loader.py
│   ├── audio_io.py
│   ├── text_normalization.py
│   ├── alignment.py
│   ├── segmentation.py
│   ├── cross_model_filter.py
│   ├── splitting.py
│   ├── dataset.py
│   ├── augmentation.py
│   ├── train_utils.py
│   ├── eval_utils.py
│   ├── mlflow_utils.py
│   └── viz.py
├── data/
│   ├── raw/                # downloaded audio (.wav) + raw caption file (.srt/.vtt)
│   ├── processed/          # normalized full transcript, alignment output
│   ├── segments/           # final audio clips (read-only once Notebook 0 completes)
│   └── splits/             # train.csv / val.csv / test.csv manifests (read-only)
├── notebooks/
│   ├── 00_data_pipeline.ipynb
│   ├── 01_finetune_whisper_frozen.ipynb
│   ├── 02_finetune_whisper_lora.ipynb
│   ├── 03_finetune_mms_frozen.ipynb
│   ├── 04_finetune_mms_lora.ipynb
│   └── 05_compare_results.ipynb        # optional, see 03_finetuning_notebooks_spec.md §7
├── runs/
│   ├── whisper_frozen/
│   │   ├── checkpoints/
│   │   ├── mlruns/
│   │   └── run_id.txt
│   ├── whisper_lora/      (same sub-structure)
│   ├── mms_frozen/        (same sub-structure)
│   └── mms_lora/          (same sub-structure)
└── reports/
    └── figures/           # all exported plots, one subfolder per notebook
```

**Rule:** each of the 4 experiment runs writes *exclusively* inside its own `runs/<experiment_name>/` subtree. No two notebooks ever write to the same file. `data/segments/` and `data/splits/` are written once by Notebook 0 and are **read-only** afterward — all 4 fine-tuning notebooks only read from them, which makes concurrent reads from multiple Colab accounts safe.

---

## 3. Central config system

- Format: **YAML**, loaded with a small loader in `src/config_loader.py` that merges layers in this fixed order (later overrides earlier):
  1. `configs/base.yaml` — global paths, random seed, project root, sample rate
  2. `configs/data_pipeline.yaml` — all Phase 1 parameters
  3. `configs/models/<model>.yaml` — architecture-specific fixed settings (checkpoint name, target modules for LoRA, freezing rules)
  4. `configs/experiments/<experiment>.yaml` — the one file that varies per notebook (which model config + which approach + this run's output paths)
- Every notebook's first code cell loads exactly one experiment config by name (e.g. `"whisper_lora"`) and gets back one merged config object. **No notebook should hardcode a parameter that already exists in a config file.**

**`configs/base.yaml` — required keys:**
```yaml
project_root: "/content/drive/MyDrive/az-asr-align-demo"
seed: 42
sample_rate: 16000
video_url: "https://www.youtube.com/watch?v=LPre3ILXY1k"
```

**`configs/data_pipeline.yaml` — required keys:**
```yaml
segment_min_duration_sec: 3
segment_target_duration_sec: [5, 10]
segment_max_duration_sec: 12
alignment_confidence_drop_percentile: 10
cross_model_agreement_drop_percentile: 10
qa_sample_size: 18
qa_sample_strata: 3            # early / middle / late thirds of the video
qa_problem_rate_retune_threshold: 0.25
split_ratios: [0.8, 0.1, 0.1]  # train, val, test
```

**`configs/experiments/*.yaml` — required keys (example for `whisper_lora.yaml`):**
```yaml
experiment_name: "whisper_lora"
model_config: "whisper"
approach: "lora"                # "frozen" | "lora"
output_dir: "runs/whisper_lora"
learning_rate: 2.0e-4
```
(`whisper_frozen.yaml` is identical except `approach: "frozen"`, `output_dir: "runs/whisper_frozen"`, `learning_rate: 5.0e-5`. The `mms_*.yaml` files mirror this but with `model_config: "mms"` — exact learning rates for all four are fixed in `03_finetuning_notebooks_spec.md`, Section 3.)

---

## 4. Standard Colab notebook boilerplate (first 2 cells, every notebook)

Describe, do not write as code — the implementing agent writes the actual cell:

**Cell 1 — Mount & install:**
1. Mount Google Drive at `/content/drive`.
2. `cd` to the canonical project root.
3. Install dependencies from `requirements.txt` (only the packages not already present in the Colab image — check before reinstalling to save time).
4. Add `src/` to `sys.path` so the shared modules are importable.

**Cell 2 — Load config:**
1. Import `config_loader`.
2. Load the merged config for this notebook's experiment name (Notebooks 1–4 only; Notebook 0 loads `base.yaml` + `data_pipeline.yaml` only, no experiment file).
3. Print the fully merged config so the run log shows exactly what ran.

**`requirements.txt` contents (package names; let pip resolve compatible versions against Colab's preinstalled torch):**
```
transformers
datasets
accelerate
peft
jiwer
mlflow
ctc-segmentation
yt-dlp
soundfile
librosa
pandas
matplotlib
pyyaml
evaluate
```

---

## 5. MLflow setup (and the Drive-pathing problem, solved)

**The problem this section exists to prevent:** MLflow's default file-based backend records absolute paths into each run's `meta.yaml` at creation time. If the Drive mount path differs across sessions (different account, different mount order) or if `runs/` ever gets moved, old runs become unreadable, and concurrent writers to one shared store can corrupt it.

**The fix, applied identically in every fine-tuning notebook:**

1. Set `MLFLOW_TRACKING_URI` to a path **inside that notebook's own isolated experiment folder**, never to a shared top-level `mlruns/`:
   ```
   file:///content/drive/MyDrive/az-asr-align-demo/runs/<experiment_name>/mlruns
   ```
2. Artifact storage uses the same isolated folder — do not point artifacts at a separate shared location.
3. Use the exact canonical project root from Section 1 every time. Never use a relative path, never use a symlink.
4. **Resume-safety:** at the start of each notebook run, check whether `runs/<experiment_name>/run_id.txt` exists.
   - If it exists, read the MLflow run ID from it and resume logging to that same run (`mlflow.start_run(run_id=...)`).
   - If it does not exist, start a new MLflow run and immediately write its run ID to `run_id.txt`.
   This means re-running the notebook after a Colab disconnect continues the same MLflow run instead of creating a duplicate.
5. Because each of the 4 experiments has its own isolated `mlruns/` folder, **running all 4 notebooks simultaneously across different Google accounts on the same shared Drive is safe** — there is no shared mutable file any two notebooks ever touch at the same time.

---

## 6. Checkpointing (training resumability)

1. Each fine-tuning notebook's training step uses `output_dir = runs/<experiment_name>/checkpoints`, written directly to Drive (never to local Colab disk).
2. Save a checkpoint every epoch; keep at most the 3 most recent (`save_total_limit=3`) to control storage.
3. At the start of the training cell, check whether `checkpoints/` already contains a checkpoint:
   - If yes, resume training from the latest checkpoint (do not redo already-completed steps).
   - If no, start fresh.
4. This combined with the MLflow run-id resume logic in Section 5 means a dropped Colab session loses **zero** training progress beyond the current unsaved step.

---

## 7. Parallel execution across multiple Google accounts (operational note)

Since all accounts mount the same Drive, true parallelism means **opening different notebook files** (e.g. account A opens `01_finetune_whisper_frozen.ipynb`, account B opens `03_finetune_mms_frozen.ipynb`) in separate Colab sessions — not opening the same notebook twice. Because each experiment's read/write footprint is fully isolated (Section 2's rule), this requires no extra coordination logic beyond what's already specified above.

---

## 8. Visualization output convention (applies to every notebook)

Every plot generated anywhere in this project is:
1. Saved as a PNG to `reports/figures/<notebook_name>/`
2. Logged to MLflow as an artifact (fine-tuning notebooks only — Notebook 0 has no MLflow run, just saves to `reports/figures/`)
3. Displayed inline in the notebook

This guarantees the figures survive independently of any single notebook's session state.
