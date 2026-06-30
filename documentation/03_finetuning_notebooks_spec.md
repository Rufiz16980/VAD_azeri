# 03 — Fine-Tuning Notebooks Spec
## `notebooks/01–04_finetune_*.ipynb` (shared template, 4 config-driven variants) + optional `05_compare_results.ipynb`

> Read the previous three documents first. The 4 fine-tuning notebooks share **one identical structure** (this document) and differ only through which experiment config they load (`configs/experiments/*.yaml`, defined in `01_repo_structure_and_infrastructure.md` §3). Do not write 4 divergent notebooks — write one template and instantiate it 4 times with different config names.

---

## 1. Cell order (identical across all 4 notebooks)

| # | Stage |
|---|---|
| 1 | Mount Drive, install deps, load experiment config |
| 2 | Set up MLflow (isolated per-experiment path, resume-safe) |
| 3 | Load train/val/test manifests (read-only) |
| 4 | Load pretrained model + processor for this notebook's architecture |
| 5 | Baseline zero-shot evaluation (before any training) |
| 6 | Apply this notebook's adaptation approach (frozen or LoRA) |
| 7 | Configure training arguments |
| 8 | Train (with checkpoint-resume) |
| 9 | Load best checkpoint, run final test-set evaluation |
| 10 | Generate and save all required visualizations |
| 11 | Print final summary block (with the generalization caveat) |
| 12 | Save final weights + run summary file |

---

## 2. Stage detail

### Stage 4 — Model loading
- Whisper notebooks: load `openai/whisper-small` + its processor (feature extractor + tokenizer).
- MMS notebooks: load `facebook/mms-1b-all` + processor, with the same Azerbaijani-adapter verification step described in `00_overview_and_task_definition.md` §5.

### Stage 5 — Baseline zero-shot evaluation
- Before touching any weights, run the untouched pretrained model on the **test** split.
- Apply the Stage-4 text normalization (from the data pipeline doc) identically to both model output and reference text before scoring.
- Compute WER and CER (use the `jiwer` library) → log as `baseline_wer`, `baseline_cer` to MLflow.
- Print explicitly: *"Baseline evaluated zero-shot. Any improvement reported later in this notebook should be read against this baseline, on a same-video holdout — see the generalization caveat in Stage 11."*

### Stage 6 — Adaptation approach (architecture-specific, per `00_overview_and_task_definition.md` §6)

| Notebook | Frozen | Trainable |
|---|---|---|
| `whisper_frozen` | Entire audio encoder | Decoder + LM head |
| `whisper_lora` | Entire base model | LoRA adapters on `q_proj`, `v_proj` in encoder **and** decoder attention, `r=8`, `alpha=16`, `dropout=0.05` |
| `mms_frozen` | Conv feature encoder + all transformer layers except the top 2 | Top 2 transformer layers + CTC head |
| `mms_lora` | Entire base model | LoRA adapters on `q_proj`, `v_proj` in the transformer encoder's attention, `r=8`, `alpha=16`, `dropout=0.05` |

- After applying freezing or LoRA injection, **print the trainable-vs-total parameter count** for the run. This number is itself part of the demo's story (LoRA should show a dramatically smaller trainable count than the frozen approach) and must be logged to MLflow as a parameter (`trainable_params`, `total_params`).
- **LoRA module-name verification step (not a decision):** before setting `target_modules`, the agent should print `model.named_modules()` (or equivalent) for whichever architecture is loaded and confirm `q_proj`/`v_proj` exist under those exact names for that model's attention blocks. If the names differ for MMS's attention implementation, use the actual names found — do not silently assume Whisper's naming applies unchanged.

### Stage 6b — Architecture-correct regularization (replaces a one-size-fits-all "SpecAugment everywhere" approach)
This project's source material originally suggested applying SpecAugment uniformly to both models. That doesn't transfer cleanly, because the two architectures don't share an input representation (§5 of the overview doc). The correct, architecture-specific approach:

- **Whisper (operates on log-mel spectrograms):** apply SpecAugment to the extracted log-mel features during training only (never during evaluation). Parameters: frequency masking max width `F=10`, time masking max width `T=20`, `num_freq_masks=2`, `num_time_masks=2`. Implement via a custom data collator that masks the feature tensor after extraction, before it reaches the model.
- **MMS (operates on raw waveform via a learned CNN front-end):** classic mel-band SpecAugment does not apply here. Instead, use the regularization mechanism wav2vec2-family models already provide internally: set `mask_time_prob` and `mask_feature_prob` in the model config to `0.05` each during fine-tuning. This is the architecturally-correct analog — do not attempt to bolt mel-spectrogram-style masking onto a raw-waveform model.

### Stage 7 — Training arguments (loaded from model YAML configs under `training:` key)

| Parameter | Frozen approach | LoRA approach |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Learning rate | `5e-5` | `2e-4` |
| Weight decay | `0.01` | `0.01` |
| Batch size | Driven by `training.per_device_train_batch_size` (`8` Whisper / `4` MMS) | Driven by config (`8` Whisper / `4` MMS) |
| Gradient accumulation steps | Driven by `training.gradient_accumulation_steps` (`2` Whisper / `4` MMS) | Driven by config (`2` Whisper / `4` MMS) |
| Warmup ratio | `0.1` | `0.1` |
| Max epochs | `15` | `15` |
| Early stopping patience | `3` epochs (monitor val loss) | `3` epochs |
| Gradient clipping | `max_grad_norm = 1.0` | same |
| Mixed precision | `fp16=True` (T4 does not support bf16 efficiently) | same |
| Save strategy | every epoch, `save_total_limit=3` | same |
| Logging frequency | every 5 steps (fine-grained curves given the small dataset) | same |

*(Note: AdamW is used uniformly; the Muon optimizer is intentionally not used here — at this scale (LoRA / small frozen subset, single 20-minute video) optimizer choice is not the bottleneck, and Muon's demonstrated track record is in pretraining hidden layers, not fine-tuning Adam-pretrained foundation models. Not worth the added complexity for this demo.)*

### Stage 8 — Training loop, with checkpoint-resume
- Before starting, check `runs/<experiment_name>/checkpoints/` for an existing checkpoint (per `01_repo_structure_and_infrastructure.md` §6). Resume if found.
- Per-epoch: log train loss, validation loss, validation WER, validation CER to MLflow.
- Per-step (every 5 steps): log train loss for fine-grained curve resolution.

### Stage 9 — Final test evaluation
- Reload the checkpoint with the **lowest validation loss** (not necessarily the last one — early stopping may have continued a few epochs past the best point).
- Run on the test split, applying the same normalization as Stage 5.
- Log `final_test_wer`, `final_test_cer` to MLflow.

### Stage 10 — Required visualizations (every one of these, every notebook, saved + logged + displayed per `01_repo_structure_and_infrastructure.md` §8)
1. Train loss vs. training step
2. Train loss vs. validation loss, by epoch (the core overfitting diagnostic)
3. Validation WER and CER vs. epoch
4. Bar chart: baseline (zero-shot) WER/CER vs. final fine-tuned WER/CER, this run only
5. A small annotation/bar showing trainable-vs-total parameter count for this run

### Stage 11 — Final summary block (printed in the notebook, not just logged)
Must explicitly restate, every time: *"This run's test split is a 10% holdout from the same video used for training. Improvement over baseline here primarily reflects rapid adaptation to this speaker's voice and this recording's acoustic conditions, not a general improvement in Azerbaijani ASR. No independent held-out video was used in this project (an accepted, deliberate scope reduction — see `00_overview_and_task_definition.md` §8)."*

### Stage 12 — Save outputs
- Save final model weights (or, for LoRA, just the adapter weights) to `runs/<experiment_name>/checkpoints/final/`.
- Write a short JSON or Markdown run summary (`runs/<experiment_name>/run_summary.md`) containing: approach, architecture, baseline metrics, final metrics, trainable parameter count, and the caveat text from Stage 11.

---

## 3. Known risks to note in each notebook's markdown (not blockers, just documented)

- **Catastrophic forgetting** is not tested. Fine-tuning on ~18 minutes of one speaker's narrow-domain speech could degrade the base model's broader Azerbaijani/multilingual ability outside this video's domain — more of a risk for `*_frozen` (more trainable parameters) than `*_lora`. Out of scope for this demo; note it as a one-line caveat only.
- **LoRA tooling maturity differs by architecture.** Whisper + PEFT LoRA is a well-documented, common recipe. wav2vec2/MMS + PEFT LoRA is less commonly demonstrated in public examples — if `mms_lora` hits PEFT compatibility issues that `whisper_lora` doesn't, that's a known asymmetry in tooling maturity, not a bug in this spec.

---

## 4. Optional Notebook 5 — `05_compare_results.ipynb` (recommended addition beyond the original 4)

A short notebook that:
1. Reads `run_summary.md` (or the equivalent logged MLflow metrics) from all 4 `runs/*/` directories.
2. Produces one consolidated bar chart: baseline vs. fine-tuned WER (and separately CER) across all 4 runs, grouped by model (Whisper, MMS) and colored by approach (frozen, LoRA).
3. Produces one consolidated bar chart of trainable-parameter counts across all 4 runs (the LoRA-vs-frozen efficiency story in one image).
4. Prints the same generalization caveat from Stage 11 once, at the top, since it applies to every number on the page.

This notebook only reads already-produced artifacts — it performs no training and is safe to run any time after all 4 fine-tuning notebooks have completed at least one full pass.
