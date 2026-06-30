# 00 — Project Overview & Task Definition
## Azerbaijani ASR: Forced-Alignment Dataset Construction + Fine-Tuning Ablation Demo

> **Audience note:** This document set (4 files) is written for an implementing coding agent that has **zero prior context** on this project. Do not assume the agent has read any prior conversation. Every design decision below is final — the agent's job is to implement exactly what is specified, not to choose between alternatives. Where a value must be verified at runtime rather than hardcoded (e.g. a model's exact language code), this is explicitly flagged as a **verification step**, not a design decision.

---

## 1. What this project actually is

This is a **learning/demonstration project**, not a production system and not a research contribution. The person commissioning it has a CS background but **zero prior experience with speech processing** — their prior ML work was CNN image classification and XGBoost tabular regression. This is their first exposure to:

- Audio/speech as a data modality
- Forced alignment
- ASR model architectures (encoder-decoder vs CTC)
- Parameter-efficient fine-tuning (LoRA vs targeted full fine-tuning)
- Experiment tracking (MLflow) and resumable training pipelines

**Success criterion for this project is a working, well-instrumented, clearly-explained end-to-end pipeline — not a state-of-the-art WER number.** Do not optimize for accuracy at the expense of clarity, modularity, or robustness to interruption. If a choice trades off "slightly better metric" against "simpler to understand / more robust to a dropped Colab session," choose the latter.

---

## 2. Correct terminology (use this consistently in code comments, notebook markdown, and any generated reports)

| Phase | What's actually happening | Correct term |
|---|---|---|
| Phase 1 (Notebook 0) | Pretrained models used purely for **inference** to build a labeled dataset from one video | **Automated forced-alignment-based ASR dataset construction** |
| Phase 2 (Notebooks 1–4) | Models are **trained** on the Phase-1-generated pairs | **Parameter-efficient fine-tuning** (this is where "weakly supervised" becomes an accurate description, since training now happens on noisy/heuristic labels) |

Do not call Phase 1 alone "weakly supervised speech recognition" — no learning happens in that phase, only inference and alignment.

---

## 3. Source material

- **One YouTube video**, ~20 minutes, single speaker, sitting and talking directly to camera (interview-style, no second speaker, no music/effects expected).
- **URL:** `https://www.youtube.com/watch?v=LPre3ILXY1k`
- **Captions:** Azerbaijani, confirmed **not** auto-generated (YouTube's caption menu does not append "(auto-generated)" to the language label, which is how YouTube distinguishes machine captions from uploaded ones).

**Standing caveat — carry this through every phase of the project:** even manually-authored captions are not guaranteed to be verbatim. Caption authors routinely drop filler words, false starts, and clean up phrasing for reading speed. This pipeline does **not** algorithmically correct for that. The human spot-check in Phase 1 (Section 6 of `02_data_pipeline_notebook_spec.md`) exists specifically to catch gross cases of this — it is a coarse go/no-go gate, not a transcription-quality guarantee. All WER/CER numbers produced anywhere in this project should be reported with the caveat: *"measured against caption text, which may itself contain minor edits relative to verbatim speech."*

---

## 4. Two-phase structure

### Phase 1 — Dataset construction (1 notebook: `00_data_pipeline.ipynb`)
Extract audio + captions → normalize text → forced-align full audio to full text using a CTC model → derive 5–10s segments from the alignment (not from raw silence detection) → cross-check segment quality using agreement between two independent models → human spot-check → split into train/val/test → package as the frozen, reusable dataset for Phase 2.

### Phase 2 — Fine-tuning ablation (4 notebooks + 1 optional comparison notebook)
Two pretrained model families (Whisper, MMS) × two adaptation strategies (targeted/frozen, LoRA) = 4 independent training runs, each producing baseline-vs-fine-tuned WER/CER and full training diagnostics. An optional 5th notebook aggregates all 4 results into one comparison view.

---

## 5. The two model families, and why this pairing is meaningful

| | Whisper | MMS |
|---|---|---|
| Architecture | Attention-based Transformer, encoder-decoder | CTC-based, wav2vec2-style encoder only |
| Loss | Cross-entropy (next-token prediction) | CTC loss (frame-to-character alignment) |
| Input representation | Log-mel spectrogram (hand-crafted features) | **Raw waveform** — a learned convolutional feature encoder, no hand-crafted spectrogram step |
| Why included | Demonstrates how a general multilingual foundation model handles a low-resource language it saw little of | Has an Azerbaijani-specific checkpoint; CTC models are the natural tool for forced alignment, which is why MMS also does double duty as the Phase-1 aligner |

> **Correction to earlier draft material:** an earlier draft of this plan stated that *all* audio must be converted to a mel-spectrogram before a model can use it. That is true for Whisper but **not** for MMS — wav2vec2-family models consume the raw waveform directly through a learned CNN front-end. Treat "feature extraction" as architecture-specific, not universal. This has downstream consequences for how augmentation is applied in Phase 2 (see `03_finetuning_notebooks_spec.md`, Section 4).

**Model checkpoints (fixed, not a decision point):**
- Whisper: `openai/whisper-small` (244M params — fits comfortably on a free-tier T4 with room for LoRA/frozen fine-tuning).
- MMS: `facebook/mms-1b-all`, using the Azerbaijani-language adapter.
  - **Verification step (not a design decision):** MMS uses per-language adapter weights selected by a language code. Before loading, the agent must enumerate the model's supported target languages (via the processor/tokenizer's adapter vocabulary, not from memory) and select the Azerbaijani entry programmatically — do not hardcode a guessed code. If more than one Azerbaijani-related entry exists (e.g. variants for different scripts), prefer the Latin-script entry, since the source captions are in Latin script.

---

## 6. The two adaptation approaches — correctly redefined per architecture

The original idea ("freeze the encoder, fine-tune the rest" vs. LoRA) only maps cleanly onto Whisper, which has a real encoder/decoder split. MMS has no decoder to freeze-and-spare. To keep the ablation a fair, like-for-like comparison, **"targeted/frozen" is redefined per architecture** as follows. This redefinition is final — do not improvise a different split.

| Approach | Whisper | MMS |
|---|---|---|
| **A — Targeted/Frozen** | Freeze the entire audio encoder. Fine-tune the decoder + LM head only. | Freeze the convolutional feature encoder and all but the top 2 transformer layers. Fine-tune the top 2 layers + the CTC head. |
| **B — LoRA** | Inject LoRA adapters into the attention projection matrices (`q_proj`, `v_proj`) of both encoder and decoder. Base model frozen otherwise. | Inject LoRA adapters into the attention projection matrices (`q_proj`, `v_proj`) of the transformer encoder. Base model frozen otherwise. |

This gives **4 notebooks**: `whisper_frozen`, `whisper_lora`, `mms_frozen`, `mms_lora`.

---

## 7. What "success" looks like for this demo

1. A working pipeline that produces a clean, segment-level Azerbaijani audio/text dataset from one video, with visualized quality diagnostics at every filtering step.
2. Four trained model variants, each reporting: zero-shot baseline WER/CER → fine-tuned WER/CER, full train/val loss curves, and trainable-parameter-count comparisons (frozen vs LoRA).
3. **Honest framing, stated explicitly in every notebook's output, not just in this document:** because validation/test data comes from a 10% holdout of the *same* video, any WER/CER improvement primarily reflects rapid speaker- and channel-adaptation, not general-purpose Azerbaijani ASR improvement. This project deliberately does not include a second, independently-recorded held-out video — that's an explicit, accepted scope reduction for this demo, not an oversight.
4. (Optional, recommended) One consolidated comparison view across all 4 runs — useful for presenting the ablation result in one glance. See `03_finetuning_notebooks_spec.md`, Section 7.

---

## 8. Known and accepted limitations — do not "fix" these, they are intentionally out of scope

- Single speaker, single recording session, presumed clean studio/near-field audio — no diarization, no noise-robustness testing.
- No second held-out video for true generalization testing (Section 7.3 above).
- No catastrophic-forgetting evaluation against a broader Azerbaijani benchmark.
- No automated hyperparameter search (Optuna etc.) — fixed defaults are specified in `03_finetuning_notebooks_spec.md` and are final.
- Caption fidelity is checked only by a small human spot-check, not a rigorous transcription audit.

---

## 9. Document map

| File | Covers |
|---|---|
| `00_overview_and_task_definition.md` | This file — context, terminology, scope |
| `01_repo_structure_and_infrastructure.md` | Repository layout, config system, Drive/Colab/MLflow setup, checkpointing, parallel-run safety |
| `02_data_pipeline_notebook_spec.md` | Full cell-by-cell spec for `00_data_pipeline.ipynb` |
| `03_finetuning_notebooks_spec.md` | Full spec for the 4 fine-tuning notebooks + optional comparison notebook |
