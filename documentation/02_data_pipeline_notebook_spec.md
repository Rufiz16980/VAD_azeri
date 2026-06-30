# 02 — Data Pipeline Notebook Spec
## `notebooks/00_data_pipeline.ipynb` — full cell-by-cell specification

> Read `00_overview_and_task_definition.md` and `01_repo_structure_and_infrastructure.md` first. This notebook implements all of Phase 1. Output is the frozen, read-only dataset that all 4 fine-tuning notebooks consume. All parameters referenced below live in `configs/data_pipeline.yaml` (see previous document) — do not hardcode them inline.

---

## 1. Cell order (high level)

| # | Stage |
|---|---|
| 1 | Mount Drive, install deps, load config |
| 2 | Download audio + captions from the source video |
| 3 | Parse captions → build full reference transcript |
| 4 | Text normalization (Azerbaijani-specific) |
| 5 | Tokenizer/vocabulary coverage diagnostic |
| 6 | Load MMS model/processor for alignment |
| 7 | Full-audio forced alignment via `ctc-segmentation` |
| 8 | Derive 5–10s segments from alignment output |
| 9 | Visualize segmentation quality |
| 10 | Zero-shot decode every segment with both Whisper and MMS |
| 11 | Cross-model agreement filtering |
| 12 | Human spot-check export (pause point) |
| 13 | Train/val/test split |
| 14 | Package final dataset + write summary report |

---

## 2. Stage detail

### Stage 2 — Extraction
- Use `yt-dlp` to download:
  - Best-quality audio only, converted to **16kHz mono WAV** (the sample rate both Whisper and MMS expect) → `data/raw/audio.wav`
  - The Azerbaijani subtitle track, in `.vtt` or `.srt` (whichever `yt-dlp` provides for this track) → `data/raw/captions.vtt`
- Since these are confirmed non-auto-generated captions, the "rolling/duplicated caption block" artifact common to YouTube's live auto-captions is **not expected** — but the parser in Stage 3 should still defensively skip exact-duplicate consecutive caption blocks in case any exist, rather than assuming the file is perfectly clean.

### Stage 3 — Caption parsing
- Parse the caption file into an ordered list of `(start_time, end_time, text)` triples.
- Concatenate all text fields, in order, into one full reference transcript string. Preserve sentence-level boundaries (insert a single space or newline between caption blocks — be consistent, document the choice in code comments) since this full string is what gets forced-aligned against the full audio in Stage 7.
- Keep the original per-block timestamps too (saved separately) — they are **not** used as ground-truth segment boundaries (that would reproduce the "VAD-first" mistake this design avoids), but they are useful as a sanity-check reference when visualizing alignment quality later.

### Stage 4 — Text normalization (Azerbaijani-specific — do not skip any of these steps)
Apply, in this exact order, to the full reference transcript:
1. **Turkic case-folding fix.** Standard `str.lower()` is locale-naive and will incorrectly collapse Azerbaijani's dotted/dotless İ–I distinction (it maps both `İ` and `I` to ASCII `i`, destroying a real phonemic distinction: `İ/i` = dotted, `I/ı` = dotless). Apply an explicit character mapping **before** any other lowercasing: `İ → i`, `I → ı`, then lowercase everything else normally. Do not rely on Python's default `.lower()` or `.casefold()` for this step.
2. Strip punctuation (`. , ! ? ; : ' " … ( ) —` etc.), but preserve all Azerbaijani-specific letters as-is: `ə ğ ı ö ü ş ç` (and their uppercase forms, before step 1 is applied).
3. Collapse multiple whitespace characters into a single space; strip leading/trailing whitespace.
4. **Leave digits as written** (do not spell them out) — digit-format mismatches are a known, accepted minor source of WER/CER noise for this project, not worth solving here.
- Save the normalized full transcript to `data/processed/transcript_normalized.txt`. Keep the raw (pre-normalization) version too, for the human spot-check stage later.

### Stage 5 — Tokenizer/vocabulary coverage diagnostic
- Tokenize the normalized full transcript with Whisper's tokenizer.
- Compute and print/plot: average subword tokens per word, and a histogram of tokens-per-word across the transcript.
- This is descriptive only — it tells you whether Azerbaijani morphology is fragmenting heavily into subwords (expected, given Whisper's tokenizer wasn't trained on much Azerbaijani data), but requires no action. Include the resulting histogram in the final report (Stage 14).

### Stage 6 — Load MMS for alignment
- Load `facebook/mms-1b-all` and its processor.
- Per the verification step in `00_overview_and_task_definition.md` §5: programmatically select the Azerbaijani (Latin-script) language adapter from the model's supported target languages — do not hardcode a guessed language code.

### Stage 7 — Forced alignment via `ctc-segmentation`
- Run the MMS model once over the full audio to obtain per-frame log-probabilities (not a greedy/beam decode — the raw CTC output distribution is what `ctc-segmentation` needs).
- Feed the log-probabilities, the model's character vocabulary, and the **full normalized transcript** (Stage 4 output) into the `ctc-segmentation` library to obtain:
  - Word-level (or short-phrase-level) start/end timestamps across the entire 20-minute audio
  - A per-segment confidence score (provided directly by the library's output)
- This is the step that replaces naive VAD/silence-based chunking — segmentation in Stage 8 is derived from this alignment, not from raw amplitude.

### Stage 8 — Segment derivation
- Using the alignment output, group consecutive aligned words into segments targeting **5–10 seconds** (`segment_target_duration_sec`), with a hard floor of 3s and hard ceiling of 12s (`segment_min/max_duration_sec`).
- Prefer to break segments at points where the alignment shows a natural gap (a small silence/low-activity span between words in the alignment), so breaks fall between words/clauses, never mid-word.
- For each resulting segment, store: `start_time`, `end_time`, `duration`, the exact corresponding slice of normalized text, and an aggregated alignment confidence score (e.g. the mean of the per-word confidence scores it's built from).
- Cut the corresponding audio clips from `data/raw/audio.wav` into `data/segments/audio/<segment_id>.wav`.

### Stage 9 — Segmentation visualization
Produce and save (per Section 8 of the infra doc):
1. Histogram of segment durations (confirm most fall in the 5–10s target band)
2. Histogram/distribution of alignment confidence scores across all segments
3. A small number (e.g. 5) of example plots: waveform with the aligned word boundaries overlaid, for visual sanity-checking

### Stage 10 — Zero-shot decoding for cross-model agreement
- For every segment produced in Stage 8, run **both**:
  - Whisper (`openai/whisper-small`) zero-shot transcription
  - MMS zero-shot transcription (greedy/beam CTC decode this time — a real decode, not the raw log-probs used for alignment in Stage 7)
- This is a separate, independent decode pass from the alignment pass — do not reuse Stage 7's alignment output as if it were a transcription.

### Stage 11 — Cross-model agreement filtering
- For each segment, compute a normalized word-level edit distance between the Whisper output and the MMS output (apply the same Stage-4 text normalization to both before comparing).
- Convert to an agreement score (e.g. `1 - normalized_edit_distance`).
- Drop the bottom `cross_model_agreement_drop_percentile` (10%) of segments by this score — segments where the two architecturally different models substantially disagree are the least trustworthy as clean training pairs.
- Also separately apply `alignment_confidence_drop_percentile` (10%) from Stage 7/8's confidence score, dropping the lowest decile.
- Visualize: the agreement-score distribution before/after filtering, and print a summary table (segments in → segments kept → segments dropped, broken down by which filter caused the drop).

### Stage 12 — Human spot-check export (pause point)
- From the **surviving** (post-filter) segments, draw a stratified random sample of `qa_sample_size` (18) segments: split the video into `qa_sample_strata` (3) equal thirds by timestamp, sample an equal number from each third, using the fixed project seed.
- Export a simple review sheet (CSV or Markdown table) with columns: `segment_id`, path to the audio clip, the caption text for that segment, alignment confidence, agreement score, and an empty `verdict` column for the human to fill in (`OK` / `Problem`).
- **This is a manual go/no-go gate, not manual relabeling.** The human listens to each clip and checks only: does the audio roughly start/end where the text starts/ends, with no obvious mid-word cut? They do not correct or retype any text.
- **Notebook pauses here.** Print clear instructions for what to do with the exported sheet and that execution should not continue until the human has filled in the `verdict` column and confirmed.
- **Decision rule on return:** if more than `qa_problem_rate_retune_threshold` (25%) of the sampled segments are marked `Problem`, the correct response is to tighten `alignment_confidence_drop_percentile` and/or `cross_model_agreement_drop_percentile` (e.g. raise each from 10 to 20) in `configs/data_pipeline.yaml` and re-run from Stage 8 onward — not to manually edit any segment's text.

### Stage 13 — Train/val/test split
- Split the final, QA-passed segment set 80/10/10 (`split_ratios`).
- Stratify the split across the same early/middle/late thirds used in Stage 12, so all three splits are representative of the whole video rather than, say, the test set being entirely "early video" content.
- Use the fixed project seed for reproducibility.
- Write `data/splits/train.csv`, `val.csv`, `test.csv`, each listing `segment_id`, audio path, and normalized text.

### Stage 14 — Final packaging + summary report
- Confirm `data/segments/` and `data/splits/` are complete and treat them as **read-only** from this point on (no notebook in Phase 2 should ever write into these folders).
- Generate one summary report (Markdown, saved to `reports/figures/data_pipeline/summary.md`) containing: total usable audio duration, segment counts per split, average segment duration, vocabulary size, the tokens-per-word histogram from Stage 5, and the before/after filtering counts from Stage 11.

---

## 3. Output manifest schema (used by every fine-tuning notebook)

| Column | Description |
|---|---|
| `segment_id` | Unique identifier |
| `audio_path` | Path to the `.wav` clip under `data/segments/audio/` |
| `start_time`, `end_time`, `duration` | From the source video timeline |
| `normalized_text` | Stage 4 normalized transcript slice — this is the training target |
| `raw_text` | Pre-normalization text, kept for reference/debugging |
| `alignment_confidence` | From Stage 7/8 |
| `agreement_score` | From Stage 11 |
| `split` | `train` / `val` / `test` |
