# Azerbaijani ASR Forced-Alignment and Fine-Tuning Ablation Pipeline

An end-to-end pipeline for constructing a custom Azerbaijani Automatic Speech Recognition (ASR) dataset from raw video and conducting fine-tuning ablation runs. This project compares full-layer freezing and Parameter-Efficient Fine-Tuning (PEFT/LoRA) on both autoregressive (Whisper) and Connectionist Temporal Classification (MMS) architectures.

---

## Table of Contents

- [Introduction](#introduction)
- [Core Concepts Explained](#core-concepts-explained)
  - [Automatic Speech Recognition (ASR)](#automatic-speech-recognition-asr)
  - [Voice Activity Detection (VAD) and Segmentation](#voice-activity-detection-vad-and-segmentation)
  - [Forced Alignment](#forced-alignment)
  - [Evaluation Metrics: WER and CER](#evaluation-metrics-wer-and-cer)
- [Task Definition and Methodology](#task-definition-and-methodology)
  - [Why Manually Written Subtitles?](#why-manually-written-subtitles)
  - [How Success is Measured](#how-success-is-measured)
  - [Why Ablation?](#why-ablation)
- [Model Architectures and Adaptation Approaches](#model-architectures-and-adaptation-approaches)
  - [Whisper-small (Encoder-Decoder)](#whisper-small-encoder-decoder)
  - [MMS-1B-All (Wav2Vec2 CTC)](#mms-1b-all-wav2vec2-ctc)
  - [Optimization and Training Setup](#optimization-and-training-setup)
- [Repository Structure](#repository-structure)
- [Setup and Infrastructure](#setup-and-infrastructure)
- [Ablation Results Summary](#ablation-results-summary)

---

## Introduction

This project implements a complete speech processing and model training pipeline. It starts with raw video and text subtitles, performs audio-text synchronization and quality filtering, splits the resulting dataset, and conducts a structured ablation study. The ablation study compares two models (OpenAI's Whisper and Meta's MMS) across two adaptation approaches (Layer Freezing and LoRA PEFT) to determine the most parameter-efficient and accurate method for speaker and acoustic adaptation.

---

## Core Concepts Explained

This section introduces the foundational speech processing and deep learning concepts utilized in this pipeline.

### Automatic Speech Recognition (ASR)

ASR is the process of converting an audio signal (containing spoken language) into written text. Modern deep learning architectures generally approach this in one of two ways:

1. **Connectionist Temporal Classification (CTC) Models**:
   CTC is an algorithm used to train sequence-to-sequence models where the input sequence (audio frames) is much longer than the output sequence (text characters). A CTC network outputs a probability distribution over the characters (including a special "blank" token) for every audio frame. The CTC decoding step then collapses repeated characters and removes blanks. CTC models do not have an autoregressive decoder, making them extremely fast to run but unable to capture language syntax as effectively as autoregressive models.
2. **Autoregressive Encoder-Decoder Models**:
   These models consist of an audio encoder that processes raw audio into representations, and a text decoder that generates text tokens one by one (autoregressively). The decoder utilizes self-attention over the generated text and cross-attention over the encoder's representations. While computationally heavier, these models excel at transcription because the decoder acts as an implicit, powerful language model.

### Voice Activity Detection (VAD) and Segmentation

VAD is the process of identifying regions of an audio signal that contain human speech versus silence or background noise. In practical speech pipelines:
- Processing hours of continuous audio is memory-prohibitive.
- VAD is used to segment continuous audio streams into short, manageable speech clips (typically 2 to 15 seconds long).
- These slices serve as the individual training samples for our neural networks.

### Forced Alignment

Forced alignment is the process of automatically synchronizing a text transcript with its corresponding audio file by determining the exact start and end times of each spoken word or phoneme.
- Standard ASR datasets require paired audio-text clips. Continuous transcripts do not have timestamp information.
- Forced alignment maps the transcript characters to the acoustic frames of the audio using algorithms like Dynamic Time Warping (DTW) or CTC-based segmentation.
- This creates clean, millisecond-accurate segments for downstream model training.

### Evaluation Metrics: WER and CER

We evaluate ASR model transcriptions using two standard metrics based on Levenshtein distance:

1. **Word Error Rate (WER)**:
   WER measures the percentage of words that are incorrectly transcribed compared to the reference text:
   $$\text{WER} = \frac{S + D + I}{N}$$
   Where $S$ is the number of substitutions, $D$ is deletions, $I$ is insertions, and $N$ is the total number of words in the reference transcript.

2. **Character Error Rate (CER)**:
   CER uses the same formula but operates at the individual character level. It is highly useful for morphologically rich or low-resource languages where spelling variations can skew word-level metrics.

---

## Task Definition and Methodology

The goal of this task is to perform rapid speaker adaptation for Azerbaijani ASR, starting from raw audio and ground-truth text transcripts.

### Why Manually Written Subtitles?

Autogenerated subtitles contain spelling mistakes, lack punctuation, and omit grammatical structures. To train or adapt an ASR model, we need high-quality ground-truth targets. We chose a video with manually written subtitles because they provide:
- High-quality grammatical and orthographic representations of the spoken content.
- Correct spelling of specialized terminology and proper names.
- A cleaner starting point than autogenerated captions, minimizing alignment errors while acknowledging that manual subtitles may still omit filler words or contain minor paraphrasing (which is subsequently flagged via coarse spot-check verification rather than directly corrected).

### How Success is Measured

Success is evaluated along two dimensions:
- **Accuracy Improvement**: The absolute and relative reduction in WER and CER on the test holdout set compared to the zero-shot baseline of the pre-trained models.
- **Parametric Efficiency**: The number of trainable parameters required to achieve the accuracy gain. A method is successful if it achieves a low error rate while tuning as few parameters as possible.

### Why Ablation?

An ablation study systematically disables or modifies components of a system to understand their individual contribution. Here, we conduct an ablation study across:
- **Model Architectures**: Whisper (Encoder-Decoder) vs. MMS (CTC).
- **Adaptation Techniques**: Layer Freezing (tuning a subset of base layers) vs. PEFT LoRA (injecting low-rank adapters).
This allows us to analyze the trade-offs between parameter size, compute requirements, and error rate reductions.

---

## Model Architectures and Adaptation Approaches

We evaluate two state-of-the-art models using two distinct adaptation strategies.

### Whisper-small (Encoder-Decoder)

OpenAI's Whisper is a sequence-to-sequence model trained on 680,000 hours of multilingual and multitask speech data. It processes audio features using a convolutional encoder followed by a transformer encoder and generates text using a transformer decoder.
- **Frozen Encoder Approach**: The entire audio encoder is frozen. Only the decoder and language modeling projection heads are allowed to train. This forces the model to leverage its pre-trained acoustic representations while adapting its text generator to Azerbaijani grammar and pronunciation.
- **LoRA Approach**: Low-Rank Adaptation (LoRA) freezes the entire base model and injects trainable rank-8 decomposition matrices into the attention projection layers (`q_proj`, `v_proj`) of both the encoder and decoder.

### MMS-1B-All (Wav2Vec2 CTC)

Meta's Massively Multilingual Speech (MMS) is a Wav2Vec2-based model pre-trained on over 1,400 languages. It utilizes a convolutional feature extractor followed by a transformer encoder and a CTC projection head.
- **Frozen Approach**: The feature extractor and all but the top 2 layers of the transformer encoder are frozen. Only the top 2 transformer layers and the CTC head are allowed to train.
- **LoRA Approach**: Trainable low-rank projection matrices (`q_proj`, `v_proj`) are injected into the transformer encoder self-attention blocks, leaving the 1-Billion parameter base weights completely frozen.

### Optimization and Training Setup

To train the models stably under hardware constraints, we set up:
- **Optimizer**: AdamW with a linear learning rate warmup and decay.
- **Precision**: Half-precision floating-point format (FP16) to reduce memory usage and accelerate matrix multiplications on CUDA cores.
- **VRAM Constraints**: For the 1-Billion parameter MMS model, we enabled **Gradient Checkpointing** and reduced the batch size to 4 with 4 gradient accumulation steps. This keeps the peak memory footprint within Colab's T4 GPU limits while maintaining an effective batch size of 16.
- **PEFT Saving Integration**: Because Wav2Vec2 speech models process raw audio and lack a standard text embedding lookup table, they raise a `NotImplementedError` when PEFT tries to call `get_input_embeddings()`. We configured the `Wav2Vec2ForCTC` class to return `None` for input embeddings, allowing checkpoints to save successfully.

---

## Repository Structure

```
├── archive/              # Ignored folder containing legacy or outdated files
├── configs/              # Central configuration YAML files
│   ├── base.yaml         # Project-wide data and video URLs
│   ├── data_pipeline.yaml# Segmentation, VAD, and alignment parameters
│   └── experiments/      # Model-specific training configurations
├── data/                 # Raw and processed datasets (ignored except splits/)
│   └── splits/           # Train, validation, and test CSV manifest files
├── documentation/        # Detailed technical specification documents
├── notebooks/            # Executable notebook pipeline
│   ├── 00_data_pipeline.ipynb        # Downloads, aligns, filters, and splits data
│   ├── 01_finetune_whisper_frozen.ipynb# Whisper frozen encoder fine-tuning
│   ├── 02_finetune_whisper_lora.ipynb  # Whisper LoRA fine-tuning
│   ├── 03_finetune_mms_frozen.ipynb    # MMS frozen encoder fine-tuning
│   ├── 04_finetune_mms_lora.ipynb      # MMS LoRA fine-tuning
│   └── 05_compare_results.ipynb        # Consolidation and dashboard visualization
├── reports/              # Figures, charts, and QA worksheets
│   └── figures/          # Exported training curves and comparison plots
├── runs/                 # Training runs, MLflow logs, and summaries (checkpoints ignored)
├── src/                  # Helper Python source modules
└── requirements.txt      # Python dependencies manifest
```

---

## Setup and Infrastructure

The pipeline is designed to run in a hybrid Google Colab and Google Drive environment:
- **Storage**: Google Drive is mounted inside Google Colab (using `%cd /content/drive/MyDrive/az-asr-align-demo/`). This absolute path must remain fixed and non-negotiable because MLflow bakes absolute directory paths into run metadata at creation time.
- **Compute**: Standard Google Colab GPU runtimes (specifically Nvidia Tesla T4 GPU instances with 15-16 GB VRAM) are used for model training and evaluation.
- **Environment Syncing**: Runtimes are verified for package dependencies (such as NumPy 1.x compatibility with `ctc-segmentation`). If conflicts occur, the setup script uninstalls incompatible versions and programmatically restarts the Python kernel (via `os.kill(os.getpid(), 9)`) to apply changes automatically.

---

## Ablation Results Summary

The ablation study yielded the following performance metrics on the Azerbaijani test holdout set:

| Model | Adaptation Approach | Baseline WER | Fine-Tuned WER | Trainable Parameters | Total Parameters | % Trainable |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Whisper-small** | Frozen Encoder | 51.90% | 34.18% | 153,580,800 | 241,734,912 | 63.53% |
| **Whisper-small** | LoRA (`q_proj`, `v_proj`) | 51.90% | 48.10% | 884,736 | 242,619,648 | 0.36% |
| **MMS-1B-All** | Frozen (top-2 layers + CTC) | 38.61% | 35.44% | 52,646,630 | 964,738,246 | 5.46% |
| **MMS-1B-All** | **LoRA (`q_proj`, `v_proj`)** | 38.61% | **27.85%** | 1,966,080 | 966,704,326 | **0.20%** |

### Key Observations
- **MMS LoRA** achieved the best absolute performance (**27.85%** WER), requiring only **1.97 Million** parameters to adapt the 1-Billion parameter base model (a 27.87% relative improvement over zero-shot).
- **Whisper Frozen** showed the largest absolute improvement, dropping WER from 51.90% to **34.18%** (a 34.14% relative improvement), proving that its decoder layer parameters are highly receptive to transcription style adaptation.
- **Generalization Caveat**: Because the test set is a 10% holdout split from the same continuous video recording as the training data, these improvements primarily reflect rapid acoustic, channel, and speaker adaptation rather than a general performance boost for open-domain Azerbaijani ASR.
- The results demonstrate that the data pipeline successfully extracted and filtered aligned audio segments suitable for this targeted speaker adaptation task.


