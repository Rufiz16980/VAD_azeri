# Run Summary: whisper_lora
- **Approach**: lora
- **Baseline Metrics**: WER=0.5190, CER=0.1786
- **Fine-tuned Metrics**: WER=0.4810, CER=0.1594
- **Trainable Parameters**: 884,736 (Total: 242,619,648)

## Generalization Caveat
This run's test split is a 10% holdout from the same video used for training.
Improvement over baseline here primarily reflects rapid adaptation to this speaker's voice
and this recording's acoustic conditions, not a general improvement in Azerbaijani ASR.
