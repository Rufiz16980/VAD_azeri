# Run Summary: whisper_frozen
- **Approach**: frozen
- **Baseline Metrics**: WER=0.5190, CER=0.1786
- **Fine-tuned Metrics**: WER=0.3418, CER=0.1322
- **Trainable Parameters**: 153,580,800 (Total: 241,734,912)

## Generalization Caveat
This run's test split is a 10% holdout from the same video used for training.
Improvement over baseline here primarily reflects rapid adaptation to this speaker's voice
and this recording's acoustic conditions, not a general improvement in Azerbaijani ASR.
