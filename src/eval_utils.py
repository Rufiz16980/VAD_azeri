import jiwer
import numpy as np
from typing import Any, Dict, List
from text_normalization import normalize_azerbaijani

class ASRMetricEvaluator:
    """Computes Word Error Rate (WER) and Character Error Rate (CER) for ASR evaluations.
    
    1. Replaces padding label indices (-100) with pad token IDs.
    2. Argmaxes predictions if the model is CTC-based.
    3. Decodes token IDs back to text.
    4. Normalizes both predictions and labels using Azerbaijani case-folding.
    5. Calculates WER and CER via the jiwer library.
    """
    def __init__(self, tokenizer: Any, is_ctc: bool = False):
        self.tokenizer = tokenizer
        self.is_ctc = is_ctc

    def __call__(self, pred: Any) -> Dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Handle tuple of predictions (e.g. Whisper outputs logits as first element of a tuple)
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        # Convert logits (3D array of shape: batch_size, seq_len, vocab_size) to token IDs (2D) via argmax
        if hasattr(pred_ids, "ndim") and pred_ids.ndim == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)
        elif self.is_ctc:
            pred_ids = np.argmax(pred_ids, axis=-1)

        # Replace -100 with tokenizer pad token ID to allow clean decoding
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # Fallback if no pad token ID is set (e.g. Whisper tokenizer might use eos)
            pad_id = self.tokenizer.eos_token_id or 0
        label_ids = np.where(label_ids != -100, label_ids, pad_id)

        # Decode predictions and labels to string lists
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Apply Azerbaijani normalization to outputs before scoring
        norm_preds = [normalize_azerbaijani(p) for p in pred_str]
        norm_labels = [normalize_azerbaijani(l) for l in label_str]

        # Filter out empty reference sentences to avoid jiwer crashing on empty divisions
        clean_preds = []
        clean_labels = []
        for p, l in zip(norm_preds, norm_labels):
            if l.strip():
                clean_preds.append(p)
                clean_labels.append(l)
                
        if not clean_labels:
            # If no valid references, return zero errors
            return {"wer": 0.0, "cer": 0.0}

        # Calculate WER and CER
        wer = jiwer.wer(clean_labels, clean_preds)
        cer = jiwer.cer(clean_labels, clean_preds)

        return {
            "wer": float(wer),
            "cer": float(cer)
        }
