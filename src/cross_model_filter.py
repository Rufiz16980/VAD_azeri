import numpy as np
import jiwer
from typing import List, Dict, Any, Tuple
from transformers import pipeline
from text_normalization import normalize_azerbaijani

def get_asr_pipelines(device: str, target_lang: str) -> Tuple[Any, Any]:
    """Initializes and returns the Whisper and MMS speech recognition pipelines."""
    # Convert 'cuda' or 'cpu' device string to pipeline format (device index or -1)
    device_idx = 0 if "cuda" in device else -1
    
    print("Loading Whisper ASR pipeline (openai/whisper-small)...")
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device_idx
    )
    
    print(f"Loading MMS ASR pipeline (facebook/mms-1b-all) with adapter '{target_lang}'...")
    mms_pipe = pipeline(
        "automatic-speech-recognition",
        model="facebook/mms-1b-all",
        model_kwargs={"target_lang": target_lang, "ignore_mismatched_sizes": True},
        device=device_idx
    )
    
    return whisper_pipe, mms_pipe


def compute_agreement(whisper_text: str, mms_text: str) -> float:
    """Computes the agreement score between Whisper and MMS transcriptions.
    
    Normalizes both texts using Azerbaijani-specific normalization first.
    Calculates agreement as 1 - Word Error Rate (clamped between 0.0 and 1.0).
    """
    norm_whisper = normalize_azerbaijani(whisper_text)
    norm_mms = normalize_azerbaijani(mms_text)
    
    if not norm_whisper and not norm_mms:
        return 1.0
    if not norm_whisper or not norm_mms:
        return 0.0
        
    try:
        # jiwer.wer calculates word-level edit distance ratio (WER)
        wer = jiwer.wer(norm_whisper, norm_mms)
        return float(max(0.0, min(1.0, 1.0 - wer)))
    except Exception:
        # Fallback in case of unexpected empty lists or errors
        return 0.0


def apply_cross_model_filter(
    segments: List[Dict[str, Any]], 
    agreement_percentile: float = 10.0, 
    confidence_percentile: float = 10.0
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Applies decile filters based on cross-model agreement scores and alignment confidence scores.
    
    Discards the lowest deciles and categorizes dropped segments for the summary diagnostics.
    """
    agreement_scores = [seg["agreement_score"] for seg in segments]
    confidence_scores = [seg["alignment_confidence"] for seg in segments]
    
    # Calculate threshold values at the specified percentiles
    agreement_threshold = float(np.percentile(agreement_scores, agreement_percentile))
    confidence_threshold = float(np.percentile(confidence_scores, confidence_percentile))
    
    print(f"Filtering thresholds:")
    print(f"  - Agreement score threshold ({agreement_percentile}%): {agreement_threshold:.4f}")
    print(f"  - Alignment confidence threshold ({confidence_percentile}%): {confidence_threshold:.4f}")
    
    surviving_segments = []
    dropped_agreement_only = 0
    dropped_confidence_only = 0
    dropped_both = 0
    
    for seg in segments:
        agree_ok = seg["agreement_score"] >= agreement_threshold
        conf_ok = seg["alignment_confidence"] >= confidence_threshold
        
        if agree_ok and conf_ok:
            surviving_segments.append(seg)
        elif not agree_ok and not conf_ok:
            dropped_both += 1
        elif not agree_ok:
            dropped_agreement_only += 1
        else:
            dropped_confidence_only += 1
            
    stats = {
        "total_in": len(segments),
        "total_kept": len(surviving_segments),
        "dropped_agreement_only": dropped_agreement_only,
        "dropped_confidence_only": dropped_confidence_only,
        "dropped_both": dropped_both,
        "total_dropped": len(segments) - len(surviving_segments),
        "agreement_threshold": agreement_threshold,
        "confidence_threshold": confidence_threshold
    }
    
    return surviving_segments, stats
