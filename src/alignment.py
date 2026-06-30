import os
import torch
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2ForCTC, AutoProcessor
from ctc_segmentation import CtcSegmentationParameters, ctc_segmentation, determine_utterance_segments, prepare_text

def select_azerbaijani_adapter(processor) -> str:
    """Programmatically enumerates the tokenizer vocabulary keys to select the Azerbaijani Latin language code.
    
    Prefers 'azj' (North Azerbaijani, Latin) or any Azerbaijani code with Latin preference.
    """
    tokenizer = processor.tokenizer
    vocab = getattr(tokenizer, "vocab", {})
    if not vocab and hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
    vocab_keys = list(vocab.keys())
    
    # Azerbaijani ISO codes: azj (Latin), aze (general), azb (Arabic script)
    az_codes = [k for k in vocab_keys if k.startswith("az") or k.startswith("aze")]
    
    if not az_codes:
        raise ValueError("No Azerbaijani language adapter found in MMS tokenizer vocabulary.")
        
    print(f"Found Azerbaijani adapter codes in vocabulary: {az_codes}")
    
    # 1. Prefer 'azj' which is North Azerbaijani (Latin script)
    if "azj" in az_codes:
        selected = "azj"
    # 2. Check for anything with 'latn' or 'latin'
    else:
        latin_codes = [c for c in az_codes if "lat" in c.lower() or "latn" in c.lower()]
        if latin_codes:
            selected = latin_codes[0]
        else:
            # Fallback to the first Azerbaijani code
            selected = az_codes[0]
            
    print(f"Programmatically selected Azerbaijani adapter: '{selected}'")
    return selected


def get_mms_alignment(audio_path: str, normalized_text: str, device: str = "cuda"):
    """Loads MMS model, selects Azerbaijani adapter programmatically, runs inference in chunks,

    and performs CTC forced alignment to return word-level timestamps and confidence scores.
    """
    model_id = "facebook/mms-1b-all"
    
    print(f"Loading MMS model and processor: {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Programmatic adapter selection
    target_lang = select_azerbaijani_adapter(processor)
    
    # Load processor with selected adapter
    processor = AutoProcessor.from_pretrained(model_id, target_lang=target_lang)
    
    # Load model with target adapter
    model = Wav2Vec2ForCTC.from_pretrained(
        model_id, 
        target_lang=target_lang, 
        ignore_mismatched_sizes=True
    )
    
    # Load audio
    print(f"Loading audio file: {audio_path}...")
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        raise ValueError(f"Expected sample rate of 16000Hz, got {sr}Hz.")
        
    # Run chunked inference to prevent GPU OOM
    # At 16kHz, Wav2Vec2 downsamples by 320.
    sample_rate = 16000
    chunk_size_sec = 120
    overlap_sec = 5
    
    chunk_len = chunk_size_sec * sample_rate
    overlap_len = overlap_sec * sample_rate
    total_len = len(audio)
    
    log_probs_list = []
    
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device_obj)
    model.eval()
    
    step = chunk_len - overlap_len
    start = 0
    
    print("Running chunked inference to extract CTC log-probabilities...")
    while start < total_len:
        end = min(start + chunk_len, total_len)
        chunk_audio = audio[start:end]
        
        if len(chunk_audio) < sample_rate:
            break
            
        inputs = processor(chunk_audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device_obj) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs).logits
            # Log-probabilities: shape [seq_len, vocab_size]
            chunk_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[0].cpu().numpy()
            
        if len(log_probs_list) > 0:
            overlap_frames = int(overlap_len / 320)
            half_overlap = overlap_frames // 2
            
            # Stitch chunks by trimming overlap boundaries
            log_probs_list[-1] = log_probs_list[-1][:-half_overlap]
            chunk_log_probs = chunk_log_probs[half_overlap:]
            
        log_probs_list.append(chunk_log_probs)
        
        if end == total_len:
            break
        start += step
        
    log_probs = np.concatenate(log_probs_list, axis=0)
    print(f"Extracted CTC log-probabilities with shape {log_probs.shape}")

    # Set up CTC-Segmentation configuration
    vocab = processor.tokenizer.get_vocab()
    char_list = [token for token, idx in sorted(vocab.items(), key=lambda x: x[1])]
    
    config = CtcSegmentationParameters(char_list=char_list)
    config.index_duration = 320 / sample_rate  # 0.02s per frame
    
    # Select blank token index
    config.blank = processor.tokenizer.pad_token_id or 0
    
    # Determine the character representing space
    space_char = "|" if "|" in vocab else " "
    config.space = space_char
    
    # Prepare text: split transcript into words
    words_list = normalized_text.split()
    print(f"Aligning {len(words_list)} words against audio...")
    
    ground_truth_mat, utt_begin_indices = prepare_text(config, words_list)
    
    # Run alignment
    timings, char_probs, state_list = ctc_segmentation(
        config, 
        log_probs, 
        ground_truth_mat
    )
    
    # Determine segments
    raw_segments = determine_utterance_segments(
        config, 
        utt_begin_indices, 
        char_probs, 
        timings, 
        words_list
    )
    
    # Map raw segments back to words dictionary
    # raw_segments is a list of (start_time, end_time, confidence) tuples matching words_list
    word_alignments = []
    for i, (start, end, confidence) in enumerate(raw_segments):
        word_alignments.append({
            "word": words_list[i],
            "start_time": float(start),
            "end_time": float(end),
            "confidence": float(confidence)
        })
        
    print(f"Alignment completed. Successfully aligned {len(word_alignments)} words.")
    return word_alignments
