from typing import List, Dict, Any

def derive_segments(
    word_alignments: List[Dict[str, Any]], 
    target_min: float = 5.0, 
    target_max: float = 10.0, 
    floor: float = 3.0, 
    ceiling: float = 12.0
) -> List[Dict[str, Any]]:
    """Groups consecutive aligned words into segments targeting 5-10s duration.
    
    Prefers to break segments at points where the alignment shows a natural gap (a pause/silence)
    between words, ensuring boundaries fall between words, never mid-word.
    
    Args:
        word_alignments: List of dicts with keys 'word', 'start_time', 'end_time', 'confidence'.
        target_min: Target minimum duration (sec) for a segment.
        target_max: Target maximum duration (sec) for a segment.
        floor: Hard minimum duration floor (sec).
        ceiling: Hard maximum duration ceiling (sec).
        
    Returns:
        List of segment dicts with keys:
        'segment_id', 'start_time', 'end_time', 'duration', 'normalized_text', 'alignment_confidence'.
    """
    segments = []
    start_idx = 0
    n_words = len(word_alignments)
    
    seg_count = 0
    while start_idx < n_words:
        t_start = word_alignments[start_idx]['start_time']
        candidates_target = []
        candidates_all = []
        
        # Look ahead for candidate split points
        for j in range(start_idx, n_words):
            t_end = word_alignments[j]['end_time']
            # Safeguard against negative or zero durations
            t_end = max(t_end, t_start + 0.01)
            duration = t_end - t_start
            
            if duration < floor:
                continue
            if duration > ceiling:
                break
                
            # The gap after word j is the difference between its end and next word's start
            if j + 1 < n_words:
                next_start = word_alignments[j+1]['start_time']
                gap = max(0.0, next_start - t_end)
            else:
                gap = 100.0  # End of audio is an ideal boundary
                
            candidate = (j, gap, duration)
            candidates_all.append(candidate)
            if target_min <= duration <= target_max:
                candidates_target.append(candidate)
                
        if candidates_target:
            # Choose candidate with the largest gap in the target duration window
            best_j, best_gap, best_dur = max(candidates_target, key=lambda x: x[1])
        elif candidates_all:
            # Choose candidate with the largest gap within the floor-ceiling window
            best_j, best_gap, best_dur = max(candidates_all, key=lambda x: x[1])
        else:
            # Fallback: if no candidates because of duration bounds, split at the last checked word
            best_j = start_idx
            for j in range(start_idx, n_words):
                t_end = max(word_alignments[j]['end_time'], t_start + 0.01)
                if t_end - t_start > ceiling:
                    break
                best_j = j
            best_dur = max(word_alignments[best_j]['end_time'], t_start + 0.01) - t_start
            
        # Build the segment dict
        segment_words = word_alignments[start_idx : best_j + 1]
        seg_text = " ".join([w['word'] for w in segment_words])
        seg_conf = sum([w['confidence'] for w in segment_words]) / len(segment_words)
        
        seg_id = f"segment_{seg_count:04d}"
        segments.append({
            "segment_id": seg_id,
            "start_time": t_start,
            "end_time": word_alignments[best_j]['end_time'],
            "duration": best_dur,
            "normalized_text": seg_text,
            "alignment_confidence": seg_conf
        })
        
        seg_count += 1
        start_idx = best_j + 1
        
    print(f"Segment derivation complete. Created {len(segments)} segments.")
    return segments
