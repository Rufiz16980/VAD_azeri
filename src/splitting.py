import random
from typing import List, Dict, Any, Tuple

def stratified_split(
    segments: List[Dict[str, Any]], 
    split_ratios: List[float] = [0.8, 0.1, 0.1], 
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits segments into train/val/test splits (80/10/10 by default).
    
    Stratifies the split across three equal timeline blocks (early, middle, late) 
    to ensure all splits represent the entire video rather than localized sections.
    """
    if abs(sum(split_ratios) - 1.0) > 1e-5:
        raise ValueError("Split ratios must sum to 1.0")
        
    if not segments:
        return [], [], []

    # Find total video length using max end_time
    total_duration = max(seg["end_time"] for seg in segments)
    third_duration = total_duration / 3.0
    
    early_stratum = []
    middle_stratum = []
    late_stratum = []
    
    # Categorize segments by start time
    for seg in segments:
        t_start = seg["start_time"]
        if t_start < third_duration:
            early_stratum.append(seg)
        elif t_start < 2 * third_duration:
            middle_stratum.append(seg)
        else:
            late_stratum.append(seg)
            
    print(f"Stratification counts:")
    print(f"  - Early stratum: {len(early_stratum)} segments")
    print(f"  - Middle stratum: {len(middle_stratum)} segments")
    print(f"  - Late stratum: {len(late_stratum)} segments")
    
    train_segments = []
    val_segments = []
    test_segments = []
    
    # Instantiate random generator with seed for reproducibility
    rng = random.Random(seed)
    
    # Split each stratum and accumulate
    for stratum_name, stratum_list in [("Early", early_stratum), ("Middle", middle_stratum), ("Late", late_stratum)]:
        # Shuffle in-place
        shuffled = list(stratum_list)
        rng.shuffle(shuffled)
        
        n = len(shuffled)
        if n == 0:
            continue
            
        # Calculate indices
        idx_train = int(round(n * split_ratios[0]))
        idx_val = int(round(n * split_ratios[1]))
        
        # Ensure at least 1 element in train, val, and test if the stratum has enough elements (n >= 3)
        if n >= 3:
            if idx_train < 1:
                idx_train = 1
            if idx_val < 1:
                idx_val = 1
            # Adjust so test has at least 1 segment
            while idx_train + idx_val >= n:
                if idx_train > 1:
                    idx_train -= 1
                elif idx_val > 1:
                    idx_val -= 1
                else:
                    break
        elif n == 2:
            idx_train = 1
            idx_val = 1
        elif n == 1:
            idx_train = 1
            idx_val = 0
            
        stratum_train = shuffled[:idx_train]
        stratum_val = shuffled[idx_train : idx_train + idx_val]
        stratum_test = shuffled[idx_train + idx_val :]
        
        print(f"  - {stratum_name} split: train={len(stratum_train)}, val={len(stratum_val)}, test={len(stratum_test)}")
        
        train_segments.extend(stratum_train)
        val_segments.extend(stratum_val)
        test_segments.extend(stratum_test)
        
    print(f"Final dataset splits:")
    print(f"  - Train split: {len(train_segments)} segments")
    print(f"  - Val split: {len(val_segments)} segments")
    print(f"  - Test split: {len(test_segments)} segments")
    
    return train_segments, val_segments, test_segments
