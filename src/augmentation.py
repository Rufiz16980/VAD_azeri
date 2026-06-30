import random
import torch
from typing import Dict, List, Any, Union

class WhisperSpecAugmentCollator:
    """Custom data collator for Whisper that pads speech inputs and labels, 
    and applies SpecAugment (frequency and time masking) during training.
    """
    def __init__(
        self, 
        processor: Any, 
        F: int = 10, 
        T: int = 20, 
        num_freq_masks: int = 2, 
        num_time_masks: int = 2, 
        is_training: bool = True
    ):
        self.processor = processor
        self.F = F
        self.T = T
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.is_training = is_training

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different padding needs
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels with pad token
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 so that PyTorch cross-entropy loss ignores it
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is present at the beginning of all labels, strip it
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels

        # Apply SpecAugment masking during training only
        if self.is_training:
            x = batch["input_features"]  # Shape: [batch_size, num_features, seq_len]
            batch_size, num_features, seq_len = x.shape
            
            # Apply frequency masking
            for _ in range(self.num_freq_masks):
                w = random.randint(0, self.F)
                f0 = random.randint(0, num_features - w)
                if w > 0:
                    x[:, f0 : f0 + w, :] = 0.0
                    
            # Apply time masking
            for _ in range(self.num_time_masks):
                w = random.randint(0, self.T)
                t0 = random.randint(0, seq_len - w)
                if w > 0:
                    x[:, :, t0 : t0 + w] = 0.0
                    
            batch["input_features"] = x

        return batch
