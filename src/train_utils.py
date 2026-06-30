import os
from typing import Any, Tuple, Optional
from peft import LoraConfig, get_peft_model

def get_last_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Scans the checkpoint directory and returns the path to the latest checkpoint if it exists.
    
    Returns None if no checkpoints are found.
    """
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoints = [
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    
    if not checkpoints:
        return None
        
    # Sort by the step number (e.g. checkpoint-1200 -> 1200)
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    print(f"Found existing checkpoints. Latest checkpoint resolved: '{latest_checkpoint}'")
    return latest_checkpoint


def count_parameters(model: Any) -> Tuple[int, int]:
    """Calculates and returns (trainable_parameters, total_parameters) for the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


def configure_adaptation(model: Any, approach: str, model_type: str) -> Tuple[Any, int, int]:
    """Configures parameter-efficient fine-tuning (PEFT/LoRA or Layer Freezing) on the model.
    
    Whisper:
      - Frozen: Freeze entire audio encoder. Decoders and LM head remain trainable.
      - LoRA: Inject LoRA adapters to attention projections ('q_proj', 'v_proj') of both encoder/decoder.
      
    MMS:
      - Frozen: Freeze feature encoder and all but the top 2 transformer layers. 
                Top 2 layers and CTC head remain trainable.
      - LoRA: Inject LoRA adapters to attention projections ('q_proj', 'v_proj') of the transformer encoder.
      
    Returns:
      (adapted_model, trainable_params, total_params)
    """
    approach = approach.lower()
    model_type = model_type.lower()
    
    if model_type == "whisper":
        if approach == "frozen":
            print("Configuring Whisper with Frozen Encoder approach...")
            # Freeze the entire audio encoder
            for param in model.model.encoder.parameters():
                param.requires_grad = False
            # Ensure decoder and LM head remain trainable
            for param in model.model.decoder.parameters():
                param.requires_grad = True
            for param in model.proj_out.parameters():
                param.requires_grad = True
                
        elif approach == "lora":
            print("Configuring Whisper with LoRA adaptation...")
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none"
            )
            model = get_peft_model(model, peft_config)
            
        else:
            raise ValueError(f"Unknown adaptation approach '{approach}' for model '{model_type}'")
            
    elif model_type == "mms":
        if approach == "frozen":
            print("Configuring MMS with top-2 layers and CTC head fine-tuning...")
            # 1. Freeze the feature extractor (convolutional encoder)
            for param in model.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
            # 2. Freeze the feature projection layers
            if hasattr(model.wav2vec2, "feature_projection"):
                for param in model.wav2vec2.feature_projection.parameters():
                    param.requires_grad = False
                    
            # 3. Freeze all transformer layers EXCEPT the top 2
            layers = model.wav2vec2.encoder.layers
            num_layers = len(layers)
            for i in range(num_layers - 2):
                for param in layers[i].parameters():
                    param.requires_grad = False
                    
            # Ensure top 2 transformer layers are explicitly trainable
            for i in range(num_layers - 2, num_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
                    
            # Ensure CTC LM head is trainable
            if hasattr(model, "lm_head"):
                for param in model.lm_head.parameters():
                    param.requires_grad = True
                    
        elif approach == "lora":
            print("Configuring MMS with LoRA adaptation...")
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none"
            )
            model = get_peft_model(model, peft_config)
            
        else:
            raise ValueError(f"Unknown adaptation approach '{approach}' for model '{model_type}'")
            
    else:
        raise ValueError(f"Unsupported model type '{model_type}'")
        
    trainable_params, total_params = count_parameters(model)
    print(f"Model configured successfully:")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Total parameters    : {total_params:,}")
    print(f"  - Percent trainable   : {100 * trainable_params / total_params:.4f}%")
    
    return model, trainable_params, total_params
