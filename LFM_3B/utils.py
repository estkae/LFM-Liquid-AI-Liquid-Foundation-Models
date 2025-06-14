"""
Utility functions for LFM-3B model
"""

import torch
import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

try:
    from .config import LFM3BConfig
    from .model import LFM3BForCausalLM, LFM3BModel
except ImportError:
    from config import LFM3BConfig
    from model import LFM3BForCausalLM, LFM3BModel


def save_model(
    model: Union[LFM3BModel, LFM3BForCausalLM],
    save_directory: str,
    config: Optional[LFM3BConfig] = None,
    safe_serialization: bool = True,
) -> None:
    """
    Save model weights and configuration to directory
    
    Args:
        model: The model to save
        save_directory: Directory to save the model
        config: Model configuration (if None, uses model.config)
        safe_serialization: Whether to save in safetensors format
    """
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    
    # Get config
    if config is None:
        config = model.config
    
    # Save configuration
    config_dict = config.__dict__
    with open(save_directory / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Save model weights
    if safe_serialization:
        try:
            from safetensors.torch import save_file
            state_dict = model.state_dict()
            save_file(state_dict, save_directory / "model.safetensors")
        except ImportError:
            print("safetensors not installed, falling back to PyTorch format")
            torch.save(model.state_dict(), save_directory / "pytorch_model.bin")
    else:
        torch.save(model.state_dict(), save_directory / "pytorch_model.bin")
    
    print(f"Model saved to {save_directory}")


def load_model(
    model_class: type,
    load_directory: str,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Union[LFM3BModel, LFM3BForCausalLM]:
    """
    Load model from directory
    
    Args:
        model_class: The model class to instantiate (LFM3BModel or LFM3BForCausalLM)
        load_directory: Directory containing the saved model
        device: Device to load the model on
        dtype: Data type for model weights
        
    Returns:
        Loaded model
    """
    load_directory = Path(load_directory)
    
    # Load configuration
    with open(load_directory / "config.json", "r") as f:
        config_dict = json.load(f)
    
    config = LFM3BConfig(**config_dict)
    
    # Initialize model
    model = model_class(config)
    
    # Load weights
    if (load_directory / "model.safetensors").exists():
        try:
            from safetensors.torch import load_file
            state_dict = load_file(load_directory / "model.safetensors")
            model.load_state_dict(state_dict)
        except ImportError:
            print("safetensors not installed, trying PyTorch format")
            state_dict = torch.load(load_directory / "pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state_dict)
    elif (load_directory / "pytorch_model.bin").exists():
        state_dict = torch.load(load_directory / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"No model weights found in {load_directory}")
    
    # Move to device and dtype
    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
    
    print(f"Model loaded from {load_directory}")
    return model


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
        "total_billions": total_params / 1e9,
        "trainable_billions": trainable_params / 1e9,
    }


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Get model size in megabytes
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_summary(model: Union[LFM3BModel, LFM3BForCausalLM]) -> None:
    """
    Print a summary of the model architecture and parameters
    
    Args:
        model: The model to summarize
    """
    config = model.config if hasattr(model, 'config') else model.model.config
    
    print("=" * 60)
    print("LFM-3B Model Summary")
    print("=" * 60)
    
    print("\nArchitecture:")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention Heads: {config.num_attention_heads}")
    print(f"  Experts: {config.num_experts} (top-{config.num_experts_per_token})")
    print(f"  Vocabulary: {config.vocab_size:,}")
    print(f"  Max Context: {config.max_position_embeddings:,}")
    
    print("\nLiquid Neural Network:")
    print(f"  Enabled: {config.use_liquid_layers}")
    print(f"  Reservoirs: {config.liquid_num_reservoirs}")
    print(f"  Reservoir Size: {config.liquid_reservoir_size}")
    print(f"  Update Interval: every {config.liquid_update_interval} layers")
    
    # Count parameters
    param_stats = count_parameters(model)
    print("\nParameters:")
    print(f"  Total: {param_stats['total']:,} ({param_stats['total_billions']:.2f}B)")
    print(f"  Trainable: {param_stats['trainable']:,} ({param_stats['trainable_billions']:.2f}B)")
    
    # Model size
    size_mb = get_model_size_mb(model)
    print(f"\nModel Size: {size_mb:.2f} MB")
    
    print("=" * 60)


def create_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    Create attention mask from input IDs
    
    Args:
        input_ids: Input token IDs
        pad_token_id: ID of padding token
        
    Returns:
        Attention mask (1 for real tokens, 0 for padding)
    """
    return (input_ids != pad_token_id).long()


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: The number of steps for the warmup phase
        num_training_steps: The total number of training steps
        last_epoch: The index of the last epoch when resuming training
        
    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)