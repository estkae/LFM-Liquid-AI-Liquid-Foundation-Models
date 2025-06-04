import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lfm.model_v2 import create_lfm_model
from lfm.config import get_config
from loguru import logger


def main():
    # Set memory optimization flags
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Use smaller model for 8GB GPU
    model_name = "LFM-1B"  # Changed from LFM-7B
    logger.info(f"Creating {model_name} model...")
    model = create_lfm_model(model_name)
    
    # Print model configuration
    config = get_config(model_name)
    logger.info(f"Model Configuration:")
    logger.info(f"  - Hidden dimension: {config.hidden_dim}")
    logger.info(f"  - Number of layers: {config.num_layers}")
    logger.info(f"  - Number of attention heads: {config.num_heads}")
    logger.info(f"  - Number of experts: {config.num_experts}")
    logger.info(f"  - Intermediate dimension: {config.intermediate_dim}")
    logger.info(f"  - Vocabulary size: {config.vocab_size}")
    logger.info(f"  - Max sequence length: {config.max_position_embeddings}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nTotal parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e9:.2f}B)")
    
    # Estimate memory usage
    param_memory_gb = (total_params * 4) / (1024**3)  # 4 bytes per float32 parameter
    logger.info(f"Estimated parameter memory: {param_memory_gb:.2f} GB")
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Enable mixed precision for memory efficiency
    model = model.to(device)
    
    # Use smaller batch size and sequence length for testing
    batch_size = 1  # Reduced from 2
    seq_length = 128  # Can be reduced further if needed
    
    logger.info(f"\nTesting forward pass with batch_size={batch_size}, seq_length={seq_length}...")
    
    # Generate random input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    
    # Forward pass with mixed precision
    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        with torch.no_grad():
            loss, logits = model(input_ids)
    
    logger.info(f"Forward pass successful!")
    logger.info(f"Output logits shape: {logits.shape}")
    logger.info(f"Expected shape: [{batch_size}, {seq_length}, {config.vocab_size}]")
    
    # Memory usage after forward pass
    if torch.cuda.is_available():
        logger.info(f"\nGPU Memory Usage:")
        logger.info(f"  - Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        logger.info(f"  - Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        logger.info(f"  - Max Allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
    
    # Test generation with smaller sequence
    logger.info("\nTesting text generation...")
    model.eval()
    
    # Start with a prompt token
    prompt = torch.tensor([[1]], device=device)
    max_new_tokens = 20  # Reduced from 50
    
    generated = prompt
    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for the last position
                _, logits = model(generated)
                next_token_logits = logits[:, -1, :]
                
                # Greedy selection
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we hit an end token
                if next_token.item() == 2:
                    break
    
    logger.info(f"Generated sequence length: {generated.shape[1]}")
    logger.info(f"Generated token IDs: {generated[0].tolist()}")
    
    # Final memory usage
    if torch.cuda.is_available():
        logger.info(f"\nFinal GPU Memory Usage:")
        logger.info(f"  - Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        logger.info(f"  - Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        logger.info("\nCleared GPU cache")


if __name__ == "__main__":
    main()